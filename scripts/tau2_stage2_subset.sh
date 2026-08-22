#!/bin/bash
# STAGE 2 of the arm-A ramp: 24 diagnostic episodes (8 tasks x 3 test domains) fanned out
# over all 8 GPUs of one node.
#
# Task ids come from scripts/tau2_select_subset.py, which strides the grouped task lists so
# the subset actually covers distinct scenarios -- `--num-tasks 8` would take a prefix and
# cover 1 of telecom's 3. Stage 1's retail task 7 is the selector's first retail pick, so
# stage 1 is a strict subset of this run.
#
# ONE PROCESS PER EPISODE, and that is not an accident. `connect()` builds DuplexSTTModel
# in-process, and the model is NOT released when an episode ends: the stage-1 run that
# re-ran an episode in-process went 79 GB -> 115 GB -> 135 GB of 143 GB and then collapsed
# ~20x into allocator thrashing. A second episode in the same process would leak the same
# way, so each episode gets a fresh process that takes its ~79 GB to the grave with it.
# The price is 24 model loads (~4.5 min each, ~1.8 GPU-h of the ~24 GPU-h total); the
# return is no leak, plus fault isolation -- one bad episode cannot take out its siblings.
#
# ONE --save-to PER EPISODE, also deliberate. Sharing a directory would route the second
# process into `try_resume`, which raises `ValueError: Tasks were removed from the task
# set` the moment the new `--task-ids` does not contain the previous run's task
# (checkpoint.py:145). Merge afterwards with scripts/tau2_merge_results.py, which
# concatenates the per-directory `Results` into one.
#
# WHY NOT `tau2 run --workers 8`, which is a real built-in controller/worker fan-out: it
# spawns workers with `env=os.environ.copy()` (controller.py:378) and has no per-worker GPU
# assignment, so all 8 would inherit the same CUDA_VISIBLE_DEVICES and pile 8 x 79 GB onto
# one card. Its workers are also long-lived across units, which is exactly the leak above.
# Fix those two things and this script should be deleted in favour of it.
#
# --hallucination-retries 0: the default is 3 and it re-runs the WHOLE episode, so a task
# can cost 4 episodes and OOM on the way (see above). Reviewing pre-SFT episodes for
# fabricated user content is not worth 4x on a checkpoint that barely speaks.
#
# COST: 24 episodes x 0.8-1.3 GPU-h = 19-31 GPU-h. Wall clock depends on how many slots the
# run gets: 8 slots on one node is 3 episodes per GPU and ~3-4 h; 15-16 slots across two
# nodes (see SLOT_OFFSET/TOTAL_SLOTS below) is 1-2 per GPU and ~1.2-2 h. Either fits an
# allocation many times over (7-day limit).
#
# QUOTA IS NOT A GATE. The ElevenLabs key was replaced with a paid one on 2026-08-21, and the
# measured cost is ~740 user-simulator characters per full 200 s episode (~18k for all 24) --
# trivial. Concurrency is the only live limit and it is 15, so a 16-way fan-out stays under it.
# Do not re-derive character budgets or route around ElevenLabs. If it ever does fail it fails
# LOUDLY (tts_retry reraises), and every completed episode is already on disk in its own
# directory, so nothing is lost.
#
# LAUNCH IT DETACHED. It holds 8 `srun --overlap` steps for hours, so it must not be tied to
# an interactive shell or an agent tool call:
#
#   setsid nohup scripts/tau2_stage2_subset.sh <jobid> >logs/stage2_launch.log 2>&1 </dev/null &
#
# And note the failure mode that is NOT this script's fault: every step dies with SIGTERM
# ("STEP <job>.<n> CANCELLED ... DUE to SIGNAL Terminated", rc=143) the moment the
# allocation ends, whoever ends it. Check `sacct -j <jobid>` before debugging the run --
# the first full-scale attempt was killed 3 minutes in by `CANCELLED by 0`,
# `Reason=AssocGrpNodeLimit`, which has nothing to do with the episodes.
#
# Usage: scripts/tau2_stage2_subset.sh <jobid> [extra tau2 run args...]

set -euo pipefail

JOBID="${1:?usage: $0 <jobid> [extra args...]}"
shift

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX=/fsx/home/kai.li/miniforge3/envs/voicechat

RUN="${RUN_NAME:-stage2_subset}"
CAP_SECONDS="${CAP_SECONDS:-200}"
NGPU="${NGPU:-8}"
# Two allocations, one run. Local slot k takes global slot (SLOT_OFFSET + k) of TOTAL_SLOTS,
# so 24 episodes split as 8 slots here (SLOT_OFFSET=0) and 8 on a second node
# (SLOT_OFFSET=8, TOTAL_SLOTS=16) with no overlap -- 1-2 episodes per GPU instead of 3, which
# is the difference between ~2 h and ~3-4 h of wall clock.
#
# Keep RUN the same on both nodes: --save-to is "$RUN/<domain>__<task>", one directory per
# episode, so both nodes populate one run that tau2_merge_results.py reads in a single pass.
# Give each node its own LOGDIR, or the two progress.log files collide -- and the truncation
# below would wipe the other node's record, which is exactly the "7 OKs counted twice" bug
# the comment there warns about.
SLOT_OFFSET="${SLOT_OFFSET:-0}"
TOTAL_SLOTS="${TOTAL_SLOTS:-$NGPU}"
# Slot k runs on GPU (GPU_OFFSET + k). Needed on a node where the VoiceChat NIM container
# already owns GPU 0 (fdb_v3_serve.sh pins CUDA_VISIBLE_DEVICES=0 and holds ~127 GB of 143):
# NGPU=7 GPU_OFFSET=1 uses GPUs 1-7 and leaves the server alone. An episode landing on GPU 0
# would OOM against the container rather than fail cleanly. MASTER_PORT is derived from the
# GPU index downstream, so offsetting the GPU keeps the ports collision-free too.
GPU_OFFSET="${GPU_OFFSET:-0}"
STAGGER_SECONDS="${STAGGER_SECONDS:-20}"
LOGDIR="${LOGDIR:-$HERE/../logs/$RUN}"

# Retry mode, set by scripts/tau2_watch_run.sh: run exactly the episodes in this TSV
# instead of the generated subset, keep the existing progress log, and write to distinct
# `--save-to` directories so a half-written first attempt is neither resumed nor
# overwritten (a same-named directory would route the retry into `try_resume`).
REQUEUE_FILE="${REQUEUE_FILE:-}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
if [[ -n "$REQUEUE_FILE" ]]; then
  NAME_SUFFIX="${NAME_SUFFIX:-__retry}"
fi

mkdir -p "$LOGDIR"
# Truncate rather than append: a stale progress.log from an aborted attempt makes the final
# tally read as successes that never happened (the first launch's 7 OKs survived into the
# second and were counted there). A retry pass is a continuation of the same run, so it
# appends instead -- losing the first pass's record is exactly the bug above in reverse.
if [[ -z "$REQUEUE_FILE" ]]; then
  : >"$LOGDIR/progress.log"
fi

# --- build the job list -----------------------------------------------------
# TSV rather than a shell array of ids: telecom ids contain `|` and `[]`
# (`[mms_issue]airplane_mode_on|break_app_both_permissions[PERSONA:Hard]`), so they must
# never pass through word splitting or eval. `read -r` with IFS=tab is the only safe path.
JOBS="$LOGDIR/jobs.tsv"
if [[ -n "$REQUEUE_FILE" ]]; then
  JOBS="$REQUEUE_FILE"
  echo "retry pass: $(wc -l <"$JOBS") episode(s) from $REQUEUE_FILE"
else
LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
  "$ENV_PREFIX/bin/python" - "$HERE" >"$JOBS" <<'PY'
import subprocess, sys, shlex
here = sys.argv[1]
out = subprocess.run(
    [sys.executable, f"{here}/tau2_select_subset.py", "--format", "args"],
    capture_output=True, text=True, check=True,
).stdout
for line in out.splitlines():
    if not line.strip():
        continue
    domain, rest = line.split(" ", 1)
    for task_id in shlex.split(rest):
        print(f"{domain}\t{task_id}")
PY
fi

N_JOBS=$(wc -l <"$JOBS")
MINE=0
for ((s = 0; s < NGPU; s++)); do
  for ((j = 0; j < N_JOBS; j++)); do
    ((j % TOTAL_SLOTS == SLOT_OFFSET + s)) && MINE=$((MINE + 1))
  done
done
echo "stage 2: $MINE of $N_JOBS episodes on this node ($NGPU GPUs, global slots" \
     "$SLOT_OFFSET..$((SLOT_OFFSET + NGPU - 1)) of $TOTAL_SLOTS), cap ${CAP_SECONDS}s," \
     "logs in $LOGDIR"

# --- fan out ----------------------------------------------------------------
# Static round-robin (job i -> global slot i % TOTAL_SLOTS) rather than a work queue. Episode
# costs are within ~1.6x of each other, and the domains are contiguous in the job list so
# round-robin also hands every slot a mix of the three. With TOTAL_SLOTS > N_JOBS/2 the split is
# uneven by construction (24 jobs over 15 slots = 2 each for slots 0-8, 1 for the rest), so give
# the node that frees up first the low SLOT_OFFSET.
declare -a WORKER_PIDS=()

# Read the whole job list into memory BEFORE any srun runs. This is a correctness fix, not a
# style preference. `srun` inherits and reads the worker's stdin, and with the loop written as
# `while read ... done <"$JOBS"` that stdin IS the job list -- so the first srun drains the file
# and `read` returns EOF on the next iteration. Every slot silently runs exactly one episode and
# exits 0. That cost 9 of 24 episodes on the 2026-08-21 stage-2 launch (airline 46 and all 8
# telecom) with both launchers reporting success. Also pass </dev/null to the child so it can
# never consume this shell's stdin again. Same fix, same reasoning, as tau2_stage2_gapfill.sh.
mapfile -t JOB_LINES <"$JOBS"

for ((slot = 0; slot < NGPU; slot++)); do
  (
    # Desynchronise the slots. Every episode otherwise reaches its first user utterance within
    # the same second; the paid key allows 15 concurrent requests so this is now belt-and-braces
    # rather than load-bearing, but it also spreads the ~79 GB model loads, and 20 s x 8 slots
    # costs 2.3 min of a multi-hour run.
    sleep $((slot * STAGGER_SECONDS))
    gpu=$((GPU_OFFSET + slot))
    for i in "${!JOB_LINES[@]}"; do
      ((i % TOTAL_SLOTS == SLOT_OFFSET + slot)) || continue
      line="${JOB_LINES[i]}"
      [[ -n "${line//[[:space:]]/}" ]] || continue
      IFS=$'\t' read -r domain task_id <<<"$line"

      # Directory name must be filesystem-safe: telecom ids are not.
      slug=$(printf '%s' "$task_id" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-80)
      name="$RUN/${domain}__${slug}${NAME_SUFFIX}"
      log="$LOGDIR/gpu${gpu}__${domain}__${slug}${NAME_SUFFIX}.log"

      echo "[gpu $gpu] START $domain $task_id" >>"$LOGDIR/progress.log"
      rc=0
      TAU2_DOMAIN="$domain" "$HERE/tau2_smoke_nemo.sh" "$JOBID" "$gpu" \
        --task-ids "$task_id" \
        --max-steps-seconds "$CAP_SECONDS" \
        --hallucination-retries 0 \
        --save-to "$name" \
        --log-level INFO \
        "$@" >"$log" 2>&1 </dev/null || rc=$?

      # Exit 0 is NOT enough. An episode that dies in setup is recorded as an
      # INFRASTRUCTURE_ERROR simulation, excluded from the metrics panel, and the batch then
      # reports "Successfully completed all simulations" and exits 0. The first stage-2
      # launch had 7 of 8 workers fail on a port collision and every one logged OK. So look
      # for the retry exhaustion line too; it is the thing that is actually diagnostic.
      # Deliberately does not abort the slot: the remaining episodes are independent, and a
      # partial stage 2 still answers the diagnostic question.
      if ((rc != 0)); then
        echo "[gpu $gpu] FAIL $domain $task_id rc=$rc (see $log)" >>"$LOGDIR/progress.log"
      elif grep -q "failed permanently after" "$log"; then
        echo "[gpu $gpu] INFRA $domain $task_id (see $log)" >>"$LOGDIR/progress.log"
      else
        echo "[gpu $gpu] OK $domain $task_id" >>"$LOGDIR/progress.log"
      fi
    done
  ) &
  WORKER_PIDS+=($!)
done

FAILED=0
for pid in "${WORKER_PIDS[@]}"; do
  wait "$pid" || FAILED=1
done

N_OK=$(grep -c ' OK ' "$LOGDIR/progress.log" || true)
N_FAIL=$(grep -c ' FAIL ' "$LOGDIR/progress.log" || true)
N_INFRA=$(grep -c ' INFRA ' "$LOGDIR/progress.log" || true)
echo "stage 2 fan-out finished: ${N_OK:-0} ok, ${N_INFRA:-0} infra, ${N_FAIL:-0} failed," \
     "of $MINE on this node ($N_JOBS in the run)"

# Dispatch accounting. ok+infra+failed summing to less than $MINE is the signature of the
# stdin-drain bug above, and it is the ONLY signal that catches it: every worker exits 0 and the
# summary line reads as a clean run. Coverage is the verdict, not the exit code.
N_START=$(grep -c ' START ' "$LOGDIR/progress.log" || true)
if ((${N_START:-0} != MINE)); then
  echo "WARNING: dispatched ${N_START:-0} of $MINE episodes -- $((MINE - ${N_START:-0})) were" \
       "NEVER STARTED. Do not treat this run as covering its episode list; diff the list" \
       "against $LOGDIR/progress.log and re-run the remainder." >&2
  FAILED=1
fi
echo "Merge with: scripts/tau2_merge_results.py --run $RUN"
exit "$FAILED"
