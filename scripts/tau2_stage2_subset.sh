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
# COST: 24 episodes x 0.8-1.3 GPU-h = 19-31 GPU-h. Statically round-robined 3 per GPU, so
# expect ~3-4 h of wall clock. Fits an allocation many times over (7-day limit).
#
# QUOTA: this is user-simulator ElevenLabs characters, ~200-2,000 per episode depending on
# how far the conversation gets. 6,228 remained on 2026-08-18, so 24 episodes may exhaust
# the free tier. A run that dies on quota fails LOUDLY (tts_retry reraises), and every
# completed episode is already on disk in its own directory, so nothing is lost.
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
LOGDIR="${LOGDIR:-$HERE/../logs/$RUN}"

mkdir -p "$LOGDIR"
# Truncate rather than append: a stale progress.log from an aborted attempt makes the final
# tally read as successes that never happened (the first launch's 7 OKs survived into the
# second and were counted there).
: >"$LOGDIR/progress.log"

# --- build the job list -----------------------------------------------------
# TSV rather than a shell array of ids: telecom ids contain `|` and `[]`
# (`[mms_issue]airplane_mode_on|break_app_both_permissions[PERSONA:Hard]`), so they must
# never pass through word splitting or eval. `read -r` with IFS=tab is the only safe path.
JOBS="$LOGDIR/jobs.tsv"
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

N_JOBS=$(wc -l <"$JOBS")
echo "stage 2: $N_JOBS episodes over $NGPU GPUs, cap ${CAP_SECONDS}s, logs in $LOGDIR"

# --- fan out ----------------------------------------------------------------
# Static round-robin (job i -> GPU i % NGPU) rather than a work queue. 24 jobs over 8 GPUs
# divides exactly, episode costs are within ~1.6x of each other, and the domains are
# contiguous in the job list so round-robin also hands every GPU a mix of the three.
declare -a WORKER_PIDS=()

for ((slot = 0; slot < NGPU; slot++)); do
  (
    i=0
    while IFS=$'\t' read -r domain task_id; do
      if ((i % NGPU != slot)); then
        i=$((i + 1))
        continue
      fi
      i=$((i + 1))

      # Directory name must be filesystem-safe: telecom ids are not.
      slug=$(printf '%s' "$task_id" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-80)
      name="$RUN/${domain}__${slug}"
      log="$LOGDIR/gpu${slot}__${domain}__${slug}.log"

      echo "[gpu $slot] START $domain $task_id" >>"$LOGDIR/progress.log"
      rc=0
      TAU2_DOMAIN="$domain" "$HERE/tau2_smoke_nemo.sh" "$JOBID" "$slot" \
        --task-ids "$task_id" \
        --max-steps-seconds "$CAP_SECONDS" \
        --hallucination-retries 0 \
        --save-to "$name" \
        --log-level INFO \
        "$@" >"$log" 2>&1 || rc=$?

      # Exit 0 is NOT enough. An episode that dies in setup is recorded as an
      # INFRASTRUCTURE_ERROR simulation, excluded from the metrics panel, and the batch then
      # reports "Successfully completed all simulations" and exits 0. The first stage-2
      # launch had 7 of 8 workers fail on a port collision and every one logged OK. So look
      # for the retry exhaustion line too; it is the thing that is actually diagnostic.
      # Deliberately does not abort the slot: the remaining episodes are independent, and a
      # partial stage 2 still answers the diagnostic question.
      if ((rc != 0)); then
        echo "[gpu $slot] FAIL $domain $task_id rc=$rc (see $log)" >>"$LOGDIR/progress.log"
      elif grep -q "failed permanently after" "$log"; then
        echo "[gpu $slot] INFRA $domain $task_id (see $log)" >>"$LOGDIR/progress.log"
      else
        echo "[gpu $slot] OK $domain $task_id" >>"$LOGDIR/progress.log"
      fi
    done <"$JOBS"
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
echo "stage 2 fan-out finished: ${N_OK:-0} ok, ${N_INFRA:-0} infra, ${N_FAIL:-0} failed, of $N_JOBS"
echo "Merge with: scripts/tau2_merge_results.py --run $RUN"
exit "$FAILED"
