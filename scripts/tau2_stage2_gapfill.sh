#!/bin/bash
# Run a named list of stage-2 episodes on explicitly named (jobid, gpu) placements.
#
# WHY THIS EXISTS: THE BUG IT WORKS AROUND
# `tau2_stage2_subset.sh` fans out one subshell per GPU and each subshell walks the whole
# job list off ITS OWN STDIN:
#
#     while IFS=$'\t' read -r domain task_id; do
#         ... tau2_smoke_nemo.sh ...        # <- runs srun
#     done <"$JOBS"
#
# `srun` reads stdin (it forwards it to the remote task), so the FIRST episode a slot runs
# drains the rest of jobs.tsv. Every slot therefore ran exactly ONE episode and its loop
# then hit EOF and exited cleanly. On 2026-08-21 that dropped 9 of the 24 stage-2 episodes
# -- airline 46 and all 8 telecom -- and both launchers reported rc=0. Node 1374 even
# printed "7 ok, 0 infra, 0 failed, of 8" without anyone noticing the 8th never started.
# This is the "exit 0 lies" trap again, one layer up: the *episode* accounting was right and
# the *dispatch* accounting was missing.
#
# The durable fix belongs in tau2_stage2_subset.sh (read the job list on fd 3 and give the
# child `</dev/null`). It is not applied yet because a slot of the previous launch is still
# executing that file, and bash re-reads a script it is mid-way through by byte offset --
# editing it in place would corrupt the tail of the running copy. Do it once the node is
# clear; this script covers the gap in the meantime and stays useful afterwards as the way
# to re-run a hand-picked set on hand-picked GPUs.
#
# HOW THIS ONE AVOIDS THE SAME BUG
# The job list is read into an array ONCE, before any srun exists, and every child is given
# `</dev/null`. Neither the outer loop nor the inner one depends on stdin surviving.
#
# PLACEMENTS ARE EXPLICIT, on purpose. The static round-robin in the main launcher assumes
# it owns whole nodes. Gap-filling never does: on 2026-08-21 GPU 0 of job 1374 held the
# VoiceChat NIM container (~127 GB) and GPU 7 of job 1403 was still running retail 106, so
# the free set was 1374:1-7 plus 1403:0-6 and no offset arithmetic expresses that. Listing
# `jobid:gpu` pairs also keeps MASTER_PORT collision-free, since that is derived from the
# GPU index and the pairs are unique per host.
#
#   PLACEMENTS='1374:1 1374:2 ...' scripts/tau2_stage2_gapfill.sh missing.tsv
#
# Jobs are handed to placements round-robin; with more placements than jobs each runs one.
# The TSV is `<domain>\t<task_id>`, tab-separated because telecom ids contain `|` and `[]`
# and must never see word splitting.
#
# Results land in the SAME run directory as the main launcher (`$RUN/<domain>__<slug>`, no
# suffix) because these episodes never ran -- there is no first attempt to preserve, and a
# suffix would make the scorer label them `airline__46__retry__46`. Do NOT point this at an
# episode that already has a directory: tau2 would route it into `try_resume`.

set -euo pipefail

JOBS="${1:?usage: PLACEMENTS='<jobid>:<gpu> ...' $0 <jobs.tsv> [extra tau2 run args...]}"
shift

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${RUN_NAME:-stage2_subset}"
CAP_SECONDS="${CAP_SECONDS:-200}"
LOGDIR="${LOGDIR:-$HERE/../logs/${RUN}_gapfill}"
STAGGER_SECONDS="${STAGGER_SECONDS:-20}"
: "${PLACEMENTS:?set PLACEMENTS to a space-separated list of <jobid>:<gpu>}"

read -r -a PLACE <<<"$PLACEMENTS"
# Read the whole job list up front. This is the fix, not a style choice -- see the header.
mapfile -t LINES <"$JOBS"
# Drop blank lines; a trailing newline in a hand-written TSV would otherwise dispatch an
# episode with an empty domain and burn a GPU on a tau2 usage error.
JOBLIST=()
for line in "${LINES[@]}"; do
  [[ -n "${line//[[:space:]]/}" ]] && JOBLIST+=("$line")
done

mkdir -p "$LOGDIR"
echo "gapfill: ${#JOBLIST[@]} episode(s) over ${#PLACE[@]} placement(s), cap ${CAP_SECONDS}s," \
     "run $RUN, logs in $LOGDIR"
for i in "${!JOBLIST[@]}"; do
  p="${PLACE[i % ${#PLACE[@]}]}"
  printf '  %s -> %s\n' "$p" "${JOBLIST[i]//$'\t'/ }"
done

declare -a WORKER_PIDS=()
for pi in "${!PLACE[@]}"; do
  (
    sleep $((pi * STAGGER_SECONDS))   # spread the ~79 GB model loads
    jobid="${PLACE[pi]%%:*}"
    gpu="${PLACE[pi]##*:}"
    for i in "${!JOBLIST[@]}"; do
      ((i % ${#PLACE[@]} == pi)) || continue
      line="${JOBLIST[i]}"
      domain="${line%%$'\t'*}"
      task_id="${line#*$'\t'}"

      slug=$(printf '%s' "$task_id" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-80)
      name="$RUN/${domain}__${slug}"
      log="$LOGDIR/n${jobid}_gpu${gpu}__${domain}__${slug}.log"

      echo "[$jobid:$gpu] START $domain $task_id" >>"$LOGDIR/progress.log"
      rc=0
      # `</dev/null`: srun would otherwise eat this loop's stdin. Harmless here because the
      # job list is already in an array, but leaving it out is how the original broke.
      TAU2_DOMAIN="$domain" "$HERE/tau2_smoke_nemo.sh" "$jobid" "$gpu" \
        --task-ids "$task_id" \
        --max-steps-seconds "$CAP_SECONDS" \
        --hallucination-retries 0 \
        --save-to "$name" \
        --log-level INFO \
        "$@" >"$log" 2>&1 </dev/null || rc=$?

      # rc=0 is not enough; an episode that dies in setup is written as an
      # INFRASTRUCTURE_ERROR simulation and the batch still exits 0.
      if ((rc != 0)); then
        echo "[$jobid:$gpu] FAIL $domain $task_id rc=$rc (see $log)" >>"$LOGDIR/progress.log"
      elif grep -q "failed permanently after" "$log"; then
        echo "[$jobid:$gpu] INFRA $domain $task_id (see $log)" >>"$LOGDIR/progress.log"
      else
        echo "[$jobid:$gpu] OK $domain $task_id" >>"$LOGDIR/progress.log"
      fi
    done
  ) &
  WORKER_PIDS+=($!)
done

for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || true; done

# Dispatch accounting, which is the thing the original launcher lacked. Compare STARTs to the
# job list, not OKs to expectations: an episode that was never dispatched leaves no trace at
# all and is invisible to any per-episode check.
started=$(grep -c ' START ' "$LOGDIR/progress.log" || true)
echo "gapfill finished: $started of ${#JOBLIST[@]} episodes dispatched"
printf '  %s\n' "$(grep -c ' OK ' "$LOGDIR/progress.log" || true) ok," \
  "$(grep -c ' INFRA ' "$LOGDIR/progress.log" || true) infra," \
  "$(grep -c ' FAIL ' "$LOGDIR/progress.log" || true) failed"
((started == ${#JOBLIST[@]})) || echo "WARNING: $((${#JOBLIST[@]} - started)) never dispatched"
