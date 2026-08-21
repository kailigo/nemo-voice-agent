#!/bin/bash
# Run the 100 Full-Duplex-Bench v3 examples over the 8 GPUs of an existing allocation.
#
# ONE WORKER PER GPU, NOT ONE PER EXAMPLE. This is the opposite of what
# tau2_stage2_subset.sh does, and deliberately so: there, each episode was a separate tau2
# process because tau2 owns the orchestration. Here the model load is ~10 minutes and the
# examples are ~45 seconds of audio each, so a process per example would spend 100 x 10 min
# = ~16 GPU-hours loading weights to do ~6 GPU-hours of inference. The driver keeps the
# model resident and opens a fresh StreamingFCSession per example instead.
#
# Sharding is `examples[k::8]` over a stable sorted list, so a shard that dies can be
# re-run on its own without recomputing the split, and results already on disk are skipped
# unless --force is passed through.
#
# Usage: scripts/fdb_v3_fanout.sh <jobid> [extra args for fdb_v3_nemo_infer.py...]
#   scripts/fdb_v3_fanout.sh 1303
#   scripts/fdb_v3_fanout.sh 1303 --provider nemo_bare --cascaded-config nemo-base-bare-prompt

set -uo pipefail

JOBID="${1:?usage: $0 <jobid> [extra args...]}"; shift

ENV_PREFIX=/fsx/home/kai.li/miniforge3/envs/voicechat
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPU="${NGPU:-8}"
# Shard k runs on GPU (GPU_OFFSET + k). Needed when the VoiceChat NIM container already owns
# GPU 0 (fdb_v3_serve.sh pins CUDA_VISIBLE_DEVICES=0): NGPU=7 GPU_OFFSET=1 uses GPUs 1-7 and
# leaves the server untouched. Shard indices stay 0..NGPU-1 so the split is independent of it.
GPU_OFFSET="${GPU_OFFSET:-0}"
BASE_PORT="${BASE_PORT:-29700}"
RUN_NAME="${RUN_NAME:-fdb_v3}"
LOGDIR="${LOGDIR:-$HERE/../logs/$RUN_NAME}"

mkdir -p "$LOGDIR"
PROGRESS="$LOGDIR/progress.log"
: >"$PROGRESS"

note() { echo "[fanout $(date -u +%H:%M:%S)] $*" | tee -a "$PROGRESS"; }

note "$NGPU shards on job $JOBID, extra args: $*"

pids=()
for gpu in $(seq 0 $((NGPU - 1))); do
  log="$LOGDIR/gpu${gpu}.log"
  # MASTER_PORT must differ per process: provider.py does os.environ.setdefault on a fixed
  # 29593 for the single-rank process group DuplexSTTModel needs, so concurrent workers on
  # one node otherwise die with EADDRINUSE -- which reads like a networking fault.
  srun --jobid="$JOBID" --overlap bash -c "
    export LD_LIBRARY_PATH='$ENV_PREFIX/lib:\${LD_LIBRARY_PATH:-}'
    export CUDA_VISIBLE_DEVICES=$((GPU_OFFSET + gpu))
    export MASTER_PORT=\$(($BASE_PORT + $gpu))
    exec '$ENV_PREFIX/bin/python' -u '$HERE/fdb_v3_nemo_infer.py' \
      --shard $gpu --num-shards $NGPU $(printf '%q ' "$@")
  " >"$log" 2>&1 &
  pids+=($!)
  note "shard $gpu -> gpu $((GPU_OFFSET + gpu)), log $log"
done

rc_total=0
for gpu in $(seq 0 $((NGPU - 1))); do
  wait "${pids[$gpu]}"; rc=$?
  # rc=0 is necessary but not sufficient: the driver writes a result file per example even
  # for a failed one, so the count on disk is the real verdict -- and a missing result is
  # invisible in the final score, since the evaluators drop absent scenarios from the
  # denominator. fdb_v3_evaluate.py's coverage() prints that count before scoring.
  if ((rc == 0)); then note "OK shard $gpu"; else note "FAIL shard $gpu rc=$rc"; rc_total=1; fi
done

note "all shards done"
exit "$rc_total"
