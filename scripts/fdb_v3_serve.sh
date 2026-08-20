#!/usr/bin/env bash
# Start the NIM VoiceChat server inside an existing Slurm allocation, under pyxis/enroot.
#
# The one knob that matters for FDB-v3 faithfulness is USE_JINJA_TEMPLATE_PROMPT
# (audio_server.py:53), which **defaults to "0" inside the container**:
#
#   0 -> audio_server.py:1179-1198 concatenates our instructions with TOOLS_TEMPLATE (:68),
#        whose decision-process text tells the model "DO NOT use any tools when not needed" --
#        directly contradicting the benchmark's "Execute the tool unconditionally!".
#   1 -> /s2s/prompt_template.jinja, which carries no such text. This is the faithful arm,
#        the prompt the other providers in the published table received.
#
# Confirm which branch actually ran before trusting a number -- the client cannot see it:
#   grep -c 'Preparing prompt using jinja template' <log>   # >0 iff jinja
#   grep -c 'Call a tool ONLY when the user'        <log>    # >0 iff the restraint text leaked in
#
# The deploy step (NEMO_CHECKPOINT_PATH=/checkpoint /s2s/deploy_s2s_model.sh, ~12 min) is NOT
# run here: $MODEL_REPO is already built. Rebuild it by hand if the checkpoint changes.
#
# One GPU per node is the ceiling: audio_server.py:48 hardcodes TRITON_URL to localhost:8000
# and pyxis shares the host network namespace, so two servers on one node collide.
#
# Usage: scripts/fdb_v3_serve.sh <jobid> [jinja|default] [port]
set -euo pipefail

JOBID="${1:?usage: fdb_v3_serve.sh <jobid> [jinja|default] [port]}"
MODE="${2:-jinja}"
PORT="${3:-9000}"

case "$MODE" in
  jinja)   USE_JINJA=1 ;;
  default) USE_JINJA=0 ;;
  *) echo "mode must be 'jinja' or 'default', got '$MODE'" >&2; exit 2 ;;
esac

IMAGE=/fsx/home/kai.li/data/containers/nemotron-labs-voicechat.sqsh
MODEL_REPO=/fsx/home/kai.li/data/voicechat/triton-model-repo
[[ -d "$MODEL_REPO/nemotron-voicechat" ]] || {
  echo "$MODEL_REPO/nemotron-voicechat missing -- run /s2s/deploy_s2s_model.sh first" >&2
  exit 3
}

set -x
exec srun --overlap --jobid="$JOBID" --nodes=1 --ntasks=1 \
  --container-image="$IMAGE" \
  --container-mounts="$MODEL_REPO:/data/models" \
  --container-writable bash -c "
    unset PYTHONPATH LD_LIBRARY_PATH CONDA_PREFIX VIRTUAL_ENV
    export NIM_HTTP_API_PORT=$PORT CUDA_VISIBLE_DEVICES=0 MODEL_REPOSITORY=/data/models
    export USE_JINJA_TEMPLATE_PROMPT=$USE_JINJA
    /s2s/run_s2s_server.sh"
