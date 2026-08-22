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
# LLM_MAX_MODEL_LEN: raises the vLLM context window for the TEXT backbone above the released
# 6144. Leave it unset for anything that will be compared to the model card or to the numbers
# in FDB_V3_REPRODUCTION.md -- those were all measured at 6144 and a patched server must not
# be quoted alongside them.
#
# Why it exists: `max_model_len=6144` is a literal at
# `checkpoint_utils/load_utils.py:424`, not an env var, and it is a HARD gate on tau-voice.
# retail's policy plus tool schemas tokenizes to 10978, airline 10708, telecom 14854, so all
# three canonical tau-bench domains are refused -- and refused invisibly: `session.update` is
# ACKed, the instructions are logged as accepted, and the FIRST AUDIO CHUNK closes the socket
# with WS 1011 "Internal server error" while the real ValueError stays in the container log.
# See TAU_VOICE_SFT_PLAN.md 0e-bis.
#
# Only line 424 is patched. Line 457 carries the same literal for the TTS model
# (`load_tts_model_in_vllm`), which budgets the speech-token stream and has nothing to do with
# the text prompt; raising it would change synthesis for no reason and cost memory. The sed is
# anchored on `max_model_len=6144,` (no space, engine-args style) which appears ONLY at 424 --
# 457 is a dict entry, `"max_model_len": 6144,`. The patch verifies it changed exactly one
# line and refuses to start otherwise, because a silent no-op here reads as "the window was
# raised" and would invalidate every number measured afterwards.
#
# This does NOT cost KV cache. vLLM sizes the cache from gpu_memory_utilization, not from
# max_model_len (the server logs `GPU KV cache size: 1,105,920 tokens`), so max_model_len is a
# per-request ceiling and 32768 is far inside it. `max_num_batched_tokens=768` is already
# below 6144, so chunked prefill is on and stays on.
#
# MAX_SESSION_DURATION: the container closes any session after this many seconds of AUDIO and
# the default is 300 (`audio_server.py:106`, `MAX_SESSION_DURATION = int(os.environ.get(...,
# "300"))  # 5 minutes`). This is the SECOND silent truncation gate after max_model_len, and it
# is the one that bit tau-voice on 2026-08-22: retail task 7 was cut at 300 s six times in a
# row, mid-opening-exchange, and the client saw only `WebSocket closed unexpectedly
# (code=1006)` with vLLM logging `Aborted request(s)` -- both of which read as a crash or a
# network flake. The server-side line is the only honest signal:
#
#   Client d6aa7ae1: Session audio duration limit reached (300.0s >= 300s)
#
# tau-voice episodes are capped at 1200 s, so anything above ~4 minutes of conversation is
# unmeasurable at the default. Unlike max_model_len this needs no source patch -- it is a
# supported env var -- so set it above the tau-voice cap and let tau-voice's own
# --max-steps-seconds be the only thing that ends an episode. A truncated episode is worse
# than no episode: it scores as a real failure.
#
# Watch out for `timeout_keep_alive=300` (audio_server.py:1897), which is a hardcoded uvicorn
# setting. It governs idle HTTP keep-alive rather than an active websocket, so it should not
# matter -- but if sessions still die at exactly 300 s after raising MAX_SESSION_DURATION,
# that is the next suspect and it needs a sed.
#
# MAX_DECODER_STEPS: the THIRD silent truncation gate, and the one that survives both fixes
# above. Found on 2026-08-22: with max_model_len=65536 and MAX_SESSION_DURATION=1300, retail
# task 7 still died -- this time at 417 s, with WS 1011 rather than 1006, and the real cause
# only in the server log:
#
#   ERROR - Client ec68930e: Error processing audio:
#           [500] Sequence 2873630666 exceeded max decoder steps (5000)
#
# 5000 steps / 417 s = 12.0 steps per second, i.e. the model's ~12.5 Hz frame rate: the cap is
# 5000 AUDIO FRAMES, so a session ends after ~400 s of conversation no matter what
# MAX_SESSION_DURATION says. tau-voice caps episodes at 1200 s, which needs ~15000 steps.
#
# Note this presents as 1011 "Internal server error" -- the SAME client-side code as the
# max_model_len gate -- so 1011 does not identify which gate fired. Only the server log does.
#
# The cap is not a constant but a TENSOR SHAPE: `buffer_capacity =
# seq_state.generated_text_tokens.shape[1]` (model.py:575, :627, :1429, :1520, :1785), read
# back from the four buffers preallocated in `sequence_manager.py:43-47`. So patching those
# four literals lifts all six checks at once, and `steps_exhausted` (data_types.py:87) is only
# ever latched from that same comparison rather than being an independent limit.
#
# Unlike max_model_len this DOES allocate, because the buffers are preallocated per sequence --
# but it is (steps x (3 + num_quantizers) x 8 bytes), so 5000 -> 20000 costs about 1-4 MB per
# session. Irrelevant next to the ~127 GB the server already holds.
#
# The sed anchor `, 5000,` was verified to occur exactly 4 times in sequence_manager.py and
# nowhere else in the backend; as with max_model_len it refuses to start on a miscount, because
# a silent no-op here reads as "the cap was raised" and every later episode would die at 400 s
# and score as a genuine agent failure.
#
# Usage: scripts/fdb_v3_serve.sh <jobid> [jinja|default] [port]
#        LLM_MAX_MODEL_LEN=32768 scripts/fdb_v3_serve.sh 1374 jinja 9000
#        LLM_MAX_MODEL_LEN=65536 MAX_SESSION_DURATION=1300 MAX_DECODER_STEPS=20000 \
#          SERVE_GPU=7 scripts/fdb_v3_serve.sh 1403 jinja 9000
set -euo pipefail

JOBID="${1:?usage: fdb_v3_serve.sh <jobid> [jinja|default] [port]}"
MODE="${2:-jinja}"
PORT="${3:-9000}"
LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN:-}"
MAX_SESSION_DURATION="${MAX_SESSION_DURATION:-}"
MAX_DECODER_STEPS="${MAX_DECODER_STEPS:-}"
# Which GPU the server takes. Used to be a hardcoded 0, which is wrong whenever the node is
# also running episodes: on 2026-08-22 job 1403 had GPUs 0-6 busy and only 7 free. Still one
# server per node -- audio_server.py:48 hardcodes TRITON_URL to localhost:8000 and pyxis
# shares the host network namespace -- so this picks the GPU, not the count.
SERVE_GPU="${SERVE_GPU:-0}"

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
    export NIM_HTTP_API_PORT=$PORT CUDA_VISIBLE_DEVICES=$SERVE_GPU MODEL_REPOSITORY=/data/models
    export USE_JINJA_TEMPLATE_PROMPT=$USE_JINJA
    if [[ -n '$MAX_SESSION_DURATION' ]]; then
      export MAX_SESSION_DURATION=$MAX_SESSION_DURATION
      echo \"SESSION CAP: \$MAX_SESSION_DURATION s of audio (container default is 300)\"
    else
      echo 'SESSION CAP: container default 300 s -- episodes longer than ~5 min WILL be cut'
    fi
    LU=/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/load_utils.py
    if [[ -n '$LLM_MAX_MODEL_LEN' ]]; then
      # Fail loudly on a no-op: if the anchor ever stops matching, the server would come up at
      # 6144 and every subsequent tau-voice episode would die on the first audio chunk with an
      # error that looks like a network flake.
      n=\$(grep -c 'max_model_len=6144,' \"\$LU\") || n=0
      if [[ \"\$n\" != 1 ]]; then
        echo \"REFUSING TO START: expected exactly 1 'max_model_len=6144,' in \$LU, found \$n\" >&2
        exit 4
      fi
      sed -i 's/max_model_len=6144,/max_model_len=$LLM_MAX_MODEL_LEN,/' \"\$LU\"
      echo \"PATCHED: LLM backbone max_model_len 6144 -> $LLM_MAX_MODEL_LEN (TTS left at 6144)\"
      grep -n 'max_model_len' \"\$LU\"
    else
      echo 'UNPATCHED: released max_model_len=6144'
    fi
    SM=/opt/tritonserver/backends/nemotron-voicechat/sequence_manager.py
    if [[ -n '$MAX_DECODER_STEPS' ]]; then
      # Same no-op guard as above: 4 preallocated buffers, and buffer_capacity is read back
      # from their shape, so all six decoder-step checks follow from this one sed.
      n=\$(grep -c ', 5000,' \"\$SM\") || n=0
      if [[ \"\$n\" != 4 ]]; then
        echo \"REFUSING TO START: expected exactly 4 ', 5000,' in \$SM, found \$n\" >&2
        exit 5
      fi
      sed -i 's/, 5000,/, $MAX_DECODER_STEPS,/g' \"\$SM\"
      echo \"PATCHED: max decoder steps 5000 -> $MAX_DECODER_STEPS (~\$(( $MAX_DECODER_STEPS / 12 )) s of audio at 12 Hz)\"
      grep -n '$MAX_DECODER_STEPS' \"\$SM\"
    else
      echo 'UNPATCHED: max decoder steps 5000 -- sessions WILL die at ~400 s of audio with WS 1011'
    fi
    /s2s/run_s2s_server.sh"
