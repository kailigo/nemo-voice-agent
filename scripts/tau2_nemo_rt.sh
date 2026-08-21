#!/bin/bash
# tau-voice through the NemotronLabs VoiceChat NIM container (the SERVED path), as arm A'.
#
# WHY THIS ARM EXISTS
# Arm A runs `--audio-native-provider nemo`, which is the in-process research path. FDB-v3
# already measured that path against the container on the same 100 examples and the same
# weights (FDB_V3_REPRODUCTION.md §"Where it landed"):
#
#   metric              research path   container   card
#   Tool Selection          78.3 %       73.1 %    82.5 %
#   Argument accuracy       30.3 %       51.7 %    44.2 %
#   Pass@1                  23.0 %       35.0 %    33   %
#
# The research path is 21 points worse on ARGUMENTS and 12 points worse on Pass@1. Arguments
# are precisely what tau-voice's modes A, B and C are about, so arm A may have been measuring
# a handicapped configuration rather than the model.
#
# The single most direct receipt is FDB-v3 `ecommerce_01`, whose user spells "a B C one two
# three" out loud -- the same task shape as tau-voice's spelled ids:
#
#   research path:  order_id: "LHR"          <- hallucinated, and airport-code shaped
#   container:      order_id: "A-BC123"      <- right tool, near-miss argument
#
# Compare arm A's tau-voice output on spelled ids: `SIN-555` for SI5UKW and `JFK59001` for
# anya_garcia_5901. Also airport codes. Same attractor, same path.
#
# And FDB-v3 found the container does the two things arm A never does. It ASKS for a missing
# argument ("I just need your starting address") instead of inventing one -- FDB-v3 penalises
# that because its instructions forbid clarifying questions in capitals, but tau-voice is a
# conversation and rewards it. That is mode C and mode G, path-specific.
#
# So this arm tests one thing: how much of arm A's 0-of-14 is the path.
#
# BLOCKED ON retail/airline/telecom AS THE CONTAINER IS RELEASED. Tried 2026-08-21 and it
# reproduced the wall TAU_VOICE_SFT_PLAN.md §0e-bis already measured: the LLM backbone is
# launched with `max_model_len=6144` (`checkpoint_utils/load_utils.py:424`, a literal, not an
# env var), and retail's prompt plus tool schemas tokenizes to 10978. The failure is
# invisible from here -- `session.update` is ACKed, the server logs the instructions as
# accepted, and then the FIRST AUDIO CHUNK closes the socket with WebSocket 1011 "Internal
# server error", which reads exactly like a network flake. All 5 retail episodes came back
# `infrastructure_error` with 0 ticks. Only `mock`, `banking` and `healthcare` complete, and
# none of those is a tau-bench domain.
#
# Do not re-run this on retail/airline/telecom until the window is dealt with. The options
# and their costs are in TAU_VOICE_SFT_PLAN.md §0e-bis; raising the literal deviates from the
# released config, so anything measured that way must not be quoted alongside the FDB-v3
# numbers in FDB_V3_REPRODUCTION.md.
#
# NO GPU AND NO SRUN, on purpose. The container is already resident (fdb_v3_serve.sh pins
# CUDA_VISIBLE_DEVICES=0 and holds ~127 GB); this is a websocket client that runs anywhere
# that can reach it, including the login node. It must never take a share of the allocation.
#
# CAP: 1200 s, matching arm B rather than arm A's 200 s. Arm A ran 200 s only because the
# research path infers at 12-19x realtime; the container is realtime (measured wall/input
# audio 1.001), so the cap is affordable here. It is also not the confound it looks like:
# every one of arm A's error-capped episodes died at 46-99 s, less than half the budget it
# already had, so a longer cap cannot explain a difference in argument grounding. Matching
# arm B instead makes A' vs B a direct comparison.
#
# Usage: NEMO_RT_URL=ws://ip-10-1-30-86:9000 scripts/tau2_nemo_rt.sh retail 7 35 49 64 78
#        scripts/tau2_nemo_rt.sh airline 3 15 28

set -euo pipefail

DOMAIN="${1:?usage: [NEMO_RT_URL=ws://host:port] $0 <domain> <task_id> [task_id...]}"
shift
(($#)) || { echo "give at least one task id" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAU2="${TAU2:-/fsx/home/kai.li/code/tau-voice-2}"
PYTHON="${PYTHON:-/fsx/home/kai.li/miniforge3/envs/voicechat/bin/python}"

# The container's host changes with every Slurm allocation, so there is no sane default; the
# provider reads $NEMO_RT_URL itself but fail loudly here rather than 20 minutes in.
: "${NEMO_RT_URL:?set NEMO_RT_URL, e.g. ws://ip-10-1-30-86:9000 (the node holding the NIM container)}"

USER_LLM="${USER_LLM:-bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
RUN_NAME="${RUN_NAME:-nemo_rt_0821}"
COMPLEXITY="${COMPLEXITY:-control}"
CAP_SECONDS="${CAP_SECONDS:-1200}"
# 1, not 2. The container is a single Triton/vLLM instance on one GPU serving one realtime
# session at a time, and the shared ElevenLabs plan allows 2 concurrent requests TOTAL across
# every arm running right now. Fanning out here would 429 the Gemini control as well.
CONCURRENCY="${CONCURRENCY:-1}"
VOICES="${VOICES:-$HERE/tau2_stock_voices.env}"
LOGDIR="${LOGDIR:-$HERE/../logs/$RUN_NAME}"

mkdir -p "$LOGDIR"
log="$LOGDIR/${DOMAIN}.log"
echo "arm A' (nemo_rt): $DOMAIN $* -> $NEMO_RT_URL, cap ${CAP_SECONDS}s, run $RUN_NAME"
echo "  log $log"

(
  # python-dotenv searches upward from the cwd and ELEVENLABS_API_KEY lives in tau-voice-2/.env.
  cd "$TAU2"
  # shellcheck disable=SC1090
  source "$VOICES"
  export NEMO_RT_URL
  # The user simulator's interruption/backchannel decision model is hardcoded and --user-llm
  # does not reach it; the post-episode reviewer and the nl-assertions judge are separate
  # again, and the reviewer is FATAL if it cannot authenticate -- it runs after the episode
  # has already been paid for. Redirect all three at the one model we can authenticate.
  export TAU2_VOICE_DECISION_MODEL="$USER_LLM"
  export TAU2_EVAL_USER_SIMULATOR_MODEL="$USER_LLM"
  export TAU2_NL_ASSERTIONS_MODEL="$USER_LLM"
  export TAU2_RETRY_ATTEMPTS="${TAU2_RETRY_ATTEMPTS:-8}"
  export TAU2_RETRY_MAX_WAIT="${TAU2_RETRY_MAX_WAIT:-30}"
  exec "$PYTHON" -u -m tau2.cli run \
    --domain "$DOMAIN" \
    --agent discrete_time_audio_native_agent \
    --audio-native \
    --audio-native-provider nemo_rt \
    --user voice_streaming_user_simulator \
    --user-llm "$USER_LLM" \
    --task-ids "$@" \
    --num-trials 1 \
    --max-concurrency "$CONCURRENCY" \
    --max-steps-seconds "$CAP_SECONDS" \
    --speech-complexity "$COMPLEXITY" \
    --hallucination-retries 0 \
    --save-to "$RUN_NAME/$DOMAIN" \
    --log-level INFO
) >"$log" 2>&1

echo "[$DOMAIN] finished; score with tau-voice-2/scripts/score_voice_run.py"
