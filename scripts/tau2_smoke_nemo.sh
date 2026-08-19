#!/bin/bash
# Single-episode tau2 smoke test for the `nemo` audio-native provider.
#
# This is the last seam in the arm-A path that nothing else covers: the driver, the
# provider, the FC parser and the two clocks are all validated standalone by
# scripts/check_streaming_driver.py, but DiscreteTimeNeMoAdapter had never run inside a
# live tau2 orchestrator. Everything this catches is integration, not model quality.
#
# Non-obvious settings, each of which the defaults get wrong for this arm:
#
#   * --user-llm bedrock/...  The tau2 defaults are gpt-4.1 and there is no OpenAI key.
#     Bedrock authenticates off the instance IAM role, so it needs no key at all.
#   * --max-steps-seconds     Defaults to 1200 s. Measured in this harness, inference runs
#     at ~23.5x realtime with no KV cache (Nemotron-Nano is hybrid Mamba2, see
#     TAU_VOICE_SFT_PLAN 0b-bis), so the default would cost ~8 HOURS of GPU for one
#     episode. Always pass this; 20 s of audio is ~8 min and reaches every seam.
#   * cd into tau-voice-2     python-dotenv searches upward from the cwd, and the
#     ELEVENLABS_API_KEY the user simulator needs lives in tau-voice-2/.env.
#   * --max-concurrency 1     Each concurrent episode loads its own 9B model.
#
# Usage: scripts/tau2_smoke_nemo.sh <jobid> <gpu-index> [extra tau2 run args...]

set -euo pipefail

JOBID="$1"; GPU="$2"; shift 2

ENV_PREFIX=/fsx/home/kai.li/miniforge3/envs/voicechat
TAU2=/fsx/home/kai.li/code/tau-voice-2
USER_LLM=bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0

# Overridable rather than an extra arg: `--domain` is a plain argparse store, so passing a
# second one would silently rely on last-wins. Stage 2 fans out over all three test
# domains, so it needs to set this per worker.
DOMAIN="${TAU2_DOMAIN:-retail}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

srun --jobid="$JOBID" --overlap bash -c "
  export LD_LIBRARY_PATH='$ENV_PREFIX/lib:\${LD_LIBRARY_PATH:-}'
  export CUDA_VISIBLE_DEVICES='$GPU'
  # provider.py:276 does os.environ.setdefault('MASTER_PORT', '29593') for the single-rank
  # process group DuplexSTTModel needs. It is a fixed port, so N concurrent episodes on one
  # node all race for it and N-1 die with 'EADDRINUSE ... port: 29593' -- which surfaces as
  # 4 failed attempts in 3 seconds and reads like a networking fault. Deriving the port
  # from the GPU index is collision-free by construction, because this script places
  # exactly one process per GPU. setdefault means this export wins.
  export MASTER_PORT=\$((29593 + $GPU))
  # The official tau-bench voice ids 404 on our ElevenLabs key; see the file's header.
  source '$HERE/tau2_stock_voices.env'
  # ElevenLabs allows 2 CONCURRENT requests on our plan and 429s the rest with
  # \`concurrent_limit_exceeded\`. tau2 does retry 429, but only 3 attempts at 1 s + 2 s,
  # which killed 2 of 8 fanned-out episodes within 30 s of their first user utterance -- a
  # TTS failure is fatal to the simulation, so that is an hour of GPU lost to a 3-second
  # queue. 8 attempts with a 30 s ceiling gives ~90 s of tolerance. Not a quota problem:
  # quota failures say \`character_limit\`, these said \`rate_limit_error\`.
  export TAU2_RETRY_ATTEMPTS="\${TAU2_RETRY_ATTEMPTS:-8}"
  export TAU2_RETRY_MAX_WAIT="\${TAU2_RETRY_MAX_WAIT:-30}"
  # The user simulator's interruption/backchannel decisions use their own hardcoded
  # model (config.VOICE_USER_SIMULATOR_DECISION_MODEL), which --user-llm does not touch.
  # Exported in the shell rather than put in .env because config.py reads it at import
  # time, which happens before tau2 calls load_dotenv().
  export TAU2_VOICE_DECISION_MODEL='$USER_LLM'
  # The post-episode hallucination reviewer. This one is FATAL if it cannot authenticate:
  # it runs after the episode completes and takes the whole simulation down with it, so
  # results.json lands with simulations: [] after all the GPU time has been spent.
  export TAU2_EVAL_USER_SIMULATOR_MODEL='$USER_LLM'
  export TAU2_NL_ASSERTIONS_MODEL='$USER_LLM'
  cd '$TAU2'
  exec '$ENV_PREFIX/bin/python' -u -m tau2.cli run \
    --domain '$DOMAIN' \
    --agent discrete_time_audio_native_agent \
    --audio-native \
    --audio-native-provider nemo \
    --cascaded-config nemo-base \
    --user voice_streaming_user_simulator \
    --user-llm '$USER_LLM' \
    --num-tasks 1 \
    --num-trials 1 \
    --max-concurrency 1 \
    --max-steps-seconds 120 \
    --speech-complexity control \
    $(printf '%q ' "$@")
"
