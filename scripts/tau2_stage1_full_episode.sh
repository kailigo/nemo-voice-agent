#!/bin/bash
# STAGE 1 of the arm-A ramp: ONE full-length episode, on ONE GPU.
#
# Everything validated so far ran 100-200 ticks. Arm A wants 1,000. This script is the
# cheapest thing that exercises an episode at real length, and it is deliberately the only
# stage that fits inside the remaining ElevenLabs free-tier quota (~7,900 chars as of
# 2026-08-18; this costs ~1,900). Stage 2 onward needs a paid plan.
#
# WHAT IT IS MEANT TO CATCH -- all of these are unexercised at 1,000 ticks:
#
#   * FC budget exhaustion. max_fc_total_tokens is 12,000 (plan 1c). Never driven to the
#     end of a long conversation, so we do not know the real headroom.
#   * Memory. 119,235 MiB of 143,771 MiB measured at batch 1, and inference preallocates
#     input_embeds to the full horizon T -- which is larger at a 200 s cap than anything
#     run so far. On a 140 GB card that is ~85 %, so there is not much room.
#   * The throughput extrapolation. Per-tick cost is linear in prefix length to within ~1 %
#     over the measured range (prefix 4,640 -> 5,014), and the 51 h projection for arm A
#     rests on that holding out to ~7,075. This run reaches it.
#
# WHAT IT CANNOT TELL YOU: anything about reward. The pre-SFT checkpoint emits off-task
# refusal boilerplate and calls no tools, so reward 0.0 is the expected and correct result.
# Diagnosing failure modes is stage 2's job, on a spread of tasks.
#
# COST: ~1.5 GPU-h (88 min of compute) + ~4.5 min model load. Runs on one GPU, so it does
# not need the whole node.
#
# Usage: scripts/tau2_stage1_full_episode.sh <jobid> <gpu-index> [extra tau2 run args...]

set -euo pipefail

JOBID="${1:?usage: $0 <jobid> <gpu-index> [extra args...]}"
GPU="${2:?usage: $0 <jobid> <gpu-index> [extra args...]}"
shift 2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 200 s of simulated audio = 1,000 ticks. Chosen as the arm-A cap because episode cost is
# EXACTLY this number: the pre-SFT model never terminates normally (both smoke runs ended
# on max_steps), so nothing finishes early. Cost is super-linear in the cap because the
# prefix grows through the episode -- 200 s is 1.5 GPU-h, the 1200 s default is 18.
CAP_SECONDS="${CAP_SECONDS:-200}"

# One retail task. Task 7 rather than 0: the lists are grouped, so the first few tasks are
# near-duplicate variants of one scenario (see scripts/tau2_select_subset.py). Position 7
# is what the stage-2 selector picks first for retail, so stage 1 is a strict subset of
# stage 2 and `--save-to` resume lines up if you reuse the run name.
exec "$HERE/tau2_smoke_nemo.sh" "$JOBID" "$GPU" \
  --task-ids 7 \
  --max-steps-seconds "$CAP_SECONDS" \
  --save-to stage1_full_episode \
  --log-level INFO \
  "$@"
