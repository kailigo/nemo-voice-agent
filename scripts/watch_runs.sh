#!/usr/bin/env bash
# Append a status snapshot for every live tau-voice arm, every $INTERVAL seconds.
#
# Exists because the interactive shell times out at 2 minutes but these arms run for hours, so
# polling from the foreground either blocks or samples too coarsely to catch a transition. This
# runs detached and leaves a timeline on disk that can be read back in one cheap call.
#
# It records COVERAGE (episode records written), not exit status: a tau-voice batch reports
# success and exits 0 even when every episode inside it died in setup, so the count of saved
# records is the only honest progress signal.
#
# Counting trap: episode records are `<run>/<subdir>/simulations/<uuid>.json`, but `find -path
# '*/simulations/*.json'` ALSO matches the `data/simulations/` prefix shared by every run, which
# silently doubles the count and made a 14-episode arm read as 28. Anchor on the UUID filename.
#
# Process trap: arm A's clients run inside srun steps on the compute node, so pgrep on the login
# node sees only the srun wrappers (2 per episode) and never the python client. Count the
# wrappers and halve, or just treat >0 as "alive" -- do not read it as an episode count.
#
# Usage: setsid nohup scripts/watch_runs.sh > logs/watch.log 2>&1 </dev/null & disown
set -uo pipefail

TAU2="${TAU2:-/fsx/home/kai.li/code/tau-voice-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL="${INTERVAL:-120}"
OUT="${OUT:-$HERE/../logs/run_timeline.tsv}"

# run name -> target episode count
RUNS=(
  "stage2_subset_0821:24"
  "gemini_baseline_0821b:16"
  "nemo_rt_0822c:5"
)

# The arm A' client log and the container log currently under test.
RT_LOG="${RT_LOG:-$HERE/../logs/nemo_rt_0822c/retail.log}"
SRV_LOG="${SRV_LOG:-$HERE/../logs/nemo_rt_serve_0822_v3.log}"

count_records() {
  find "$TAU2/data/simulations/$1" -name '*-*-*-*-*.json' 2>/dev/null | wc -l
}

if [[ ! -s "$OUT" ]]; then
  printf 'ts\twhat\tvalue\tnote\n' >"$OUT"
fi

while :; do
  ts=$(date -u +%H:%M:%S)

  for entry in "${RUNS[@]}"; do
    run="${entry%%:*}"; target="${entry##*:}"
    printf '%s\t%s\t%s/%s\t-\n' "$ts" "$run" "$(count_records "$run")" "$target" >>"$OUT"
  done

  # arm A' is the one with live gates to watch: ticks must keep climbing PAST 2000 (the old
  # 5000-decoder-step wall sat at ~2085 ticks) and the failure signatures must stay at zero.
  if [[ -f "$RT_LOG" ]]; then
    ticks=$(grep -c '^Tick ' "$RT_LOG" 2>/dev/null || echo 0)
    sess=$(grep -c 'session configured' "$RT_LOG" 2>/dev/null || echo 0)
    fail=$(grep -cE 'code=10[0-9][0-9]|failed \(attempt' "$RT_LOG" 2>/dev/null || echo 0)
    calls=$(grep -cE 'FunctionCall|function_call|tool_call' "$RT_LOG" 2>/dev/null || echo 0)
    printf '%s\tnemo_rt_client\t-\tticks=%s sessions=%s failures=%s toolcalls=%s\n' \
      "$ts" "$ticks" "$sess" "$fail" "$calls" >>"$OUT"
  fi

  # The three container gates, each with its own server-side signature. The client sees WS 1011
  # for BOTH max_model_len and the decoder-step cap, so only these lines identify which fired.
  if [[ -f "$SRV_LOG" ]]; then
    cuts=$(grep -c 'Session audio duration limit reached' "$SRV_LOG" 2>/dev/null || echo 0)
    steps=$(grep -c 'exceeded max decoder steps' "$SRV_LOG" 2>/dev/null || echo 0)
    ctx=$(grep -cE 'longer than the maximum model length|exceeds the maximum' "$SRV_LOG" 2>/dev/null || echo 0)
    printf '%s\tcontainer\t-\tsession_cuts=%s decoder_steps=%s ctx_overflow=%s\n' \
      "$ts" "$cuts" "$steps" "$ctx" >>"$OUT"
  fi

  printf '%s\tprocs\t-\tsrun_wrappers=%s rt_clients=%s\n' "$ts" \
    "$(pgrep -fc 'srun --jobid=1403 --overlap' 2>/dev/null || echo 0)" \
    "$(pgrep -fc 'audio-native-provider nemo_rt' 2>/dev/null || echo 0)" >>"$OUT"

  sleep "$INTERVAL"
done
