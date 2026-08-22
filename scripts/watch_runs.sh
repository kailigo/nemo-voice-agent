#!/usr/bin/env bash
# Append a one-line status snapshot for every live tau-voice arm, every $INTERVAL seconds.
#
# Exists because the interactive shell times out at 2 minutes but these arms run for hours, so
# polling from the foreground either blocks or samples too coarsely to catch a transition. This
# runs detached and leaves a timeline on disk that can be read in one cheap call.
#
# It records COVERAGE (episode records written), not exit status: a tau-voice batch reports
# success and exits 0 even when every episode inside it died in setup, so the count of saved
# records is the only honest progress signal.
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
  "nemo_rt_0822b:5"
)

count_records() {
  # UUID-named files are per-simulation records; results.json / sim_status.json are not.
  find "$TAU2/data/simulations/$1" -name '*-*-*-*-*.json' 2>/dev/null | wc -l
}

if [[ ! -s "$OUT" ]]; then
  printf 'ts\trun\trecords\ttarget\tprocs\tnote\n' >"$OUT"
fi

while :; do
  ts=$(date -u +%H:%M:%S)

  for entry in "${RUNS[@]}"; do
    run="${entry%%:*}"; target="${entry##*:}"
    n=$(count_records "$run")
    printf '%s\t%s\t%s\t%s\t-\t-\n' "$ts" "$run" "$n" "$target" >>"$OUT"
  done

  # arm A' is the one with a live gate to watch: ticks should keep climbing and the two
  # truncation signatures should stay at zero.
  rt_log="$HERE/../logs/nemo_rt_0822b/retail.log"
  if [[ -f "$rt_log" ]]; then
    ticks=$(grep -c 'Agent audio' "$rt_log" 2>/dev/null || echo 0)
    sess=$(grep -c 'session configured' "$rt_log" 2>/dev/null || echo 0)
    retry=$(grep -icE 'code=1006|failed \(attempt' "$rt_log" 2>/dev/null || echo 0)
    calls=$(grep -icE 'FunctionCall|function_call|tool_call' "$rt_log" 2>/dev/null || echo 0)
    printf '%s\tnemo_rt_0822b\t-\t-\t-\tticks=%s sessions=%s retries=%s toolcalls=%s\n' \
      "$ts" "$ticks" "$sess" "$retry" "$calls" >>"$OUT"
  fi

  srv="$HERE/../logs/nemo_rt_serve_0822_v2.log"
  if [[ -f "$srv" ]]; then
    cuts=$(grep -c 'Session audio duration limit reached' "$srv" 2>/dev/null || echo 0)
    printf '%s\tcontainer\t-\t-\t-\tsession_cuts=%s\n' "$ts" "$cuts" >>"$OUT"
  fi

  # Live episode processes, per arm, so a stall is distinguishable from a finish.
  for pat in "audio-native-provider nemo " "audio-native-provider nemo_rt" "audio-native-provider gemini"; do
    c=$(pgrep -fc "$pat" 2>/dev/null || echo 0)
    printf '%s\tprocs\t-\t-\t%s\t%s\n' "$ts" "$c" "$pat" >>"$OUT"
  done

  sleep "$INTERVAL"
done
