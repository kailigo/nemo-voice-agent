#!/bin/bash
# Watch a tau2_stage2_subset.sh fan-out and act on what it finds, rather than waiting for
# it to finish and reading the wreckage afterwards.
#
# WHAT IT ACTS ON
#   1. An episode whose PROCESS EXITED without a usable result is appended to
#      `requeue.tsv`. "Exited" is load-bearing: tau2 re-runs a failed unit up to 4 times
#      internally, so a mid-episode fault is not evidence of a dead episode. Only the
#      fan-out's own post-exit verdict in progress.log is. The fan-out does not retry
#      across episodes -- deliberately, since a retry inside the slot would delay the two
#      episodes queued behind it -- so someone has to, and doing it after the fan-out
#      drains costs nothing but the episodes' own runtime.
#   2. When the fan-out exits, any requeued episodes are re-run over the freed GPUs, once.
#      Second failures are left alone: two infrastructure failures on the same episode
#      means the cause is not transient and a third attempt is just more GPU.
#   3. Every poll writes `status.txt`: per-episode reward, tool-call counts, and the
#      invented-tool-name rate, which is the arm-A diagnostic. Cheap because
#      tau2_quick_report.py is stdlib-only; importing tau2 here would cost 90 s a poll.
#
# WHAT IT DELIBERATELY DOES NOT DO
#   Kill anything. A watchdog with a kill switch turns one bad heuristic into a lost
#   allocation, and every failure mode seen so far has been either self-terminating or
#   worth leaving alone to inspect.
#
# Usage: scripts/tau2_watch_run.sh <jobid> [run-name] [poll-seconds]
#   Launch it detached, like the fan-out:
#   setsid nohup scripts/tau2_watch_run.sh 1303 stage2_protocol >logs/watch.log 2>&1 </dev/null &

set -uo pipefail

JOBID="${1:?usage: $0 <jobid> [run-name] [poll-seconds]}"
RUN="${2:-stage2_protocol}"
POLL="${3:-120}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$HERE/../logs/$RUN"
SIMS=/fsx/home/kai.li/code/tau-voice-2/data/simulations
STATUS="$LOGDIR/status.txt"
REQUEUE="$LOGDIR/requeue.tsv"
SEEN="$LOGDIR/.watch_seen"

mkdir -p "$LOGDIR"
: >"$REQUEUE"
: >"$SEEN"

note() { echo "[watch $(date -u +%H:%M:%S)] $*"; }

note "watching $RUN on job $JOBID, poll ${POLL}s"

while :; do
  # --- 1. is the fan-out still alive? ---------------------------------------
  # Matched on the script name rather than a recorded pid so the watchdog can be started
  # before or after the fan-out, and survives it being relaunched.
  fanout_alive=0
  pgrep -f "tau2_stage2_subset.sh $JOBID" >/dev/null 2>&1 && fanout_alive=1

  # --- 2. has the allocation gone? ------------------------------------------
  # This is the failure that looks like the run breaking but is not: every srun step dies
  # with rc=143 at once. Report it and stop; there is nothing to requeue onto.
  if ! squeue -h -j "$JOBID" >/dev/null 2>&1 || [[ -z "$(squeue -h -j "$JOBID" 2>/dev/null)" ]]; then
    note "ALLOCATION $JOBID IS GONE -- $(sacct -n -j "$JOBID" --format=State%20 2>/dev/null | head -1 | tr -s ' ')"
    note "everything already on disk is intact; a new allocation is needed to continue"
    break
  fi

  # --- 3. collect newly-dead episodes --------------------------------------
  # Read the fan-out's OWN verdict from progress.log rather than grepping the live episode
  # logs. The first version of this grepped for "emergency cleanup|failed permanently
  # after|DUE TO SIGNAL" and was WRONG in the most expensive direction: tau2's
  # `runner/progress.py::run_with_retry` re-runs a failed unit up to 4 times, so an episode
  # that lost its TTS call to a 429 logs the orchestrator's emergency cleanup, then quietly
  # restarts and runs to completion. Task 21 of stage2_protocol did exactly that. Requeuing
  # it would have spent an hour of GPU duplicating an episode that was still running.
  #
  # progress.log lines are written only after the episode's process has exited, and they
  # carry the real task id verbatim -- no slug round-trip to get wrong either.
  while IFS= read -r line; do
    case "$line" in
      *'] FAIL '*) rest="${line#*'] FAIL '}"; rest="${rest%% rc=*}"; why="rc!=0" ;;
      *'] INFRA '*) rest="${line#*'] INFRA '}"; rest="${rest%% (see *}"; why="retries exhausted" ;;
      *) continue ;;
    esac
    domain="${rest%% *}"
    task_id="${rest#* }"
    key="$domain	$task_id"
    grep -qxF "$key" "$SEEN" && continue
    printf '%s\n' "$key" >>"$SEEN"
    printf '%s\n' "$key" >>"$REQUEUE"
    note "DEAD ($why): $domain $task_id -> requeued"
  done <"$LOGDIR/progress.log"

  # An episode can also exit 0 having persisted nothing -- see the "exit 0 is NOT enough"
  # note in the fan-out. Catch it by comparing OK verdicts against what is actually on disk,
  # deriving the slug exactly the way the fan-out derived the directory name.
  while IFS= read -r line; do
    case "$line" in *'] OK '*) rest="${line#*'] OK '}" ;; *) continue ;; esac
    domain="${rest%% *}"
    task_id="${rest#* }"
    key="$domain	$task_id"
    grep -qxF "$key" "$SEEN" && continue
    slug="$(printf '%s' "$task_id" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-80)"
    for cand in "$SIMS/$RUN/${domain}__${slug}" "$SIMS/$RUN/${domain}__${slug}__retry"; do
      compgen -G "$cand/simulations/*.json" >/dev/null 2>&1 && continue 2
    done
    printf '%s\n' "$key" >>"$SEEN"
    printf '%s\n' "$key" >>"$REQUEUE"
    note "DEAD (exit 0, nothing persisted): $domain $task_id -> requeued"
  done <"$LOGDIR/progress.log"

  # --- 4. is the TTS quota gone? -------------------------------------------
  # `character_limit` is the one TTS failure that is NOT transient: the month's characters
  # are spent, every subsequent episode dies at its first user utterance, and a retry pass
  # would spend GPU producing nothing. Distinct from `concurrent_limit_exceeded`, which is
  # the transient one this run has already survived twice.
  quota_gone=0
  if grep -qls "character_limit" "$LOGDIR"/gpu*.log 2>/dev/null; then
    quota_gone=1
  fi

  # --- 5. rolling result table ---------------------------------------------
  {
    echo "=== $RUN at $(date -u +%H:%M:%S) UTC (fanout_alive=$fanout_alive) ==="
    ((quota_gone)) && echo "!! ELEVENLABS CHARACTER QUOTA EXHAUSTED -- remaining episodes cannot run"
    python3 "$HERE/tau2_quick_report.py" --run "$RUN" 2>&1
    # Live progress, which the report above cannot see: `simulations/<id>.json` is written
    # only when an episode ENDS, so a 2-hour episode is invisible to it until it is over.
    # Tick depth is the intermediate signal that matters for arm A, because the control
    # episodes all died of `too_many_errors` at ~300 ticks (of a 1000-tick, 200 s cap). An
    # episode that is past ~320 ticks with no termination line has already avoided the error
    # cascade the protocol fix targets, whatever its eventual reward.
    echo
    printf 'LIVE  %-30s %8s %8s  %s\n' episode tick 'of 1000' terminated
    for log in "$LOGDIR"/gpu*.log; do
      [[ -f "$log" ]] || continue
      t=$(grep -oE '^Tick [0-9]+$' "$log" | tail -1 | awk '{print $2}')
      term=$(grep -ohE 'too_many_errors|max_steps|user_stop|agent_stop' "$log" | sort -u | paste -sd,)
      printf 'LIVE  %-30s %8s %7s%%  %s\n' \
        "$(basename "$log" .log)" "${t:-0}" "$(( ${t:-0} / 10 ))" "${term:--}"
    done
    echo
    echo "requeued: $(wc -l <"$REQUEUE") episode(s)"
  } >"$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"

  # --- 6. fan-out done: re-run the requeue, once ---------------------------
  if ((fanout_alive == 0)); then
    n_requeue=$(wc -l <"$REQUEUE")
    note "fan-out has exited; $n_requeue episode(s) to re-run"
    if ((quota_gone)); then
      note "NOT re-running: the TTS character quota is exhausted, so every retry would die"
      note "at its first user utterance. Top up the plan, then: REQUEUE_FILE=$REQUEUE \\"
      note "  RUN_NAME=$RUN LOGDIR=$LOGDIR NGPU=$n_requeue $HERE/tau2_stage2_subset.sh <jobid>"
    elif ((n_requeue > 0)); then
      # Reuse the fan-out for the retry pass so the retries inherit every setting that
      # matters (one process per episode, one --save-to, the retry backoff). JOBS is
      # overridden to the requeue list; NGPU is capped at the number of episodes so idle
      # slots do not spin.
      note "re-running: $(tr '\n' ' ' <"$REQUEUE")"
      RUN_NAME="$RUN" LOGDIR="$LOGDIR" NGPU=$((n_requeue < 8 ? n_requeue : 8)) \
        REQUEUE_FILE="$REQUEUE" STAGGER_SECONDS=20 \
        "$HERE/tau2_stage2_subset.sh" "$JOBID" 2>&1 | sed 's/^/[requeue] /'
      note "retry pass finished"
    fi
    break
  fi

  sleep "$POLL"
done

note "final state:"
cat "$STATUS"
