#!/bin/bash
# Retry the FDB-v3 examples whose result file says `inference_error`, up to N rounds.
#
# Why this exists: the released Triton backend has a race in its degenerate-tool-call
# recovery. When the model opens a tool call and never emits the end-of-tool-call token,
# "fast extract" gives up after 512 steps (`exceeded 512 steps without eotc_id`), and if the
# vLLM request has meanwhile hit EOS the next `append_request` raises
# `request '<seq>' not found (already finished or never started)` -> HTTP 500 -> WebSocket
# 1011. It killed 1 of the first 56 sessions; an earlier occurrence recovered cleanly, so it
# is a race, not a property of the example. The server stays healthy for later sessions.
#
# A crashed session is not a model-quality signal, so it must not silently score as a total
# miss -- but a *persistent* failure must not be silently retried away either. Hence: retry a
# bounded number of times, and leave whatever still fails in place, flagged, for disclosure.
#
# Usage: scripts/fdb_v3_retry_failures.sh <server> <provider> [rounds] [wait_for_pid]
set -uo pipefail

SERVER=${1:?server host:port}
PROVIDER=${2:?provider name}
ROUNDS=${3:-2}
WAIT_PID=${4:-}

HERE=$(cd "$(dirname "$0")" && pwd)
PY=/fsx/home/kai.li/miniforge3/envs/voicechat/bin/python
DATA=/fsx/home/kai.li/code/Full-Duplex-Bench/v3/fdb_v3_data_released

if [[ -n "$WAIT_PID" ]]; then
  echo "waiting for pid $WAIT_PID to finish the main pass..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
  echo "main pass done"
fi

for ((round = 1; round <= ROUNDS; round++)); do
  # Delete only the failed result files; the driver skips any example that still has one.
  failed=$($PY - "$DATA" "$PROVIDER" <<'EOF'
import json, sys, glob, os
data, provider = sys.argv[1], sys.argv[2]
n = 0
for path in sorted(glob.glob(f"{data}/*/result_{provider}.json")):
    try:
        status = json.load(open(path)).get("status")
    except Exception:
        status = "unreadable"
    if status != "completed":
        print(f"  retrying {os.path.basename(os.path.dirname(path))} (status={status})", file=sys.stderr)
        os.remove(path)
        n += 1
print(n)
EOF
)
  echo "round $round: $failed example(s) to retry"
  [[ "$failed" == "0" ]] && break
  $PY -u "$HERE/fdb_v3_realtime_infer.py" --server "$SERVER" --provider "$PROVIDER"
done

$PY - "$DATA" "$PROVIDER" <<'EOF'
import json, sys, glob, collections
data, provider = sys.argv[1], sys.argv[2]
c = collections.Counter()
for path in sorted(glob.glob(f"{data}/*/result_{provider}.json")):
    c[json.load(open(path)).get("status")] += 1
print("final status counts:", dict(c))
EOF
