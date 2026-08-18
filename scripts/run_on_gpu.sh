#!/bin/bash
# Run a python script from the `voicechat` env on an already-allocated Slurm GPU.
#
# Two things this exists to get right, both of which fail silently otherwise:
#
#   * LD_LIBRARY_PATH. The env's activate.d hook prepends $CONDA_PREFIX/lib so conda's
#     libstdc++ (CXXABI_1.3.15) wins over the system one (1.3.13). Invoking the python
#     binary directly never runs that hook, and `import sqlite3`/`torchcodec` then dies
#     with "version CXXABI_1.3.15 not found".
#   * `srun --export=ALL,VAR=val` does NOT override a variable the compute node already
#     has in its environment -- LD_LIBRARY_PATH and CUDA_VISIBLE_DEVICES both come back
#     as the node's own values. They have to be set inside the remote command.
#
# Usage: scripts/run_on_gpu.sh <jobid> <gpu-index> <script.py> [args...]

set -euo pipefail

JOBID="$1"; GPU="$2"; shift 2

ENV_PREFIX=/fsx/home/kai.li/miniforge3/envs/voicechat
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

srun --jobid="$JOBID" --overlap bash -c "
  export LD_LIBRARY_PATH='$ENV_PREFIX/lib:\${LD_LIBRARY_PATH:-}'
  export CUDA_VISIBLE_DEVICES='$GPU'
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=\${MASTER_PORT:-29591}
  cd '$REPO'
  exec '$ENV_PREFIX/bin/python' -u $(printf '%q ' "$@")
"
