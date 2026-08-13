#!/bin/bash
# Launch finetuning of Nemotron VoiceChat 11B (STT component)
#
# Prerequisites:
#   1. Checkpoint extracted: python scripts/extract_stt_checkpoint.py ...
#   2. Training data prepared: python scripts/prepare_lhotse_data.py ...
#   3. NeMo installed: pip install -e . (from repo root)
#
# Usage:
#   bash scripts/run_finetune.sh <stt_ckpt_dir> <train_shar_path> <val_shar_path> [num_gpus]
#
# Example:
#   bash scripts/run_finetune.sh \
#       /data/checkpoints/stt_extracted \
#       /data/lhotse/train/shards \
#       /data/lhotse/val/shards \
#       8

set -euo pipefail

STT_CKPT_DIR="${1:?Usage: $0 <stt_ckpt_dir> <train_shar_path> <val_shar_path> [num_gpus]}"
TRAIN_SHAR_PATH="${2:?Usage: $0 <stt_ckpt_dir> <train_shar_path> <val_shar_path> [num_gpus]}"
VAL_SHAR_PATH="${3:?Usage: $0 <stt_ckpt_dir> <train_shar_path> <val_shar_path> [num_gpus]}"
NUM_GPUS="${4:-8}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="${REPO_ROOT}/examples/speechlm2/s2s_duplex_stt_train.py"
CONFIG_PATH="conf/finetune"
CONFIG_NAME="s2s_duplex_stt_11b"

echo "============================================"
echo "Nemotron VoiceChat 11B STT Finetuning"
echo "============================================"
echo "STT checkpoint: ${STT_CKPT_DIR}"
echo "Train data:     ${TRAIN_SHAR_PATH}"
echo "Val data:       ${VAL_SHAR_PATH}"
echo "GPUs:           ${NUM_GPUS}"
echo "Config:         ${CONFIG_PATH}/${CONFIG_NAME}"
echo "============================================"

cd "${REPO_ROOT}/examples/speechlm2"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    "${TRAIN_SCRIPT}" \
    --config-path "${CONFIG_PATH}" \
    --config-name "${CONFIG_NAME}" \
    model.pretrained_s2s_model="${STT_CKPT_DIR}" \
    data.train_ds.input_cfg.0.shar_path="${TRAIN_SHAR_PATH}" \
    data.validation_ds.datasets.val_set_0.shar_path="${VAL_SHAR_PATH}" \
    trainer.devices="${NUM_GPUS}"
