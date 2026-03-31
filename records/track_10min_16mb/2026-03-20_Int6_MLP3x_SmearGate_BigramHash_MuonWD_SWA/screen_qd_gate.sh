#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-gate}"
case "${MODE}" in
  baseline)
    PREFIX_CONTROL_ENABLED="${PREFIX_CONTROL_ENABLED:-0}"
    ;;
  gate)
    PREFIX_CONTROL_ENABLED="${PREFIX_CONTROL_ENABLED:-1}"
    ;;
  *)
    echo "usage: $0 [baseline|gate]" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export ITERATIONS="${ITERATIONS:-1200}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-300}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-1048576}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export SWA_ENABLED="${SWA_ENABLED:-0}"
export PREFIX_CONTROL_SCALE="${PREFIX_CONTROL_SCALE:-0.10}"
export RUN_ID="${RUN_ID:-prefix_control_screen_${MODE}_seed${SEED:-1337}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

echo "screen mode=${MODE} prefix_control=${PREFIX_CONTROL_ENABLED} iterations=${ITERATIONS} val_max_tokens=${VAL_MAX_TOKENS} nproc=${NPROC_PER_NODE}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py
