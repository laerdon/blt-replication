#!/usr/bin/env bash
# train distilgpt2 hf stack on 2 local gpus (e.g. 2x h100) via torchrun ddp.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CONFIG="${CONFIG:-apps/main/configs/distilgpt2_hf_fineweb1p7b.yaml}"

if [[ -d "$REPO_ROOT/.venv" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.venv/bin/activate"
fi

# optional: restrict visible gpus (default: first two)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# can help ddp + flash attention stability on some setups
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --rdzv_backend=c10d \
  -m apps.main.train_distilgpt2 \
  "config=$CONFIG" \
  "$@"
