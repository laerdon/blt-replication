#!/bin/bash
# sets up blt on a fresh ubuntu 22.04 box with an nvidia gpu.
# idempotent: safe to rerun. skips steps that are already done.
#
# requirements:
#   - sudo access (for apt and cuda toolkit install)
#   - nvidia driver already present and supporting cuda 12.x (check nvidia-smi)
#   - run from anywhere inside or outside the repo

set -euo pipefail

start_time=$(date +%s)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
if [ ! -f pyproject.toml ]; then
  echo "error: expected pyproject.toml in ${REPO_ROOT}"
  exit 1
fi
echo "repo root: ${REPO_ROOT}"

mkdir -p /home/ubuntu/.ssh
chmod 700 /home/ubuntu/.ssh
printf '%s\n' 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILwKnbBe5gXovj9iUrYy5UtPw5UkZ9tMYCZj4kBMewzF elk97@cornell.edu
' 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIC7wjc4TWhGqYqhbpwzpwYv3PbQvgj1NPgEPU3r3kwUY srikarkarra@Srikars-MacBook-Pro.local
' 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHJYcf8K5lUYC2IaH2aMJPtV9/Tfoppj/M2+aZdHKeCn newkey
' 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMjEzOKiOCfWSqKWh4RTOj0Fd6b/mqU9IE7yIOE8VSSQ
'>> /home/ubuntu/.ssh/authorized_keys

if [ "$(id -u)" -eq 0 ]; then
  SUDO=()
elif command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  echo "error: this script needs root privileges for apt and cuda, but sudo was not found"
  exit 1
fi

# where to download the cuda runfile (needs ~5gb free; /tmp often too small)
# use a writable subdirectory instead of /mnt itself, since /mnt is often root-owned.
CUDA_DOWNLOAD_DIR="${CUDA_DOWNLOAD_DIR:-/mnt/cuda-downloads}"
CUDA_INSTALL_PATH="/usr/local/cuda-12.1"
CUDA_RUNFILE="cuda_12.1.1_530.30.02_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/${CUDA_RUNFILE}"
BLT_DATA_ROOT="${BLT_DATA_ROOT:-./data}"
BLT_PREPROCESS_DIR="${BLT_PREPROCESS_DIR:-${BLT_DATA_ROOT}/preprocess}"
BLT_ENTROPY_MODEL_NAME="${BLT_ENTROPY_MODEL_NAME:-transformer_100m}"
BLT_FINEWEB_ENTROPY_DIR="${BLT_FINEWEB_ENTROPY_DIR:-${BLT_PREPROCESS_DIR}/fineweb_edu_10bt/${BLT_ENTROPY_MODEL_NAME}}"
BLT_FINEWEB_DATA_DIR="${BLT_FINEWEB_DATA_DIR:-${BLT_DATA_ROOT}/fineweb_edu_10bt}"
BLT_SHUFFLED_DATASET_NAME="${BLT_SHUFFLED_DATASET_NAME:-fineweb_edu_10bt_shuffled}"
BLT_SHUFFLE_VAL_FRACTION="${BLT_SHUFFLE_VAL_FRACTION:-0.01}"
BLT_SHUFFLE_SEED="${BLT_SHUFFLE_SEED:-42}"
# Default to 1 train shard for 1-GPU runs. Increase this later to match the
# number of GPUs/chunks you want the training loader to consume.
BLT_SHUFFLE_NUM_TRAIN_SHARDS="${BLT_SHUFFLE_NUM_TRAIN_SHARDS:-1}"
BLT_SHUFFLE_NUM_VAL_SHARDS="${BLT_SHUFFLE_NUM_VAL_SHARDS:-1}"
UV_CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME}/.config}"

ensure_user_writable_dir() {
  local dir="$1"

  mkdir -p "${dir}" 2>/dev/null || "${SUDO[@]}" mkdir -p "${dir}"
  if [ ! -w "${dir}" ]; then
    if [ "${#SUDO[@]}" -gt 0 ]; then
      "${SUDO[@]}" chown -R "$(id -u):$(id -g)" "${dir}"
    else
      echo "error: ${dir} is not writable by $(id -un)"
      echo "set XDG_CONFIG_HOME to a writable path or fix the directory ownership"
      exit 1
    fi
  fi
}

# gpu arch. a6000=8.6, a100=8.0, rtx 4090=8.9, h100=9.0. adjust if needed.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

echo "preparing fineweb_edu_10bt entropy directory"
mkdir -p "${BLT_FINEWEB_ENTROPY_DIR}"
echo "entropy arrow files for fineweb_edu_10bt should go in: ${BLT_FINEWEB_ENTROPY_DIR}"

echo "[1/7] installing system build dependencies"
"${SUDO[@]}" apt-get update
if ! "${SUDO[@]}" apt-get install -y \
  build-essential \
  gcc-12 \
  g++-12 \
  ninja-build \
  wget \
  curl \
  git; then
  echo "apt install hit unmet dependencies; attempting apt-get --fix-broken install"
  "${SUDO[@]}" apt-get --fix-broken install -y
  "${SUDO[@]}" apt-get install -y \
    build-essential \
    gcc-12 \
    g++-12 \
    ninja-build \
    wget \
    curl \
    git
fi

# point gcc/g++ at version 12 for the cuda build. cuda 12.1 supports gcc <= 12.
export CC=gcc-12
export CXX=g++-12

echo "[2/7] checking cuda 12.1 toolkit"
if [ ! -x "${CUDA_INSTALL_PATH}/bin/nvcc" ]; then
  echo "installing cuda 12.1 toolkit to ${CUDA_INSTALL_PATH}"
  ensure_user_writable_dir "${CUDA_DOWNLOAD_DIR}"
  if [ ! -f "${CUDA_DOWNLOAD_DIR}/${CUDA_RUNFILE}" ]; then
    wget --continue -O "${CUDA_DOWNLOAD_DIR}/${CUDA_RUNFILE}" "${CUDA_URL}"
  fi
  mkdir -p "${CUDA_DOWNLOAD_DIR}/cuda_tmp"
  "${SUDO[@]}" sh "${CUDA_DOWNLOAD_DIR}/${CUDA_RUNFILE}" \
    --silent \
    --toolkit \
    --toolkitpath="${CUDA_INSTALL_PATH}" \
    --override \
    --tmpdir="${CUDA_DOWNLOAD_DIR}/cuda_tmp"
  # the installer sometimes leaves things root-only readable
  "${SUDO[@]}" chmod -R a+rX "${CUDA_INSTALL_PATH}"
else
  echo "cuda 12.1 already installed at ${CUDA_INSTALL_PATH}"
fi

export CUDA_HOME="${CUDA_INSTALL_PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "nvcc version:"
nvcc --version

echo "[3/7] installing uv"
ensure_user_writable_dir "${UV_CONFIG_HOME}"
ensure_user_writable_dir "${UV_CONFIG_HOME}/uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "[4/7] creating venv"
uv venv --allow-existing
# shellcheck disable=SC1091
source .venv/bin/activate
echo "venv python: $(which python)"

echo "[5/7] installing pre_build group (torch, setuptools, ninja)"
# pre_build must land before compile_xformers so the xformers build sees torch.
uv pip install --group pre_build --no-build-isolation

echo "[6/7] building xformers from source with cuda"
# flags consumed by xformers' setup.py during the compile step
export FORCE_CUDA=1
export XFORMERS_BUILD_TYPE=Release
export MAX_JOBS="${MAX_JOBS:-4}"

uv pip install --group compile_xformers --no-build-isolation

echo "[7/7] syncing remaining project dependencies"
uv sync

LAERDON_SSH_KEY="${HOME}/.ssh/laerdon_pkey"
FINEWEB_REMOTE_HOST="ubuntu@204.12.163.233"
FINEWEB_REMOTE_DIR="/mnt"
FINEWEB_CHUNK_NAME="fineweb_edu_10bt.chunk.00.jsonl"
FINEWEB_ARROW_FILE="${BLT_FINEWEB_ENTROPY_DIR}/${FINEWEB_CHUNK_NAME}.arrow"
FINEWEB_COMPLETE_FILE="${FINEWEB_ARROW_FILE}.complete"
FINEWEB_SHARD_ARROW_FILE="${BLT_FINEWEB_ENTROPY_DIR}/${FINEWEB_CHUNK_NAME}.shard_00.arrow"
FINEWEB_SHARD_COMPLETE_FILE="${FINEWEB_SHARD_ARROW_FILE}.complete"
FINEWEB_JSONL_FILE="${BLT_FINEWEB_DATA_DIR}/${FINEWEB_CHUNK_NAME}"

echo "checking for laerdon's ssh key"
mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"
if [ ! -f "${LAERDON_SSH_KEY}" ]; then
  echo "error: expected laerdon's key at ${LAERDON_SSH_KEY}"
  echo "copy it from your local machine first, then rerun this script"
  echo "example:"
  echo "  scp -P <port> -i ~/.ssh/<instance-login-key> ~/.ssh/laerdon_pkey $(id -un)@<instance-ip>:~/.ssh/laerdon_pkey"
  exit 1
fi
chmod 600 "${LAERDON_SSH_KEY}"
chmod 700 "${HOME}/.ssh"
if ! ssh-keygen -y -f "${LAERDON_SSH_KEY}" >/dev/null 2>&1; then
  echo "error: ${LAERDON_SSH_KEY} is not a valid SSH private key"
  exit 1
fi

echo "copying fineweb_edu_10bt arrow files"
scp -i "${LAERDON_SSH_KEY}" \
  "${FINEWEB_REMOTE_HOST}:${FINEWEB_REMOTE_DIR}/${FINEWEB_CHUNK_NAME}.arrow" \
  "${FINEWEB_REMOTE_HOST}:${FINEWEB_REMOTE_DIR}/${FINEWEB_CHUNK_NAME}.arrow.complete" \
  "${BLT_FINEWEB_ENTROPY_DIR}/"

echo "renaming fineweb_edu_10bt arrow files to shard format"
mv -f "${FINEWEB_ARROW_FILE}" "${FINEWEB_SHARD_ARROW_FILE}"
mv -f "${FINEWEB_COMPLETE_FILE}" "${FINEWEB_SHARD_COMPLETE_FILE}"

echo "reconstructing fineweb_edu_10bt jsonl chunk from arrow"
mkdir -p "${BLT_FINEWEB_DATA_DIR}"
bash "${SCRIPT_DIR}/arrow_to_jsonl.sh" "${FINEWEB_SHARD_ARROW_FILE}" "${FINEWEB_JSONL_FILE}"

echo "creating shuffled train/val split for fineweb_edu_10bt"
bash "${SCRIPT_DIR}/shuffle_split_arrow.sh" \
  --input-dataset-name fineweb_edu_10bt \
  --dataset-name "${BLT_SHUFFLED_DATASET_NAME}" \
  --entropy-model-name "${BLT_ENTROPY_MODEL_NAME}" \
  --val-fraction "${BLT_SHUFFLE_VAL_FRACTION}" \
  --seed "${BLT_SHUFFLE_SEED}" \
  --num-train-shards "${BLT_SHUFFLE_NUM_TRAIN_SHARDS}" \
  --num-val-shards "${BLT_SHUFFLE_NUM_VAL_SHARDS}" \
  --overwrite

echo "verification:"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())"
python -m xformers.info | grep -E "build.cuda_version|TORCH_CUDA_ARCH_LIST|cutlassF:|cutlassB:|fa2F|triton_splitK" || true

end_time=$(date +%s)
elapsed_minutes=$(( (end_time - start_time) / 60 ))
echo "done in ${elapsed_minutes} minutes"
echo "remember to 'source .venv/bin/activate' in new shells"
echo "and optionally add the cuda env vars to ~/.bashrc:"
echo "  export CUDA_HOME=${CUDA_INSTALL_PATH}"
echo "  export PATH=${CUDA_INSTALL_PATH}/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=${CUDA_INSTALL_PATH}/lib64:\$LD_LIBRARY_PATH"
