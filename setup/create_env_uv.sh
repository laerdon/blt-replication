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

if [ "$(id -u)" -eq 0 ]; then
  SUDO=()
elif command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  echo "error: this script needs root privileges for apt and cuda, but sudo was not found"
  exit 1
fi

# where to download the cuda runfile (needs ~5gb free; /tmp often too small)
CUDA_DOWNLOAD_DIR="${CUDA_DOWNLOAD_DIR:-/mnt}"
CUDA_INSTALL_PATH="/usr/local/cuda-12.1"
CUDA_RUNFILE="cuda_12.1.1_530.30.02_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/${CUDA_RUNFILE}"
BLT_DATA_ROOT="${BLT_DATA_ROOT:-./data}"
BLT_PREPROCESS_DIR="${BLT_PREPROCESS_DIR:-${BLT_DATA_ROOT}/preprocess}"
BLT_ENTROPY_MODEL_NAME="${BLT_ENTROPY_MODEL_NAME:-transformer_100m}"
BLT_FINEWEB_ENTROPY_DIR="${BLT_FINEWEB_ENTROPY_DIR:-${BLT_PREPROCESS_DIR}/fineweb_edu_10bt/${BLT_ENTROPY_MODEL_NAME}}"
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

echo "[1/7] installing system build dependencies"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  build-essential \
  gcc-12 \
  g++-12 \
  ninja-build \
  wget \
  curl \
  git

# point gcc/g++ at version 12 for the cuda build. cuda 12.1 supports gcc <= 12.
export CC=gcc-12
export CXX=g++-12

echo "[2/7] checking cuda 12.1 toolkit"
if [ ! -x "${CUDA_INSTALL_PATH}/bin/nvcc" ]; then
  echo "installing cuda 12.1 toolkit to ${CUDA_INSTALL_PATH}"
  mkdir -p "${CUDA_DOWNLOAD_DIR}"
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
uv venv
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

echo "preparing fineweb_edu_10bt entropy directory"
mkdir -p "${BLT_FINEWEB_ENTROPY_DIR}"
echo "entropy arrow files for fineweb_edu_10bt should go in: ${BLT_FINEWEB_ENTROPY_DIR}"

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
