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
BLT_TRAIN_DATASET_NAME="${BLT_TRAIN_DATASET_NAME:-fineweb_edu_10bt_mix_1p7bt}"
BLT_TRAIN_DIR="${BLT_TRAIN_DIR:-${BLT_PREPROCESS_DIR}/${BLT_TRAIN_DATASET_NAME}/${BLT_ENTROPY_MODEL_NAME}}"
BLT_VALIDATION_DIR="${BLT_VALIDATION_DIR:-${BLT_DATA_ROOT}/validation/${BLT_TRAIN_DATASET_NAME}/${BLT_ENTROPY_MODEL_NAME}}"
BLT_TRAIN_SOURCE_DIR="${BLT_TRAIN_SOURCE_DIR:-${BLT_DATA_ROOT}/${BLT_TRAIN_DATASET_NAME}}"
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

echo "preparing fineweb_edu_10bt train and validation directories"
mkdir -p "${BLT_TRAIN_DIR}" "${BLT_VALIDATION_DIR}" "${BLT_TRAIN_SOURCE_DIR}"
echo "train arrow files should go in: ${BLT_TRAIN_DIR}"
echo "validation arrow files should go in: ${BLT_VALIDATION_DIR}"

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
# Detect the GPU arch from the actual machine unless the user already
# overrode TORCH_CUDA_ARCH_LIST explicitly.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  if DETECTED_TORCH_CUDA_ARCH_LIST="$(
    python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(1)

major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
  )"; then
    export TORCH_CUDA_ARCH_LIST="${DETECTED_TORCH_CUDA_ARCH_LIST}"
    echo "detected TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} from GPU 0"
  else
    export TORCH_CUDA_ARCH_LIST="8.6"
    echo "warning: could not detect GPU compute capability; defaulting TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  fi
else
  echo "using user-provided TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
fi

# flags consumed by xformers' setup.py during the compile step
export FORCE_CUDA=1
export XFORMERS_BUILD_TYPE=Release
export MAX_JOBS="${MAX_JOBS:-4}"

uv pip install --group compile_xformers --no-build-isolation

echo "[7/7] syncing remaining project dependencies"
uv sync

LAERDON_SSH_KEY="${HOME}/.ssh/laerdon_pkey"
FINEWEB_REMOTE_HOST="ubuntu@204.12.169.234"
FINEWEB_REMOTE_DIR="/mnt"
FINEWEB_TRAIN_REMOTE_FILE="fineweb_edu_10bt_mix_1p7bt.chunk.00.jsonl.shard_00.arrow"
FINEWEB_VALIDATION_REMOTE_FILE="fineweb_edu_10bt.validation.05_06_07.5m.arrow"
FINEWEB_TRAIN_CHUNK_NAME="${FINEWEB_TRAIN_REMOTE_FILE%.shard_00.arrow}"
FINEWEB_TRAIN_ARROW_FILE="${BLT_TRAIN_DIR}/${FINEWEB_TRAIN_REMOTE_FILE}"
FINEWEB_VALIDATION_ARROW_FILE="${BLT_VALIDATION_DIR}/${FINEWEB_VALIDATION_REMOTE_FILE}"
FINEWEB_SOURCE_PLACEHOLDER="${BLT_TRAIN_SOURCE_DIR}/${FINEWEB_TRAIN_CHUNK_NAME}"

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

echo "copying fineweb_edu_10bt train and validation arrow files"
mkdir -p "${BLT_TRAIN_DIR}" "${BLT_VALIDATION_DIR}" "${BLT_TRAIN_SOURCE_DIR}"
scp -i "${LAERDON_SSH_KEY}" \
  "${FINEWEB_REMOTE_HOST}:${FINEWEB_REMOTE_DIR}/${FINEWEB_TRAIN_REMOTE_FILE}" \
  "${FINEWEB_TRAIN_ARROW_FILE}"
scp -i "${LAERDON_SSH_KEY}" \
  "${FINEWEB_REMOTE_HOST}:${FINEWEB_REMOTE_DIR}/${FINEWEB_VALIDATION_REMOTE_FILE}" \
  "${FINEWEB_VALIDATION_ARROW_FILE}"

echo "validating fineweb_edu_10bt arrow file formats"
TRAIN_ARROW_FILE="${FINEWEB_TRAIN_ARROW_FILE}" VALIDATION_ARROW_FILE="${FINEWEB_VALIDATION_ARROW_FILE}" python - <<'PY'
import os
from pathlib import Path

import pyarrow as pa


required_columns = {"sample_id", "text", "entropies"}


def validate_schema(path: Path, schema: pa.Schema) -> None:
    missing = required_columns - set(schema.names)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def ensure_ipc_file(path: Path) -> None:
    try:
        with pa.ipc.open_file(path) as reader:
            validate_schema(path, reader.schema)
        return
    except pa.ArrowInvalid:
        pass

    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with pa.ipc.open_stream(path) as reader:
        validate_schema(path, reader.schema)
        with pa.OSFile(str(tmp_path), "wb") as sink:
            with pa.ipc.new_file(sink, reader.schema) as writer:
                for batch in reader:
                    writer.write_batch(batch)
    tmp_path.replace(path)
    print(f"converted arrow stream to ipc file: {path}")


ensure_ipc_file(Path(os.environ["TRAIN_ARROW_FILE"]))
ensure_ipc_file(Path(os.environ["VALIDATION_ARROW_FILE"]))
PY

echo "marking fineweb_edu_10bt arrow files complete"
touch "${FINEWEB_TRAIN_ARROW_FILE}.complete"
touch "${FINEWEB_VALIDATION_ARROW_FILE}.complete"
touch "${FINEWEB_SOURCE_PLACEHOLDER}"

echo "verification:"
python - <<'PY'
import torch

print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
PY
python -m xformers.info | grep -E "build.cuda_version|TORCH_CUDA_ARCH_LIST|cutlassF:|cutlassB:|fa2F|triton_splitK" || true

end_time=$(date +%s)
elapsed_minutes=$(( (end_time - start_time) / 60 ))
echo "done in ${elapsed_minutes} minutes"
echo "remember to 'source .venv/bin/activate' in new shells"
echo "and optionally add the cuda env vars to ~/.bashrc:"
echo "  export CUDA_HOME=${CUDA_INSTALL_PATH}"
echo "  export PATH=${CUDA_INSTALL_PATH}/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=${CUDA_INSTALL_PATH}/lib64:\$LD_LIBRARY_PATH"
