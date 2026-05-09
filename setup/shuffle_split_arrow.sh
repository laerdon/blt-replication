#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

usage() {
  cat <<'EOF'
usage: setup/shuffle_split_arrow.sh [options] [INPUT_ARROW_OR_GLOB ...]

Shuffle entropy-annotated Arrow files into train/val splits that BLT can train on.

Defaults assume setup/create_env_uv.sh has already run and produced:
  ./data/preprocess/fineweb_edu_10bt/transformer_100m/fineweb_edu_10bt.chunk.*.jsonl.shard_00.arrow

If no input paths are provided, the script uses that default input glob.

Useful overrides:
  --dataset-name NAME         Output dataset name. default: fineweb_edu_10bt_shuffled
  --num-train-shards N        Train shard count. default: 1
  --num-val-shards N          Validation shard count. default: 1
  --val-fraction FLOAT        Validation split fraction. default: 0.01
  --seed INT                  Shuffle seed. default: 42
  --max-rows INT              Limit rows for a small smoke test
  --overwrite                 Replace existing output files

Examples:
  setup/shuffle_split_arrow.sh

  setup/shuffle_split_arrow.sh \
    --dataset-name fineweb_edu_10bt_shuffled \
    --num-train-shards 1 \
    --num-val-shards 1 \
    --overwrite

  setup/shuffle_split_arrow.sh \
    ./data/preprocess/fineweb_edu_10bt/transformer_100m/fineweb_edu_10bt.chunk.00.jsonl.shard_00.arrow \
    ./data/preprocess/fineweb_edu_10bt/transformer_100m/fineweb_edu_10bt.chunk.01.jsonl.shard_00.arrow \
    --dataset-name fineweb_edu_10bt_shuffled_multi \
    --overwrite
EOF
}

DATA_ROOT="./data"
PREPROCESS_DIR=""
VALIDATION_ROOT=""
ENTROPY_MODEL_NAME="transformer_100m"
INPUT_DATASET_NAME="fineweb_edu_10bt"
DATASET_NAME=""
TRAIN_OUTPUT_DIR=""
VAL_OUTPUT_DIR=""
SOURCE_DIR=""
VAL_FRACTION="0.01"
SEED="42"
NUM_TRAIN_SHARDS="1"
NUM_VAL_SHARDS="1"
READ_BATCH_SIZE="1024"
BUFFER_ROWS="5000"
MAX_ROWS=""
OVERWRITE=0
INPUTS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --preprocess-dir)
      PREPROCESS_DIR="$2"
      shift 2
      ;;
    --validation-root)
      VALIDATION_ROOT="$2"
      shift 2
      ;;
    --entropy-model-name)
      ENTROPY_MODEL_NAME="$2"
      shift 2
      ;;
    --input-dataset-name)
      INPUT_DATASET_NAME="$2"
      shift 2
      ;;
    --dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --train-output-dir)
      TRAIN_OUTPUT_DIR="$2"
      shift 2
      ;;
    --val-output-dir)
      VAL_OUTPUT_DIR="$2"
      shift 2
      ;;
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --val-fraction)
      VAL_FRACTION="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --num-train-shards)
      NUM_TRAIN_SHARDS="$2"
      shift 2
      ;;
    --num-val-shards)
      NUM_VAL_SHARDS="$2"
      shift 2
      ;;
    --read-batch-size)
      READ_BATCH_SIZE="$2"
      shift 2
      ;;
    --buffer-rows)
      BUFFER_ROWS="$2"
      shift 2
      ;;
    --max-rows)
      MAX_ROWS="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --)
      shift
      while [ "$#" -gt 0 ]; do
        INPUTS+=("$1")
        shift
      done
      ;;
    -*)
      echo "error: unknown option: $1"
      usage
      exit 1
      ;;
    *)
      INPUTS+=("$1")
      shift
      ;;
  esac
done

if [ -z "${PREPROCESS_DIR}" ]; then
  PREPROCESS_DIR="${DATA_ROOT}/preprocess"
fi
if [ -z "${VALIDATION_ROOT}" ]; then
  VALIDATION_ROOT="${DATA_ROOT}/validation"
fi
if [ -z "${DATASET_NAME}" ]; then
  DATASET_NAME="${INPUT_DATASET_NAME}_shuffled"
fi
if [ -z "${TRAIN_OUTPUT_DIR}" ]; then
  TRAIN_OUTPUT_DIR="${PREPROCESS_DIR}/${DATASET_NAME}/${ENTROPY_MODEL_NAME}"
fi
if [ -z "${VAL_OUTPUT_DIR}" ]; then
  VAL_OUTPUT_DIR="${VALIDATION_ROOT}/${DATASET_NAME}/${ENTROPY_MODEL_NAME}"
fi
if [ -z "${SOURCE_DIR}" ]; then
  SOURCE_DIR="${DATA_ROOT}/${DATASET_NAME}"
fi

if [ "${#INPUTS[@]}" -eq 0 ]; then
  INPUTS=(
    "${PREPROCESS_DIR}/${INPUT_DATASET_NAME}/${ENTROPY_MODEL_NAME}/${INPUT_DATASET_NAME}.chunk.*.jsonl.shard_00.arrow"
  )
fi

if ! python -c "import pyarrow" >/dev/null 2>&1; then
  echo "error: pyarrow is not installed in the active python environment"
  echo "run 'source .venv/bin/activate && uv sync' first"
  exit 1
fi

echo "shuffle_split_arrow.sh configuration:"
echo "  repo root: ${REPO_ROOT}"
echo "  inputs:"
for input_path in "${INPUTS[@]}"; do
  echo "    - ${input_path}"
done
echo "  input dataset name: ${INPUT_DATASET_NAME}"
echo "  output dataset name: ${DATASET_NAME}"
echo "  train output dir: ${TRAIN_OUTPUT_DIR}"
echo "  val output dir: ${VAL_OUTPUT_DIR}"
echo "  source dir: ${SOURCE_DIR}"
echo "  val fraction: ${VAL_FRACTION}"
echo "  seed: ${SEED}"
echo "  num train shards: ${NUM_TRAIN_SHARDS}"
echo "  num val shards: ${NUM_VAL_SHARDS}"
echo "  read batch size: ${READ_BATCH_SIZE}"
echo "  buffer rows: ${BUFFER_ROWS}"
if [ -n "${MAX_ROWS}" ]; then
  echo "  max rows: ${MAX_ROWS}"
fi
if [ "${OVERWRITE}" -eq 1 ]; then
  echo "  overwrite: enabled"
fi

CMD=(
  python
  -m
  bytelatent.preprocess.shuffle_split_arrow
)
CMD+=("${INPUTS[@]}")
CMD+=(
  --dataset-name "${DATASET_NAME}"
  --train-output-dir "${TRAIN_OUTPUT_DIR}"
  --val-output-dir "${VAL_OUTPUT_DIR}"
  --source-dir "${SOURCE_DIR}"
  --val-fraction "${VAL_FRACTION}"
  --seed "${SEED}"
  --num-train-shards "${NUM_TRAIN_SHARDS}"
  --num-val-shards "${NUM_VAL_SHARDS}"
  --read-batch-size "${READ_BATCH_SIZE}"
  --buffer-rows "${BUFFER_ROWS}"
)
if [ -n "${MAX_ROWS}" ]; then
  CMD+=(--max-rows "${MAX_ROWS}")
fi
if [ "${OVERWRITE}" -eq 1 ]; then
  CMD+=(--overwrite)
fi

"${CMD[@]}"
