#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

if [ "$#" -ne 2 ]; then
  echo "usage: $0 INPUT_ARROW OUTPUT_JSONL"
  exit 1
fi

INPUT_ARROW="$1"
OUTPUT_JSONL="$2"

if [ ! -f "${INPUT_ARROW}" ]; then
  echo "error: input arrow file not found: ${INPUT_ARROW}"
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_JSONL}")"

python - "${INPUT_ARROW}" "${OUTPUT_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow as pa
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

input_arrow = Path(sys.argv[1])
output_jsonl = Path(sys.argv[2])

with pa.memory_map(str(input_arrow), "r") as source:
    reader = pa.ipc.open_file(source)
    missing = {"sample_id", "text"} - set(reader.schema.names)
    if missing:
        raise ValueError(
            f"input schema is missing required columns: {sorted(missing)}"
        )

    total_batches = reader.num_record_batches
    sample_id_index = reader.schema.get_field_index("sample_id")
    text_index = reader.schema.get_field_index("text")

    rows_written = 0
    with output_jsonl.open("w", encoding="utf-8") as out_f:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                "reconstructing jsonl rows=0",
                total=total_batches,
            )
            for batch_index in range(total_batches):
                batch = reader.get_batch(batch_index)
                sample_ids = batch.column(sample_id_index).to_pylist()
                texts = batch.column(text_index).to_pylist()
                for sample_id, text in zip(sample_ids, texts):
                    out_f.write(
                        json.dumps(
                            {"sample_id": str(sample_id), "text": text},
                            ensure_ascii=False,
                        )
                    )
                    out_f.write("\n")
                rows_written += batch.num_rows
                progress.update(
                    task,
                    advance=1,
                    description=f"reconstructing jsonl rows={rows_written}",
                )

print(f"wrote {rows_written} rows to {output_jsonl}")
PY
