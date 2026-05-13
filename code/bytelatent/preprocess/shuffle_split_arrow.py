# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import glob
from pathlib import Path

import numpy as np
import pyarrow as pa

# pyarrow needs the initialization from this import
import pyarrow.dataset  # pyright: ignore


REQUIRED_COLUMNS = {"sample_id", "text", "entropies"}


def expand_input_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    paths = sorted(path for path in paths if path.exists() and path.suffix == ".arrow")
    if not paths:
        raise ValueError("no input arrow files found")
    return paths


def validate_schema(schema: pa.Schema) -> None:
    missing = REQUIRED_COLUMNS - set(schema.names)
    if missing:
        raise ValueError(f"input schema is missing required columns: {sorted(missing)}")


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()
    complete_path = Path(f"{path}.complete")
    if complete_path.exists():
        complete_path.unlink()


def make_writer(path: Path, schema: pa.Schema, overwrite: bool) -> pa.ipc.RecordBatchFileWriter:
    if overwrite:
        remove_if_exists(path)
    elif path.exists() or Path(f"{path}.complete").exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sink = pa.OSFile(str(path), "wb")
    return pa.ipc.new_file(sink, schema)


def write_indices(
    *,
    table: pa.Table,
    indices: np.ndarray,
    writer: pa.ipc.RecordBatchFileWriter,
) -> int:
    if len(indices) == 0:
        return 0
    index_array = pa.array(indices.astype(np.int64, copy=False))
    writer.write_table(table.take(index_array))
    return len(indices)


def flush_buffer(
    *,
    batches: list[pa.RecordBatch],
    rng: np.random.Generator,
    val_fraction: float,
    train_writers: list[pa.ipc.RecordBatchFileWriter],
    val_writers: list[pa.ipc.RecordBatchFileWriter],
) -> tuple[int, int]:
    table = pa.Table.from_batches(batches)
    row_count = table.num_rows
    if row_count == 0:
        return 0, 0

    order = rng.permutation(row_count)
    val_mask = rng.random(row_count) < val_fraction
    shuffled_val_mask = val_mask[order]
    shuffled_train_indices = order[~shuffled_val_mask]
    shuffled_val_indices = order[shuffled_val_mask]

    train_rows = 0
    if len(shuffled_train_indices) > 0:
        train_targets = rng.integers(
            low=0, high=len(train_writers), size=len(shuffled_train_indices)
        )
        for shard_id, writer in enumerate(train_writers):
            shard_indices = shuffled_train_indices[train_targets == shard_id]
            train_rows += write_indices(
                table=table,
                indices=shard_indices,
                writer=writer,
            )

    val_rows = 0
    if len(shuffled_val_indices) > 0:
        val_targets = rng.integers(
            low=0, high=len(val_writers), size=len(shuffled_val_indices)
        )
        for shard_id, writer in enumerate(val_writers):
            shard_indices = shuffled_val_indices[val_targets == shard_id]
            val_rows += write_indices(
                table=table,
                indices=shard_indices,
                writer=writer,
            )

    return train_rows, val_rows


def touch_complete(path: Path) -> None:
    Path(f"{path}.complete").touch()


def create_source_placeholders(source_dir: Path, dataset_name: str, num_train_shards: int) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    for shard_id in range(num_train_shards):
        (source_dir / f"{dataset_name}.chunk.{shard_id:02d}.jsonl").touch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="shuffle annotated entropy arrow rows into train and validation arrow files"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="input arrow files or glob patterns",
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--train-output-dir", required=True)
    parser.add_argument("--val-output-dir", required=True)
    parser.add_argument(
        "--source-dir",
        default=None,
        help="optional train source placeholder directory for arrow iterator discovery",
    )
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-shards", type=int, default=8)
    parser.add_argument("--num-val-shards", type=int, default=1)
    parser.add_argument("--read-batch-size", type=int, default=1024)
    parser.add_argument("--buffer-rows", type=int, default=5000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between 0 and 1")
    if args.num_train_shards <= 0 or args.num_val_shards <= 0:
        raise ValueError("num train shards and num val shards must be positive")
    if args.buffer_rows <= 0 or args.read_batch_size <= 0:
        raise ValueError("buffer rows and read batch size must be positive")

    input_paths = expand_input_paths(args.inputs)
    first_dataset = pa.dataset.dataset(str(input_paths[0]), format="arrow")
    schema = first_dataset.schema
    validate_schema(schema)

    train_output_dir = Path(args.train_output_dir)
    val_output_dir = Path(args.val_output_dir)
    train_paths = [
        train_output_dir / f"{args.dataset_name}.chunk.{shard_id:02d}.jsonl.shard_00.arrow"
        for shard_id in range(args.num_train_shards)
    ]
    val_paths = [
        val_output_dir / f"{args.dataset_name}.val.shard_{shard_id:02d}.arrow"
        for shard_id in range(args.num_val_shards)
    ]

    train_writers = [
        make_writer(path, schema=schema, overwrite=args.overwrite) for path in train_paths
    ]
    val_writers = [
        make_writer(path, schema=schema, overwrite=args.overwrite) for path in val_paths
    ]

    rng = np.random.default_rng(args.seed)
    buffer: list[pa.RecordBatch] = []
    buffered_rows = 0
    seen_rows = 0
    train_rows = 0
    val_rows = 0

    try:
        for input_path in input_paths:
            print(f"reading input arrow: {input_path}")
            dataset = pa.dataset.dataset(str(input_path), format="arrow")
            validate_schema(dataset.schema)
            for batch in dataset.to_batches(batch_size=args.read_batch_size):
                if args.max_rows is not None:
                    remaining_rows = args.max_rows - seen_rows
                    if remaining_rows <= 0:
                        break
                    if batch.num_rows > remaining_rows:
                        batch = batch.slice(0, remaining_rows)

                buffer.append(batch)
                buffered_rows += batch.num_rows
                seen_rows += batch.num_rows

                if buffered_rows >= args.buffer_rows:
                    flushed_train, flushed_val = flush_buffer(
                        batches=buffer,
                        rng=rng,
                        val_fraction=args.val_fraction,
                        train_writers=train_writers,
                        val_writers=val_writers,
                    )
                    train_rows += flushed_train
                    val_rows += flushed_val
                    print(
                        f"flushed rows: train={train_rows} val={val_rows} seen={seen_rows}"
                    )
                    buffer = []
                    buffered_rows = 0

            if args.max_rows is not None and seen_rows >= args.max_rows:
                break

        if buffer:
            flushed_train, flushed_val = flush_buffer(
                batches=buffer,
                rng=rng,
                val_fraction=args.val_fraction,
                train_writers=train_writers,
                val_writers=val_writers,
            )
            train_rows += flushed_train
            val_rows += flushed_val
    finally:
        for writer in train_writers + val_writers:
            writer.close()

    for path in train_paths + val_paths:
        touch_complete(path)

    if args.source_dir is not None:
        create_source_placeholders(
            Path(args.source_dir), args.dataset_name, args.num_train_shards
        )

    print(f"done. input_rows={seen_rows} train_rows={train_rows} val_rows={val_rows}")
    print(f"train output dir: {train_output_dir}")
    print(f"val output dir: {val_output_dir}")


if __name__ == "__main__":
    main()
