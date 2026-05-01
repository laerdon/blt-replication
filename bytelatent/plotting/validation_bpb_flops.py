from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"could not parse {path}:{line_number}: {exc}") from exc
    return rows


def find_metrics_files(runs_dir: Path, inputs: list[Path]) -> list[Path]:
    if inputs:
        paths = []
        for input_path in inputs:
            if input_path.is_dir():
                paths.extend(input_path.glob("metrics.jsonl"))
            else:
                paths.append(input_path)
        return sorted(path for path in paths if path.name == "metrics.jsonl")

    return sorted(runs_dir.glob("**/metrics.jsonl"))


def run_name_for_metrics(metrics_path: Path, runs_dir: Path) -> str:
    run_dir = metrics_path.parent
    try:
        return str(run_dir.relative_to(runs_dir))
    except ValueError:
        return run_dir.name


def load_training_flops(metrics_path: Path) -> pd.DataFrame:
    rows = read_jsonl(metrics_path)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    required = {"global_step", "speed/FLOPS", "speed/wps", "optim/total_tokens"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{metrics_path} is missing columns: {sorted(missing)}")

    df = df.sort_values("global_step")
    flops_per_token = df["speed/FLOPS"] / df["speed/wps"]
    df["train_flops"] = flops_per_token * df["optim/total_tokens"]
    return df[["global_step", "train_flops"]].dropna()


def validation_rows_from_metrics_validation(
    path: Path,
) -> list[dict[str, Any]]:
    output = []
    for row in read_jsonl(path):
        global_step = row.get("global_step")
        created_at = row.get("created_at")
        for source, metrics in row.items():
            if source in {"global_step", "created_at"} or not isinstance(metrics, dict):
                continue
            bpb = metrics.get("bpb")
            if bpb is None:
                continue
            output.append(
                {
                    "global_step": global_step,
                    "created_at": created_at,
                    "validation_source": source,
                    "validation_bpb": bpb,
                }
            )
    return output


def validation_rows_from_metrics_eval(path: Path) -> list[dict[str, Any]]:
    output = []
    for row in read_jsonl(path):
        global_step = row.get("global_step")
        created_at = row.get("created_at")
        ppl_results = row.get("ppl")
        if not isinstance(ppl_results, dict):
            continue
        for source, metrics in ppl_results.items():
            if not isinstance(metrics, dict):
                continue
            bpb = metrics.get("bpb")
            if bpb is None:
                continue
            output.append(
                {
                    "global_step": global_step,
                    "created_at": created_at,
                    "validation_source": source,
                    "validation_bpb": bpb,
                }
            )
    return output


def load_validation_bpb(run_dir: Path) -> pd.DataFrame:
    validation_path = run_dir / "metrics.validation.jsonl"
    eval_path = run_dir / "metrics.eval.jsonl"

    if validation_path.exists():
        rows = validation_rows_from_metrics_validation(validation_path)
    elif eval_path.exists():
        rows = validation_rows_from_metrics_eval(eval_path)
    else:
        rows = []

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("global_step")


def build_plot_frame(metrics_path: Path, runs_dir: Path) -> pd.DataFrame:
    train_df = load_training_flops(metrics_path)
    val_df = load_validation_bpb(metrics_path.parent)
    if train_df.empty or val_df.empty:
        return pd.DataFrame()

    merged = pd.merge_asof(
        val_df.sort_values("global_step"),
        train_df.sort_values("global_step"),
        on="global_step",
        direction="backward",
    )
    merged = merged.dropna(subset=["train_flops", "validation_bpb"])
    if merged.empty:
        return merged

    merged["run"] = run_name_for_metrics(metrics_path, runs_dir)
    merged["series"] = merged["run"] + " / " + merged["validation_source"]
    merged["initial_validation_bpb"] = merged.groupby("series")[
        "validation_bpb"
    ].transform("first")
    merged["delta_validation_bpb"] = (
        merged["validation_bpb"] - merged["initial_validation_bpb"]
    )
    return merged


def create_chart(df: pd.DataFrame, y_column: str) -> alt.Chart:
    y_title = (
        "validation bpb change from first eval"
        if y_column == "delta_validation_bpb"
        else "validation bpb"
    )
    tooltip = [
        alt.Tooltip("run:N", title="run"),
        alt.Tooltip("validation_source:N", title="validation source"),
        alt.Tooltip("global_step:Q", title="step"),
        alt.Tooltip("train_flops:Q", title="training flops", format=".3e"),
        alt.Tooltip("validation_bpb:Q", title="validation bpb", format=".4f"),
        alt.Tooltip(
            "delta_validation_bpb:Q",
            title="delta validation bpb",
            format=".4f",
        ),
    ]
    base = alt.Chart(df).encode(
        x=alt.X("train_flops:Q", title="training flops").scale(type="log"),
        y=alt.Y(f"{y_column}:Q", title=y_title).scale(zero=False),
        color=alt.Color("series:N", title="run / validation source"),
        tooltip=tooltip,
    )
    return (
        (base.mark_line(point=True) + base.mark_point(size=70))
        .properties(width=900, height=520)
        .interactive()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="plot validation bpb change over cumulative training flops"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="metrics.jsonl files or run directories. defaults to runs/**/metrics.jsonl",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/validation_bpb_flops.html"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="optional csv path for the joined plot data",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="plot validation bpb instead of change from the first eval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_files = find_metrics_files(args.runs_dir, args.inputs)
    if not metrics_files:
        raise ValueError("no metrics.jsonl files found")

    frames = [build_plot_frame(path, args.runs_dir) for path in metrics_files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("no validation bpb rows found next to metrics.jsonl files")

    df = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    y_column = "validation_bpb" if args.absolute else "delta_validation_bpb"
    create_chart(df, y_column).save(args.output)
    print(f"wrote chart to {args.output}")

    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv_output, index=False)
        print(f"wrote plot data to {args.csv_output}")


if __name__ == "__main__":
    main()
