#!/usr/bin/env python3
"""
Combine evaluation_results.csv files from all experiment directories
into an Excel workbook with two sheets:
  1. "All Trials" — every trial for every experiment (raw data)
  2. "Summary" — mean ± SD per experiment, one column per experiment

Handles both single-run experiments (evaluation_results.csv at top level)
and multi-trial experiments (trial_*/evaluation_results.csv).

Usage:
    python scripts/combine_results.py --results-dir full_results_arms
    python scripts/combine_results.py --results-dir full_results_arms --output full_results_arms/combined.xlsx
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


def find_eval_csvs(results_dir: Path) -> dict[str, Path]:
    """Find all evaluation_results.csv files, keyed by experiment/trial name."""
    found = {}
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        if name in ("projection-visualization",):
            continue

        trial_dirs = sorted(exp_dir.glob("trial_*/evaluation_results.csv"))
        if trial_dirs:
            for csv_path in trial_dirs:
                trial_name = csv_path.parent.name
                found[f"{name}/{trial_name}"] = csv_path
        else:
            csv_path = exp_dir / "evaluation_results.csv"
            if csv_path.exists():
                found[name] = csv_path

    return found


def read_eval_csv(path: Path) -> dict[str, str]:
    """Read an evaluation_results.csv and return {Metric: Value}."""
    metrics = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["Metric"]] = row["Value"]
    return metrics


def build_trials_dataframe(
    csvs: dict[str, Path],
) -> tuple[pd.DataFrame, list[str]]:
    """Build a DataFrame with metrics as rows and experiment/trials as columns."""
    all_metrics: list[str] = []
    all_data: dict[str, dict[str, str]] = {}

    for name, path in csvs.items():
        data = read_eval_csv(path)
        all_data[name] = data
        for metric in data:
            if metric not in all_metrics:
                all_metrics.append(metric)

    exp_names = sorted(all_data.keys())

    rows = []
    for metric in all_metrics:
        row = {"Metric": metric}
        for name in exp_names:
            val = all_data[name].get(metric, "")
            try:
                val = float(val)
            except (ValueError, TypeError):
                pass
            row[name] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("Metric")
    return df, all_metrics


def build_summary_dataframe(
    trials_df: pd.DataFrame,
    metrics_order: list[str],
) -> pd.DataFrame:
    """
    Summarise trials into mean ± SD per experiment.

    Groups columns by experiment name (stripping /trial_N suffix),
    computes mean and SD for numeric metrics, and formats as "mean ± SD".
    Non-numeric metrics (e.g. "64/64") are kept as-is.
    """
    # Group columns by experiment (strip /trial_N)
    experiments: dict[str, list[str]] = {}
    for col in trials_df.columns:
        exp_name = re.sub(r"/trial_\d+$", "", col)
        experiments.setdefault(exp_name, []).append(col)

    # Define desired column order: representation × model × input type
    repr_order = ["raw", "smoothed", "bspline", "fpc"]
    model_order = ["transformer", "mlp"]
    input_order = ["triaxial", "resultant"]

    def sort_key(name: str) -> tuple:
        parts = name.split("-")
        # e.g. "raw-transformer-triaxial" -> repr=raw, model=transformer, input=triaxial
        r = repr_order.index(parts[0]) if parts[0] in repr_order else 99
        m = model_order.index(parts[1]) if len(parts) > 1 and parts[1] in model_order else 99
        i = input_order.index(parts[2]) if len(parts) > 2 and parts[2] in input_order else 99
        return (i, r, m)

    exp_names = sorted(experiments.keys(), key=sort_key)

    summary_rows = []
    for metric in metrics_order:
        row = {"Metric": metric}
        for exp_name in exp_names:
            trial_cols = experiments[exp_name]
            values = []
            for col in trial_cols:
                val = trials_df.loc[metric, col] if metric in trials_df.index else ""
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass

            if values:
                mean = np.mean(values)
                if len(values) > 1:
                    sd = np.std(values, ddof=1)
                    row[exp_name] = f"{mean:.4f} ± {sd:.4f}"
                else:
                    row[exp_name] = f"{mean:.4f}"
            else:
                # Non-numeric (e.g. "64/64") — take first value
                first_val = trials_df.loc[metric, trial_cols[0]] if metric in trials_df.index else ""
                row[exp_name] = first_val

        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    df = df.set_index("Metric")
    return df


# Metrics to exclude from the summary (redundant full-precision duplicates)
EXCLUDE_FROM_SUMMARY = {
    "Jump Height R² (valid only)",
    "Peak Power R² (valid only)",
}


def main():
    parser = argparse.ArgumentParser(description="Combine evaluation results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_both"),
        help="Directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: <results-dir>/combined_evaluation_results.xlsx)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.results_dir / "combined_evaluation_results.xlsx"

    # Ensure .xlsx extension
    if args.output.suffix != ".xlsx":
        args.output = args.output.with_suffix(".xlsx")

    csvs = find_eval_csvs(args.results_dir)
    if not csvs:
        print(f"No evaluation_results.csv files found in {args.results_dir}")
        return

    print(f"Found {len(csvs)} result files")

    # Build DataFrames
    trials_df, metrics_order = build_trials_dataframe(csvs)

    summary_metrics = [m for m in metrics_order if m not in EXCLUDE_FROM_SUMMARY]
    summary_df = build_summary_dataframe(trials_df, summary_metrics)

    # Write Excel workbook with two sheets
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary")
        trials_df.to_excel(writer, sheet_name="All Trials")

        # Format both sheets
        from openpyxl.styles import Alignment
        from openpyxl.worksheet.properties import WorksheetProperties

        col_widths = {"Summary": 15, "All Trials": 12}

        for sheet_name in ("Summary", "All Trials"):
            ws = writer.sheets[sheet_name]

            # Auto-fit first column (metric names) to content
            max_len = 0
            for cell in ws["A"]:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions["A"].width = max_len + 2

            # Fixed width for data columns
            data_width = col_widths[sheet_name]
            for col_cells in list(ws.columns)[1:]:
                ws.column_dimensions[col_cells[0].column_letter].width = data_width

            # Rotate column headers to 45 degrees (row 1, skip first column)
            for cell in list(ws.iter_rows(min_row=1, max_row=1))[0][1:]:
                cell.alignment = Alignment(text_rotation=45, horizontal="center")

            # Set a modest default window size
            ws.sheet_view.windowProtection = False

        # Set workbook window size (width, height in twips: 1 twip = 1/20 pt)
        writer.book.properties
        wb = writer.book
        wb.views[0].windowWidth = 20000
        wb.views[0].windowHeight = 12000

    n_experiments = len(summary_df.columns)
    n_trials = len(trials_df.columns)
    print(f"Written to: {args.output}")
    print(f"  Summary:    {len(summary_metrics)} metrics × {n_experiments} experiments")
    print(f"  All Trials: {len(metrics_order)} metrics × {n_trials} trial columns")



if __name__ == "__main__":
    main()
