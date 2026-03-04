#!/usr/bin/env python3
"""
Combine evaluation_results.csv files from all experiment directories
into a single wide-format CSV for easy comparison.

Handles both single-run experiments (evaluation_results.csv at top level)
and multi-trial experiments (trial_*/evaluation_results.csv).

Usage:
    python scripts/combine_results.py --results-dir results_both
    python scripts/combine_results.py --results-dir results_both --output results_both/combined.csv
"""

import argparse
import csv
from pathlib import Path


def find_eval_csvs(results_dir: Path) -> dict[str, Path]:
    """Find all evaluation_results.csv files, keyed by experiment/trial name."""
    found = {}
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        # Skip non-experiment directories
        if name in ("projection-visualization",):
            continue

        # Check for multi-trial structure
        trial_dirs = sorted(exp_dir.glob("trial_*/evaluation_results.csv"))
        if trial_dirs:
            for csv_path in trial_dirs:
                trial_name = csv_path.parent.name
                found[f"{name}/{trial_name}"] = csv_path
        else:
            # Single-run experiment
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
        help="Output CSV path (default: <results-dir>/combined_evaluation_results.csv)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.results_dir / "combined_evaluation_results.csv"

    csvs = find_eval_csvs(args.results_dir)
    if not csvs:
        print(f"No evaluation_results.csv files found in {args.results_dir}")
        return

    print(f"Found {len(csvs)} result files:")
    for name in csvs:
        print(f"  {name}")

    # Collect all metrics (preserving order from first file)
    all_metrics: list[str] = []
    all_data: dict[str, dict[str, str]] = {}
    for name, path in csvs.items():
        data = read_eval_csv(path)
        all_data[name] = data
        for metric in data:
            if metric not in all_metrics:
                all_metrics.append(metric)

    # Sort experiment names for consistent column order
    exp_names = sorted(all_data.keys())

    # Write combined CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric"] + exp_names)
        for metric in all_metrics:
            row = [metric]
            for name in exp_names:
                row.append(all_data[name].get(metric, ""))
            writer.writerow(row)

    print(f"\nCombined results written to: {args.output}")
    print(f"  {len(all_metrics)} metrics × {len(exp_names)} experiments")


if __name__ == "__main__":
    main()
