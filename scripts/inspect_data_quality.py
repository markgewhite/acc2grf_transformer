#!/usr/bin/env python3
"""
Inspect CMJ Dataset for Accelerometer Data Quality Issues

Checks for two categories of sensor artefact:
1. Miscalibrated sensors — participants whose peak resultant acceleration
   is systematically too low relative to their jump performance
2. ADC clipping — trials where an accelerometer axis shows runs of
   consecutive identical values, indicating analogue-to-digital converter
   saturation

Outputs a summary table and identifies specific trials/participants
for potential exclusion.

Usage:
    python scripts/inspect_data_quality.py
    python scripts/inspect_data_quality.py --data-path data/cmj_dataset_both.npz
    python scripts/inspect_data_quality.py --clip-run-length 15
"""

import argparse
from pathlib import Path

import numpy as np


DEFAULT_DATA_PATH = str(
    Path(__file__).parent.parent / "data" / "cmj_dataset_both.npz"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect CMJ dataset for accelerometer quality issues"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to .npz dataset file (default: data/cmj_dataset_both.npz)",
    )
    parser.add_argument(
        "--clip-run-length",
        type=int,
        default=10,
        help="Minimum consecutive identical samples to flag as clipping (default: 10)",
    )
    parser.add_argument(
        "--low-acc-percentile",
        type=float,
        default=5.0,
        help="Percentile threshold for flagging low peak acceleration (default: 5)",
    )
    return parser.parse_args()


def longest_constant_run(signal_1d):
    """Find the longest run of consecutive identical values in a 1D signal."""
    if len(signal_1d) < 2:
        return 1, signal_1d[0] if len(signal_1d) == 1 else 0.0

    diffs = np.diff(signal_1d)
    max_run = 1
    current_run = 1
    max_value = signal_1d[0]

    for i, d in enumerate(diffs):
        if d == 0.0:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
                max_value = signal_1d[i + 1]
        else:
            current_run = 1

    return max_run, max_value


def check_clipping(acc_signals, clip_threshold):
    """Check each trial for ADC clipping on any axis.

    Returns:
        list of dicts with trial index, axis, run length, and clipped value
        for every trial that exceeds the clip_threshold.
    """
    clipped_trials = []
    axis_names = ["X", "Y", "Z"]

    for i, sig in enumerate(acc_signals):
        for ax in range(sig.shape[1]):
            run_len, run_val = longest_constant_run(sig[:, ax])
            if run_len >= clip_threshold:
                clipped_trials.append({
                    "trial": i,
                    "axis": axis_names[ax],
                    "run_length": run_len,
                    "clipped_value": run_val,
                })

    return clipped_trials


def check_low_acceleration(acc_signals, subject_ids, original_ids,
                           jump_height, percentile):
    """Identify participants with systematically low peak resultant acceleration.

    Returns:
        list of dicts with participant info and statistics for those whose
        mean peak resultant is below the given percentile.
    """
    # Compute peak resultant for each trial
    peak_resultants = np.array([
        np.max(np.sqrt(np.sum(sig ** 2, axis=1)))
        for sig in acc_signals
    ])

    threshold = np.percentile(peak_resultants, percentile)

    # Group by participant
    unique_subjects = np.unique(subject_ids)
    flagged = []

    for sid in unique_subjects:
        mask = subject_ids == sid
        subj_peaks = peak_resultants[mask]
        subj_jh = jump_height[mask]
        orig_id = original_ids[mask][0]

        # Flag if ALL trials are below the percentile threshold
        if np.all(subj_peaks < threshold):
            flagged.append({
                "subject_id": int(sid),
                "original_id": int(orig_id),
                "n_trials": int(np.sum(mask)),
                "peak_acc_range": (float(subj_peaks.min()),
                                   float(subj_peaks.max())),
                "jump_height_range": (float(subj_jh.min()),
                                      float(subj_jh.max())),
                "mean_peak_acc": float(subj_peaks.mean()),
            })

    return flagged, peak_resultants, threshold


def main():
    args = parse_args()
    data_path = Path(args.data_path)

    print("=" * 70)
    print("CMJ Dataset Quality Inspection")
    print(f"Data: {data_path}")
    print("=" * 70)

    # Load data
    data = np.load(str(data_path), allow_pickle=True)
    acc_signals = data["acc_signals"]
    subject_ids = data["subject_ids"]
    original_ids = data["original_participant_ids"]
    jump_height = data["jump_height"]
    condition_labels = data.get("condition_labels", None)

    n_trials = len(acc_signals)
    n_subjects = int(data["n_subjects"])
    print(f"\nLoaded {n_trials} trials from {n_subjects} participants")

    # --- Check 1: ADC clipping ---
    print(f"\n{'─' * 70}")
    print(f"CHECK 1: ADC Clipping (>= {args.clip_run_length} consecutive "
          f"identical samples)")
    print(f"{'─' * 70}")

    clipped = check_clipping(acc_signals, args.clip_run_length)

    if clipped:
        # Group by participant
        clipped_trials = set(c["trial"] for c in clipped)
        clipped_subjects = {}
        for c in clipped:
            sid = int(subject_ids[c["trial"]])
            orig = int(original_ids[c["trial"]])
            key = (sid, orig)
            if key not in clipped_subjects:
                clipped_subjects[key] = {"trials": [], "details": []}
            clipped_subjects[key]["trials"].append(c["trial"])
            clipped_subjects[key]["details"].append(c)

        print(f"\nFound {len(clipped_trials)} trials with clipping "
              f"across {len(clipped_subjects)} participant(s):\n")

        for (sid, orig), info in sorted(clipped_subjects.items()):
            unique_trials = sorted(set(info["trials"]))
            print(f"  Participant {orig} (ID {sid}): "
                  f"{len(unique_trials)} trial(s) affected")
            for detail in info["details"]:
                cond = ""
                if condition_labels is not None:
                    cl = condition_labels[detail["trial"]]
                    cond = f" [{'noarms' if cl == 1 else 'arms'}]"
                print(f"    Trial {detail['trial']}{cond}: "
                      f"{detail['axis']}-axis, {detail['run_length']} samples "
                      f"at {detail['clipped_value']:.6f} g")
    else:
        print("\nNo clipping detected.")

    # --- Check 2: Systematically low peak acceleration ---
    print(f"\n{'─' * 70}")
    print(f"CHECK 2: Systematically Low Peak Acceleration "
          f"(all trials below {args.low_acc_percentile}th percentile)")
    print(f"{'─' * 70}")

    flagged, peak_resultants, threshold = check_low_acceleration(
        acc_signals, subject_ids, original_ids, jump_height,
        args.low_acc_percentile
    )

    print(f"\nDataset peak resultant statistics:")
    print(f"  Min:    {peak_resultants.min():.2f} g")
    print(f"  Median: {np.median(peak_resultants):.2f} g")
    print(f"  Max:    {peak_resultants.max():.2f} g")
    print(f"  {args.low_acc_percentile}th percentile threshold: "
          f"{threshold:.2f} g")

    if flagged:
        print(f"\nFlagged {len(flagged)} participant(s):\n")
        for f in flagged:
            print(f"  Participant {f['original_id']} (ID {f['subject_id']}): "
                  f"{f['n_trials']} trials")
            print(f"    Peak ACC: {f['peak_acc_range'][0]:.2f} - "
                  f"{f['peak_acc_range'][1]:.2f} g "
                  f"(mean {f['mean_peak_acc']:.2f} g)")
            print(f"    Jump height: {f['jump_height_range'][0]:.3f} - "
                  f"{f['jump_height_range'][1]:.3f} m")

            # Compare to dataset average for similar jump heights
            jh_lo, jh_hi = f["jump_height_range"]
            similar_mask = (jump_height >= jh_lo) & (jump_height <= jh_hi)
            if np.sum(similar_mask) > f["n_trials"]:
                similar_peaks = peak_resultants[similar_mask]
                print(f"    Expected peak ACC for JH {jh_lo:.2f}-{jh_hi:.2f} m: "
                      f"{similar_peaks.mean():.2f} g "
                      f"(dataset average)")
    else:
        print("\nNo participants flagged.")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    clipped_trial_set = set(c["trial"] for c in clipped) if clipped else set()
    flagged_trial_set = set()
    for f in flagged:
        mask = subject_ids == f["subject_id"]
        flagged_trial_set.update(np.where(mask)[0])

    all_flagged = clipped_trial_set | flagged_trial_set
    print(f"  Trials with clipping artefacts:      {len(clipped_trial_set)}")
    print(f"  Trials from low-ACC participants:     {len(flagged_trial_set)}")
    print(f"  Total unique trials flagged:          {len(all_flagged)}")
    print(f"  Remaining trials after exclusion:     "
          f"{n_trials - len(all_flagged)}")


if __name__ == "__main__":
    main()
