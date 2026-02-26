#!/usr/bin/env python3
"""
Prepare CMJ Dataset from MATLAB Source Files

Loads AccelerometerSignals.mat and GRFFeatures.mat, extracts CMJ data for
selected conditions (arms, noarms, or both), merges duplicate participants,
applies quality filters, and saves a portable .npz file.

Source data:
    - AccelerometerSignals.mat: ACC signals, takeoff indices, outcomes
    - GRFFeatures.mat: GRF curves, takeoff indices, subject/jump IDs
    - processedjumpdata.mat: Attribute data for participant ID mapping

Conditions:
    - arms (index 0): CMJs with arm swing
    - noarms (index 1): CMJs without arm swing

Duplicate participants (same person, two sessions with different IDs):
    21 <-> 121, 56 <-> 156, 63 <-> 163, 96 <-> 196
These are merged to give 69 unique participants from the original 73 internal IDs.

Quality exclusions (applied by default, see EXCLUSIONS.md):
    - Participant 39: all trials excluded (suspected sensor miscalibration)
    - Participant 4: arms trials excluded (X-axis ADC clipping)
    - Participant 19: all trials excluded (X-axis ADC clipping)

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --conditions both --output data/cmj_dataset_both.npz
    python scripts/prepare_dataset.py --conditions noarms --output data/cmj_dataset.npz
    python scripts/prepare_dataset.py --no-exclude-quality  # skip quality exclusions
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat


# Duplicate participant mappings (remap higher ID -> lower ID)
DUPLICATE_MAP = {121: 21, 156: 56, 163: 63, 196: 96}

# Condition indices in the MATLAB files
# Index 0 = all (combined), index 1 = noarms, index 2 = arms
CONDITION_INDICES = {'arms': 2, 'noarms': 1}

# Quality exclusions — see EXCLUSIONS.md for full analysis
# Participant 39: suspected sensor miscalibration (all trials)
# Participant 4: X-axis ADC clipping (arms trials only)
# Participant 19: X-axis ADC clipping (all trials)
EXCLUDE_ALL_TRIALS = {39, 19}  # Original participant IDs to exclude entirely
EXCLUDE_ARMS_TRIALS = {4}  # Original participant IDs to exclude arms trials only


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare CMJ dataset from MATLAB source files'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/Users/markgewhite/ARCHIVE/Data/Processed/All',
        help='Directory containing MATLAB source files'
    )
    parser.add_argument(
        '--conditions',
        type=str,
        default='both',
        choices=['both', 'arms', 'noarms'],
        help='Which CMJ conditions to include (default: both)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for .npz file (default: auto-named based on condition)'
    )
    parser.add_argument(
        '--acc-threshold',
        type=float,
        default=100.0,
        help='Exclude jumps with ACC resultant > threshold in g (default: 100)'
    )
    parser.add_argument(
        '--no-exclude-quality',
        action='store_true',
        default=False,
        help='Skip quality exclusions (miscalibrated/clipped trials). '
             'See EXCLUSIONS.md for details.'
    )
    return parser.parse_args()


def load_acc_data(data_dir: Path, condition_idx: int, condition_name: str):
    """Load accelerometer signals and outcomes for a given condition.

    Args:
        data_dir: Directory containing MATLAB source files
        condition_idx: Index into MATLAB arrays (0=arms, 1=noarms)
        condition_name: Name of condition ('arms' or 'noarms')
    """
    acc_path = data_dir / 'AccelerometerSignals.mat'
    print(f"Loading {acc_path} [{condition_name}]...")
    mat = loadmat(str(acc_path), squeeze_me=False, struct_as_record=False)

    # ACC signals: signal.raw[0, condition_idx] -> cell array of (n_timesteps, 3)
    signal_struct = mat['signal'][0, 0]
    acc_cell = signal_struct.raw[0, condition_idx]

    # ACC takeoff indices: signal.takeoff[condition_idx, 0] -> (N, 1)
    acc_takeoff = signal_struct.takeoff[condition_idx, 0].flatten().astype(np.int32)

    # Outcomes: outcomes.<condition_name>
    outcomes = mat['outcomes'][0, 0]
    cond_outcomes = getattr(outcomes, condition_name)[0, 0]
    jump_height = cond_outcomes.jumpHeight.flatten().astype(np.float32)
    peak_power = cond_outcomes.peakPower.flatten().astype(np.float32)

    # Extract individual ACC signals from cell array
    n_jumps = len(acc_takeoff)
    acc_signals = []
    for i in range(n_jumps):
        sig = acc_cell[i, 0]
        if isinstance(sig, np.ndarray) and sig.ndim == 0:
            sig = sig.item()
        acc_signals.append(np.array(sig, dtype=np.float32))

    print(f"  Loaded {n_jumps} ACC signals [{condition_name}]")
    return acc_signals, acc_takeoff, jump_height, peak_power


def load_grf_data(data_dir: Path, condition_idx: int, condition_name: str):
    """Load GRF curves and subject IDs for a given condition.

    Args:
        data_dir: Directory containing MATLAB source files
        condition_idx: Index into MATLAB arrays (0=arms, 1=noarms)
        condition_name: Name of condition ('arms' or 'noarms')
    """
    grf_path = data_dir / 'GRFFeatures.mat'
    print(f"Loading {grf_path} [{condition_name}]...")
    mat = loadmat(str(grf_path), squeeze_me=False, struct_as_record=False)

    # GRF signals: curveSet[0, condition_idx] -> cell array of (n_timesteps, 1)
    grf_cell = mat['curveSet'][0, condition_idx]

    # GRF takeoff indices: curveTOSet[0, condition_idx] -> (N, 1)
    grf_takeoff = mat['curveTOSet'][0, condition_idx].flatten().astype(np.int32)

    # Subject/jump IDs: curveIDSet[0, condition_idx] -> (N, 2)
    curve_ids = mat['curveIDSet'][0, condition_idx].astype(np.int32)

    # Extract individual GRF signals (stored at 1000 Hz)
    n_jumps = grf_cell.shape[0]
    grf_signals = []
    for i in range(n_jumps):
        sig = grf_cell[i, 0]
        if isinstance(sig, np.ndarray) and sig.ndim == 0:
            sig = sig.item()
        sig = np.array(sig, dtype=np.float32).flatten()
        grf_signals.append(sig)

    # Internal subject IDs (1-indexed) and jump IDs
    internal_subject_ids = curve_ids[:, 0]  # 1-indexed internal ID (1-73)

    print(f"  Loaded {n_jumps} GRF signals (1000 Hz) [{condition_name}]")
    return grf_signals, grf_takeoff, internal_subject_ids


def load_participant_mapping(data_dir: Path):
    """Load sDataID from processedjumpdata.mat to map internal -> original IDs."""
    mat_path = data_dir / 'processedjumpdata.mat'
    print(f"Loading participant mapping from {mat_path}...")
    mat = loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)

    # sDataID is a top-level variable (not inside the attribute struct).
    # It maps internal subject index (0-based, length 73) to original participant ID.
    # Note: attribute.sex and attribute.age are swapped in this file.
    s_data_id = mat['sDataID'].flatten().astype(np.int32)

    print(f"  Found {len(s_data_id)} internal-to-original ID mappings")
    return s_data_id


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    # Determine output path
    if args.output is not None:
        output_path = Path(args.output)
    elif args.conditions == 'both':
        output_path = Path('data/cmj_dataset_both.npz')
    elif args.conditions == 'noarms':
        output_path = Path('data/cmj_dataset.npz')
    else:
        output_path = Path(f'data/cmj_dataset_{args.conditions}.npz')

    # Determine which conditions to load
    if args.conditions == 'both':
        conditions = ['arms', 'noarms']
    else:
        conditions = [args.conditions]

    print("=" * 60)
    print("CMJ Dataset Preparation")
    print(f"Conditions: {', '.join(conditions)}")
    print("=" * 60)

    # Load participant mapping (shared across conditions)
    s_attribute_id = load_participant_mapping(data_dir)

    # Load and concatenate data from all selected conditions
    all_acc_signals = []
    all_acc_takeoff = []
    all_jump_height = []
    all_peak_power = []
    all_grf_signals = []
    all_grf_takeoff = []
    all_internal_subject_ids = []
    all_condition_labels = []

    for cond_name in conditions:
        cond_idx = CONDITION_INDICES[cond_name]
        cond_label = cond_idx  # 0=arms, 1=noarms

        acc_signals, acc_takeoff, jump_height, peak_power = load_acc_data(
            data_dir, cond_idx, cond_name
        )
        grf_signals, grf_takeoff, internal_subject_ids = load_grf_data(
            data_dir, cond_idx, cond_name
        )

        n_cond = len(acc_signals)
        assert len(grf_signals) == n_cond, (
            f"ACC ({len(acc_signals)}) and GRF ({len(grf_signals)}) counts don't match "
            f"for condition '{cond_name}'"
        )

        all_acc_signals.extend(acc_signals)
        all_acc_takeoff.append(acc_takeoff)
        all_jump_height.append(jump_height)
        all_peak_power.append(peak_power)
        all_grf_signals.extend(grf_signals)
        all_grf_takeoff.append(grf_takeoff)
        all_internal_subject_ids.append(internal_subject_ids)
        all_condition_labels.append(np.full(n_cond, cond_label, dtype=np.int32))

    # Concatenate across conditions
    acc_signals = all_acc_signals
    acc_takeoff = np.concatenate(all_acc_takeoff)
    jump_height = np.concatenate(all_jump_height)
    peak_power = np.concatenate(all_peak_power)
    grf_signals = all_grf_signals
    grf_takeoff = np.concatenate(all_grf_takeoff)
    internal_subject_ids = np.concatenate(all_internal_subject_ids)
    condition_labels = np.concatenate(all_condition_labels)

    n_total = len(acc_signals)

    # Map internal subject IDs (1-indexed) to original participant IDs
    print("\n--- Mapping Subject IDs ---")
    original_ids = np.array([
        s_attribute_id[sid - 1] for sid in internal_subject_ids  # 1-indexed -> 0-indexed
    ], dtype=np.int32)

    # Merge duplicate participants
    merged_ids = original_ids.copy()
    for old_id, new_id in DUPLICATE_MAP.items():
        n_remapped = np.sum(merged_ids == old_id)
        if n_remapped > 0:
            merged_ids[merged_ids == old_id] = new_id
            print(f"  Remapped {n_remapped} jumps: participant {old_id} -> {new_id}")

    unique_participants = np.unique(merged_ids)
    print(f"  {len(unique_participants)} unique participants after merging")

    # Re-index to contiguous 0-based IDs
    id_map = {orig: new for new, orig in enumerate(sorted(unique_participants))}
    subject_ids_0based = np.array([id_map[pid] for pid in merged_ids], dtype=np.int32)
    n_subjects = len(unique_participants)
    print(f"  Re-indexed to 0-{n_subjects - 1}")

    # Apply quality filters
    print("\n--- Applying Quality Filters ---")
    valid_mask = np.ones(n_total, dtype=bool)

    # Filter 1: ACC resultant > threshold (sensor artifacts)
    n_acc_excluded = 0
    for i in range(n_total):
        resultant = np.sqrt(np.sum(acc_signals[i] ** 2, axis=1))
        if np.max(resultant) > args.acc_threshold:
            valid_mask[i] = False
            n_acc_excluded += 1
    print(f"  Excluded {n_acc_excluded} jumps: ACC resultant > {args.acc_threshold} g")

    # Filter 2: Minimum pre-takeoff samples
    n_short_excluded = 0
    for i in range(n_total):
        if valid_mask[i] and acc_takeoff[i] < 100:
            valid_mask[i] = False
            n_short_excluded += 1
    print(f"  Excluded {n_short_excluded} jumps: ACC takeoff < 100 samples")

    # Filter 3: Quality exclusions (miscalibration and ADC clipping)
    n_quality_excluded = 0
    if not args.no_exclude_quality:
        for i in range(n_total):
            if not valid_mask[i]:
                continue
            orig_id = int(merged_ids[i])
            cond = int(condition_labels[i])
            # Exclude all trials for miscalibrated/fully clipped participants
            if orig_id in EXCLUDE_ALL_TRIALS:
                valid_mask[i] = False
                n_quality_excluded += 1
            # Exclude arms trials only for partially clipped participants
            elif orig_id in EXCLUDE_ARMS_TRIALS and cond == CONDITION_INDICES['arms']:
                valid_mask[i] = False
                n_quality_excluded += 1
        print(f"  Excluded {n_quality_excluded} jumps: quality issues "
              f"(see EXCLUSIONS.md)")
    else:
        print("  Quality exclusions: SKIPPED (--no-exclude-quality)")

    n_valid = np.sum(valid_mask)
    n_excluded = n_total - n_valid
    print(f"\n  Total: {n_total} -> {n_valid} valid jumps ({n_excluded} excluded)")

    # Apply filter
    valid_indices = np.where(valid_mask)[0]

    # Build output arrays
    out_acc_signals = np.empty(n_valid, dtype=object)
    out_grf_signals = np.empty(n_valid, dtype=object)
    out_acc_takeoff = np.zeros(n_valid, dtype=np.int32)
    out_grf_takeoff = np.zeros(n_valid, dtype=np.int32)
    out_subject_ids = np.zeros(n_valid, dtype=np.int32)
    out_original_ids = np.zeros(n_valid, dtype=np.int32)
    out_jump_height = np.zeros(n_valid, dtype=np.float32)
    out_peak_power = np.zeros(n_valid, dtype=np.float32)
    out_condition_labels = np.zeros(n_valid, dtype=np.int32)

    for out_i, src_i in enumerate(valid_indices):
        out_acc_signals[out_i] = acc_signals[src_i]
        out_grf_signals[out_i] = grf_signals[src_i]
        out_acc_takeoff[out_i] = acc_takeoff[src_i]
        out_grf_takeoff[out_i] = grf_takeoff[src_i]
        out_subject_ids[out_i] = subject_ids_0based[src_i]
        out_original_ids[out_i] = merged_ids[src_i]
        out_jump_height[out_i] = jump_height[src_i]
        out_peak_power[out_i] = peak_power[src_i]
        out_condition_labels[out_i] = condition_labels[src_i]

    # Save .npz
    print(f"\n--- Saving to {output_path} ---")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(output_path),
        acc_signals=out_acc_signals,
        acc_takeoff=out_acc_takeoff,
        grf_signals=out_grf_signals,
        grf_takeoff=out_grf_takeoff,
        subject_ids=out_subject_ids,
        original_participant_ids=out_original_ids,
        jump_height=out_jump_height,
        peak_power=out_peak_power,
        condition_labels=out_condition_labels,
        acc_sampling_rate=np.int32(250),
        grf_sampling_rate=np.int32(1000),
        n_subjects=np.int32(n_subjects),
        allow_pickle=True,
    )

    # Verify
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Conditions:     {', '.join(conditions)}")
    print(f"  Valid jumps:    {n_valid}")
    if len(conditions) > 1:
        for cond_name in conditions:
            cond_idx = CONDITION_INDICES[cond_name]
            n_cond = np.sum(out_condition_labels == cond_idx)
            print(f"    {cond_name}: {n_cond}")
    excl_parts = []
    if n_acc_excluded:
        excl_parts.append(f"{n_acc_excluded} sensor artifacts")
    if n_short_excluded:
        excl_parts.append(f"{n_short_excluded} too short")
    if n_quality_excluded:
        excl_parts.append(f"{n_quality_excluded} quality issues")
    excl_detail = ", ".join(excl_parts) if excl_parts else "none"
    print(f"  Excluded:       {n_excluded} ({excl_detail})")
    print(f"  Unique subjects: {n_subjects}")
    print(f"  Jump height:    [{out_jump_height.min():.3f}, {out_jump_height.max():.3f}] m")
    print(f"  Peak power:     [{out_peak_power.min():.1f}, {out_peak_power.max():.1f}] W/kg")
    print(f"  ACC sampling:   250 Hz")
    print(f"  GRF sampling:   1000 Hz")
    print(f"  Output:         {output_path}")


if __name__ == '__main__':
    main()
