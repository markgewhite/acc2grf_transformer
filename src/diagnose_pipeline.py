#!/usr/bin/env python3
"""
Diagnostic script to trace through the pipeline and identify where things go wrong.
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_loader import CMJDataLoader, DEFAULT_DATA_PATH, SAMPLING_RATE
from src.biomechanics import compute_jump_height, compute_peak_power


def main():
    print("=" * 70)
    print("PIPELINE DIAGNOSTIC")
    print("=" * 70)

    # Load data
    loader = CMJDataLoader(
        data_path=DEFAULT_DATA_PATH,
        use_resultant=True,
        input_transform='raw',
        output_transform='raw',
    )
    loader.load_data()

    # Get raw data before any normalization (with skip_normalization flag)
    print("\n--- RAW DATA (skip_normalization=True) ---")
    acc_raw, grf_raw = loader.preprocess(skip_normalization=True)
    print(f"ACC shape: {acc_raw.shape}")
    print(f"GRF shape: {grf_raw.shape}")
    print(f"GRF range: [{grf_raw.min():.4f}, {grf_raw.max():.4f}] (should be ~0-3 BW)")
    print(f"GRF mean: {grf_raw.mean():.4f}")

    # Check a few individual samples
    print("\nSample GRF curves (first 3):")
    for i in range(3):
        grf_i = grf_raw[i].flatten()
        print(f"  Sample {i}: min={grf_i.min():.3f}, max={grf_i.max():.3f}, mean={grf_i.mean():.3f}")

    # Compute jump metrics from raw data
    print("\n--- BIOMECHANICS FROM RAW GRF ---")
    for i in range(3):
        jh = compute_jump_height(grf_raw[i], SAMPLING_RATE)
        pp = compute_peak_power(grf_raw[i], SAMPLING_RATE)
        print(f"Sample {i}: JH = {jh:.3f} m, PP = {pp:.1f} W/kg")

    # Now preprocess (normalize) - need a fresh loader
    print("\n--- PREPROCESSING (normalize) ---")
    loader2 = CMJDataLoader(
        data_path=DEFAULT_DATA_PATH,
        use_resultant=True,
        input_transform='raw',
        output_transform='raw',
    )
    loader2.load_data()
    acc_norm, grf_norm = loader2.preprocess()
    print(f"ACC normalized shape: {acc_norm.shape}")
    print(f"GRF normalized shape: {grf_norm.shape}")
    print(f"GRF normalized range: [{grf_norm.min():.4f}, {grf_norm.max():.4f}]")
    print(f"GRF normalized mean: {grf_norm.mean():.4f}")
    print(f"\nNormalization params:")
    print(f"  grf_mean_function shape: {loader2.grf_mean_function.shape}")
    print(f"  grf_mean_function range: [{loader2.grf_mean_function.min():.4f}, {loader2.grf_mean_function.max():.4f}]")
    print(f"  grf_std: {loader2.grf_std:.4f}")

    # Denormalize and verify
    print("\n--- DENORMALIZATION CHECK ---")
    grf_denorm = loader2.denormalize_grf(grf_norm[:10])
    print(f"GRF denormalized shape: {grf_denorm.shape}")
    print(f"GRF denormalized range: [{grf_denorm.min():.4f}, {grf_denorm.max():.4f}]")

    # Check round-trip error
    roundtrip_error = np.abs(grf_denorm - grf_raw[:10]).max()
    print(f"Round-trip max error: {roundtrip_error:.6f} (should be ~0)")

    # Compute jump metrics from denormalized data
    print("\n--- BIOMECHANICS FROM DENORMALIZED GRF ---")
    for i in range(3):
        jh = compute_jump_height(grf_denorm[i], SAMPLING_RATE)
        pp = compute_peak_power(grf_denorm[i], SAMPLING_RATE)
        print(f"Sample {i}: JH = {jh:.3f} m, PP = {pp:.1f} W/kg")

    # Now test: what if the model predicts the mean (constant output)?
    print("\n--- WHAT IF MODEL PREDICTS ZEROS (normalized space)? ---")
    zero_pred_norm = np.zeros_like(grf_norm[:10])
    zero_pred_denorm = loader2.denormalize_grf(zero_pred_norm)
    print(f"Zero pred denormalized = grf_mean_function")
    print(f"Range: [{zero_pred_denorm.min():.4f}, {zero_pred_denorm.max():.4f}]")
    for i in range(3):
        jh = compute_jump_height(zero_pred_denorm[i], SAMPLING_RATE)
        pp = compute_peak_power(zero_pred_denorm[i], SAMPLING_RATE)
        print(f"Sample {i}: JH = {jh:.3f} m, PP = {pp:.1f} W/kg")

    # Check ground truth values from .mat file (need to use get_train_val_split to populate these)
    print("\n--- GROUND TRUTH (from .mat file) ---")
    print("Getting train/val split to populate ground truth...")
    _, _, info = loader2.get_train_val_split()
    if loader2.val_gt_jump_height is not None:
        print(f"Val GT JH range: [{loader2.val_gt_jump_height.min():.3f}, {loader2.val_gt_jump_height.max():.3f}] m")
        print(f"Val GT PP range: [{loader2.val_gt_peak_power.min():.1f}, {loader2.val_gt_peak_power.max():.1f}] W")
        print(f"Train GT JH range: [{loader2.train_gt_jump_height.min():.3f}, {loader2.train_gt_jump_height.max():.3f}] m")
    else:
        print("No ground truth loaded")

    # Compare computed JH vs ground truth JH for first few samples
    print("\n--- COMPUTED VS GROUND TRUTH JUMP HEIGHT ---")
    print("(Compute JH from 500ms pre-takeoff GRF vs GT from full signal)")
    # Use training set GT
    n_compare = min(10, len(loader2.train_gt_jump_height))
    for i in range(n_compare):
        jh_computed = compute_jump_height(grf_raw[i], SAMPLING_RATE)
        jh_gt = loader2.train_gt_jump_height[i]
        print(f"Sample {i}: computed={jh_computed:.3f}m, GT={jh_gt:.3f}m, diff={jh_computed-jh_gt:.3f}m")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
