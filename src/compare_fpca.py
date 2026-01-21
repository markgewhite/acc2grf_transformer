#!/usr/bin/env python3
"""
Compare Custom FPCA vs scikit-fda FPCA implementations.

This script loads the jump data and compares the two FPCA implementations
to understand why they produce different results.

Usage:
    python src/compare_fpca.py [--n-components 15] [--use-triaxial]
"""

import argparse
import numpy as np
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def normalize_signals(signals: np.ndarray) -> tuple:
    """Z-score normalize signals."""
    mean = np.mean(signals)
    std = np.std(signals)
    normalized = (signals - mean) / std
    return normalized, mean, std


def main():
    parser = argparse.ArgumentParser(description='Compare FPCA implementations')
    parser.add_argument('--data-path', type=str,
                        default='/Users/markgewhite/ARCHIVE/Data/Processed/All/processedjumpdata.mat')
    parser.add_argument('--n-components', type=int, default=15)
    parser.add_argument('--use-triaxial', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("FPCA Implementation Comparison: Custom (Discrete) vs scikit-fda (L²)")
    print("=" * 70)

    # Load data using existing data loader
    print("\nLoading data using CMJDataLoader...")
    from src.data_loader import CMJDataLoader

    loader = CMJDataLoader(
        data_path=args.data_path,
        use_resultant=not args.use_triaxial,
        input_transform='raw',
        output_transform='raw',
    )
    loader.load_data()
    acc, grf = loader.preprocess(skip_normalization=True)

    print(f"  ACC shape: {acc.shape}")
    print(f"  GRF shape: {grf.shape}")

    # Normalize (simple z-score, not the robust method from data_loader)
    print("\nNormalizing signals with z-score...")
    acc_norm, acc_mean, acc_std = normalize_signals(acc)
    grf_norm, grf_mean, grf_std = normalize_signals(grf)
    print(f"  ACC mean: {acc_mean:.4f}, std: {acc_std:.4f}")
    print(f"  GRF mean: {grf_mean:.4f}, std: {grf_std:.4f}")

    # Import transformers
    from src.transformations_custom import CustomFPCATransformer
    from src.transformations import FPCATransformer as SkfdaFPCATransformer

    # Fit both on GRF (single channel, easier to compare)
    print(f"\nFitting FPCA with {args.n_components} components on GRF...")

    custom = CustomFPCATransformer(n_components=args.n_components, use_varimax=False)
    skfda = SkfdaFPCATransformer(n_components=args.n_components, use_varimax=False)

    custom.fit(grf_norm)
    skfda.fit(grf_norm)

    # Get scores
    custom_scores = custom.transform(grf_norm)
    skfda_scores = skfda.transform(grf_norm)

    print("\n" + "=" * 70)
    print("SCORE STATISTICS")
    print("=" * 70)

    print("\nCustom (Discrete dot product):")
    print(f"  Score shape: {custom_scores.shape}")
    print(f"  Score range: [{custom_scores.min():.4f}, {custom_scores.max():.4f}]")
    print(f"  Score std:   {np.std(custom_scores):.4f}")
    for i in range(min(5, args.n_components)):
        print(f"  FPC{i+1} std: {np.std(custom_scores[:, i, 0]):.4f}")

    print("\nscikit-fda (L² inner product):")
    print(f"  Score shape: {skfda_scores.shape}")
    print(f"  Score range: [{skfda_scores.min():.4f}, {skfda_scores.max():.4f}]")
    print(f"  Score std:   {np.std(skfda_scores):.4f}")
    for i in range(min(5, args.n_components)):
        print(f"  FPC{i+1} std: {np.std(skfda_scores[:, i, 0]):.4f}")

    # Variance explained
    print("\n" + "=" * 70)
    print("VARIANCE EXPLAINED")
    print("=" * 70)

    custom_var = custom.get_variance_explained()[0]
    skfda_var = skfda.get_variance_explained()[0]

    print("\n  Component | Custom    | scikit-fda")
    print("  " + "-" * 40)
    for i in range(min(10, args.n_components)):
        c_var = custom_var[i] if i < len(custom_var) else 0
        s_var = skfda_var[i] if i < len(skfda_var) else 0
        print(f"  FPC{i+1:2d}     | {c_var:.4f}    | {s_var:.4f}")

    # Reconstruction error
    print("\n" + "=" * 70)
    print("RECONSTRUCTION ERROR")
    print("=" * 70)

    custom_recon = custom.inverse_transform(custom_scores)
    skfda_recon = skfda.inverse_transform(skfda_scores)

    custom_rmse = np.sqrt(np.mean((grf_norm - custom_recon) ** 2))
    skfda_rmse = np.sqrt(np.mean((grf_norm - skfda_recon) ** 2))

    print(f"\nCustom RMSE:     {custom_rmse:.6f}")
    print(f"scikit-fda RMSE: {skfda_rmse:.6f}")

    # Eigenfunction comparison
    print("\n" + "=" * 70)
    print("EIGENFUNCTION COMPARISON")
    print("=" * 70)

    custom_ef = custom.get_eigenfunctions()[0]
    skfda_ef = skfda.get_eigenfunctions()[0]

    print(f"\nEigenfunction shape: Custom {custom_ef.shape}, scikit-fda {skfda_ef.shape}")

    # Compare first few eigenfunctions
    print("\nCorrelation between eigenfunctions:")
    for i in range(min(5, args.n_components)):
        # Eigenfunctions may have opposite signs, so use absolute correlation
        corr = np.abs(np.corrcoef(custom_ef[:, i], skfda_ef[:, i])[0, 1])
        print(f"  FPC{i+1}: {corr:.4f}")

    # Score correlation
    print("\nCorrelation between scores:")
    for i in range(min(5, args.n_components)):
        corr = np.corrcoef(custom_scores[:, i, 0], skfda_scores[:, i, 0])[0, 1]
        print(f"  FPC{i+1}: {corr:.4f}")

    # Mean function comparison
    print("\n" + "=" * 70)
    print("MEAN FUNCTION COMPARISON")
    print("=" * 70)

    custom_mean = custom._mean_function[:, 0]
    # scikit-fda stores mean differently
    skfda_mean = skfda._fpca_objects[0].mean_.data_matrix[0, :, 0]

    mean_corr = np.corrcoef(custom_mean, skfda_mean)[0, 1]
    mean_rmse = np.sqrt(np.mean((custom_mean - skfda_mean) ** 2))

    print(f"\nMean function correlation: {mean_corr:.6f}")
    print(f"Mean function RMSE: {mean_rmse:.6f}")
    print(f"Custom mean range: [{custom_mean.min():.4f}, {custom_mean.max():.4f}]")
    print(f"scikit-fda mean range: [{skfda_mean.min():.4f}, {skfda_mean.max():.4f}]")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The key differences between implementations:

1. COVARIANCE COMPUTATION:
   - Custom: cov = X.T @ X / (n-1)  [discrete sum over samples]
   - scikit-fda: Uses L² inner products with numerical integration

2. SCORE PROJECTION:
   - Custom: scores = centered_X @ eigenfunctions  [discrete dot product]
   - scikit-fda: Uses L² inner product ∫ X(t) * φ(t) dt

3. MEAN FUNCTION:
   - Custom: Pointwise mean across samples at each time point
   - scikit-fda: L² mean (may differ slightly)

The discrete approach treats all time points equally, while L² inner products
implicitly weight by the integration rule. This can lead to different
eigenfunctions and score distributions.
""")

    print("=" * 70)


if __name__ == '__main__':
    main()
