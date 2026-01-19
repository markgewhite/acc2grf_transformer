"""
Model Evaluation Module

Functions to evaluate model performance on signal prediction and
derived biomechanical metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import csv

from .biomechanics import (
    compute_metrics_comparison,
    print_metrics_summary,
    compute_jump_metrics_batch,
)
from .data_loader import CMJDataLoader, SAMPLING_RATE


def compute_signal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute signal-level metrics between predicted and actual GRF.

    Args:
        y_true: Actual GRF signals, shape (n_samples, seq_len, 1)
        y_pred: Predicted GRF signals, shape (n_samples, seq_len, 1)

    Returns:
        Dictionary with RMSE, MAE, R^2
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # R^2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    data_loader: CMJDataLoader,
    sampling_rate: float = SAMPLING_RATE,
    ground_truth_jh: np.ndarray = None,
    ground_truth_pp: np.ndarray = None,
    body_mass: np.ndarray = None,
) -> dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained SignalTransformer model
        X: Input accelerometer data (normalized)
        y: Target GRF data (normalized)
        data_loader: CMJDataLoader with normalization parameters
        sampling_rate: Sampling rate in Hz
        ground_truth_jh: Pre-computed jump heights from full signal (meters)
        ground_truth_pp: Pre-computed peak power from full signal (Watts)
        body_mass: Body mass in kg (for converting PP to W/kg)

    Returns:
        Dictionary with all evaluation metrics
    """
    # Get predictions (normalized)
    y_pred_normalized = model.predict(X, verbose=0)

    # Denormalize for biomechanical analysis
    y_true_bw = data_loader.denormalize_grf(y)
    y_pred_bw = data_loader.denormalize_grf(y_pred_normalized)

    # Signal-level metrics (on normalized data)
    signal_metrics = compute_signal_metrics(y, y_pred_normalized)

    # Signal-level metrics (on BW units)
    signal_metrics_bw = compute_signal_metrics(y_true_bw, y_pred_bw)

    # Biomechanical metrics (predicted vs actual 500ms curves)
    bio_metrics = compute_metrics_comparison(y_true_bw, y_pred_bw, sampling_rate)

    results = {
        'signal': {
            'normalized': signal_metrics,
            'body_weight': signal_metrics_bw,
        },
        'biomechanics': bio_metrics,
        'predictions': {
            'normalized': y_pred_normalized,
            'body_weight': y_pred_bw,
        },
        'actual': {
            'normalized': y,
            'body_weight': y_true_bw,
        },
    }

    # Add reference comparison: actual 500ms curve vs ground truth (full signal)
    if ground_truth_jh is not None and ground_truth_pp is not None:
        # Get metrics computed from actual 500ms curves
        actual_500ms_jh = bio_metrics['actual']['jump_height']
        actual_500ms_pp = bio_metrics['actual']['peak_power']

        # Convert ground truth peak power from Watts to W/kg
        if body_mass is not None:
            gt_pp_per_kg = ground_truth_pp / body_mass
        else:
            gt_pp_per_kg = ground_truth_pp  # Assume already in W/kg

        # Compute reference comparison statistics
        jh_diff = actual_500ms_jh - ground_truth_jh
        pp_diff = actual_500ms_pp - gt_pp_per_kg

        results['reference'] = {
            'ground_truth': {
                'jump_height': ground_truth_jh,
                'peak_power': gt_pp_per_kg,
            },
            'actual_500ms': {
                'jump_height': actual_500ms_jh,
                'peak_power': actual_500ms_pp,
            },
            'jump_height': {
                'rmse': np.sqrt(np.mean(jh_diff ** 2)),
                'mae': np.mean(np.abs(jh_diff)),
                'median_ae': np.median(np.abs(jh_diff)),
                'bias': np.mean(jh_diff),
                'r2': _compute_r2(ground_truth_jh, actual_500ms_jh),
            },
            'peak_power': {
                'rmse': np.sqrt(np.mean(pp_diff ** 2)),
                'mae': np.mean(np.abs(pp_diff)),
                'median_ae': np.median(np.abs(pp_diff)),
                'bias': np.mean(pp_diff),
                'r2': _compute_r2(gt_pp_per_kg, actual_500ms_pp),
            },
        }

    return results


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def print_evaluation_summary(results: dict) -> None:
    """Print a formatted evaluation summary."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)

    print("\n--- Signal Prediction Metrics ---")
    print("\nNormalized (z-score):")
    sig_norm = results['signal']['normalized']
    print(f"  RMSE: {sig_norm['rmse']:.4f}")
    print(f"  MAE:  {sig_norm['mae']:.4f}")
    print(f"  R^2:  {sig_norm['r2']:.4f}")

    print("\nBody Weight Units:")
    sig_bw = results['signal']['body_weight']
    print(f"  RMSE: {sig_bw['rmse']:.4f} BW")
    print(f"  MAE:  {sig_bw['mae']:.4f} BW")
    print(f"  R^2:  {sig_bw['r2']:.4f}")

    # Reference comparison: 500ms curve vs full signal ground truth
    if 'reference' in results:
        print("\n" + "-" * 60)
        print("REFERENCE: 500ms Actual Curve vs Full Signal Ground Truth")
        print("-" * 60)
        print("(Shows baseline error from using truncated 500ms signal)")

        ref = results['reference']
        ref_jh = ref['jump_height']
        ref_pp = ref['peak_power']

        print("\nJump Height (500ms vs Ground Truth):")
        print(f"  RMSE:      {ref_jh['rmse']:.4f} m")
        print(f"  Median AE: {ref_jh['median_ae']:.4f} m")
        print(f"  Bias:      {ref_jh['bias']:.4f} m")
        print(f"  R^2:       {ref_jh['r2']:.4f}")
        gt_jh = ref['ground_truth']['jump_height']
        print(f"  GT range:  [{gt_jh.min():.3f}, {gt_jh.max():.3f}] m")

        print("\nPeak Power (500ms vs Ground Truth):")
        print(f"  RMSE:      {ref_pp['rmse']:.2f} W/kg")
        print(f"  Median AE: {ref_pp['median_ae']:.2f} W/kg")
        print(f"  Bias:      {ref_pp['bias']:.2f} W/kg")
        print(f"  R^2:       {ref_pp['r2']:.4f}")
        gt_pp = ref['ground_truth']['peak_power']
        print(f"  GT range:  [{gt_pp.min():.1f}, {gt_pp.max():.1f}] W/kg")

    print_metrics_summary(results['biomechanics'])


def plot_predictions(
    results: dict,
    n_samples: int = 5,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """
    Plot predicted vs actual GRF curves.

    Args:
        results: Results dictionary from evaluate_model
        n_samples: Number of samples to plot
        save_path: Path to save figure
        figsize: Figure size per sample

    Returns:
        Matplotlib figure
    """
    y_true = results['actual']['body_weight']
    y_pred = results['predictions']['body_weight']

    n_total = len(y_true)
    indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

    fig, axes = plt.subplots(n_samples, 1, figsize=(figsize[0], figsize[1] * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        ax = axes[i]

        # Time axis (assuming 500 points normalized)
        time = np.arange(len(y_true[idx].flatten()))

        ax.plot(time, y_true[idx].flatten(), 'b-', linewidth=2, label='Actual')
        ax.plot(time, y_pred[idx].flatten(), 'r--', linewidth=2, label='Predicted')

        # Compute sample RMSE
        sample_rmse = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))

        ax.set_xlabel('Sample')
        ax.set_ylabel('GRF (BW)')
        ax.set_title(f'Sample {idx} - RMSE: {sample_rmse:.4f} BW')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_outliers(
    results: dict,
    X: np.ndarray,
    metric: str = 'jump_height',
    n_outliers: int = 5,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 3),
) -> plt.Figure:
    """
    Plot the worst outliers with input ACC and actual/predicted GRF.

    Args:
        results: Results dictionary from evaluate_model
        X: Input accelerometer data (normalized)
        metric: 'jump_height' or 'peak_power'
        n_outliers: Number of outliers to plot
        save_path: Path to save figure
        figsize: Figure size per outlier

    Returns:
        Matplotlib figure
    """
    bio = results['biomechanics']
    y_true = results['actual']['body_weight']
    y_pred = results['predictions']['body_weight']

    # Get outlier indices and info
    outliers = bio['outliers'][metric]
    indices = outliers['indices'][:n_outliers]
    actual_vals = outliers['actual'][:n_outliers]
    pred_vals = outliers['predicted'][:n_outliers]
    errors = outliers['errors'][:n_outliers]

    fig, axes = plt.subplots(n_outliers, 2, figsize=(figsize[0], figsize[1] * n_outliers))
    if n_outliers == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        # Time axis
        time = np.arange(500)

        # Plot input ACC (resultant if 1D, or show all 3 axes)
        ax_acc = axes[i, 0]
        acc = X[idx]
        if acc.shape[-1] == 1:
            ax_acc.plot(time, acc.flatten(), 'b-', linewidth=1.5)
        else:
            ax_acc.plot(time, acc[:, 0], 'r-', alpha=0.7, label='X')
            ax_acc.plot(time, acc[:, 1], 'g-', alpha=0.7, label='Y')
            ax_acc.plot(time, acc[:, 2], 'b-', alpha=0.7, label='Z')
            resultant = np.sqrt(np.sum(acc ** 2, axis=1))
            ax_acc.plot(time, resultant, 'k-', linewidth=1.5, label='Resultant')
            ax_acc.legend(loc='lower left', fontsize=7)

        ax_acc.set_xlabel('Sample')
        ax_acc.set_ylabel('ACC (normalized)')
        ax_acc.set_title(f'Sample {idx} - Input Accelerometer')
        ax_acc.grid(True, alpha=0.3)

        # Plot GRF comparison
        ax_grf = axes[i, 1]
        ax_grf.plot(time, y_true[idx].flatten(), 'b-', linewidth=2, label='Actual')
        ax_grf.plot(time, y_pred[idx].flatten(), 'r--', linewidth=2, label='Predicted')

        ax_grf.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax_grf.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5)

        # Add metric info
        if metric == 'jump_height':
            title = f'Sample {idx} | JH: actual={actual_vals[i]:.3f}m, pred={pred_vals[i]:.3f}m, error={errors[i]:.3f}m'
        else:
            title = f'Sample {idx} | PP: actual={actual_vals[i]:.1f}, pred={pred_vals[i]:.1f}, error={errors[i]:.1f} W/kg'

        ax_grf.set_xlabel('Sample')
        ax_grf.set_ylabel('GRF (BW)')
        ax_grf.set_title(title)
        ax_grf.legend(loc='lower left', fontsize=8)
        ax_grf.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_scatter_metrics(
    results: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create scatter plots for jump height and peak power.

    Args:
        results: Results dictionary from evaluate_model
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    bio = results['biomechanics']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Jump Height scatter
    ax = axes[0]
    actual_jh = bio['actual']['jump_height']
    pred_jh = bio['predicted']['jump_height']

    ax.scatter(actual_jh, pred_jh, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Identity line
    min_val, max_val = min(actual_jh.min(), pred_jh.min()), max(actual_jh.max(), pred_jh.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Identity')

    ax.set_xlabel('Actual Jump Height (m)')
    ax.set_ylabel('Predicted Jump Height (m)')
    ax.set_title(f"Jump Height: R² = {bio['jump_height']['r2']:.3f}, RMSE = {bio['jump_height']['rmse']:.4f} m")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # Peak Power scatter
    ax = axes[1]
    actual_pp = bio['actual']['peak_power']
    pred_pp = bio['predicted']['peak_power']

    ax.scatter(actual_pp, pred_pp, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Identity line
    min_val, max_val = min(actual_pp.min(), pred_pp.min()), max(actual_pp.max(), pred_pp.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Identity')

    ax.set_xlabel('Actual Peak Power (W/kg)')
    ax.set_ylabel('Predicted Peak Power (W/kg)')
    ax.set_title(f"Peak Power: R² = {bio['peak_power']['r2']:.3f}, RMSE = {bio['peak_power']['rmse']:.2f} W/kg")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_bland_altman(
    results: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create Bland-Altman plots for jump height and peak power.

    Args:
        results: Results dictionary from evaluate_model
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    bio = results['biomechanics']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Jump Height Bland-Altman
    ax = axes[0]
    actual_jh = bio['actual']['jump_height']
    pred_jh = bio['predicted']['jump_height']

    mean_jh = (actual_jh + pred_jh) / 2
    diff_jh = pred_jh - actual_jh
    bias_jh = np.mean(diff_jh)
    loa_jh = 1.96 * np.std(diff_jh)

    ax.scatter(mean_jh, diff_jh, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(y=bias_jh, color='red', linestyle='-', label=f'Bias: {bias_jh:.4f} m')
    ax.axhline(y=bias_jh + loa_jh, color='red', linestyle='--', label=f'+1.96 SD: {bias_jh + loa_jh:.4f} m')
    ax.axhline(y=bias_jh - loa_jh, color='red', linestyle='--', label=f'-1.96 SD: {bias_jh - loa_jh:.4f} m')

    ax.set_xlabel('Mean Jump Height (m)')
    ax.set_ylabel('Difference (Predicted - Actual) (m)')
    ax.set_title('Jump Height - Bland-Altman Plot')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Peak Power Bland-Altman
    ax = axes[1]
    actual_pp = bio['actual']['peak_power']
    pred_pp = bio['predicted']['peak_power']

    mean_pp = (actual_pp + pred_pp) / 2
    diff_pp = pred_pp - actual_pp
    bias_pp = np.mean(diff_pp)
    loa_pp = 1.96 * np.std(diff_pp)

    ax.scatter(mean_pp, diff_pp, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(y=bias_pp, color='red', linestyle='-', label=f'Bias: {bias_pp:.2f} W/kg')
    ax.axhline(y=bias_pp + loa_pp, color='red', linestyle='--', label=f'+1.96 SD: {bias_pp + loa_pp:.2f} W/kg')
    ax.axhline(y=bias_pp - loa_pp, color='red', linestyle='--', label=f'-1.96 SD: {bias_pp - loa_pp:.2f} W/kg')

    ax.set_xlabel('Mean Peak Power (W/kg)')
    ax.set_ylabel('Difference (Predicted - Actual) (W/kg)')
    ax.set_title('Peak Power - Bland-Altman Plot')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def save_results_csv(
    results: dict,
    save_path: str,
) -> None:
    """
    Save evaluation results to CSV file.

    Args:
        results: Results dictionary from evaluate_model
        save_path: Path to save CSV
    """
    bio = results['biomechanics']
    jh = bio['jump_height']
    pp = bio['peak_power']
    valid = bio.get('valid_samples', {})

    rows = [
        ['Metric', 'Category', 'Value', 'Unit'],
        ['Signal RMSE (normalized)', 'Signal', f"{results['signal']['normalized']['rmse']:.6f}", 'z-score'],
        ['Signal MAE (normalized)', 'Signal', f"{results['signal']['normalized']['mae']:.6f}", 'z-score'],
        ['Signal R²', 'Signal', f"{results['signal']['normalized']['r2']:.6f}", '-'],
        ['Signal RMSE (BW)', 'Signal', f"{results['signal']['body_weight']['rmse']:.6f}", 'BW'],
        ['Signal MAE (BW)', 'Signal', f"{results['signal']['body_weight']['mae']:.6f}", 'BW'],
        ['Valid Samples', 'Data Quality', f"{valid.get('n_valid', 'N/A')}/{valid.get('n_total', 'N/A')}", 'count'],
        ['Jump Height RMSE', 'Biomechanics', f"{jh['rmse']:.6f}", 'm'],
        ['Jump Height MAE', 'Biomechanics', f"{jh['mae']:.6f}", 'm'],
        ['Jump Height Median AE', 'Biomechanics', f"{jh['median_ae']:.6f}", 'm'],
        ['Jump Height Bias', 'Biomechanics', f"{jh['bias']:.6f}", 'm'],
        ['Jump Height R²', 'Biomechanics', f"{jh['r2']:.6f}", '-'],
        ['Jump Height 90th %ile Error', 'Biomechanics', f"{jh['p90_error']:.6f}", 'm'],
        ['Jump Height R² (valid only)', 'Biomechanics', f"{jh.get('r2_valid', 'N/A')}", '-'],
        ['Peak Power RMSE', 'Biomechanics', f"{pp['rmse']:.4f}", 'W/kg'],
        ['Peak Power MAE', 'Biomechanics', f"{pp['mae']:.4f}", 'W/kg'],
        ['Peak Power Median AE', 'Biomechanics', f"{pp['median_ae']:.4f}", 'W/kg'],
        ['Peak Power Bias', 'Biomechanics', f"{pp['bias']:.4f}", 'W/kg'],
        ['Peak Power R²', 'Biomechanics', f"{pp['r2']:.6f}", '-'],
        ['Peak Power 90th %ile Error', 'Biomechanics', f"{pp['p90_error']:.4f}", 'W/kg'],
        ['Peak Power R² (valid only)', 'Biomechanics', f"{pp.get('r2_valid', 'N/A')}", '-'],
    ]

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Results saved to {save_path}")
