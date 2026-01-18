"""
Biomechanics Module

Functions to compute jump performance metrics from GRF signals,
matching the MATLAB implementations (jumpheight.m and jumppeakpower.m).
"""

import numpy as np
from typing import Union

from .data_loader import SAMPLING_RATE

# Constants
GRAVITY = 9.812  # m/s^2 (standard gravity)


def compute_jump_height(
    grf: np.ndarray,
    sampling_rate: float = SAMPLING_RATE,
) -> float:
    """
    Compute jump height from vertical GRF signal.

    Uses the impulse-momentum method matching MATLAB jumpheight.m:
    1. Convert GRF (in BW) to net acceleration
    2. Integrate acceleration to get velocity
    3. Integrate velocity to get position
    4. Add kinetic energy contribution at takeoff

    Args:
        grf: GRF signal in body weight (BW) units, shape (n_timesteps,) or (n_timesteps, 1)
        sampling_rate: Sampling rate in Hz (default 1000)

    Returns:
        Jump height in meters
    """
    grf = np.asarray(grf).flatten()

    # Net GRF (subtract 1 BW for body weight)
    net_grf = grf - 1.0

    # Convert to acceleration (multiply by g since GRF is in BW units)
    # F = m*a, and F/BW = a/g, so a = (F/BW) * g
    acceleration = net_grf * GRAVITY  # m/s^2

    # Integrate acceleration to get velocity using cumulative trapezoid
    dt = 1.0 / sampling_rate
    velocity = np.cumsum(acceleration) * dt  # Simple cumulative sum (matches MATLAB cumtrapz behavior closely)

    # Integrate velocity to get position
    position = np.cumsum(velocity) * dt

    # Final height = position at takeoff + additional height from takeoff velocity
    # h_total = h_position + v^2 / (2g)
    takeoff_velocity = velocity[-1]
    jump_height = position[-1] + 0.5 * takeoff_velocity ** 2 / GRAVITY

    return jump_height


def compute_peak_power(
    grf: np.ndarray,
    sampling_rate: float = SAMPLING_RATE,
) -> float:
    """
    Compute peak power from vertical GRF signal.

    Uses the method matching MATLAB jumppeakpower.m:
    1. Integrate net force to get velocity
    2. Compute instantaneous power as P = F * v
    3. Return peak power in W/kg

    Args:
        grf: GRF signal in body weight (BW) units, shape (n_timesteps,) or (n_timesteps, 1)
        sampling_rate: Sampling rate in Hz (default 1000)

    Returns:
        Peak power in W/kg (watts per kilogram of body mass)
    """
    grf = np.asarray(grf).flatten()

    # Net GRF
    net_grf = grf - 1.0

    # Integrate net force (in BW) to get velocity
    # v = integral(a) = integral(F/m) = integral((F/BW) * g)
    dt = 1.0 / sampling_rate
    velocity = GRAVITY * np.cumsum(net_grf) * dt  # m/s

    # Instantaneous power: P = F * v
    # With F in BW units, P = grf * v gives power in BW*m/s
    # To get W/kg: multiply by g (since BW = m*g)
    power = grf * velocity  # BW * m/s

    # Peak power in W/kg
    peak_power = GRAVITY * np.max(power)

    return peak_power


def compute_jump_metrics_batch(
    grf_batch: np.ndarray,
    sampling_rate: float = SAMPLING_RATE,
) -> dict:
    """
    Compute jump height and peak power for a batch of GRF signals.

    Args:
        grf_batch: Batch of GRF signals, shape (n_samples, seq_len) or (n_samples, seq_len, 1)
        sampling_rate: Sampling rate in Hz

    Returns:
        Dictionary with 'jump_height' and 'peak_power' arrays
    """
    grf_batch = np.asarray(grf_batch)
    if grf_batch.ndim == 3:
        grf_batch = grf_batch.squeeze(-1)

    n_samples = len(grf_batch)

    jump_heights = np.zeros(n_samples)
    peak_powers = np.zeros(n_samples)

    for i in range(n_samples):
        jump_heights[i] = compute_jump_height(grf_batch[i], sampling_rate)
        peak_powers[i] = compute_peak_power(grf_batch[i], sampling_rate)

    return {
        'jump_height': jump_heights,
        'peak_power': peak_powers,
    }


def compute_metrics_comparison(
    grf_actual: np.ndarray,
    grf_predicted: np.ndarray,
    sampling_rate: float = SAMPLING_RATE,
) -> dict:
    """
    Compare jump metrics between actual and predicted GRF.

    Args:
        grf_actual: Actual GRF signals, shape (n_samples, seq_len, 1)
        grf_predicted: Predicted GRF signals, shape (n_samples, seq_len, 1)
        sampling_rate: Sampling rate in Hz

    Returns:
        Dictionary with actual/predicted metrics and comparison statistics
    """
    actual_metrics = compute_jump_metrics_batch(grf_actual, sampling_rate)
    predicted_metrics = compute_jump_metrics_batch(grf_predicted, sampling_rate)

    # Compute errors
    jh_errors = predicted_metrics['jump_height'] - actual_metrics['jump_height']
    pp_errors = predicted_metrics['peak_power'] - actual_metrics['peak_power']

    # Identify valid samples (actual jump height >= 0)
    valid_mask = actual_metrics['jump_height'] >= 0
    n_valid = np.sum(valid_mask)
    n_total = len(valid_mask)

    # Compute statistics on all samples
    results = {
        'actual': actual_metrics,
        'predicted': predicted_metrics,
        'jump_height': {
            'rmse': np.sqrt(np.mean(jh_errors ** 2)),
            'mae': np.mean(np.abs(jh_errors)),
            'median_ae': np.median(np.abs(jh_errors)),
            'bias': np.mean(jh_errors),
            'r2': _compute_r2(actual_metrics['jump_height'], predicted_metrics['jump_height']),
            'p90_error': np.percentile(np.abs(jh_errors), 90),
            'errors': jh_errors,
        },
        'peak_power': {
            'rmse': np.sqrt(np.mean(pp_errors ** 2)),
            'mae': np.mean(np.abs(pp_errors)),
            'median_ae': np.median(np.abs(pp_errors)),
            'bias': np.mean(pp_errors),
            'r2': _compute_r2(actual_metrics['peak_power'], predicted_metrics['peak_power']),
            'p90_error': np.percentile(np.abs(pp_errors), 90),
            'errors': pp_errors,
        },
        'valid_samples': {
            'n_valid': n_valid,
            'n_total': n_total,
            'mask': valid_mask,
        },
    }

    # Compute robust statistics on valid samples only
    if n_valid > 0:
        jh_errors_valid = jh_errors[valid_mask]
        pp_errors_valid = pp_errors[valid_mask]
        jh_actual_valid = actual_metrics['jump_height'][valid_mask]
        jh_pred_valid = predicted_metrics['jump_height'][valid_mask]
        pp_actual_valid = actual_metrics['peak_power'][valid_mask]
        pp_pred_valid = predicted_metrics['peak_power'][valid_mask]

        results['jump_height']['rmse_valid'] = np.sqrt(np.mean(jh_errors_valid ** 2))
        results['jump_height']['median_ae_valid'] = np.median(np.abs(jh_errors_valid))
        results['jump_height']['r2_valid'] = _compute_r2(jh_actual_valid, jh_pred_valid)

        results['peak_power']['rmse_valid'] = np.sqrt(np.mean(pp_errors_valid ** 2))
        results['peak_power']['median_ae_valid'] = np.median(np.abs(pp_errors_valid))
        results['peak_power']['r2_valid'] = _compute_r2(pp_actual_valid, pp_pred_valid)

    # Find worst outliers (by absolute error)
    worst_jh_idx = np.argsort(np.abs(jh_errors))[-5:][::-1]
    worst_pp_idx = np.argsort(np.abs(pp_errors))[-5:][::-1]

    results['outliers'] = {
        'jump_height': {
            'indices': worst_jh_idx,
            'actual': actual_metrics['jump_height'][worst_jh_idx],
            'predicted': predicted_metrics['jump_height'][worst_jh_idx],
            'errors': jh_errors[worst_jh_idx],
        },
        'peak_power': {
            'indices': worst_pp_idx,
            'actual': actual_metrics['peak_power'][worst_pp_idx],
            'predicted': predicted_metrics['peak_power'][worst_pp_idx],
            'errors': pp_errors[worst_pp_idx],
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


def print_metrics_summary(metrics: dict) -> None:
    """Print a formatted summary of metrics comparison."""
    print("\n" + "=" * 60)
    print("Jump Performance Metrics Comparison")
    print("=" * 60)

    # Valid samples info
    valid = metrics.get('valid_samples', {})
    if valid:
        n_valid = valid.get('n_valid', 0)
        n_total = valid.get('n_total', 0)
        n_invalid = n_total - n_valid
        print(f"\nSamples: {n_total} total, {n_valid} valid, {n_invalid} invalid (negative JH)")

    print("\nJump Height (meters):")
    jh = metrics['jump_height']
    print(f"  RMSE:       {jh['rmse']:.4f} m")
    print(f"  MAE:        {jh['mae']:.4f} m")
    print(f"  Median AE:  {jh['median_ae']:.4f} m")
    print(f"  Bias:       {jh['bias']:.4f} m")
    print(f"  R^2:        {jh['r2']:.4f}")
    print(f"  90th %ile:  {jh['p90_error']:.4f} m")
    if 'r2_valid' in jh:
        print(f"  --- Valid samples only ---")
        print(f"  RMSE:       {jh['rmse_valid']:.4f} m")
        print(f"  Median AE:  {jh['median_ae_valid']:.4f} m")
        print(f"  R^2:        {jh['r2_valid']:.4f}")
    print(f"  Actual range: [{metrics['actual']['jump_height'].min():.3f}, {metrics['actual']['jump_height'].max():.3f}] m")

    print("\nPeak Power (W/kg):")
    pp = metrics['peak_power']
    print(f"  RMSE:       {pp['rmse']:.2f} W/kg")
    print(f"  MAE:        {pp['mae']:.2f} W/kg")
    print(f"  Median AE:  {pp['median_ae']:.2f} W/kg")
    print(f"  Bias:       {pp['bias']:.2f} W/kg")
    print(f"  R^2:        {pp['r2']:.4f}")
    print(f"  90th %ile:  {pp['p90_error']:.2f} W/kg")
    if 'r2_valid' in pp:
        print(f"  --- Valid samples only ---")
        print(f"  RMSE:       {pp['rmse_valid']:.2f} W/kg")
        print(f"  Median AE:  {pp['median_ae_valid']:.2f} W/kg")
        print(f"  R^2:        {pp['r2_valid']:.4f}")
    print(f"  Actual range: [{metrics['actual']['peak_power'].min():.1f}, {metrics['actual']['peak_power'].max():.1f}] W/kg")

    # Print worst outliers
    if 'outliers' in metrics:
        print("\n--- Worst Jump Height Outliers ---")
        ol_jh = metrics['outliers']['jump_height']
        for i, idx in enumerate(ol_jh['indices']):
            print(f"  Sample {idx}: actual={ol_jh['actual'][i]:.3f}m, pred={ol_jh['predicted'][i]:.3f}m, error={ol_jh['errors'][i]:.3f}m")

        print("\n--- Worst Peak Power Outliers ---")
        ol_pp = metrics['outliers']['peak_power']
        for i, idx in enumerate(ol_pp['indices']):
            print(f"  Sample {idx}: actual={ol_pp['actual'][i]:.1f}, pred={ol_pp['predicted'][i]:.1f}, error={ol_pp['errors'][i]:.1f} W/kg")

    print("=" * 60)
