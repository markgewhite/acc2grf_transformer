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

    # Compute statistics
    results = {
        'actual': actual_metrics,
        'predicted': predicted_metrics,
        'jump_height': {
            'rmse': np.sqrt(np.mean(jh_errors ** 2)),
            'mae': np.mean(np.abs(jh_errors)),
            'bias': np.mean(jh_errors),
            'r2': _compute_r2(actual_metrics['jump_height'], predicted_metrics['jump_height']),
            'errors': jh_errors,
        },
        'peak_power': {
            'rmse': np.sqrt(np.mean(pp_errors ** 2)),
            'mae': np.mean(np.abs(pp_errors)),
            'bias': np.mean(pp_errors),
            'r2': _compute_r2(actual_metrics['peak_power'], predicted_metrics['peak_power']),
            'errors': pp_errors,
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

    print("\nJump Height (meters):")
    jh = metrics['jump_height']
    print(f"  RMSE:  {jh['rmse']:.4f} m")
    print(f"  MAE:   {jh['mae']:.4f} m")
    print(f"  Bias:  {jh['bias']:.4f} m")
    print(f"  R^2:   {jh['r2']:.4f}")
    print(f"  Actual range: [{metrics['actual']['jump_height'].min():.3f}, {metrics['actual']['jump_height'].max():.3f}] m")

    print("\nPeak Power (W/kg):")
    pp = metrics['peak_power']
    print(f"  RMSE:  {pp['rmse']:.2f} W/kg")
    print(f"  MAE:   {pp['mae']:.2f} W/kg")
    print(f"  Bias:  {pp['bias']:.2f} W/kg")
    print(f"  R^2:   {pp['r2']:.4f}")
    print(f"  Actual range: [{metrics['actual']['peak_power'].min():.1f}, {metrics['actual']['peak_power'].max():.1f}] W/kg")

    print("=" * 60)
