"""
Custom Loss Functions for Biomechanics-Aware Training

These loss functions compute jump height and peak power from predicted GRF,
allowing the model to be trained directly on the metrics that matter.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data_loader import SAMPLING_RATE

# Constants
GRAVITY = 9.812  # m/s^2


class JumpHeightLoss(keras.losses.Loss):
    """
    Loss function based on jump height computed from GRF.

    Computes jump height using impulse-momentum method and returns
    MSE between predicted and actual jump heights.

    Args:
        sampling_rate: Sampling rate in Hz
        grf_mean_function: Mean function used for GRF normalization (shape: seq_len, 1)
        grf_std: Std used for GRF normalization (to denormalize)
        name: Loss name
    """

    def __init__(
        self,
        sampling_rate: float = SAMPLING_RATE,
        grf_mean_function: np.ndarray = None,
        grf_std: float = 0.5,
        name: str = "jump_height_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.sampling_rate = sampling_rate
        # Convert mean function to tensor; default to 1.0 if not provided
        if grf_mean_function is None:
            self.grf_mean_function = tf.constant(1.0, dtype=tf.float32)
        else:
            self.grf_mean_function = tf.constant(grf_mean_function, dtype=tf.float32)
        self.grf_std = grf_std
        self.dt = 1.0 / sampling_rate

    def call(self, y_true, y_pred):
        """
        Compute loss based on jump height difference.

        Args:
            y_true: Actual GRF (normalized), shape (batch, seq_len, 1)
            y_pred: Predicted GRF (normalized), shape (batch, seq_len, 1)

        Returns:
            Scalar loss value
        """
        # Denormalize to body weight units using mean function
        y_true_bw = y_true * self.grf_std + self.grf_mean_function
        y_pred_bw = y_pred * self.grf_std + self.grf_mean_function

        # Compute jump heights
        jh_true = self._compute_jump_height(y_true_bw)
        jh_pred = self._compute_jump_height(y_pred_bw)

        # MSE loss on jump height
        return tf.reduce_mean(tf.square(jh_true - jh_pred))

    def _compute_jump_height(self, grf):
        """Compute jump height from GRF tensor."""
        # grf shape: (batch, seq_len, 1)
        grf = tf.squeeze(grf, axis=-1)  # (batch, seq_len)

        # Net GRF (subtract 1 BW)
        net_grf = grf - 1.0

        # Acceleration
        acceleration = net_grf * GRAVITY

        # Integrate to velocity
        velocity = tf.cumsum(acceleration, axis=1) * self.dt

        # Integrate to position
        position = tf.cumsum(velocity, axis=1) * self.dt

        # Jump height = final position + kinetic energy contribution
        final_velocity = velocity[:, -1]
        final_position = position[:, -1]

        jump_height = final_position + 0.5 * tf.square(final_velocity) / GRAVITY

        return jump_height

    def get_config(self):
        config = super().get_config()
        config.update({
            'sampling_rate': self.sampling_rate,
            'grf_std': self.grf_std,
            # Note: grf_mean_function not serialized (numpy array)
        })
        return config


class PeakPowerLoss(keras.losses.Loss):
    """
    Loss function based on peak power computed from GRF.

    Args:
        sampling_rate: Sampling rate in Hz
        grf_mean_function: Mean function used for GRF normalization (shape: seq_len, 1)
        grf_std: Std used for GRF normalization
        name: Loss name
    """

    def __init__(
        self,
        sampling_rate: float = SAMPLING_RATE,
        grf_mean_function: np.ndarray = None,
        grf_std: float = 0.5,
        name: str = "peak_power_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.sampling_rate = sampling_rate
        # Convert mean function to tensor; default to 1.0 if not provided
        if grf_mean_function is None:
            self.grf_mean_function = tf.constant(1.0, dtype=tf.float32)
        else:
            self.grf_mean_function = tf.constant(grf_mean_function, dtype=tf.float32)
        self.grf_std = grf_std
        self.dt = 1.0 / sampling_rate

    def call(self, y_true, y_pred):
        """Compute loss based on peak power difference."""
        # Denormalize using mean function
        y_true_bw = y_true * self.grf_std + self.grf_mean_function
        y_pred_bw = y_pred * self.grf_std + self.grf_mean_function

        # Compute peak powers
        pp_true = self._compute_peak_power(y_true_bw)
        pp_pred = self._compute_peak_power(y_pred_bw)

        # MSE loss on peak power (normalized by typical range ~50 W/kg)
        return tf.reduce_mean(tf.square((pp_true - pp_pred) / 50.0))

    def _compute_peak_power(self, grf):
        """Compute peak power from GRF tensor."""
        grf = tf.squeeze(grf, axis=-1)  # (batch, seq_len)

        # Net GRF
        net_grf = grf - 1.0

        # Velocity from integration
        velocity = GRAVITY * tf.cumsum(net_grf, axis=1) * self.dt

        # Instantaneous power: P = F * v (in BW * m/s)
        power = grf * velocity

        # Peak power in W/kg
        peak_power = GRAVITY * tf.reduce_max(power, axis=1)

        return peak_power

    def get_config(self):
        config = super().get_config()
        config.update({
            'sampling_rate': self.sampling_rate,
            'grf_std': self.grf_std,
            # Note: grf_mean_function not serialized (numpy array)
        })
        return config


class TemporalWeightedMSELoss(keras.losses.Loss):
    """
    MSE loss with temporal weights emphasizing high-jerk regions.

    Uses precomputed weights derived from the second derivative of ACC data
    to emphasize biomechanically important phases (countermovement, propulsion)
    over quiet standing periods.

    Args:
        temporal_weights: Array of shape (seq_len,) with per-timestep weights
        name: Loss name
    """

    def __init__(
        self,
        temporal_weights: tf.Tensor = None,
        name: str = "temporal_weighted_mse_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        if temporal_weights is not None:
            self.temporal_weights = tf.constant(temporal_weights, dtype=tf.float32)
        else:
            self.temporal_weights = None

    def call(self, y_true, y_pred):
        """
        Compute temporally weighted MSE loss.

        Args:
            y_true: Actual GRF, shape (batch, seq_len, 1)
            y_pred: Predicted GRF, shape (batch, seq_len, 1)

        Returns:
            Scalar loss value
        """
        # Compute squared error per timestep
        squared_error = tf.square(y_true - y_pred)  # (batch, seq_len, 1)
        squared_error = tf.squeeze(squared_error, axis=-1)  # (batch, seq_len)

        if self.temporal_weights is not None:
            # Apply temporal weights (broadcast across batch)
            # weights shape: (seq_len,) -> (1, seq_len) for broadcasting
            weights = tf.reshape(self.temporal_weights, (1, -1))
            weighted_se = squared_error * weights
        else:
            weighted_se = squared_error

        # Mean over all timesteps and samples
        return tf.reduce_mean(weighted_se)

    def get_config(self):
        config = super().get_config()
        # Note: temporal_weights not serialized (must be passed at construction)
        return config


class CombinedBiomechanicsLoss(keras.losses.Loss):
    """
    Combined loss: MSE + jump height + peak power.

    Balances signal reconstruction with biomechanics accuracy.

    Args:
        sampling_rate: Sampling rate in Hz
        grf_mean_function: Mean function used for GRF normalization (shape: seq_len, 1)
        grf_std: Std used for GRF normalization
        mse_weight: Weight for MSE component
        jh_weight: Weight for jump height component
        pp_weight: Weight for peak power component
        name: Loss name
    """

    def __init__(
        self,
        sampling_rate: float = SAMPLING_RATE,
        grf_mean_function: np.ndarray = None,
        grf_std: float = 0.5,
        mse_weight: float = 1.0,
        jh_weight: float = 1.0,
        pp_weight: float = 1.0,
        name: str = "combined_biomechanics_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.sampling_rate = sampling_rate
        self.grf_mean_function = grf_mean_function
        self.grf_std = grf_std
        self.mse_weight = mse_weight
        self.jh_weight = jh_weight
        self.pp_weight = pp_weight
        self.dt = 1.0 / sampling_rate

        # Sub-losses
        self.jh_loss = JumpHeightLoss(sampling_rate, grf_mean_function, grf_std)
        self.pp_loss = PeakPowerLoss(sampling_rate, grf_mean_function, grf_std)

    def call(self, y_true, y_pred):
        """Compute combined loss."""
        # MSE component (on normalized data)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Biomechanics components
        jh_loss = self.jh_loss(y_true, y_pred)
        pp_loss = self.pp_loss(y_true, y_pred)

        # Weighted combination
        total_loss = (
            self.mse_weight * mse +
            self.jh_weight * jh_loss +
            self.pp_weight * pp_loss
        )

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'sampling_rate': self.sampling_rate,
            'grf_std': self.grf_std,
            'mse_weight': self.mse_weight,
            'jh_weight': self.jh_weight,
            'pp_weight': self.pp_weight,
            # Note: grf_mean_function not serialized (numpy array)
        })
        return config


class EigenvalueWeightedMSELoss(keras.losses.Loss):
    """
    MSE loss weighted by eigenvalues for FPC score prediction.

    When predicting FPC scores, earlier components explain more variance
    and are more important for reconstruction. This loss weights each
    component's error by its eigenvalue (variance explained), so errors
    in important components contribute more to the loss.

    Args:
        eigenvalues: Array of shape (n_components,) or (n_components, n_channels)
            containing eigenvalues (variance explained) for each FPC.
            Will be normalized to sum to 1.
        name: Loss name
    """

    def __init__(
        self,
        eigenvalues: np.ndarray = None,
        name: str = "eigenvalue_weighted_mse_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        if eigenvalues is not None:
            # Normalize eigenvalues to sum to 1 (per channel if multi-channel)
            eigenvalues = np.array(eigenvalues, dtype=np.float32)
            if eigenvalues.ndim == 1:
                # Single channel: (n_components,)
                eigenvalues = eigenvalues / np.sum(eigenvalues)
            else:
                # Multi-channel: (n_components, n_channels)
                eigenvalues = eigenvalues / np.sum(eigenvalues, axis=0, keepdims=True)
            self.eigenvalues = tf.constant(eigenvalues, dtype=tf.float32)
        else:
            self.eigenvalues = None

    def call(self, y_true, y_pred):
        """
        Compute eigenvalue-weighted MSE loss.

        Args:
            y_true: Actual FPC scores, shape (batch, n_components, n_channels)
            y_pred: Predicted FPC scores, shape (batch, n_components, n_channels)

        Returns:
            Scalar loss value
        """
        # Compute squared error per component
        squared_error = tf.square(y_true - y_pred)  # (batch, n_components, n_channels)

        if self.eigenvalues is not None:
            # Weight by eigenvalues
            # eigenvalues shape: (n_components,) or (n_components, n_channels)
            if len(self.eigenvalues.shape) == 1:
                # Expand for broadcasting: (n_components,) -> (1, n_components, 1)
                weights = tf.reshape(self.eigenvalues, (1, -1, 1))
            else:
                # (n_components, n_channels) -> (1, n_components, n_channels)
                weights = tf.expand_dims(self.eigenvalues, axis=0)

            weighted_se = squared_error * weights
        else:
            # Fall back to unweighted MSE
            weighted_se = squared_error

        # Mean over all dimensions
        return tf.reduce_mean(weighted_se)

    def get_config(self):
        config = super().get_config()
        # Note: eigenvalues not serialized (must be passed at construction)
        return config


class SmoothnessRegularizationLoss(keras.losses.Loss):
    """
    MSE loss with smoothness regularization penalizing second derivative.

    Encourages smooth predictions by penalizing roughness (large second
    derivatives), which is appropriate for biomechanical signals that
    should vary continuously.

    Args:
        lambda_smooth: Weight for smoothness penalty (default: 0.1)
        name: Loss name
    """

    def __init__(
        self,
        lambda_smooth: float = 0.1,
        name: str = "smoothness_regularization_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.lambda_smooth = lambda_smooth

    def call(self, y_true, y_pred):
        """
        Compute MSE + smoothness penalty.

        Args:
            y_true: Actual GRF, shape (batch, seq_len, 1)
            y_pred: Predicted GRF, shape (batch, seq_len, 1)

        Returns:
            Scalar loss value
        """
        # MSE component
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Second derivative via finite differences: d2y = y[i+1] - 2*y[i] + y[i-1]
        d2y = y_pred[:, 2:, :] - 2 * y_pred[:, 1:-1, :] + y_pred[:, :-2, :]
        roughness = tf.reduce_mean(tf.square(d2y))

        return mse + self.lambda_smooth * roughness

    def get_config(self):
        config = super().get_config()
        config.update({
            'lambda_smooth': self.lambda_smooth,
        })
        return config


def get_loss_function(
    loss_type: str,
    grf_mean_function: np.ndarray = None,
    grf_std: float = 0.5,
    sampling_rate: float = SAMPLING_RATE,
    mse_weight: float = 1.0,
    jh_weight: float = 1.0,
    pp_weight: float = 1.0,
    temporal_weights: tf.Tensor = None,
    lambda_smooth: float = 0.1,
    eigenvalues: np.ndarray = None,
) -> keras.losses.Loss:
    """
    Factory function to get loss by name.

    Args:
        loss_type: One of 'mse', 'jump_height', 'peak_power', 'combined',
            'weighted', 'smooth', 'eigenvalue_weighted'
        grf_mean_function: Mean function for denormalization (shape: seq_len, 1)
        grf_std: Std for denormalization
        sampling_rate: Sampling rate in Hz
        mse_weight: Weight for MSE component (combined/weighted loss)
        jh_weight: Weight for jump height component (combined/weighted loss)
        pp_weight: Weight for peak power component (combined/weighted loss)
        temporal_weights: Per-timestep weights for weighted loss type
        lambda_smooth: Smoothness regularization weight
        eigenvalues: Eigenvalues for eigenvalue_weighted loss (from FPCA)

    Returns:
        Keras loss function
    """
    if loss_type == 'mse':
        return keras.losses.MeanSquaredError()
    elif loss_type == 'jump_height':
        return JumpHeightLoss(sampling_rate, grf_mean_function, grf_std)
    elif loss_type == 'peak_power':
        return PeakPowerLoss(sampling_rate, grf_mean_function, grf_std)
    elif loss_type == 'combined':
        return CombinedBiomechanicsLoss(
            sampling_rate, grf_mean_function, grf_std,
            mse_weight=mse_weight,
            jh_weight=jh_weight,
            pp_weight=pp_weight,
        )
    elif loss_type == 'weighted':
        return TemporalWeightedMSELoss(temporal_weights)
    elif loss_type == 'smooth':
        return SmoothnessRegularizationLoss(lambda_smooth=lambda_smooth)
    elif loss_type == 'eigenvalue_weighted':
        if eigenvalues is None:
            raise ValueError("eigenvalue_weighted loss requires eigenvalues parameter")
        return EigenvalueWeightedMSELoss(eigenvalues=eigenvalues)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Choose from: mse, jump_height, peak_power, combined, "
                        f"weighted, smooth, eigenvalue_weighted")
