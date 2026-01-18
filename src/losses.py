"""
Custom Loss Functions for Biomechanics-Aware Training

These loss functions compute jump height and peak power from predicted GRF,
allowing the model to be trained directly on the metrics that matter.
"""

import tensorflow as tf
from tensorflow import keras

from .data_loader import SAMPLING_RATE

# Constants
GRAVITY = 9.812  # m/s^2


class JumpHeightLoss(keras.losses.Loss):
    """
    Loss function based on jump height computed from GRF.

    Computes jump height using impulse-momentum method and returns
    MSE between predicted and actual jump heights.

    Args:
        sampling_rate: Sampling rate in Hz
        grf_mean: Mean used for GRF normalization (to denormalize)
        grf_std: Std used for GRF normalization (to denormalize)
        name: Loss name
    """

    def __init__(
        self,
        sampling_rate: float = SAMPLING_RATE,
        grf_mean: float = 1.0,
        grf_std: float = 0.5,
        name: str = "jump_height_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.sampling_rate = sampling_rate
        self.grf_mean = grf_mean
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
        # Denormalize to body weight units
        y_true_bw = y_true * self.grf_std + self.grf_mean
        y_pred_bw = y_pred * self.grf_std + self.grf_mean

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
            'grf_mean': self.grf_mean,
            'grf_std': self.grf_std,
        })
        return config


class PeakPowerLoss(keras.losses.Loss):
    """
    Loss function based on peak power computed from GRF.

    Args:
        sampling_rate: Sampling rate in Hz
        grf_mean: Mean used for GRF normalization
        grf_std: Std used for GRF normalization
        name: Loss name
    """

    def __init__(
        self,
        sampling_rate: float = SAMPLING_RATE,
        grf_mean: float = 1.0,
        grf_std: float = 0.5,
        name: str = "peak_power_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.sampling_rate = sampling_rate
        self.grf_mean = grf_mean
        self.grf_std = grf_std
        self.dt = 1.0 / sampling_rate

    def call(self, y_true, y_pred):
        """Compute loss based on peak power difference."""
        # Denormalize
        y_true_bw = y_true * self.grf_std + self.grf_mean
        y_pred_bw = y_pred * self.grf_std + self.grf_mean

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
            'grf_mean': self.grf_mean,
            'grf_std': self.grf_std,
        })
        return config


class CombinedBiomechanicsLoss(keras.losses.Loss):
    """
    Combined loss: MSE + jump height + peak power.

    Balances signal reconstruction with biomechanics accuracy.

    Args:
        sampling_rate: Sampling rate in Hz
        grf_mean: Mean used for GRF normalization
        grf_std: Std used for GRF normalization
        mse_weight: Weight for MSE component
        jh_weight: Weight for jump height component
        pp_weight: Weight for peak power component
        name: Loss name
    """

    def __init__(
        self,
        sampling_rate: float = SAMPLING_RATE,
        grf_mean: float = 1.0,
        grf_std: float = 0.5,
        mse_weight: float = 1.0,
        jh_weight: float = 1.0,
        pp_weight: float = 1.0,
        name: str = "combined_biomechanics_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.sampling_rate = sampling_rate
        self.grf_mean = grf_mean
        self.grf_std = grf_std
        self.mse_weight = mse_weight
        self.jh_weight = jh_weight
        self.pp_weight = pp_weight
        self.dt = 1.0 / sampling_rate

        # Sub-losses
        self.jh_loss = JumpHeightLoss(sampling_rate, grf_mean, grf_std)
        self.pp_loss = PeakPowerLoss(sampling_rate, grf_mean, grf_std)

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
            'grf_mean': self.grf_mean,
            'grf_std': self.grf_std,
            'mse_weight': self.mse_weight,
            'jh_weight': self.jh_weight,
            'pp_weight': self.pp_weight,
        })
        return config


def get_loss_function(
    loss_type: str,
    grf_mean: float = 1.0,
    grf_std: float = 0.5,
    sampling_rate: float = SAMPLING_RATE,
) -> keras.losses.Loss:
    """
    Factory function to get loss by name.

    Args:
        loss_type: One of 'mse', 'jump_height', 'peak_power', 'combined'
        grf_mean: Mean for denormalization
        grf_std: Std for denormalization
        sampling_rate: Sampling rate in Hz

    Returns:
        Keras loss function
    """
    if loss_type == 'mse':
        return keras.losses.MeanSquaredError()
    elif loss_type == 'jump_height':
        return JumpHeightLoss(sampling_rate, grf_mean, grf_std)
    elif loss_type == 'peak_power':
        return PeakPowerLoss(sampling_rate, grf_mean, grf_std)
    elif loss_type == 'combined':
        return CombinedBiomechanicsLoss(sampling_rate, grf_mean, grf_std)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Choose from: mse, jump_height, peak_power, combined")
