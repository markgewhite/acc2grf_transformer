"""
Alternative Model Architectures for Signal-to-Signal Regression

Provides simpler baseline models (MLP) as alternatives to the transformer
for mapping accelerometer coefficients to GRF coefficients.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model


class SignalMLP(Model):
    """
    Simple feedforward MLP for coefficient-to-coefficient mapping.

    Replicates the approach that worked well in MATLAB: a single hidden layer
    MLP operating on B-spline or FPC coefficients.

    Architecture:
        Flatten → Dense(hidden, relu) → Dropout → Dense(output) → Reshape

    Args:
        input_seq_len: Number of input coefficients (e.g., n_basis or n_components)
        output_seq_len: Number of output coefficients
        input_dim: Input channels (1 for resultant, 3 for triaxial)
        output_dim: Output channels (1 for GRF)
        hidden_size: Hidden layer width (default: 64)
        dropout_rate: Dropout rate (default: 0.1)

    Note:
        This model does not support scalar_prediction or scalar_only modes
        (transformer-specific features). It accepts these parameters for
        API compatibility but ignores them.
    """

    def __init__(
        self,
        input_seq_len: int,
        output_seq_len: int,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_size: int = 64,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        # Remove transformer-specific kwargs to avoid passing to Model.__init__
        kwargs.pop('d_model', None)
        kwargs.pop('num_heads', None)
        kwargs.pop('num_layers', None)
        kwargs.pop('d_ff', None)
        kwargs.pop('scalar_prediction', None)
        kwargs.pop('scalar_only', None)

        super().__init__(**kwargs)

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # Flatten input: (batch, seq_len, dim) -> (batch, seq_len * dim)
        self.flatten = layers.Flatten()

        # Hidden layer
        self.hidden = layers.Dense(hidden_size, activation='relu')

        # Dropout
        self.dropout = layers.Dropout(dropout_rate)

        # Output layer: produces flattened output
        self.output_dense = layers.Dense(output_seq_len * output_dim)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, input_seq_len, input_dim)
            training: Whether in training mode

        Returns:
            Output tensor of shape (batch_size, output_seq_len, output_dim)
        """
        batch_size = tf.shape(x)[0]

        # Flatten input
        x = self.flatten(x)  # (batch, input_seq_len * input_dim)

        # Hidden layer
        x = self.hidden(x)  # (batch, hidden_size)

        # Dropout
        x = self.dropout(x, training=training)

        # Output layer
        x = self.output_dense(x)  # (batch, output_seq_len * output_dim)

        # Reshape to (batch, output_seq_len, output_dim)
        x = tf.reshape(x, (batch_size, self.output_seq_len, self.output_dim))

        return x

    def get_config(self):
        return {
            'input_seq_len': self.input_seq_len,
            'output_seq_len': self.output_seq_len,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_size': self.hidden_size,
            'dropout_rate': self.dropout_rate,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
