"""
Alternative Model Architectures for Signal-to-Signal Regression

Provides simpler baseline models (MLP) and hybrid architectures as alternatives
to the transformer for mapping accelerometer coefficients to GRF coefficients.
"""

import numpy as np
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


class HybridProjectionMLP(Model):
    """
    Hybrid model combining physics-informed linear projection with learned refinement.

    This implements the MATLAB-inspired approach where a functional projection matrix
    provides an interpretable linear baseline, and an MLP learns nonlinear corrections.

    Architecture options:
        - "residual": output = P @ x + MLP(x)
            Linear projection plus learned residual correction.
        - "sequential": output = MLP(P @ x)
            MLP refines the projected scores.
        - "parallel": output = α * (P @ x) + (1-α) * MLP(x)
            Weighted combination of linear and learned paths.

    Args:
        input_seq_len: Number of input FPC components per channel
        output_seq_len: Number of output FPC components
        input_dim: Number of input channels (1 for resultant, 3 for triaxial)
        output_dim: Number of output channels (typically 1 for GRF)
        projection_matrix: Pre-computed projection matrix P of shape
            (input_seq_len * input_dim, output_seq_len * output_dim).
            If None, initialized randomly.
        rescale_factor: Magnitude rescaling factor (default: 1.0)
        hidden_size: MLP hidden layer size (default: 64)
        dropout_rate: Dropout rate (default: 0.1)
        architecture: One of "residual", "sequential", "parallel" (default: "residual")
        freeze_projection: If True, P is fixed; if False, P can be fine-tuned (default: True)
        parallel_alpha: Initial mixing coefficient for parallel mode (default: 0.5)
        trainable_alpha: Whether alpha is trainable in parallel mode (default: True)
    """

    def __init__(
        self,
        input_seq_len: int,
        output_seq_len: int,
        input_dim: int = 1,
        output_dim: int = 1,
        projection_matrix: np.ndarray = None,
        rescale_factor: float = 1.0,
        hidden_size: int = 64,
        dropout_rate: float = 0.1,
        architecture: str = "residual",
        freeze_projection: bool = True,
        parallel_alpha: float = 0.5,
        trainable_alpha: bool = True,
        **kwargs
    ):
        # Remove transformer-specific kwargs
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
        self.architecture = architecture
        self.freeze_projection = freeze_projection
        self._rescale_factor = rescale_factor

        # Validate architecture
        valid_architectures = ["residual", "sequential", "parallel"]
        if architecture not in valid_architectures:
            raise ValueError(f"architecture must be one of {valid_architectures}")

        # Compute dimensions
        input_features = input_seq_len * input_dim
        output_features = output_seq_len * output_dim

        # Initialize or use provided projection matrix
        if projection_matrix is not None:
            expected_shape = (input_features, output_features)
            if projection_matrix.shape != expected_shape:
                raise ValueError(
                    f"projection_matrix shape {projection_matrix.shape} doesn't match "
                    f"expected shape {expected_shape}"
                )
            P_init = projection_matrix.astype(np.float32)
        else:
            # Random initialization (will be learned)
            P_init = np.random.randn(input_features, output_features).astype(np.float32) * 0.1

        # Create projection matrix as trainable or non-trainable variable
        self.projection_matrix = tf.Variable(
            P_init,
            trainable=not freeze_projection,
            name='projection_matrix',
            dtype=tf.float32
        )

        # Rescale factor
        self.rescale = tf.Variable(
            rescale_factor,
            trainable=False,  # Rescale is always fixed
            name='rescale_factor',
            dtype=tf.float32
        )

        # MLP components
        self.flatten = layers.Flatten()

        if architecture == "sequential":
            # MLP takes projected features as input
            mlp_input_size = output_features
        else:
            # MLP takes original features as input
            mlp_input_size = input_features

        self.hidden = layers.Dense(hidden_size, activation='relu', name='mlp_hidden')
        self.dropout = layers.Dropout(dropout_rate)
        self.output_dense = layers.Dense(output_features, name='mlp_output')

        # For parallel architecture: learnable mixing coefficient
        if architecture == "parallel":
            self.alpha = tf.Variable(
                parallel_alpha,
                trainable=trainable_alpha,
                name='parallel_alpha',
                dtype=tf.float32
            )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the hybrid model.

        Args:
            x: Input tensor of shape (batch_size, input_seq_len, input_dim)
            training: Whether in training mode

        Returns:
            Output tensor of shape (batch_size, output_seq_len, output_dim)
        """
        batch_size = tf.shape(x)[0]

        # Flatten input: (batch, seq_len, dim) -> (batch, seq_len * dim)
        x_flat = self.flatten(x)

        # Linear projection: (batch, input_features) @ (input_features, output_features)
        linear_proj = self.rescale * tf.matmul(x_flat, self.projection_matrix)

        if self.architecture == "residual":
            # output = P @ x + MLP(x)
            mlp_out = self.hidden(x_flat)
            mlp_out = self.dropout(mlp_out, training=training)
            mlp_out = self.output_dense(mlp_out)
            output = linear_proj + mlp_out

        elif self.architecture == "sequential":
            # output = MLP(P @ x)
            mlp_out = self.hidden(linear_proj)
            mlp_out = self.dropout(mlp_out, training=training)
            output = self.output_dense(mlp_out)

        elif self.architecture == "parallel":
            # output = α * (P @ x) + (1-α) * MLP(x)
            mlp_out = self.hidden(x_flat)
            mlp_out = self.dropout(mlp_out, training=training)
            mlp_out = self.output_dense(mlp_out)
            # Constrain alpha to [0, 1] using sigmoid
            alpha = tf.sigmoid(self.alpha)
            output = alpha * linear_proj + (1 - alpha) * mlp_out

        # Reshape to (batch, output_seq_len, output_dim)
        output = tf.reshape(output, (batch_size, self.output_seq_len, self.output_dim))

        return output

    def get_linear_prediction(self, x: tf.Tensor) -> tf.Tensor:
        """
        Get only the linear projection output (for analysis/comparison).

        Args:
            x: Input tensor of shape (batch_size, input_seq_len, input_dim)

        Returns:
            Linear projection of shape (batch_size, output_seq_len, output_dim)
        """
        batch_size = tf.shape(x)[0]
        x_flat = self.flatten(x)
        linear_proj = self.rescale * tf.matmul(x_flat, self.projection_matrix)
        return tf.reshape(linear_proj, (batch_size, self.output_seq_len, self.output_dim))

    def get_mlp_prediction(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Get only the MLP output (for analysis/comparison).

        Args:
            x: Input tensor of shape (batch_size, input_seq_len, input_dim)
            training: Whether in training mode

        Returns:
            MLP output of shape (batch_size, output_seq_len, output_dim)
        """
        batch_size = tf.shape(x)[0]
        x_flat = self.flatten(x)

        if self.architecture == "sequential":
            # MLP takes projected input
            linear_proj = self.rescale * tf.matmul(x_flat, self.projection_matrix)
            mlp_input = linear_proj
        else:
            mlp_input = x_flat

        mlp_out = self.hidden(mlp_input)
        mlp_out = self.dropout(mlp_out, training=training)
        mlp_out = self.output_dense(mlp_out)

        return tf.reshape(mlp_out, (batch_size, self.output_seq_len, self.output_dim))

    def get_effective_alpha(self) -> float:
        """Get the effective mixing coefficient (parallel architecture only)."""
        if self.architecture != "parallel":
            return None
        return float(tf.sigmoid(self.alpha).numpy())

    def get_config(self):
        return {
            'input_seq_len': self.input_seq_len,
            'output_seq_len': self.output_seq_len,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_size': self.hidden_size,
            'dropout_rate': self.dropout_rate,
            'architecture': self.architecture,
            'freeze_projection': self.freeze_projection,
            'rescale_factor': float(self._rescale_factor),
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
