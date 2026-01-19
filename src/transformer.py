"""
Transformer Model for Signal-to-Signal Regression

Encoder-only transformer architecture for mapping accelerometer
signals to ground reaction force during countermovement jumps.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

from .attention import MultiHeadSelfAttention


class PositionalEncoding(layers.Layer):
    """
    Learnable positional encoding for sequence data.

    Unlike fixed sinusoidal encodings, learnable embeddings allow the model
    to adapt positional information to the specific characteristics of
    biomechanical signals.

    Args:
        seq_len: Maximum sequence length
        d_model: Embedding dimension
    """

    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model

        # Learnable position embeddings
        self.pos_embedding = layers.Embedding(
            input_dim=seq_len,
            output_dim=d_model
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_encoding = self.pos_embedding(positions)
        return x + pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'd_model': self.d_model,
        })
        return config


class FeedForwardNetwork(layers.Layer):
    """
    Position-wise feed-forward network.

    Applies two linear transformations with a ReLU activation in between.
    FFN(x) = max(0, xW1 + b1)W2 + b2

    Args:
        d_model: Model dimension
        d_ff: Hidden dimension of feed-forward network
        dropout_rate: Dropout rate
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.dense1 = layers.Dense(d_ff, activation='relu')
        self.dense2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through FFN."""
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
        })
        return config


class TransformerEncoderBlock(layers.Layer):
    """
    Single transformer encoder block.

    Consists of:
    1. Multi-head self-attention with residual connection and layer norm
    2. Feed-forward network with residual connection and layer norm

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Multi-head self-attention
        self.mha = MultiHeadSelfAttention(d_model, num_heads, dropout_rate)

        # Feed-forward network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)

        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout for residual connections
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(
        self,
        x: tf.Tensor,
        mask: tf.Tensor = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass through encoder block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            training: Whether in training mode

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Multi-head self-attention with residual connection
        attn_output = self.mha(x, mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    @property
    def attention_weights(self) -> tf.Tensor:
        """Return attention weights from this block."""
        return self.mha.attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
        })
        return config


class SignalTransformer(Model):
    """
    Encoder-only transformer for sequence-to-sequence signal regression.

    Maps accelerometer signals to ground reaction force predictions.
    Input: (batch_size, input_seq_len, input_dim) - accelerometer data
    Output: (batch_size, output_seq_len, output_dim) - predicted GRF

    Args:
        input_seq_len: Input sequence length (ACC, may include post-takeoff)
        output_seq_len: Output sequence length (GRF, up to takeoff only)
        input_dim: Input dimension (1 for resultant, 3 for triaxial)
        output_dim: Output dimension (1 for raw GRF, >1 for FPC scores)
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of encoder blocks
        d_ff: Feed-forward hidden dimension
        dropout_rate: Dropout rate

    Note:
        When input_seq_len > output_seq_len, the model uses the full input
        context but only outputs predictions for the first output_seq_len
        positions (corresponding to the pre-takeoff period).

        For FDA transformations:
        - With B-spline transform: input_seq_len = n_basis, output_seq_len = n_basis
        - With FPC transform: input_seq_len = n_components, output_seq_len = n_components
    """

    def __init__(
        self,
        input_seq_len: int = 500,
        output_seq_len: int = None,
        input_dim: int = 1,
        output_dim: int = 1,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 128,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Handle backward compatibility: if output_seq_len not specified, use input_seq_len
        if output_seq_len is None:
            output_seq_len = input_seq_len

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Input projection: (batch, input_seq_len, input_dim) -> (batch, input_seq_len, d_model)
        self.input_projection = layers.Dense(d_model)

        # Positional encoding for full input sequence
        self.positional_encoding = PositionalEncoding(input_seq_len, d_model)

        # Input dropout
        self.input_dropout = layers.Dropout(dropout_rate)

        # Stack of encoder blocks
        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]

        # Output projection: (batch, output_seq_len, d_model) -> (batch, output_seq_len, output_dim)
        self.output_projection = layers.Dense(output_dim)

    def call(
        self,
        x: tf.Tensor,
        mask: tf.Tensor = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass through the transformer.

        Args:
            x: Input tensor of shape (batch_size, input_seq_len, input_dim)
            mask: Optional attention mask
            training: Whether in training mode

        Returns:
            Output tensor of shape (batch_size, output_seq_len, output_dim)
        """
        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Input dropout
        x = self.input_dropout(x, training=training)

        # Pass through encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask=mask, training=training)

        # Take only the first output_seq_len positions (pre-takeoff period)
        # The post-takeoff positions provide context but don't need GRF predictions
        if self.output_seq_len < self.input_seq_len:
            x = x[:, :self.output_seq_len, :]

        # Output projection to single GRF value per timestep
        output = self.output_projection(x)

        return output

    def get_attention_weights(self, layer_idx: int = -1) -> tf.Tensor:
        """
        Get attention weights from a specific encoder layer.

        Args:
            layer_idx: Index of encoder layer (-1 for last layer)

        Returns:
            Attention weights tensor
        """
        return self.encoder_blocks[layer_idx].attention_weights

    def get_config(self):
        return {
            'input_seq_len': self.input_seq_len,
            'output_seq_len': self.output_seq_len,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_signal_transformer(
    input_seq_len: int = 500,
    output_seq_len: int = None,
    input_dim: int = 1,
    output_dim: int = 1,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    d_ff: int = 128,
    dropout_rate: float = 0.1,
    learning_rate: float = 1e-4,
) -> SignalTransformer:
    """
    Build and compile a SignalTransformer model.

    Args:
        input_seq_len: Input sequence length (ACC)
        output_seq_len: Output sequence length (GRF), defaults to input_seq_len
        input_dim: Input dimension (1 for resultant, 3 for triaxial)
        output_dim: Output dimension (1 for raw GRF, >1 for FPC scores)
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of encoder blocks
        d_ff: Feed-forward hidden dimension
        dropout_rate: Dropout rate
        learning_rate: Learning rate for Adam optimizer

    Returns:
        Compiled SignalTransformer model
    """
    model = SignalTransformer(
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )

    return model
