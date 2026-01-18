"""
Multi-Head Self-Attention Module

Implements scaled dot-product attention from scratch using TensorFlow.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers


class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-head self-attention layer implemented from scratch.

    Computes scaled dot-product attention across multiple heads,
    allowing the model to attend to information from different
    representation subspaces.

    Args:
        d_model: Dimensionality of the model (embedding dimension)
        num_heads: Number of attention heads
        dropout_rate: Dropout rate for attention weights
    """

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.dropout_rate = dropout_rate

        # Linear projections for Q, K, V
        self.W_q = layers.Dense(d_model, use_bias=False)
        self.W_k = layers.Dense(d_model, use_bias=False)
        self.W_v = layers.Dense(d_model, use_bias=False)

        # Output projection
        self.W_o = layers.Dense(d_model, use_bias=False)

        # Dropout for attention weights
        self.dropout = layers.Dropout(dropout_rate)

        # Store attention weights for visualization
        self._attention_weights = None

    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            batch_size: Batch size

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(
        self,
        Q: tf.Tensor,
        K: tf.Tensor,
        V: tf.Tensor,
        mask: tf.Tensor = None,
        training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Compute scaled dot-product attention.

        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

        Args:
            Q: Query tensor (batch_size, num_heads, seq_len_q, d_k)
            K: Key tensor (batch_size, num_heads, seq_len_k, d_k)
            V: Value tensor (batch_size, num_heads, seq_len_v, d_k)
            mask: Optional mask tensor
            training: Whether in training mode

        Returns:
            Tuple of (attention output, attention weights)
        """
        # Compute attention scores: Q @ K^T
        matmul_qk = tf.matmul(Q, K, transpose_b=True)

        # Scale by sqrt(d_k)
        scale = tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        scaled_attention_logits = matmul_qk / scale

        # Apply mask if provided (for padding or causal attention)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply dropout to attention weights
        attention_weights_dropped = self.dropout(attention_weights, training=training)

        # Weighted sum of values
        output = tf.matmul(attention_weights_dropped, V)

        return output, attention_weights

    def call(
        self,
        x: tf.Tensor,
        mask: tf.Tensor = None,
        training: bool = False,
        return_attention_weights: bool = False
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass of multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            training: Whether in training mode
            return_attention_weights: Whether to return attention weights

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
            Optionally also returns attention weights
        """
        batch_size = tf.shape(x)[0]

        # Linear projections
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads
        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask=mask, training=training
        )

        # Store attention weights for visualization
        self._attention_weights = attention_weights

        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.d_model))

        # Final linear projection
        output = self.W_o(concat_attention)

        if return_attention_weights:
            return output, attention_weights
        return output

    @property
    def attention_weights(self) -> tf.Tensor:
        """Return the most recent attention weights for visualization."""
        return self._attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
        })
        return config
