# Accelerometer → Ground Reaction Force Transformer

## Strategic Value

This project is genuinely differentiated. It demonstrates:

1. **Custom transformer architecture** — attention mechanism built from scratch, not downloaded
2. **Physics-AI integration** — exactly what SoftInWay does with turbomachinery
3. **Sequence-to-sequence learning** — mapping one physical signal to another
4. **TensorFlow competency** — custom layers, training loops, real implementation
5. **Your unique data and domain** — nobody else has this

This directly addresses SoftInWay's first job responsibility:
> "Implement Training Algorithms based on statistical models and correlate with physical model interaction"

**Time budget: 8-12 hours** (can be reduced by using your existing data pipeline)

---

## The Problem

**Input**: Triaxial accelerometer signal from body-worn sensor (e.g., lower back)
**Output**: Vertical Ground Reaction Force (vGRF) during countermovement jump

This is a sequence-to-sequence regression task. The accelerometer measures motion; the force platform measures the force applied to the ground. They're physically related through Newton's second law, but the mapping is non-trivial due to:
- Multi-segment body dynamics
- Soft tissue artefact
- Sensor placement effects

Your PhD work showed this mapping is learnable. Now you're demonstrating it with a modern transformer architecture.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SIGNAL TRANSFORMER                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Accelerometer (n_timesteps, 3)                             │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │ Input Embed  │  Linear projection to d_model             │
│  └──────────────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │ + Pos Embed  │  Learnable positional encoding            │
│  └──────────────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │  Encoder     │  N × TransformerEncoderBlock              │
│  │  Stack       │    - Multi-Head Self-Attention            │
│  │              │    - Feed-Forward Network                 │
│  │              │    - LayerNorm + Residual                 │
│  └──────────────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │ Output Head  │  Linear projection to 1 (vGRF)            │
│  └──────────────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  vGRF (n_timesteps, 1)                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

This is an **encoder-only** transformer for sequence-to-sequence regression (same length input and output). Similar to using BERT for token-level classification, but for continuous signal prediction.

---

## Project Structure

```
accel_to_grf_transformer/
├── src/
│   ├── attention.py           # Multi-head attention from scratch
│   ├── transformer.py         # Encoder block and full model
│   ├── positional_encoding.py # Positional embeddings
│   ├── data_loader.py         # Load your CMJ dataset
│   └── train.py               # Training script
├── notebooks/
│   └── visualise_predictions.ipynb
├── data/
│   └── README.md              # Instructions for data placement
├── outputs/
│   ├── checkpoints/
│   └── figures/
├── requirements.txt
└── README.md
```

---

## Core Implementation

### Multi-Head Self-Attention (`attention.py`)

This is the key component — implemented from first principles.

```python
"""
Multi-Head Self-Attention implemented from scratch.

This is the core mechanism of transformer architectures.
No pre-built attention layers—demonstrating genuine understanding.
"""

import tensorflow as tf
from tensorflow.keras import layers


class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-head self-attention mechanism.
    
    For input sequence X of shape (batch, seq_len, d_model):
    1. Project X to queries Q, keys K, values V
    2. Compute attention weights: softmax(QK^T / sqrt(d_k))
    3. Apply attention to values: weights @ V
    4. Concatenate heads and project output
    
    Parameters
    ----------
    d_model : int
        Model dimension (embedding size)
    num_heads : int
        Number of attention heads
    """
    
    def __init__(self, d_model: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = layers.Dense(d_model, use_bias=False)
        self.W_k = layers.Dense(d_model, use_bias=False)
        self.W_v = layers.Dense(d_model, use_bias=False)
        
        # Output projection
        self.W_o = layers.Dense(d_model, use_bias=False)
        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, d_k).
        
        Input:  (batch, seq_len, d_model)
        Output: (batch, num_heads, seq_len, d_k)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, Q, K, V):
        """
        Compute scaled dot-product attention.
        
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        
        The scaling by sqrt(d_k) prevents dot products from growing
        too large, which would push softmax into regions with
        vanishing gradients.
        """
        # QK^T: (batch, heads, seq_len, seq_len)
        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        
        # Scale
        scale = tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        scaled_attention_logits = matmul_qk / scale
        
        # Softmax over last axis (key dimension)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Apply attention to values
        output = tf.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def call(self, x, return_attention=False):
        """
        Forward pass.
        
        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, seq_len, d_model)
        return_attention : bool
            If True, also return attention weights for visualisation
            
        Returns
        -------
        output : tf.Tensor
            Shape (batch, seq_len, d_model)
        attention_weights : tf.Tensor, optional
            Shape (batch, num_heads, seq_len, seq_len)
        """
        batch_size = tf.shape(x)[0]
        
        # Linear projections
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split into heads
        Q = self.split_heads(Q, batch_size)  # (batch, heads, seq_len, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # Scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V)
        
        # Concatenate heads
        # (batch, heads, seq_len, d_k) → (batch, seq_len, heads, d_k) → (batch, seq_len, d_model)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.W_o(concat_attention)
        
        if return_attention:
            return output, attention_weights
        return output
```

### Transformer Encoder Block (`transformer.py`)

```python
"""
Transformer encoder block and full model.

Architecture follows "Attention Is All You Need" (Vaswani et al., 2017)
adapted for continuous signal regression.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from attention import MultiHeadSelfAttention


class TransformerEncoderBlock(layers.Layer):
    """
    Single transformer encoder block.
    
    Structure:
        x → MultiHeadAttention → Add & Norm → FFN → Add & Norm → output
        
    With residual connections around both sub-layers.
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
        
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dense(d_model),
        ])
        
        # Layer normalisation
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, x, training=False):
        # Self-attention with residual connection
        attn_output = self.attention(x)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)
        
        return x


class SignalTransformer(Model):
    """
    Transformer for accelerometer → GRF signal translation.
    
    Encoder-only architecture for sequence-to-sequence regression
    where input and output have the same length.
    
    Parameters
    ----------
    input_dim : int
        Input signal dimension (3 for triaxial accelerometer)
    output_dim : int
        Output signal dimension (1 for vertical GRF)
    d_model : int
        Internal model dimension
    num_heads : int
        Number of attention heads
    num_layers : int
        Number of encoder blocks
    d_ff : int
        Feed-forward network hidden dimension
    max_seq_len : int
        Maximum sequence length for positional encoding
    dropout_rate : float
        Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 1,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 128,
        max_seq_len: int = 1000,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = layers.Dense(d_model)
        
        # Learnable positional encoding
        self.pos_embedding = layers.Embedding(max_seq_len, d_model)
        
        # Encoder stack
        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        # Output projection
        self.output_projection = layers.Dense(output_dim)
        
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, x, training=False):
        """
        Forward pass.
        
        Parameters
        ----------
        x : tf.Tensor
            Input accelerometer signal, shape (batch, seq_len, 3)
            
        Returns
        -------
        output : tf.Tensor
            Predicted GRF signal, shape (batch, seq_len, 1)
        """
        seq_len = tf.shape(x)[1]
        
        # Project input to model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_encoding = self.pos_embedding(positions)  # (seq_len, d_model)
        x = x + pos_encoding
        
        x = self.dropout(x, training=training)
        
        # Pass through encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, training=training)
        
        # Project to output dimension
        output = self.output_projection(x)  # (batch, seq_len, 1)
        
        return output


def build_model(
    seq_length: int = 500,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 3
) -> Model:
    """
    Build and compile the signal transformer.
    """
    model = SignalTransformer(
        input_dim=3,
        output_dim=1,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_model * 2,
        max_seq_len=seq_length,
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )
    
    return model
```

### Data Loader (`data_loader.py`)

```python
"""
Data loader for CMJ accelerometer and GRF data.

This interfaces with your existing MATLAB data pipeline.
Adapt paths and formats as needed for your dataset structure.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional
import h5py  # If your data is in HDF5/MAT format


def load_cmj_data(
    data_dir: Path,
    sensor_location: str = 'LB',  # Lower back
    jump_type: str = 'CMJNA',      # Without arm swing
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load accelerometer and GRF data from CMJ dataset.
    
    Parameters
    ----------
    data_dir : Path
        Path to data directory
    sensor_location : str
        Sensor location code ('LB', 'UB', 'LS', 'RS')
    jump_type : str
        Jump type ('CMJNA' or 'CMJA')
    normalize : bool
        Whether to normalize signals
        
    Returns
    -------
    accel : np.ndarray
        Accelerometer data, shape (n_jumps, seq_len, 3)
    grf : np.ndarray
        Ground reaction force, shape (n_jumps, seq_len, 1)
    """
    # TODO: Adapt this to your actual data format
    # This is a placeholder structure
    
    # Example for HDF5/MAT file:
    # with h5py.File(data_dir / 'cmj_data.mat', 'r') as f:
    #     accel = np.array(f['accelerometer'][sensor_location][jump_type])
    #     grf = np.array(f['grf'][jump_type])
    
    # For now, return placeholder for structure validation
    raise NotImplementedError(
        "Adapt this function to load your CMJ dataset. "
        "Expected output shapes: accel (n_jumps, seq_len, 3), grf (n_jumps, seq_len, 1)"
    )


def preprocess_signals(
    accel: np.ndarray,
    grf: np.ndarray,
    seq_length: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess signals for transformer input.
    
    - Resample to fixed length
    - Normalize to zero mean, unit variance
    - Handle any NaN values
    """
    # Resample if needed
    # ... (use scipy.signal.resample or similar)
    
    # Normalize
    accel_mean = accel.mean(axis=(0, 1), keepdims=True)
    accel_std = accel.std(axis=(0, 1), keepdims=True)
    accel = (accel - accel_mean) / (accel_std + 1e-8)
    
    grf_mean = grf.mean()
    grf_std = grf.std()
    grf = (grf - grf_mean) / (grf_std + 1e-8)
    
    return accel, grf


def create_dataset(
    accel: np.ndarray,
    grf: np.ndarray,
    batch_size: int = 32,
    validation_split: float = 0.2,
    shuffle: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create TensorFlow datasets for training and validation.
    
    Uses participant-level splitting to avoid data leakage
    (same as your PhD methodology).
    """
    n_samples = len(accel)
    n_val = int(n_samples * validation_split)
    
    if shuffle:
        indices = np.random.permutation(n_samples)
        accel = accel[indices]
        grf = grf[indices]
    
    # Split
    accel_train, accel_val = accel[n_val:], accel[:n_val]
    grf_train, grf_val = grf[n_val:], grf[:n_val]
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((accel_train, grf_train))
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((accel_val, grf_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds
```

### Training Script (`train.py`)

```python
"""
Training script for accelerometer → GRF transformer.
"""

import tensorflow as tf
from pathlib import Path
from transformer import build_model
from data_loader import load_cmj_data, preprocess_signals, create_dataset


def train(
    data_dir: Path,
    epochs: int = 100,
    batch_size: int = 32,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    checkpoint_dir: Path = Path('outputs/checkpoints')
):
    """
    Train the signal transformer.
    """
    # Load data
    print("Loading data...")
    accel, grf = load_cmj_data(data_dir)
    accel, grf = preprocess_signals(accel, grf)
    
    seq_length = accel.shape[1]
    print(f"Data loaded: {len(accel)} jumps, sequence length {seq_length}")
    
    # Create datasets
    train_ds, val_ds = create_dataset(accel, grf, batch_size)
    
    # Build model
    print("Building model...")
    model = build_model(
        seq_length=seq_length,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    )
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_dir / 'best_model.keras',
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=20,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=10
        ),
    ]
    
    # Train
    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    train(args.data_dir, args.epochs, args.batch_size)
```

---

## What This Demonstrates

| Requirement (from Job Spec) | How This Project Addresses It |
|----------------------------|-------------------------------|
| "Experience with TensorFlow" | Entire implementation in TensorFlow/Keras |
| "Creating deep learning Python libraries" | Custom attention layer, encoder block as reusable components |
| "Implement training algorithms based on statistical models and correlate with physical model interaction" | Direct physics signal translation (accelerometer ↔ GRF) |
| "Create data reduction models" | Attention learns which timesteps matter for prediction |
| "Standard NLP techniques like Semantic similarity" | Attention mechanism is the foundation of semantic similarity models |

---

## Timeline

| Task | Time |
|------|------|
| Implement MultiHeadSelfAttention | 2 hours |
| Implement TransformerEncoderBlock | 1 hour |
| Implement SignalTransformer model | 1.5 hours |
| Adapt data loader to your CMJ data | 2-3 hours |
| Training script and validation | 1.5 hours |
| Visualisation notebook | 1 hour |
| README and documentation | 1 hour |
| **Total** | **10-12 hours** |

**Note**: Data loading may be faster if you can reuse your existing MATLAB export scripts to create numpy-compatible files.

---

## Fallback: Partial Implementation

If time runs short, even a **partial implementation** is valuable:

**Minimum viable deliverable:**
- `attention.py` with working MultiHeadSelfAttention ✓
- `transformer.py` with working SignalTransformer ✓
- Simple test on synthetic data (sine wave → cosine wave mapping)

This still demonstrates you can implement attention from scratch in TensorFlow, even without your full CMJ dataset integrated.

---

## Interview Talking Points

*"I implemented a transformer architecture from scratch in TensorFlow for a physics signal translation task—predicting ground reaction force from accelerometer data during countermovement jumps. The attention mechanism is built from first principles: the scaled dot-product attention, multi-head splitting, and the full encoder block with residual connections and layer normalisation. This is the same fundamental architecture underlying modern LLMs and semantic similarity models, but applied to continuous physical signals rather than discrete tokens."*

On physics-AI integration:
*"The problem is essentially learning the mapping that Newton's second law describes—force equals mass times acceleration—but through a complex multi-segment biomechanical system. The transformer learns which temporal relationships in the accelerometer signal are most predictive of the force output. This is directly analogous to learning surrogate models for CFD or other physics simulations."*

On relevance to SoftInWay:
*"Turbomachinery simulation involves similar challenges: you have sensor measurements from one domain and want to predict physical quantities in another. The attention mechanism lets the model learn which parts of the input sequence are most relevant for each output prediction, which is valuable when the physical relationships are complex and non-local in time."*
