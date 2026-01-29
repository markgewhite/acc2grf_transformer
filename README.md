# Accelerometer to GRF Prediction

Predicting vertical ground reaction force (vGRF) from triaxial accelerometer signals during countermovement jumps using Functional Principal Component Analysis (FPCA) and a simple MLP.

## Key Finding

**A simple MLP (~12K parameters) with FPC representation massively outperforms a transformer (~750K parameters).** The data representation matters far more than model architecture complexity.

### Results (5-trial validation, triaxial input)

| Metric | Value | Reference Baseline |
|--------|-------|--------------------|
| **Jump Height R²** | **0.82 ± 0.03** | 0.87 |
| **Jump Height Median Error** | **3.5 cm** | — |
| **Peak Power R²** | **0.80 ± 0.03** | 0.99 |
| **Peak Power Median Error** | **2.7 W/kg** | — |
| **Signal R² (BW)** | **0.971** | — |
| **Invalid predictions** | **0** | — |

The reference baseline is the theoretical maximum achievable from the 500ms signal window.

## Overview

This project maps accelerometer signals to GRF curves, enabling force plate-quality biomechanical metrics (jump height, peak power) from a single wearable sensor. The approach uses:

1. **Functional Principal Component Analysis (FPCA)** to represent both input (ACC) and output (GRF) signals as low-dimensional score vectors
2. **A simple MLP** (single hidden layer) to learn the mapping between FPC scores
3. **Triaxial accelerometer input** which preserves directional information critical for prediction

## Project Structure

```
acc_grf_transformer/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── attention.py          # Multi-head self-attention (legacy)
│   ├── transformer.py        # Transformer model (legacy)
│   ├── mlp.py                # MLP model (recommended)
│   ├── transformations.py    # FPCA and B-spline transforms
│   ├── data_loader.py        # MATLAB data loading and preprocessing
│   ├── visualize_data.py     # Data inspection and debugging plots
│   ├── biomechanics.py       # Jump height and peak power calculations
│   ├── losses.py             # Custom loss functions
│   ├── evaluate.py           # Model evaluation metrics
│   └── train.py              # Training script with CLI
├── notebooks/
│   └── visualise_predictions.ipynb
├── outputs/
│   ├── checkpoints/          # Saved models
│   └── figures/              # Generated plots
├── requirements.txt
├── EXPERIMENTS.md            # Detailed experiment log
└── README.md
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Recommended Configuration (Best Results)

```bash
python -m src.train \
    --model-type mlp --mlp-hidden 128 \
    --use-triaxial \
    --input-transform fpc --output-transform fpc \
    --loss reconstruction \
    --simple-normalization \
    --n-trials 5 --seed 42 \
    --epochs 200
```

This achieves JH R² = 0.82 ± 0.03 and PP R² = 0.80 ± 0.03.

### Data Visualization (Recommended First Step)

Before training, verify your data loading is working correctly:

```bash
python -m src.visualize_data
```

This generates diagnostic plots in `outputs/figures/` and runs sanity checks.

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | Auto | Path to processedjumpdata.mat |
| `--model-type` | transformer | Model type: `mlp` (recommended) or `transformer` |
| `--mlp-hidden` | 128 | MLP hidden layer size |
| `--use-triaxial` | False | Use 3D acceleration (recommended: True) |
| `--input-transform` | raw | Input transform: `raw`, `bspline`, or `fpc` |
| `--output-transform` | raw | Output transform: `raw`, `bspline`, or `fpc` |
| `--loss` | mse | Loss function: `mse`, `reconstruction`, `signal_space` |
| `--simple-normalization` | False | Use global z-score (recommended: True) |
| `--n-trials` | 1 | Number of trials for statistical validation |
| `--epochs` | 100 | Maximum training epochs |
| `--batch-size` | 32 | Training batch size |
| `--learning-rate` | 1e-4 | Adam learning rate |
| `--patience` | 15 | Early stopping patience |
| `--output-dir` | outputs | Output directory |
| `--run-name` | timestamp | Experiment name |

## Model Architecture

### Recommended: MLP with FPC Transforms

The best-performing architecture is surprisingly simple:

```
ACC signal (500×3) → FPCA → FPC scores (45) → MLP → FPC scores (15) → Inverse FPCA → GRF signal (500×1)
```

**MLP Architecture:**
- Input: 45 features (15 FPCs × 3 channels for triaxial)
- Hidden: 128 neurons with ReLU activation
- Output: 15 features (15 FPCs for GRF)
- Parameters: ~12K

**Why MLP beats Transformer:**
1. **FPC representation does the heavy lifting** — the mean function captures the typical CMJ shape, so the model only learns deviations
2. **Attention adds no value** — the mapping from ACC FPCs to GRF FPCs doesn't benefit from temporal attention
3. **Simpler models generalize better** with limited data (896 training samples)

### Legacy: Transformer Architecture

The transformer architecture (~750K parameters) is still available but not recommended:

1. **Input Projection**: Linear layer mapping input dimension to d_model
2. **Positional Encoding**: Learnable position embeddings for 500 timesteps
3. **Encoder Stack**: N transformer encoder blocks with multi-head self-attention
4. **Output Projection**: Linear layer mapping d_model to output dimension

## Data

### Input Format
- **Accelerometer**: Lower back sensor, triaxial (x, y, z) in g units
- **Signal Mode**: Resultant acceleration √(x² + y² + z²) or raw triaxial
- **Preprocessing**: Padded/truncated to 500 points, z-score normalized

### Output Format
- **GRF**: Vertical ground reaction force in body weight (BW) units
- **Preprocessing**: Normalized by body weight, z-score normalized

### Data Splits
- Participant-level train/validation split (no data leakage)
- Default: 80% train, 20% validation

## Evaluation Metrics

### Signal-Level
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

### Biomechanical Metrics
Derived from predicted GRF using impulse-momentum method:

- **Jump Height**: Computed via double integration of net force
- **Peak Power**: Maximum instantaneous power (F × v)

Both metrics compared against ground truth with RMSE, MAE, R², and Bland-Altman analysis.

## Output Files

After training, the following files are generated:

```
outputs/<run_name>/
├── config.json              # Training configuration
├── data_info.json           # Dataset statistics
├── evaluation_results.csv   # All metrics in CSV format
├── checkpoints/
│   ├── best_model.keras     # Best validation model
│   ├── final_model.keras    # Final epoch model
│   └── training_log.csv     # Epoch-by-epoch metrics
└── figures/
    ├── prediction_curves.png  # Predicted vs actual GRF
    ├── scatter_metrics.png    # Jump height/power scatter
    ├── bland_altman.png       # Agreement analysis
    └── training_history.png   # Loss curves
```

## Python API

```python
from src.data_loader import CMJDataLoader
from src.mlp import build_mlp_model
from src.evaluate import evaluate_model, print_evaluation_summary

# Load data with FPC transforms (recommended)
loader = CMJDataLoader(
    use_triaxial=True,
    input_transform='fpc',
    output_transform='fpc',
    simple_normalization=True
)
train_ds, val_ds, info = loader.create_datasets()

# Build and train MLP model
model = build_mlp_model(
    input_dim=info['input_dim'],  # 45 for triaxial FPC
    output_dim=info['output_dim'],  # 15 for GRF FPC
    hidden_dim=128
)
model.fit(train_ds, validation_data=val_ds, epochs=200)

# Evaluate
results = evaluate_model(model, X_val, y_val, loader)
print_evaluation_summary(results)
```

## Key Insights

From extensive experimentation (see `EXPERIMENTS.md`):

1. **Representation matters more than architecture**: A simple MLP with 12K parameters outperforms a 750K-parameter transformer
2. **FPC representation is the key**: Functional Principal Components capture biomechanically relevant features that raw signals and B-splines miss
3. **Triaxial input preserves critical information**: Directional acceleration data improves JH R² by 0.15 compared to resultant magnitude
4. **Simple normalization works best**: Global z-score outperforms sophisticated robust normalization
5. **Temporal weighting doesn't help**: Contrary to intuition, jerk-based weighting provides no benefit

### Why FPC Works

1. **Mean function captures the template** — the typical CMJ shape is baked in; the model only learns deviations
2. **Variance-ordered components** naturally weight importance
3. **Massive dimensionality reduction**: 15 FPCs vs 500 raw samples
4. **Quiet standing is error-free** — errors don't compound through double integration

## Requirements

- Python 3.9+
- TensorFlow 2.10+
- NumPy
- SciPy (for MATLAB file loading)
- scikit-fda (for FPCA transforms)
- Matplotlib
- scikit-learn
