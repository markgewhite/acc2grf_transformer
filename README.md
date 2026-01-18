# Accelerometer to GRF Transformer

A TensorFlow transformer model for sequence-to-sequence regression, mapping triaxial accelerometer signals from a lower back sensor to vertical ground reaction force (vGRF) during countermovement jumps.

## Overview

This project implements an encoder-only transformer architecture that learns to predict ground reaction force from accelerometer data. The model processes normalized signals of 500 timesteps and outputs predicted GRF curves that can be used to derive biomechanical metrics like jump height and peak power.

## Project Structure

```
acc_grf_transformer/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── attention.py          # Multi-head self-attention from scratch
│   ├── transformer.py        # Encoder blocks and SignalTransformer model
│   ├── data_loader.py        # MATLAB data loading and preprocessing
│   ├── visualize_data.py     # Data inspection and debugging plots
│   ├── biomechanics.py       # Jump height and peak power calculations
│   ├── evaluate.py           # Model evaluation metrics
│   └── train.py              # Training script with CLI
├── notebooks/
│   └── visualise_predictions.ipynb
├── outputs/
│   ├── checkpoints/          # Saved models
│   └── figures/              # Generated plots
├── requirements.txt
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

### Data Visualization (Recommended First Step)

Before training, verify your data loading is working correctly:

```bash
python -m src.visualize_data
```

This generates diagnostic plots in `outputs/figures/` and runs sanity checks.

### Training

Basic training with default parameters:

```bash
python -m src.train
```

Custom training configuration:

```bash
python -m src.train \
    --d-model 64 \
    --num-heads 4 \
    --num-layers 3 \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --run-name my_experiment
```

Use triaxial input instead of resultant acceleration:

```bash
python -m src.train --use-triaxial
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | Auto | Path to processedjumpdata.mat |
| `--use-triaxial` | False | Use 3D acceleration (otherwise resultant) |
| `--d-model` | 64 | Transformer model dimension |
| `--num-heads` | 4 | Number of attention heads |
| `--num-layers` | 3 | Number of encoder layers |
| `--d-ff` | 128 | Feed-forward hidden dimension |
| `--dropout` | 0.1 | Dropout rate |
| `--epochs` | 100 | Maximum training epochs |
| `--batch-size` | 32 | Training batch size |
| `--learning-rate` | 1e-4 | Adam learning rate |
| `--patience` | 15 | Early stopping patience |
| `--output-dir` | outputs | Output directory |
| `--run-name` | timestamp | Experiment name |

## Model Architecture

The SignalTransformer uses an encoder-only architecture:

1. **Input Projection**: Linear layer mapping input dimension (1 or 3) to d_model
2. **Positional Encoding**: Learnable position embeddings for 500 timesteps
3. **Encoder Stack**: N transformer encoder blocks, each with:
   - Multi-head self-attention (implemented from scratch)
   - Position-wise feed-forward network
   - Layer normalization and residual connections
4. **Output Projection**: Linear layer mapping d_model to 1 (GRF prediction)

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
from src.transformer import build_signal_transformer
from src.evaluate import evaluate_model, print_evaluation_summary

# Load data
loader = CMJDataLoader(use_resultant=True)
train_ds, val_ds, info = loader.create_datasets()

# Build and train model
model = build_signal_transformer(
    input_dim=1,
    d_model=64,
    num_heads=4,
    num_layers=3,
)
model.fit(train_ds, validation_data=val_ds, epochs=50)

# Evaluate
results = evaluate_model(model, X_val, y_val, loader)
print_evaluation_summary(results)
```

## Requirements

- Python 3.9+
- TensorFlow 2.10+
- NumPy
- SciPy (for MATLAB file loading)
- Matplotlib
- scikit-learn
