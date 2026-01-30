"""Test if a learned linear projection works (vs eigenfunction inner products)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from src.data_loader import CMJDataLoader, DEFAULT_DATA_PATH

loader = CMJDataLoader(
    data_path=DEFAULT_DATA_PATH,
    use_resultant=False,
    input_transform='fpc',
    output_transform='fpc',
    n_components=15,
    variance_threshold=None,
    use_varimax=True,
)
train_ds, val_ds, info = loader.create_datasets(test_size=0.2, batch_size=32, random_state=42)

# Get data as arrays
X_train, y_train = [], []
for X, y in train_ds:
    X_train.append(X.numpy())
    y_train.append(y.numpy())
X_train = np.concatenate(X_train).reshape(-1, 45)  # 15*3
y_train = np.concatenate(y_train).reshape(-1, 15)

X_val, y_val = [], []
for X, y in val_ds:
    X_val.append(X.numpy())
    y_val.append(y.numpy())
X_val = np.concatenate(X_val).reshape(-1, 45)
y_val = np.concatenate(y_val).reshape(-1, 15)

print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'X_val: {X_val.shape}, y_val: {y_val.shape}')

# Learn linear projection via Ridge regression
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

print(f'\nLearned linear projection R²: {r2_score(y_val, y_pred):.4f}')

# Per-component R²
print('\nPer-component R²:')
for j in range(15):
    r2_j = r2_score(y_val[:, j], y_pred[:, j])
    print(f'  GRF FPC {j+1}: {r2_j:.4f}')
