"""Debug the eigenfunction projection computation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import CMJDataLoader, DEFAULT_DATA_PATH

# Load data with FPC transforms - NO VARIMAX (matching MATLAB)
loader = CMJDataLoader(
    data_path=DEFAULT_DATA_PATH,
    use_resultant=False,  # Triaxial
    input_transform='fpc',
    output_transform='fpc',
    n_components=15,
    variance_threshold=None,
    use_varimax=False,  # CRITICAL: MATLAB uses unrotated!
)
train_ds, val_ds, info = loader.create_datasets(test_size=0.2, batch_size=32, random_state=42)

input_transformer = info['input_transformer']
output_transformer = info['output_transformer']

# Get eigenfunctions (rotated to match scores)
input_eigenfuncs = input_transformer.get_eigenfunctions(rotated=True)
output_eigenfuncs = output_transformer.get_eigenfunctions(rotated=True)

time_points = output_transformer.get_time_points()

print("=" * 60)
print("EIGENFUNCTION ANALYSIS")
print("=" * 60)

print(f"\nTime points: {len(time_points)} points, range [{time_points[0]:.3f}, {time_points[-1]:.3f}]")
print(f"\nInput (ACC) eigenfunctions: {len(input_eigenfuncs)} channels")
for ch, ef in enumerate(input_eigenfuncs):
    print(f"  Channel {ch}: shape {ef.shape}")

print(f"\nOutput (GRF) eigenfunctions: {len(output_eigenfuncs)} channels")
for ch, ef in enumerate(output_eigenfuncs):
    print(f"  Channel {ch}: shape {ef.shape}")

# Check eigenfunction norms (should be ~1 for L²-normalized)
print("\n" + "-" * 60)
print("EIGENFUNCTION NORMS (should be ~1 for L²-normalized)")
print("-" * 60)

for ch, ef in enumerate(input_eigenfuncs):
    norms = [np.trapz(ef[:, j]**2, time_points) for j in range(ef.shape[1])]
    print(f"ACC channel {ch}: norms = {norms[:5]}...")  # First 5

for ch, ef in enumerate(output_eigenfuncs):
    norms = [np.trapz(ef[:, j]**2, time_points) for j in range(ef.shape[1])]
    print(f"GRF channel {ch}: norms = {norms[:5]}...")

# Compare ACC z-axis (channel 2) with GRF eigenfunctions
print("\n" + "-" * 60)
print("INNER PRODUCTS: ACC z-axis (ch2) vs GRF eigenfunctions")
print("-" * 60)

acc_z_ef = input_eigenfuncs[2]  # Z-axis (vertical)
grf_ef = output_eigenfuncs[0]

print("\nInner product matrix (ACC_z FPCs vs GRF FPCs):")
inner_products = np.zeros((min(5, acc_z_ef.shape[1]), min(5, grf_ef.shape[1])))
for i in range(inner_products.shape[0]):
    for j in range(inner_products.shape[1]):
        inner_products[i, j] = np.trapz(acc_z_ef[:, i] * grf_ef[:, j], time_points)

print("       ", "  ".join([f"GRF{j+1:2d}" for j in range(inner_products.shape[1])]))
for i in range(inner_products.shape[0]):
    print(f"ACC_z{i+1}: " + "  ".join([f"{inner_products[i,j]:6.3f}" for j in range(inner_products.shape[1])]))

# Plot first few eigenfunctions to visually compare
print("\n" + "-" * 60)
print("PLOTTING EIGENFUNCTIONS")
print("-" * 60)

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for j in range(5):
    # Row 0: GRF eigenfunctions
    axes[0, j].plot(time_points, grf_ef[:, j], 'b-', linewidth=2)
    axes[0, j].set_title(f'GRF FPC {j+1}')
    axes[0, j].set_ylabel('GRF' if j == 0 else '')

    # Row 1: ACC z-axis eigenfunctions
    axes[1, j].plot(time_points, acc_z_ef[:, j], 'r-', linewidth=2)
    axes[1, j].set_title(f'ACC-Z FPC {j+1}')
    axes[1, j].set_ylabel('ACC-Z' if j == 0 else '')

    # Row 2: Overlay (normalized for comparison)
    grf_norm = grf_ef[:, j] / np.max(np.abs(grf_ef[:, j]))
    acc_norm = acc_z_ef[:, j] / np.max(np.abs(acc_z_ef[:, j]))
    axes[2, j].plot(time_points, grf_norm, 'b-', label='GRF', linewidth=2)
    axes[2, j].plot(time_points, acc_norm, 'r--', label='ACC-Z', linewidth=2)
    axes[2, j].set_title(f'Overlay FPC {j+1}')
    axes[2, j].set_ylabel('Normalized' if j == 0 else '')
    if j == 4:
        axes[2, j].legend()

plt.tight_layout()
plt.savefig('outputs/projection_analysis/eigenfunction_comparison.png', dpi=150)
print("Saved eigenfunction comparison to outputs/projection_analysis/eigenfunction_comparison.png")

# Check mean functions
print("\n" + "-" * 60)
print("MEAN FUNCTION ANALYSIS")
print("-" * 60)

input_components = input_transformer.get_inverse_transform_components()
output_components = output_transformer.get_inverse_transform_components()

input_means = input_components['mean_functions']
output_means = output_components['mean_functions']

print(f"\nACC mean functions (per channel):")
for ch, mean in enumerate(input_means):
    print(f"  Channel {ch}: range [{mean.min():.4f}, {mean.max():.4f}], L² norm = {np.sqrt(np.trapz(mean**2, time_points)):.4f}")

print(f"\nGRF mean function:")
for ch, mean in enumerate(output_means):
    print(f"  Channel {ch}: range [{mean.min():.4f}, {mean.max():.4f}], L² norm = {np.sqrt(np.trapz(mean**2, time_points)):.4f}")

# Compute rescale factor as in my implementation
output_mean_sq = sum(np.trapz(m**2, time_points) for m in output_means)
input_mean_sq = sum(np.trapz(m**2, time_points) for m in input_means)
rescale = np.sqrt(output_mean_sq / input_mean_sq)
print(f"\nRescale factor (from mean functions): {rescale:.4f}")

# Check actual score statistics
print("\n" + "-" * 60)
print("SCORE STATISTICS")
print("-" * 60)

X_train, y_train = [], []
for X, y in train_ds:
    X_train.append(X.numpy())
    y_train.append(y.numpy())
X_train = np.concatenate(X_train)  # (n_samples, 15, 3)
y_train = np.concatenate(y_train)  # (n_samples, 15, 1)

print(f"\nACC scores shape: {X_train.shape}")
print(f"GRF scores shape: {y_train.shape}")

print(f"\nACC score statistics (per channel):")
for ch in range(3):
    scores = X_train[:, :, ch]
    print(f"  Channel {ch}: mean={scores.mean():.4f}, std={scores.std():.4f}, range=[{scores.min():.4f}, {scores.max():.4f}]")

print(f"\nGRF score statistics:")
scores = y_train[:, :, 0]
print(f"  mean={scores.mean():.4f}, std={scores.std():.4f}, range=[{scores.min():.4f}, {scores.max():.4f}]")

# Check correlation between ACC-Z scores and GRF scores
print("\n" + "-" * 60)
print("SCORE CORRELATIONS (ACC-Z vs GRF)")
print("-" * 60)

acc_z_scores = X_train[:, :, 2]  # Z-axis scores
grf_scores = y_train[:, :, 0]

print("\nCorrelation matrix (first 5 FPCs):")
print("       ", "  ".join([f"GRF{j+1:2d}" for j in range(5)]))
for i in range(5):
    corrs = [np.corrcoef(acc_z_scores[:, i], grf_scores[:, j])[0, 1] for j in range(5)]
    print(f"ACC_z{i+1}: " + "  ".join([f"{c:6.3f}" for c in corrs]))

plt.show()
