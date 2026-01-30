"""
Analyze FPC Projection Matrix

Computes and visualizes the projection matrix between ACC and GRF FPCs,
compares linear-only vs MLP-only vs hybrid model performance, and
analyzes which ACC FPCs contribute most to which GRF FPCs.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from src.data_loader import CMJDataLoader, DEFAULT_DATA_PATH
from src.transformations import compute_fpc_projection_matrix


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze FPC projection matrix and model performance'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=DEFAULT_DATA_PATH,
        help='Path to processedjumpdata.mat file'
    )
    parser.add_argument(
        '--use-triaxial',
        action='store_true',
        help='Use triaxial acceleration instead of resultant'
    )
    parser.add_argument(
        '--n-components',
        type=int,
        default=15,
        help='Number of FPC components (default: 15)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/projection_analysis',
        help='Output directory for figures (default: outputs/projection_analysis)'
    )
    parser.add_argument(
        '--variance-threshold',
        type=float,
        default=None,
        help='Use variance threshold instead of fixed n_components (e.g., 0.99)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    return parser.parse_args()


def plot_projection_matrix(P: np.ndarray, n_channels: int, n_components: int,
                          save_path: str = None):
    """
    Plot the projection matrix as a heatmap.

    Args:
        P: Projection matrix of shape (n_input_features, n_output_components)
        n_channels: Number of input channels
        n_components: Number of FPC components per channel
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(P, aspect='auto', cmap='RdBu_r', vmin=-np.abs(P).max(), vmax=np.abs(P).max())

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Projection Coefficient')

    # Labels
    ax.set_xlabel('GRF FPC Component')
    ax.set_ylabel('ACC FPC Component')
    ax.set_title('FPC Projection Matrix: ACC → GRF')

    # Add channel separators for multi-channel input
    if n_channels > 1:
        for ch in range(1, n_channels):
            ax.axhline(y=ch * n_components - 0.5, color='black', linewidth=2)

        # Add channel labels
        channel_labels = ['X', 'Y', 'Z'] if n_channels == 3 else [f'Ch{i}' for i in range(n_channels)]
        for ch, label in enumerate(channel_labels):
            y_pos = ch * n_components + n_components / 2 - 0.5
            ax.text(-1.5, y_pos, label, ha='right', va='center', fontweight='bold')

    # Tick labels
    ax.set_xticks(range(n_components))
    ax.set_xticklabels([f'{i+1}' for i in range(n_components)])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Projection matrix heatmap saved to {save_path}")

    return fig


def plot_fpc_contributions(P: np.ndarray, n_channels: int, n_components_per_channel: list,
                          top_k: int = 5, save_path: str = None):
    """
    Plot which ACC FPCs contribute most to each GRF FPC.

    Args:
        P: Projection matrix
        n_channels: Number of input channels
        n_components_per_channel: List of FPC components per channel
        top_k: Number of top contributors to show
        save_path: Path to save figure
    """
    n_grf_components = P.shape[1]

    # Build index-to-channel mapping for variable components per channel
    idx_to_channel = []
    idx_to_fpc = []
    for ch, n_comp in enumerate(n_components_per_channel):
        for fpc in range(n_comp):
            idx_to_channel.append(ch)
            idx_to_fpc.append(fpc + 1)  # 1-indexed

    # Create figure with subplots for each GRF FPC
    n_cols = 5
    n_rows = (n_grf_components + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    channel_labels = ['X', 'Y', 'Z'] if n_channels == 3 else [f'Ch{i}' for i in range(n_channels)]

    for j in range(n_grf_components):
        ax = axes[j]

        # Get absolute contributions and sort
        contributions = np.abs(P[:, j])
        top_indices = np.argsort(contributions)[-top_k:][::-1]

        # Create labels for top contributors
        labels = []
        values = []
        for idx in top_indices:
            ch_idx = idx_to_channel[idx]
            fpc_idx = idx_to_fpc[idx]
            sign = '+' if P[idx, j] > 0 else '-'
            labels.append(f'{channel_labels[ch_idx]}-FPC{fpc_idx} ({sign})')
            values.append(contributions[idx])

        # Plot horizontal bar chart
        colors = ['green' if P[idx, j] > 0 else 'red' for idx in top_indices]
        y_pos = range(len(labels))
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('|Coefficient|')
        ax.set_title(f'GRF FPC {j+1}')
        ax.invert_yaxis()

    # Hide unused subplots
    for j in range(n_grf_components, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Top ACC FPC Contributors to Each GRF FPC', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"FPC contribution plot saved to {save_path}")

    return fig


def evaluate_linear_projection(P: np.ndarray, rescale: float,
                              X_val: np.ndarray, y_val: np.ndarray,
                              n_input_components: list = None,
                              n_output_components: int = None) -> dict:
    """
    Evaluate pure linear projection performance.

    Args:
        P: Projection matrix
        rescale: Rescaling factor
        X_val: Validation inputs (n_samples, max_features, n_channels) or flattened
        y_val: Validation targets (n_samples, max_features, n_channels) or flattened
        n_input_components: List of actual components per input channel (for extracting non-padded)
        n_output_components: Actual output components (for extracting non-padded)

    Returns:
        Dictionary with R², RMSE, MAE per component
    """
    # Extract non-padded features if component counts provided
    if X_val.ndim == 3 and n_input_components is not None:
        # X_val shape: (n_samples, max_features, n_channels)
        n_samples = X_val.shape[0]
        X_flat = []
        for ch, n_comp in enumerate(n_input_components):
            X_flat.append(X_val[:, :n_comp, ch])
        X_flat = np.concatenate(X_flat, axis=1)  # (n_samples, total_components)
    elif X_val.ndim == 3:
        X_flat = X_val.reshape(X_val.shape[0], -1)
    else:
        X_flat = X_val

    if y_val.ndim == 3 and n_output_components is not None:
        y_flat = y_val[:, :n_output_components, 0]  # Assume single output channel
    elif y_val.ndim == 3:
        y_flat = y_val.reshape(y_val.shape[0], -1)
    else:
        y_flat = y_val

    # Linear prediction
    y_pred = rescale * (X_flat @ P)

    # Overall metrics
    r2 = r2_score(y_flat, y_pred)
    rmse = np.sqrt(np.mean((y_flat - y_pred) ** 2))
    mae = np.mean(np.abs(y_flat - y_pred))

    # Per-component metrics
    r2_per_comp = []
    for j in range(y_flat.shape[1]):
        r2_j = r2_score(y_flat[:, j], y_pred[:, j])
        r2_per_comp.append(r2_j)

    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'r2_per_component': np.array(r2_per_comp),
        'y_pred': y_pred,
        'y_true': y_flat,  # Return the actual (non-padded) targets used
    }


def plot_linear_vs_target(y_val: np.ndarray, y_pred_linear: np.ndarray,
                         save_path: str = None):
    """
    Plot linear projection predictions vs targets.

    Args:
        y_val: Validation targets
        y_pred_linear: Linear projection predictions
        save_path: Path to save figure
    """
    if y_val.ndim == 3:
        y_val = y_val.reshape(y_val.shape[0], -1)
    if y_pred_linear.ndim == 3:
        y_pred_linear = y_pred_linear.reshape(y_pred_linear.shape[0], -1)

    n_components = y_val.shape[1]
    n_cols = 5
    n_rows = (n_components + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    for j in range(n_components):
        ax = axes[j]
        ax.scatter(y_val[:, j], y_pred_linear[:, j], alpha=0.5, s=10)

        # Add diagonal line
        lims = [
            min(y_val[:, j].min(), y_pred_linear[:, j].min()),
            max(y_val[:, j].max(), y_pred_linear[:, j].max())
        ]
        ax.plot(lims, lims, 'r--', alpha=0.5)

        # Compute R²
        r2 = r2_score(y_val[:, j], y_pred_linear[:, j])
        ax.set_title(f'GRF FPC {j+1} (R²={r2:.3f})')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')

    # Hide unused subplots
    for j in range(n_components, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Linear Projection: Predicted vs True FPC Scores', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Linear vs target plot saved to {save_path}")

    return fig


def main():
    """Main analysis function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FPC Projection Matrix Analysis")
    print("=" * 60)

    # Set random seed
    np.random.seed(args.seed)

    # Load data with FPC transforms
    print("\n--- Loading Data with FPC Transforms ---")
    use_resultant = not args.use_triaxial
    loader = CMJDataLoader(
        data_path=args.data_path,
        use_resultant=use_resultant,
        input_transform='fpc',
        output_transform='fpc',
        n_components=args.n_components,
        variance_threshold=args.variance_threshold,  # None = fixed n_components
        use_varimax=True,
    )

    train_ds, val_ds, info = loader.create_datasets(
        test_size=0.2,
        batch_size=32,
        random_state=args.seed,
    )

    # Get transformers
    input_transformer = info.get('input_transformer')
    output_transformer = info.get('output_transformer')

    if input_transformer is None or output_transformer is None:
        raise ValueError("FPC transformers not found in data info")

    # Compute projection matrix
    print("\n--- Computing Projection Matrix ---")
    P, rescale = compute_fpc_projection_matrix(input_transformer, output_transformer)
    print(f"Projection matrix shape: {P.shape}")
    print(f"Rescale factor: {rescale:.4f}")

    # Save projection matrix
    np.save(output_dir / 'projection_matrix.npy', P)
    print(f"Projection matrix saved to {output_dir}/projection_matrix.npy")

    # Determine dimensions - get actual components per channel from transformer
    n_channels = 1 if use_resultant else 3
    n_components_per_channel = input_transformer._actual_n_components
    n_output_components = output_transformer._actual_n_components[0]

    print(f"Input FPCs per channel: {n_components_per_channel}")
    print(f"Output FPCs: {n_output_components}")

    # Plot projection matrix heatmap
    print("\n--- Generating Visualizations ---")
    plot_projection_matrix(
        P, n_channels, max(n_components_per_channel),
        save_path=str(output_dir / 'projection_matrix_heatmap.png')
    )

    # Plot FPC contributions
    plot_fpc_contributions(
        P, n_channels, n_components_per_channel,
        top_k=5,
        save_path=str(output_dir / 'fpc_contributions.png')
    )

    # Get validation data as arrays
    X_val_list, y_val_list = [], []
    for X_batch, y_batch in val_ds:
        X_val_list.append(X_batch.numpy())
        y_val_list.append(y_batch.numpy())
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)

    # Evaluate linear projection
    print("\n--- Evaluating Linear Projection ---")
    linear_results = evaluate_linear_projection(
        P, rescale, X_val, y_val,
        n_input_components=n_components_per_channel,
        n_output_components=n_output_components
    )
    print(f"Linear projection R²: {linear_results['r2']:.4f}")
    print(f"Linear projection RMSE: {linear_results['rmse']:.4f}")
    print(f"Linear projection MAE: {linear_results['mae']:.4f}")

    # Per-component R²
    print("\nPer-component R²:")
    for j, r2_j in enumerate(linear_results['r2_per_component']):
        print(f"  GRF FPC {j+1}: {r2_j:.4f}")

    # Plot linear predictions vs targets
    plot_linear_vs_target(
        linear_results['y_true'], linear_results['y_pred'],
        save_path=str(output_dir / 'linear_vs_target.png')
    )

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input: {'Resultant' if use_resultant else 'Triaxial'} ACC "
          f"({n_channels} channel(s), FPCs per channel: {n_components_per_channel}, total: {P.shape[0]} features)")
    print(f"Output: GRF ({P.shape[1]} FPCs)")
    print(f"\nLinear projection provides R² = {linear_results['r2']:.4f}")
    print(f"This represents the 'interpretable baseline' performance.")
    print(f"\nThe MLP learns nonlinear corrections to improve upon this baseline.")

    # Build index-to-channel mapping for variable components per channel
    idx_to_channel = []
    idx_to_fpc = []
    for ch, n_comp in enumerate(n_components_per_channel):
        for fpc in range(n_comp):
            idx_to_channel.append(ch)
            idx_to_fpc.append(fpc + 1)  # 1-indexed

    # Identify most important ACC FPCs
    print("\n--- Most Important ACC FPCs (by total absolute contribution) ---")
    total_contrib = np.sum(np.abs(P), axis=1)
    top_indices = np.argsort(total_contrib)[-10:][::-1]

    channel_labels = ['X', 'Y', 'Z'] if n_channels == 3 else ['R']
    for rank, idx in enumerate(top_indices, 1):
        ch_idx = idx_to_channel[idx]
        fpc_idx = idx_to_fpc[idx]
        print(f"  {rank}. {channel_labels[ch_idx]}-FPC{fpc_idx}: {total_contrib[idx]:.4f}")

    print(f"\nAnalysis complete. Results saved to {output_dir}/")


if __name__ == '__main__':
    main()
