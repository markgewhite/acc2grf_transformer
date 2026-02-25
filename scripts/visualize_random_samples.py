"""
Visualize Random Samples: ACC and GRF Signal Grid

Displays randomly selected samples with ACC signals stacked above GRF signals
in a configurable grid layout. Useful for visual inspection of data quality
and signal characteristics.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from src.data_loader import CMJDataLoader, DEFAULT_DATA_PATH, SAMPLING_RATE


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize random ACC/GRF sample pairs in a grid layout'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=DEFAULT_DATA_PATH,
        help='Path to cmj_dataset.npz file'
    )
    parser.add_argument(
        '--n-rows',
        type=int,
        default=2,
        help='Number of rows in the grid (default: 2)'
    )
    parser.add_argument(
        '--n-cols',
        type=int,
        default=3,
        help='Number of columns in the grid (default: 3)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/random_samples',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for saved figures (default: 150)'
    )
    return parser.parse_args()


def get_truncated_acc(acc: np.ndarray, takeoff_idx: int) -> tuple:
    """Get ACC data truncated to last 2000ms with time array."""
    time_acc = (np.arange(len(acc)) - takeoff_idx + 1) / SAMPLING_RATE * 1000
    mask = time_acc >= -2000
    return time_acc[mask], acc[mask]


def get_truncated_grf(grf: np.ndarray) -> tuple:
    """Get GRF data truncated to last 2000ms with time array."""
    time_grf = (np.arange(len(grf)) - len(grf) + 1) / SAMPLING_RATE * 1000
    mask = time_grf >= -2000
    return time_grf[mask], grf[mask]


def plot_acc_signal(ax: plt.Axes, acc: np.ndarray, takeoff_idx: int,
                    show_legend: bool = False, show_ylabel: bool = True,
                    ylim: tuple = None) -> None:
    """
    Plot accelerometer signal with triaxial components and resultant.

    Args:
        ax: Matplotlib axis
        acc: Accelerometer data of shape (n_timesteps, 3)
        takeoff_idx: Index of takeoff in the signal
        show_legend: Whether to show legend
        show_ylabel: Whether to show y-axis label and tick labels
        ylim: Y-axis limits as (min, max) tuple
    """
    # Get truncated data
    time_acc, acc = get_truncated_acc(acc, takeoff_idx)

    # Plot triaxial components (non-blue colors - GRF is blue)
    ax.plot(time_acc, acc[:, 0], color='#d62728', alpha=0.7, linewidth=0.8, label='X')  # red
    ax.plot(time_acc, acc[:, 1], color='#ff7f0e', alpha=0.7, linewidth=0.8, label='Y')  # orange
    ax.plot(time_acc, acc[:, 2], color='#9467bd', alpha=0.7, linewidth=0.8, label='Z')  # purple

    # Plot resultant
    resultant = np.sqrt(np.sum(acc ** 2, axis=1))
    ax.plot(time_acc, resultant, 'k-', linewidth=1.0, label='R')

    # Reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, linewidth=0.5)

    # Formatting
    ax.set_xlim(-2000, 0)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis='both', labelsize=7)
    if show_ylabel:
        ax.set_ylabel('ACC (g)', fontsize=7)
    else:
        ax.tick_params(labelleft=False)

    if show_legend:
        ax.legend(loc='upper left', fontsize=6, framealpha=0.8, ncol=4)


def plot_grf_signal(ax: plt.Axes, grf: np.ndarray,
                    show_legend: bool = False, show_ylabel: bool = True,
                    show_xlabel: bool = True, ylim: tuple = None) -> None:
    """
    Plot ground reaction force signal.

    Args:
        ax: Matplotlib axis
        grf: GRF data of shape (n_timesteps,)
        show_legend: Whether to show legend
        show_ylabel: Whether to show y-axis label and tick labels
        show_xlabel: Whether to show x-axis label
        ylim: Y-axis limits as (min, max) tuple
    """
    # Get truncated data
    time_grf, grf = get_truncated_grf(grf)

    # Plot GRF (blue - the only blue signal)
    ax.plot(time_grf, grf, 'b-', linewidth=1.0, label='GRF')

    # Reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, linewidth=0.5)

    # Formatting
    ax.set_xlim(-2000, 0)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis='both', labelsize=7)
    if show_ylabel:
        ax.set_ylabel('GRF (BW)', fontsize=7)
    else:
        ax.tick_params(labelleft=False)
    if show_xlabel:
        ax.set_xlabel('Time (ms)', fontsize=7)

    if show_legend:
        ax.legend(loc='upper left', fontsize=6, framealpha=0.8)


def create_sample_grid(loader: CMJDataLoader, sample_indices: np.ndarray,
                       n_rows: int, n_cols: int) -> plt.Figure:
    """
    Create grid figure with ACC stacked above GRF for each sample.

    Args:
        loader: CMJDataLoader with loaded data
        sample_indices: Array of sample indices to display
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid

    Returns:
        Matplotlib figure
    """
    # First pass: compute global y-axis ranges
    acc_min, acc_max = np.inf, -np.inf
    grf_min, grf_max = np.inf, -np.inf

    for idx in sample_indices:
        acc_item = loader.acc_data[idx]
        if isinstance(acc_item, tuple):
            acc, takeoff_idx = acc_item
        else:
            acc = acc_item
            takeoff_idx = len(acc)

        grf = loader.grf_data[idx]

        # Get truncated data for range calculation
        _, acc_trunc = get_truncated_acc(acc, takeoff_idx)
        _, grf_trunc = get_truncated_grf(grf)

        # ACC: include all components and resultant
        resultant = np.sqrt(np.sum(acc_trunc ** 2, axis=1))
        acc_min = min(acc_min, acc_trunc.min(), resultant.min())
        acc_max = max(acc_max, acc_trunc.max(), resultant.max())

        # GRF
        grf_min = min(grf_min, grf_trunc.min())
        grf_max = max(grf_max, grf_trunc.max())

    # Add small margin to y-limits
    acc_margin = (acc_max - acc_min) * 0.05
    grf_margin = (grf_max - grf_min) * 0.05
    acc_ylim = (acc_min - acc_margin, acc_max + acc_margin)
    grf_ylim = (grf_min - grf_margin, grf_max + grf_margin)

    # Figure size: 2.8 inches per column, 2.5 inches per row (tighter)
    fig = plt.figure(figsize=(2.8 * n_cols, 2.5 * n_rows))

    # Outer grid for sample pairs (tighter spacing)
    outer_gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.25, wspace=0.15)

    for i, idx in enumerate(sample_indices):
        row = i // n_cols
        col = i % n_cols

        # Determine which labels to show
        is_left_col = (col == 0)
        is_bottom_row = (row == n_rows - 1)

        # Get data for this sample
        acc_item = loader.acc_data[idx]
        if isinstance(acc_item, tuple):
            acc, takeoff_idx = acc_item
        else:
            acc = acc_item
            takeoff_idx = len(acc)

        grf = loader.grf_data[idx]
        jh = loader.ground_truth_jump_height[idx]
        pp = loader.ground_truth_peak_power[idx]  # Already in W/kg

        # Inner grid for ACC/GRF stacking
        inner_gs = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[row, col],
            hspace=0.08, height_ratios=[1, 1]
        )

        ax_acc = fig.add_subplot(inner_gs[0])
        ax_grf = fig.add_subplot(inner_gs[1])

        # Show legend only for first box pair (top-left)
        show_legend = (i == 0)

        # Plot signals with conditional labels and standardized y-limits
        plot_acc_signal(ax_acc, acc, takeoff_idx,
                        show_legend=show_legend, show_ylabel=is_left_col,
                        ylim=acc_ylim)
        plot_grf_signal(ax_grf, grf,
                        show_legend=show_legend, show_ylabel=is_left_col,
                        show_xlabel=is_bottom_row, ylim=grf_ylim)

        # Title above box pair
        title = f'#{idx} JH={jh:.2f}, PP={pp:.1f}'
        ax_acc.set_title(title, fontsize=8, fontweight='bold', pad=3)

        # Remove x-axis tick labels from ACC plot
        ax_acc.tick_params(labelbottom=False)

        # Remove x-axis tick labels from GRF if not bottom row
        if not is_bottom_row:
            ax_grf.tick_params(labelbottom=False)

    return fig


def main() -> None:
    """Main function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Random Sample Visualization")
    print("=" * 60)

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    # Load data (triaxial for full visualization)
    print("\n--- Loading Data ---")
    loader = CMJDataLoader(
        data_path=args.data_path,
        use_resultant=False,  # Always use triaxial for this visualization
    )
    loader.load_data()

    n_total = len(loader.acc_data)
    n_samples = args.n_rows * args.n_cols

    print(f"Total samples available: {n_total}")
    print(f"Grid size: {args.n_rows} rows x {args.n_cols} cols = {n_samples} samples")

    # Select random samples
    sample_indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)
    print(f"Selected sample indices: {sample_indices}")

    # Create grid figure
    print("\n--- Creating Visualization ---")
    fig = create_sample_grid(loader, sample_indices, args.n_rows, args.n_cols)

    # Save figure
    output_path = output_dir / 'random_samples.png'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to {output_path}")

    plt.show()


if __name__ == '__main__':
    main()
