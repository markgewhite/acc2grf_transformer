"""
Data Visualization Module

Tools for visualizing and debugging the CMJ data loading process.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import CMJDataLoader, SAMPLING_RATE


def plot_random_samples(
    loader: CMJDataLoader,
    n_samples: int = 5,
    save_path: Optional[str] = None,
    figsize: tuple = (13, 2),
) -> plt.Figure:
    """
    Plot random ACC/GRF sample pairs for visual inspection.

    Args:
        loader: CMJDataLoader with loaded data
        n_samples: Number of random samples to plot
        save_path: Path to save figure (optional)
        figsize: Figure size per sample

    Returns:
        Matplotlib figure
    """
    if loader.acc_data is None:
        raise ValueError("No data loaded. Call loader.load_data() first.")

    # Select random indices
    n_total = len(loader.acc_data)
    indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

    fig, axes = plt.subplots(n_samples, 2, figsize=(figsize[0], figsize[1] * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        # Handle new format: acc_data is (signal, takeoff_idx) tuple
        acc_item = loader.acc_data[idx]
        if isinstance(acc_item, tuple):
            acc, acc_takeoff_idx = acc_item
        else:
            acc = acc_item
            acc_takeoff_idx = len(acc)  # Assume takeoff at end

        grf = loader.grf_data[idx]
        subj_id = loader.subject_ids[idx]
        jump_idx = loader.jump_indices[idx]

        # Time axis relative to takeoff (t=0 at takeoff)
        # ACC: takeoff is at acc_takeoff_idx
        time_acc = (np.arange(len(acc)) - acc_takeoff_idx) / SAMPLING_RATE * 1000  # ms
        # GRF: takeoff is at the end
        time_grf = (np.arange(len(grf)) - len(grf)) / SAMPLING_RATE * 1000  # ms

        # Plot accelerometer
        ax_acc = axes[i, 0]
        if acc.ndim > 1 and acc.shape[1] == 3:
            ax_acc.plot(time_acc, acc[:, 0], 'r-', alpha=0.7, label='X')
            ax_acc.plot(time_acc, acc[:, 1], 'g-', alpha=0.7, label='Y')
            ax_acc.plot(time_acc, acc[:, 2], 'b-', alpha=0.7, label='Z')
            resultant = np.sqrt(np.sum(acc ** 2, axis=1))
            ax_acc.plot(time_acc, resultant, 'k-', linewidth=1.5, label='Resultant')
            ax_acc.legend(loc='lower left', fontsize=8)
        else:
            ax_acc.plot(time_acc, acc, 'b-')

        ax_acc.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='1g')
        ax_acc.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='Takeoff')
        ax_acc.set_xlabel('Time (ms)')
        ax_acc.set_ylabel('Acceleration (g)')
        ax_acc.set_title(f'Subject {subj_id}, Jump {jump_idx} - Accelerometer')
        ax_acc.grid(True, alpha=0.3)

        # Plot GRF
        ax_grf = axes[i, 1]
        ax_grf.plot(time_grf, grf, 'b-', linewidth=1.5)
        ax_grf.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='1 BW')
        ax_grf.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5)
        ax_grf.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='Takeoff')
        ax_grf.set_xlabel('Time (ms)')
        ax_grf.set_ylabel('GRF (BW)')
        ax_grf.set_title(f'Subject {subj_id}, Jump {jump_idx} - Ground Reaction Force')
        ax_grf.grid(True, alpha=0.3)
        ax_grf.legend(loc='lower left', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_signal_length_distribution(
    loader: CMJDataLoader,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot distribution of signal lengths before padding.

    Args:
        loader: CMJDataLoader with loaded data
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    if loader.acc_data is None:
        raise ValueError("No data loaded. Call loader.load_data() first.")

    # Handle new format: acc_data may be (signal, takeoff_idx) tuples
    lengths = []
    for acc_item in loader.acc_data:
        if isinstance(acc_item, tuple):
            lengths.append(len(acc_item[0]))
        else:
            lengths.append(len(acc_item))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=500, color='red', linestyle='--', linewidth=2, label='Target length (500)')
    ax.axvline(x=np.mean(lengths), color='green', linestyle='-', linewidth=2, label=f'Mean ({np.mean(lengths):.0f})')

    ax.set_xlabel('Signal Length (samples)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Signal Lengths Before Padding')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Min: {min(lengths)}\nMax: {max(lengths)}\nMean: {np.mean(lengths):.1f}\nStd: {np.std(lengths):.1f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_value_ranges(
    loader: CMJDataLoader,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot value ranges for ACC and GRF signals.

    Args:
        loader: CMJDataLoader with loaded data
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    if loader.acc_data is None:
        raise ValueError("No data loaded. Call loader.load_data() first.")

    # Compute statistics
    acc_mins, acc_maxs = [], []
    grf_mins, grf_maxs = [], []
    acc_starts, grf_starts = [], []

    for acc_item, grf in zip(loader.acc_data, loader.grf_data):
        # Handle new format: acc_data may be (signal, takeoff_idx) tuples
        acc = acc_item[0] if isinstance(acc_item, tuple) else acc_item
        resultant = np.sqrt(np.sum(acc ** 2, axis=1))
        acc_mins.append(np.min(resultant))
        acc_maxs.append(np.max(resultant))
        acc_starts.append(resultant[0])

        grf_mins.append(np.min(grf))
        grf_maxs.append(np.max(grf))
        grf_starts.append(grf[0])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # ACC range histograms
    axes[0, 0].hist(acc_mins, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Min Resultant ACC (g)')
    axes[0, 0].set_title('Minimum ACC Values')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(acc_maxs, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Max Resultant ACC (g)')
    axes[0, 1].set_title('Maximum ACC Values')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].hist(acc_starts, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(x=1.0, color='red', linestyle='--', label='Expected (1g)')
    axes[0, 2].set_xlabel('Initial Resultant ACC (g)')
    axes[0, 2].set_title('Starting ACC Values (should be ~1g)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # GRF range histograms
    axes[1, 0].hist(grf_mins, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Min GRF (BW)')
    axes[1, 0].set_title('Minimum GRF Values')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(grf_maxs, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Max GRF (BW)')
    axes[1, 1].set_title('Maximum GRF Values')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].hist(grf_starts, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 2].axvline(x=1.0, color='red', linestyle='--', label='Expected (1 BW)')
    axes[1, 2].set_xlabel('Initial GRF (BW)')
    axes[1, 2].set_title('Starting GRF Values (should be ~1 BW)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_samples_per_subject(
    loader: CMJDataLoader,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot number of samples per subject.

    Args:
        loader: CMJDataLoader with loaded data
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    if loader.subject_ids is None:
        raise ValueError("No data loaded. Call loader.load_data() first.")

    unique_subjects, counts = np.unique(loader.subject_ids, return_counts=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(range(len(unique_subjects)), counts, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Subject Index')
    ax.set_ylabel('Number of Jumps')
    ax.set_title(f'Jumps per Subject (Total: {len(loader.subject_ids)} jumps from {len(unique_subjects)} subjects)')
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean line
    ax.axhline(y=np.mean(counts), color='red', linestyle='--', label=f'Mean ({np.mean(counts):.1f})')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def run_sanity_checks(loader: CMJDataLoader) -> dict:
    """
    Run sanity checks on loaded data.

    Checks:
    - ACC should start near 1g (quiet standing)
    - GRF should start near 1 BW (quiet standing)
    - Both should have characteristic CMJ pattern

    Args:
        loader: CMJDataLoader with loaded data

    Returns:
        Dictionary with check results
    """
    if loader.acc_data is None:
        raise ValueError("No data loaded. Call loader.load_data() first.")

    results = {'passed': [], 'warnings': [], 'errors': []}

    # Check ACC starting values
    acc_starts = []
    for acc_item in loader.acc_data:
        # Handle new format: acc_data may be (signal, takeoff_idx) tuples
        acc = acc_item[0] if isinstance(acc_item, tuple) else acc_item
        resultant = np.sqrt(np.sum(acc ** 2, axis=1))
        acc_starts.append(resultant[0])

    mean_acc_start = np.mean(acc_starts)
    if 0.8 <= mean_acc_start <= 1.2:
        results['passed'].append(f"ACC starts near 1g (mean: {mean_acc_start:.3f}g)")
    else:
        results['warnings'].append(f"ACC starting value unexpected (mean: {mean_acc_start:.3f}g, expected ~1g)")

    # Check GRF starting values
    grf_starts = [grf[0] for grf in loader.grf_data]
    mean_grf_start = np.mean(grf_starts)
    if 0.8 <= mean_grf_start <= 1.2:
        results['passed'].append(f"GRF starts near 1 BW (mean: {mean_grf_start:.3f} BW)")
    else:
        results['warnings'].append(f"GRF starting value unexpected (mean: {mean_grf_start:.3f} BW, expected ~1 BW)")

    # Check GRF ending values (should be near 0 at takeoff)
    grf_ends = [grf[-1] for grf in loader.grf_data]
    mean_grf_end = np.mean(grf_ends)
    if mean_grf_end < 0.3:
        results['passed'].append(f"GRF ends near 0 at takeoff (mean: {mean_grf_end:.3f} BW)")
    else:
        results['warnings'].append(f"GRF ending value may not be at takeoff (mean: {mean_grf_end:.3f} BW)")

    # Check for peak GRF > 1 BW (characteristic of CMJ)
    grf_peaks = [np.max(grf) for grf in loader.grf_data]
    mean_peak = np.mean(grf_peaks)
    if mean_peak > 1.5:
        results['passed'].append(f"GRF shows characteristic peak (mean peak: {mean_peak:.3f} BW)")
    else:
        results['warnings'].append(f"GRF peak may be lower than expected (mean: {mean_peak:.3f} BW)")

    # Print results
    print("\n=== Data Sanity Check Results ===")
    print(f"\nPassed ({len(results['passed'])}):")
    for msg in results['passed']:
        print(f"  [OK] {msg}")

    if results['warnings']:
        print(f"\nWarnings ({len(results['warnings'])}):")
        for msg in results['warnings']:
            print(f"  [!] {msg}")

    if results['errors']:
        print(f"\nErrors ({len(results['errors'])}):")
        for msg in results['errors']:
            print(f"  [X] {msg}")

    return results


def main():
    """Run all visualization and sanity checks."""
    from pathlib import Path

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    loader = CMJDataLoader()
    loader.load_data()

    # Print summary
    stats = loader.get_summary_stats()
    print("\n=== Data Summary ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Run sanity checks
    run_sanity_checks(loader)

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_random_samples(loader, n_samples=5, save_path=output_dir / 'random_samples.png')
    plot_signal_length_distribution(loader, save_path=output_dir / 'signal_lengths.png')
    plot_value_ranges(loader, save_path=output_dir / 'value_ranges.png')
    plot_samples_per_subject(loader, save_path=output_dir / 'samples_per_subject.png')

    print("\nVisualization complete!")
    plt.show()


if __name__ == '__main__':
    main()
