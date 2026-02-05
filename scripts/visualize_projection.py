"""
Visualize FPC Projection Matrix with Eigenfunctions

Creates a comprehensive figure showing:
1. ACC eigenfunctions (top components per channel)
2. Projection matrix heat maps (per channel for triaxial)
3. GRF eigenfunctions (top components)
4. Biomechanical phase annotations

This visualization helps interpret how ACC movement patterns
map to GRF force characteristics through the learned projection.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from src.data_loader import CMJDataLoader, DEFAULT_DATA_PATH, SAMPLING_RATE
from src.transformations import learn_fpc_projection_matrix


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize FPC projection matrix with eigenfunctions'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=DEFAULT_DATA_PATH,
        help='Path to processedjumpdata.mat file'
    )
    parser.add_argument(
        '--n-components',
        type=int,
        default=15,
        help='Number of FPC components (default: 15)'
    )
    parser.add_argument(
        '--n-display',
        type=int,
        default=3,
        help='Number of FPC components to display (default: 3)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/projection_visualization',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained hybrid model (optional, uses learned projection if provided)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top ACC contributors to show per GRF FPC (default: 3)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for saved figures (default: 150)'
    )
    return parser.parse_args()


def get_acc_component_name(flat_idx: int, n_channels: int = 3,
                           channel_labels: list = None) -> tuple:
    """
    Resolve a flat ACC component index to channel and component within channel.

    The projection matrix uses C-order flattening of shape (n_components, n_channels),
    which interleaves channels: [X0, Y0, Z0, X1, Y1, Z1, ...].

    Args:
        flat_idx: Flat index into the projection matrix rows
        n_channels: Number of ACC channels (3 for triaxial, 1 for resultant)
        channel_labels: List of channel labels (default: ['X', 'Y', 'Z'] or ['R'])

    Returns:
        Tuple of (channel_idx, component_idx, label_string)
        e.g., (1, 5, 'Y-FPC6') for flat_idx=16 with n_channels=3
    """
    if channel_labels is None:
        channel_labels = ['X', 'Y', 'Z'] if n_channels == 3 else ['R']

    channel_idx = flat_idx % n_channels
    component_idx = flat_idx // n_channels
    label = f'{channel_labels[channel_idx]}-FPC{component_idx + 1}'

    return channel_idx, component_idx, label


def get_biomechanical_phases(seq_len: int, sampling_rate: float = SAMPLING_RATE):
    """
    Define biomechanical phase boundaries for CMJ.

    Approximate phases based on typical CMJ timing (2000ms pre-takeoff):
    - Quiet standing: 0-500ms (samples 0-125)
    - Unweighting: 500-1000ms (samples 125-250)
    - Braking: 1000-1500ms (samples 250-375)
    - Propulsion: 1500-2000ms (samples 375-500)

    Returns:
        Dictionary with phase names and sample boundaries
    """
    ms_per_sample = 1000 / sampling_rate
    total_ms = seq_len * ms_per_sample

    # Define phases as fractions of total time
    phases = {
        'Quiet\nStanding': (0, int(0.25 * seq_len)),
        'Unweighting': (int(0.25 * seq_len), int(0.5 * seq_len)),
        'Braking': (int(0.5 * seq_len), int(0.75 * seq_len)),
        'Propulsion': (int(0.75 * seq_len), seq_len),
    }

    return phases


def plot_eigenfunctions(ax, eigenfuncs: np.ndarray, n_display: int,
                       title: str, ylabel: str, phases: dict = None,
                       show_xlabel: bool = True):
    """
    Plot eigenfunctions as time series.

    Args:
        ax: Matplotlib axis
        eigenfuncs: Array of shape (seq_len, n_components)
        n_display: Number of components to display
        title: Plot title
        ylabel: Y-axis label
        phases: Dict of biomechanical phases for shading
        show_xlabel: Whether to show x-axis label
    """
    seq_len = eigenfuncs.shape[0]
    time_ms = np.arange(seq_len) * (1000 / SAMPLING_RATE)

    # Add phase shading if provided
    if phases is not None:
        colors = ['#f0f0f0', '#e0e0e0', '#d0d0d0', '#c0c0c0']
        for i, (phase_name, (start, end)) in enumerate(phases.items()):
            ax.axvspan(time_ms[start], time_ms[min(end, seq_len-1)],
                      alpha=0.3, color=colors[i % len(colors)], label=phase_name)

    # Plot eigenfunctions
    colors = plt.cm.tab10(np.linspace(0, 1, n_display))
    for i in range(min(n_display, eigenfuncs.shape[1])):
        ax.plot(time_ms, eigenfuncs[:, i], color=colors[i],
               linewidth=1.5, label=f'FPC {i+1}')

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9)
    if show_xlabel:
        ax.set_xlabel('Time (ms)', fontsize=9)
    ax.set_xlim(0, time_ms[-1])
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax.tick_params(labelsize=8)


def plot_projection_heatmap(ax, P: np.ndarray, channel_label: str,
                           n_input: int, n_output: int):
    """
    Plot projection matrix as heatmap.

    Args:
        ax: Matplotlib axis
        P: Projection matrix slice for this channel
        channel_label: Label for this channel (X, Y, Z, or R)
        n_input: Number of input components
        n_output: Number of output components
    """
    # Symmetric colormap centered at 0
    vmax = np.abs(P).max()
    im = ax.imshow(P.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    ax.set_xlabel(f'ACC-{channel_label} FPC', fontsize=9)
    ax.set_ylabel('GRF FPC', fontsize=9)
    ax.set_title(f'{channel_label} → GRF', fontsize=10, fontweight='bold')

    # Tick labels
    ax.set_xticks(range(0, n_input, max(1, n_input // 5)))
    ax.set_xticklabels([f'{i+1}' for i in range(0, n_input, max(1, n_input // 5))], fontsize=8)
    ax.set_yticks(range(0, n_output, max(1, n_output // 5)))
    ax.set_yticklabels([f'{i+1}' for i in range(0, n_output, max(1, n_output // 5))], fontsize=8)

    return im


def create_combined_figure(input_transformer, output_transformer, P: np.ndarray,
                          n_display: int = 3, save_path: str = None, dpi: int = 150):
    """
    Create combined figure with eigenfunctions and projection matrix.

    Layout for triaxial:
    ┌─────────────────────────────────────────────────────────────┐
    │  ACC-X EFs  │  ACC-Y EFs  │  ACC-Z EFs  │                   │
    ├─────────────────────────────────────────────────────────────┤
    │  P(X→GRF)   │  P(Y→GRF)   │  P(Z→GRF)   │  Colorbar        │
    ├─────────────────────────────────────────────────────────────┤
    │              GRF Eigenfunctions                             │
    └─────────────────────────────────────────────────────────────┘
    """
    # Get eigenfunctions
    acc_eigenfuncs = input_transformer.get_eigenfunctions()  # List per channel
    grf_eigenfuncs = output_transformer.get_eigenfunctions()  # List per channel (usually 1)

    n_channels = len(acc_eigenfuncs)
    n_input_per_channel = [ef.shape[1] for ef in acc_eigenfuncs]
    n_output = grf_eigenfuncs[0].shape[1]
    seq_len = acc_eigenfuncs[0].shape[0]

    # Get biomechanical phases
    phases = get_biomechanical_phases(seq_len)

    # Determine layout
    if n_channels == 3:
        channel_labels = ['X', 'Y', 'Z']
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1],
                     width_ratios=[1, 1, 1, 0.1], hspace=0.35, wspace=0.3)
    else:
        channel_labels = ['R']
        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1],
                     width_ratios=[1, 0.05], hspace=0.35, wspace=0.2)

    # Row 1: ACC Eigenfunctions
    acc_axes = []
    for ch in range(n_channels):
        if n_channels == 3:
            ax = fig.add_subplot(gs[0, ch])
        else:
            ax = fig.add_subplot(gs[0, 0])

        plot_eigenfunctions(ax, acc_eigenfuncs[ch], n_display,
                           f'ACC-{channel_labels[ch]} Eigenfunctions',
                           'Amplitude', phases, show_xlabel=False)
        acc_axes.append(ax)

    # Row 2: Projection Matrix Heat Maps
    heatmap_axes = []
    im = None
    idx_offset = 0
    for ch in range(n_channels):
        if n_channels == 3:
            ax = fig.add_subplot(gs[1, ch])
        else:
            ax = fig.add_subplot(gs[1, 0])

        # Extract projection matrix slice for this channel
        n_comp_ch = n_input_per_channel[ch]
        P_ch = P[idx_offset:idx_offset + n_comp_ch, :n_output]
        idx_offset += n_comp_ch

        im = plot_projection_heatmap(ax, P_ch, channel_labels[ch], n_comp_ch, n_output)
        heatmap_axes.append(ax)

    # Colorbar
    if n_channels == 3:
        cbar_ax = fig.add_subplot(gs[1, 3])
    else:
        cbar_ax = fig.add_subplot(gs[1, 1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Projection\nCoefficient', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Row 3: GRF Eigenfunctions (spanning all columns)
    if n_channels == 3:
        ax_grf = fig.add_subplot(gs[2, :3])
    else:
        ax_grf = fig.add_subplot(gs[2, 0])

    plot_eigenfunctions(ax_grf, grf_eigenfuncs[0], n_display,
                       'GRF Eigenfunctions', 'Amplitude', phases)

    # Add phase legend
    phase_handles = []
    colors = ['#f0f0f0', '#e0e0e0', '#d0d0d0', '#c0c0c0']
    for i, phase_name in enumerate(phases.keys()):
        phase_handles.append(mpatches.Patch(color=colors[i], alpha=0.5,
                                           label=phase_name.replace('\n', ' ')))

    fig.legend(handles=phase_handles, loc='upper right', fontsize=8,
              title='Biomechanical Phases', title_fontsize=9,
              bbox_to_anchor=(0.98, 0.98))

    # Main title
    fig.suptitle('FPC Projection Matrix: ACC → GRF Mapping\n'
                f'(showing top {n_display} components)',
                fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Combined figure saved to {save_path}")

    return fig


def create_detailed_contribution_figure(input_transformer, output_transformer,
                                        P: np.ndarray, n_display: int = 3,
                                        top_k: int = 5, save_path: str = None,
                                        dpi: int = 150):
    """
    Create figure showing how each GRF FPC is constructed from ACC FPCs.

    For each displayed GRF FPC, shows:
    - The GRF eigenfunction
    - Top contributing ACC eigenfunctions (weighted)
    - Bar chart of contribution magnitudes
    """
    acc_eigenfuncs = input_transformer.get_eigenfunctions()
    grf_eigenfuncs = output_transformer.get_eigenfunctions()

    n_channels = len(acc_eigenfuncs)
    channel_labels = ['X', 'Y', 'Z'] if n_channels == 3 else ['R']

    n_output = min(n_display, grf_eigenfuncs[0].shape[1])
    seq_len = grf_eigenfuncs[0].shape[0]
    time_ms = np.arange(seq_len) * (1000 / SAMPLING_RATE)
    phases = get_biomechanical_phases(seq_len)

    fig, axes = plt.subplots(n_output, 3, figsize=(14, 3.5 * n_output))
    if n_output == 1:
        axes = axes.reshape(1, -1)

    for j in range(n_output):
        # Column 1: GRF eigenfunction
        ax1 = axes[j, 0]

        # Phase shading
        colors = ['#f0f0f0', '#e0e0e0', '#d0d0d0', '#c0c0c0']
        for i, (phase_name, (start, end)) in enumerate(phases.items()):
            ax1.axvspan(time_ms[start], time_ms[min(end, seq_len-1)],
                       alpha=0.3, color=colors[i % len(colors)])

        ax1.plot(time_ms, grf_eigenfuncs[0][:, j], 'b-', linewidth=2)
        ax1.set_title(f'GRF FPC {j+1}', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Amplitude', fontsize=9)
        ax1.set_xlabel('Time (ms)', fontsize=9)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax1.set_xlim(0, time_ms[-1])

        # Column 2: Top contributing ACC eigenfunctions
        ax2 = axes[j, 1]

        contributions = np.abs(P[:, j])
        top_indices = np.argsort(contributions)[-top_k:][::-1]

        # Phase shading
        for i, (phase_name, (start, end)) in enumerate(phases.items()):
            ax2.axvspan(time_ms[start], time_ms[min(end, seq_len-1)],
                       alpha=0.3, color=colors[i % len(colors)])

        line_colors = plt.cm.Set1(np.linspace(0, 1, top_k))
        for rank, idx in enumerate(top_indices):
            ch, fpc, label = get_acc_component_name(idx, n_channels, channel_labels)
            coef = P[idx, j]
            sign = '+' if coef > 0 else '-'

            # Get the eigenfunction
            ef = acc_eigenfuncs[ch][:, fpc]

            ax2.plot(time_ms, ef * np.sign(coef), color=line_colors[rank],
                    linewidth=1.5, alpha=0.8,
                    label=f'{label} ({sign}{abs(coef):.2f})')

        ax2.set_title(f'Top {top_k} ACC Contributors', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Amplitude (signed)', fontsize=9)
        ax2.set_xlabel('Time (ms)', fontsize=9)
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax2.legend(loc='upper left', fontsize=7, framealpha=0.9)
        ax2.set_xlim(0, time_ms[-1])

        # Column 3: Contribution bar chart
        ax3 = axes[j, 2]

        labels = []
        values = []
        for rank, idx in enumerate(top_indices):
            ch, fpc, label = get_acc_component_name(idx, n_channels, channel_labels)
            coef = P[idx, j]
            labels.append(label)
            values.append(coef)

        y_pos = range(len(labels))
        # Use same colors as the curves
        ax3.barh(y_pos, values, color=line_colors[:len(labels)], alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels, fontsize=9)
        ax3.set_xlabel('Coefficient', fontsize=9)
        ax3.set_title('Contribution Weights', fontsize=10, fontweight='bold')
        ax3.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
        ax3.invert_yaxis()

    plt.suptitle('GRF FPC Construction from ACC FPCs', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Detailed contribution figure saved to {save_path}")

    return fig


def plot_fpc_with_deviation(ax, mean_function: np.ndarray, eigenfunction: np.ndarray,
                            score_std: float, sd_multiplier: float = 2.0,
                            colors: tuple = ('red', 'blue'),
                            fill_colors: tuple = ('#ffcccc', '#cce0ff'),
                            phases: dict = None, title: str = '',
                            show_xlabel: bool = True, show_ylabel: bool = True,
                            show_xtick_labels: bool = True, show_legend: bool = False):
    """
    Plot FPC in traditional biomechanics style: mean ± SD bands.

    Args:
        ax: Matplotlib axis
        mean_function: Mean curve of shape (seq_len,)
        eigenfunction: Eigenfunction of shape (seq_len,)
        score_std: Standard deviation of scores for this FPC
        sd_multiplier: Number of SDs for deviation bands (default 2.0)
        colors: Tuple of (positive_color, negative_color) for deviation lines
        fill_colors: Tuple of (positive_fill, negative_fill) for shaded bands
        phases: Dict of biomechanical phases for background shading
        title: Plot title
        show_xlabel: Whether to show x-axis label
        show_ylabel: Whether to show y-axis label
        show_xtick_labels: Whether to show x-axis tick labels
        show_legend: Whether to show the legend
    """
    seq_len = len(mean_function)
    time_ms = np.arange(seq_len) * (1000 / SAMPLING_RATE)

    # Add phase shading in background
    if phases is not None:
        phase_colors = ['#f5f5f5', '#ebebeb', '#e0e0e0', '#d6d6d6']
        for i, (phase_name, (start, end)) in enumerate(phases.items()):
            ax.axvspan(time_ms[start], time_ms[min(end, seq_len-1)],
                      alpha=0.5, color=phase_colors[i % len(phase_colors)], zorder=0)

    # Compute deviation curves
    deviation = sd_multiplier * score_std * eigenfunction
    plus_curve = mean_function + deviation
    minus_curve = mean_function - deviation

    # Fill between mean and deviation curves
    ax.fill_between(time_ms, mean_function, plus_curve,
                   color=fill_colors[0], alpha=0.6, zorder=1)
    ax.fill_between(time_ms, mean_function, minus_curve,
                   color=fill_colors[1], alpha=0.6, zorder=1)

    # Plot lines
    ax.plot(time_ms, mean_function, 'k-', linewidth=1.5, label='Mean', zorder=3)
    ax.plot(time_ms, plus_curve, color=colors[0], linewidth=1.2,
           linestyle='-', label=f'+{sd_multiplier}SD', zorder=2)
    ax.plot(time_ms, minus_curve, color=colors[1], linewidth=1.2,
           linestyle='-', label=f'−{sd_multiplier}SD', zorder=2)

    ax.set_title(title, fontsize=9, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('Amplitude', fontsize=8)
    if show_xlabel:
        ax.set_xlabel('Time (ms)', fontsize=8)
    ax.set_xlim(0, time_ms[-1])
    ax.tick_params(labelsize=7)
    if not show_xtick_labels:
        ax.tick_params(labelbottom=False)
    if show_legend:
        ax.legend(loc='upper left', fontsize=6, framealpha=0.9)


def create_biomechanics_fpc_figure(input_transformer, output_transformer,
                                    P: np.ndarray, X_train: np.ndarray,
                                    y_train: np.ndarray, n_display: int = 3,
                                    top_k: int = 3, sd_multiplier: float = 2.0,
                                    save_path: str = None, dpi: int = 150):
    """
    Create figure showing FPCs in traditional biomechanics style.

    Layout (portrait):
    - Top row: GRF FPCs (n_display columns)
    - Rows below: Top-k ACC contributors for each GRF FPC

    Args:
        input_transformer: Fitted ACC FPC transformer
        output_transformer: Fitted GRF FPC transformer
        P: Projection matrix from ACC to GRF FPCs
        X_train: Training ACC scores for computing score std
        y_train: Training GRF scores for computing score std
        n_display: Number of GRF FPCs to display (columns)
        top_k: Number of top ACC contributors per GRF FPC
        sd_multiplier: Number of SDs for deviation bands
        save_path: Path to save figure
        dpi: DPI for saved figure
    """
    # Get mean functions and eigenfunctions
    grf_components = output_transformer.get_inverse_transform_components()
    acc_components = input_transformer.get_inverse_transform_components()

    grf_mean = grf_components['mean_functions'][0]  # (seq_len,)
    grf_eigenfuncs = output_transformer.get_eigenfunctions(rotated=True)[0]  # (seq_len, n_comp)

    acc_means = acc_components['mean_functions']  # List of (seq_len,) per channel
    acc_eigenfuncs = input_transformer.get_eigenfunctions(rotated=True)  # List per channel

    n_channels = len(acc_eigenfuncs)
    channel_labels = ['X', 'Y', 'Z'] if n_channels == 3 else ['R']

    # Compute score standard deviations from training data
    # y_train has shape (n_samples, n_grf_components, 1) - squeeze last dim
    y_train_flat = y_train.squeeze(-1) if y_train.ndim == 3 else y_train
    grf_score_std = np.std(y_train_flat, axis=0)  # (n_grf_components,)

    # X_train has shape (n_samples, components_per_channel, n_channels)
    # Compute std per channel: shape (components_per_channel, n_channels)
    acc_score_std = np.std(X_train, axis=0)  # (n_comp, n_channels)

    # Get biomechanical phases
    seq_len = grf_mean.shape[0]
    phases = get_biomechanical_phases(seq_len)

    # Limit display
    n_grf_display = min(n_display, grf_eigenfuncs.shape[1])
    n_rows = 1 + top_k  # 1 row for GRF + top_k rows for ACC contributors

    # Create figure with tighter layout
    fig = plt.figure(figsize=(4 * n_grf_display, 2.5 * n_rows))
    gs = GridSpec(n_rows, n_grf_display, figure=fig, hspace=0.25, wspace=0.2)

    # GRF color scheme
    grf_colors = ('darkred', 'darkblue')
    grf_fills = ('#ffcccc', '#cce0ff')

    # ACC color scheme (distinct from GRF)
    acc_colors = ('darkorange', 'darkcyan')
    acc_fills = ('#ffe6cc', '#ccf2f2')

    # Row 0: GRF FPCs
    for j in range(n_grf_display):
        ax = fig.add_subplot(gs[0, j])
        is_left_col = (j == 0)
        plot_fpc_with_deviation(
            ax, grf_mean, grf_eigenfuncs[:, j],
            score_std=grf_score_std[j],
            sd_multiplier=sd_multiplier,
            colors=grf_colors, fill_colors=grf_fills,
            phases=phases, title=f'GRF FPC-{j+1}',
            show_xlabel=False, show_ylabel=is_left_col,
            show_xtick_labels=False, show_legend=is_left_col
        )

    # Rows 1 to top_k: ACC contributors for each GRF FPC
    for j in range(n_grf_display):
        # Get top-k ACC contributors for GRF FPC j
        contributions = np.abs(P[:, j])
        top_indices = np.argsort(contributions)[-top_k:][::-1]

        for rank, idx in enumerate(top_indices):
            ax = fig.add_subplot(gs[1 + rank, j])

            # Resolve flat index to channel and component
            ch, fpc, label = get_acc_component_name(idx, n_channels, channel_labels)
            weight = P[idx, j]

            # Get ACC mean and eigenfunction
            acc_mean = acc_means[ch]
            acc_ef = acc_eigenfuncs[ch][:, fpc]

            # Title with weight - format: "ACC-Z FPC-4 (w=0.42)"
            acc_title = f'ACC-{channel_labels[ch]} FPC-{fpc+1} (w={weight:.2f})'

            is_left_col = (j == 0)
            is_bottom_row = (rank == top_k - 1)
            is_top_left_acc = (j == 0 and rank == 0)

            plot_fpc_with_deviation(
                ax, acc_mean, acc_ef,
                score_std=acc_score_std[fpc, ch],
                sd_multiplier=sd_multiplier,
                colors=acc_colors, fill_colors=acc_fills,
                phases=phases, title=acc_title,
                show_xlabel=is_bottom_row, show_ylabel=is_left_col,
                show_xtick_labels=is_bottom_row, show_legend=is_top_left_acc
            )

    # Add phase legend - positioned closer to corner
    phase_handles = []
    phase_colors = ['#f5f5f5', '#ebebeb', '#e0e0e0', '#d6d6d6']
    for i, phase_name in enumerate(phases.keys()):
        phase_handles.append(mpatches.Patch(color=phase_colors[i], alpha=0.7,
                                           label=phase_name.replace('\n', ' ')))

    fig.legend(handles=phase_handles, loc='upper right', fontsize=6,
              title='Phases', title_fontsize=7,
              bbox_to_anchor=(0.905, 0.995), ncol=2)

    # Main title - positioned closer to plots
    fig.suptitle(f'FPC Variation: Mean ± {sd_multiplier}SD\n'
                f'GRF FPCs (top row) with Top-{top_k} ACC Contributors',
                fontsize=11, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Biomechanics FPC figure saved to {save_path}")

    return fig


def main():
    """Main visualization function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FPC Projection Matrix Visualization")
    print("=" * 60)

    # Set random seed
    np.random.seed(args.seed)

    # Load data with FPC transforms (triaxial)
    print("\n--- Loading Data with FPC Transforms ---")
    loader = CMJDataLoader(
        data_path=args.data_path,
        use_resultant=False,  # Always use triaxial for this visualization
        input_transform='fpc',
        output_transform='fpc',
        n_components=args.n_components,
        variance_threshold=0.99,
        use_varimax=True,
        simple_normalization=True,
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

    # Get training data for learning projection
    print("\n--- Learning Projection Matrix ---")
    X_train_list, y_train_list = [], []
    for X_batch, y_batch in train_ds:
        X_train_list.append(X_batch.numpy())
        y_train_list.append(y_batch.numpy())
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    # Learn projection matrix using Ridge regression
    P = learn_fpc_projection_matrix(X_train, y_train, alpha=1.0)

    print(f"Projection matrix shape: {P.shape}")

    # Save projection matrix
    np.save(output_dir / 'projection_matrix.npy', P)
    print(f"Projection matrix saved to {output_dir}/projection_matrix.npy")

    # Create combined figure
    print("\n--- Creating Combined Visualization ---")
    create_combined_figure(
        input_transformer, output_transformer, P,
        n_display=args.n_display,
        save_path=str(output_dir / 'projection_combined.png'),
        dpi=args.dpi
    )

    # Create detailed contribution figure
    print("\n--- Creating Detailed Contribution Figure ---")
    create_detailed_contribution_figure(
        input_transformer, output_transformer, P,
        n_display=args.n_display,
        top_k=args.top_k,
        save_path=str(output_dir / 'projection_contributions.png'),
        dpi=args.dpi
    )

    # Create biomechanics FPC figure
    print("\n--- Creating Biomechanics FPC Figure ---")
    create_biomechanics_fpc_figure(
        input_transformer, output_transformer, P,
        X_train=X_train, y_train=y_train,
        n_display=args.n_display, top_k=args.top_k,
        save_path=str(output_dir / 'biomechanics_fpc.png'),
        dpi=args.dpi
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input: Triaxial ACC ({len(input_transformer._actual_n_components)} channels)")
    print(f"  Components per channel: {input_transformer._actual_n_components}")
    print(f"  Total input features: {P.shape[0]}")
    print(f"Output: GRF ({P.shape[1]} FPCs)")
    print(f"\nFigures saved to {output_dir}/")
    print("  - projection_combined.png: Overview with eigenfunctions and heatmaps")
    print("  - projection_contributions.png: Detailed per-GRF-FPC breakdown")
    print("  - biomechanics_fpc.png: Traditional mean ± SD FPC visualization")

    plt.show()


if __name__ == '__main__':
    main()
