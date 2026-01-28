"""
Training Script for Signal Transformer

Command-line interface for training the accelerometer-to-GRF transformer model.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data_loader import (
    CMJDataLoader,
    DEFAULT_DATA_PATH,
    SAMPLING_RATE,
    DEFAULT_PRE_TAKEOFF_MS,
    DEFAULT_POST_TAKEOFF_MS,
)
from src.transformer import SignalTransformer
from src.losses import get_loss_function
from src.evaluate import (
    evaluate_model,
    print_evaluation_summary,
    plot_predictions,
    plot_outliers,
    plot_scatter_metrics,
    plot_bland_altman,
    save_results_csv,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train SignalTransformer for ACC to GRF prediction'
    )

    # Data arguments
    parser.add_argument(
        '--data-path',
        type=str,
        default=DEFAULT_DATA_PATH,
        help='Path to processedjumpdata.mat file'
    )
    parser.add_argument(
        '--use-resultant',
        action='store_true',
        default=True,
        help='Use resultant acceleration (default: True)'
    )
    parser.add_argument(
        '--use-triaxial',
        action='store_true',
        help='Use triaxial acceleration instead of resultant'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of subjects for validation (default: 0.2)'
    )
    parser.add_argument(
        '--pre-takeoff-ms',
        type=int,
        default=DEFAULT_PRE_TAKEOFF_MS,
        help=f'Duration before takeoff in ms (default: {DEFAULT_PRE_TAKEOFF_MS})'
    )
    parser.add_argument(
        '--post-takeoff-ms',
        type=int,
        default=DEFAULT_POST_TAKEOFF_MS,
        help=f'Duration after takeoff for ACC input in ms (default: {DEFAULT_POST_TAKEOFF_MS})'
    )

    # FDA transformation arguments
    parser.add_argument(
        '--input-transform',
        type=str,
        default='raw',
        choices=['raw', 'bspline', 'fpc'],
        help='Input signal transformation: raw, bspline, fpc (default: raw)'
    )
    parser.add_argument(
        '--output-transform',
        type=str,
        default='raw',
        choices=['raw', 'bspline', 'fpc'],
        help='Output signal transformation: raw, bspline, fpc (default: raw)'
    )
    parser.add_argument(
        '--n-basis',
        type=int,
        default=30,
        help='Number of B-spline basis functions (default: 30)'
    )
    parser.add_argument(
        '--bspline-lambda',
        type=float,
        default=1e-4,
        help='B-spline smoothing parameter (default: 1e-4)'
    )
    parser.add_argument(
        '--n-components',
        type=int,
        default=15,
        help='Number of FPC components (default: 15)'
    )
    parser.add_argument(
        '--variance-threshold',
        type=float,
        default=0.99,
        help='Cumulative variance threshold for automatic FPC selection (default: 0.99)'
    )
    parser.add_argument(
        '--fixed-components',
        action='store_true',
        help='Use fixed n_components instead of variance threshold for FPC'
    )
    parser.add_argument(
        '--use-varimax',
        action='store_true',
        default=True,
        help='Apply varimax rotation to FPCs (default: True)'
    )
    parser.add_argument(
        '--no-varimax',
        action='store_true',
        help='Disable varimax rotation'
    )
    parser.add_argument(
        '--fpc-smooth-lambda',
        type=float,
        default=None,
        help='Pre-FPCA smoothing parameter (default: None, no smoothing)'
    )
    parser.add_argument(
        '--fpc-n-basis-smooth',
        type=int,
        default=50,
        help='Number of basis functions for pre-FPCA smoothing (default: 50)'
    )
    parser.add_argument(
        '--score-scale',
        type=str,
        default='1.0',
        help='FPC score scale factor: "auto" for sqrt(N), or a number (default: 1.0). '
             'Use "auto" to match discrete normalization convention.'
    )
    parser.add_argument(
        '--use-custom-fpca',
        action='store_true',
        help='Use custom FPCA implementation (discrete dot products) instead of scikit-fda'
    )
    parser.add_argument(
        '--acc-max-threshold',
        type=float,
        default=100.0,
        help='Exclude samples with ACC > threshold (in g) as sensor artifacts (default: 100g)'
    )
    parser.add_argument(
        '--simple-normalization',
        action='store_true',
        help='Use simple global z-score normalization instead of mean function + MAD'
    )

    # Scalar prediction arguments
    parser.add_argument(
        '--scalar-prediction',
        type=str,
        default='none',
        choices=['none', 'jump_height'],
        help='Scalar prediction branch type (default: none)'
    )
    parser.add_argument(
        '--scalar-loss-weight',
        type=float,
        default=1.0,
        help='Weight for scalar prediction loss (default: 1.0)'
    )

    # Model arguments
    parser.add_argument(
        '--d-model',
        type=int,
        default=64,
        help='Model dimension (default: 64)'
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        default=4,
        help='Number of attention heads (default: 4)'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=3,
        help='Number of encoder layers (default: 3)'
    )
    parser.add_argument(
        '--d-ff',
        type=int,
        default=128,
        help='Feed-forward dimension (default: 128)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate (default: 0.1)'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum number of epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Early stopping patience (default: 15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='mse',
        choices=['mse', 'jump_height', 'peak_power', 'combined', 'weighted', 'smooth', 'eigenvalue_weighted', 'signal_space', 'reconstruction'],
        help='Loss function: mse, jump_height, peak_power, combined, weighted, smooth, eigenvalue_weighted, signal_space, reconstruction (default: mse)'
    )
    parser.add_argument(
        '--mse-weight',
        type=float,
        default=1.0,
        help='Weight for MSE component in combined loss (default: 1.0)'
    )
    parser.add_argument(
        '--jh-weight',
        type=float,
        default=1.0,
        help='Weight for jump height component in combined loss (default: 1.0)'
    )
    parser.add_argument(
        '--pp-weight',
        type=float,
        default=1.0,
        help='Weight for peak power component in combined loss (default: 1.0)'
    )
    parser.add_argument(
        '--smooth-lambda',
        type=float,
        default=0.1,
        help='Smoothness penalty weight for smooth loss (default: 0.1)'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory (default: outputs)'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Run name for outputs (default: timestamp)'
    )

    return parser.parse_args()


def setup_output_dirs(base_dir: str, run_name: str) -> dict:
    """Create output directories and return paths."""
    base_path = Path(base_dir) / run_name

    paths = {
        'base': base_path,
        'checkpoints': base_path / 'checkpoints',
        'figures': base_path / 'figures',
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return {k: str(v) for k, v in paths.items()}


def create_callbacks(checkpoint_dir: str, patience: int) -> list:
    """Create training callbacks."""
    callbacks = [
        # Model checkpoint - save best model
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
        # CSV logger
        keras.callbacks.CSVLogger(
            os.path.join(checkpoint_dir, 'training_log.csv'),
            separator=',',
            append=False,
        ),
    ]

    return callbacks


def main():
    """Main training function."""
    args = parse_args()

    # Handle triaxial flag
    use_resultant = not args.use_triaxial

    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Setup output directories
    run_name = args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    paths = setup_output_dirs(args.output_dir, run_name)

    print("\n" + "=" * 60)
    print("ACC -> GRF Transformer Training")
    print("=" * 60)

    # Handle varimax flag
    use_varimax = args.use_varimax and not args.no_varimax

    # Handle scalar prediction flag ('none' -> None)
    scalar_prediction = None if args.scalar_prediction == 'none' else args.scalar_prediction

    # Save configuration
    config = vars(args)
    config['run_name'] = run_name
    config['use_resultant'] = use_resultant
    config['use_varimax'] = use_varimax
    with open(os.path.join(paths['base'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to {paths['base']}/config.json")

    # Load data
    print("\n--- Loading Data ---")
    print(f"Pre-takeoff: {args.pre_takeoff_ms} ms, Post-takeoff: {args.post_takeoff_ms} ms")
    print(f"Input transform: {args.input_transform}, Output transform: {args.output_transform}")

    # Use None for variance_threshold if --fixed-components is set
    variance_threshold = None if args.fixed_components else args.variance_threshold

    # Parse score_scale: can be 'auto' or a float
    score_scale = args.score_scale if args.score_scale == 'auto' else float(args.score_scale)

    loader = CMJDataLoader(
        data_path=args.data_path,
        pre_takeoff_ms=args.pre_takeoff_ms,
        post_takeoff_ms=args.post_takeoff_ms,
        use_resultant=use_resultant,
        input_transform=args.input_transform,
        output_transform=args.output_transform,
        n_basis=args.n_basis,
        n_components=args.n_components,
        variance_threshold=variance_threshold,
        bspline_lambda=args.bspline_lambda,
        use_varimax=use_varimax,
        fpc_smooth_lambda=args.fpc_smooth_lambda,
        fpc_n_basis_smooth=args.fpc_n_basis_smooth,
        acc_max_threshold=args.acc_max_threshold,
        score_scale=score_scale,
        use_custom_fpca=args.use_custom_fpca,
        simple_normalization=args.simple_normalization,
        scalar_prediction=scalar_prediction,
    )
    train_ds, val_ds, info = loader.create_datasets(
        test_size=args.test_size,
        batch_size=args.batch_size,
        random_state=args.seed,
    )

    # Save data info (excluding non-serializable objects like transformers and arrays)
    data_info = {k: v for k, v in info.items()
                 if not isinstance(v, np.ndarray) and k not in ['input_transformer', 'output_transformer']}
    data_info['acc_std'] = float(info['acc_std'])
    data_info['grf_std'] = float(info['grf_std'])
    with open(os.path.join(paths['base'], 'data_info.json'), 'w') as f:
        json.dump(data_info, f, indent=2)

    # Save mean functions as numpy files (they're arrays, not scalars)
    np.save(os.path.join(paths['base'], 'acc_mean_function.npy'), info['acc_mean_function'])
    np.save(os.path.join(paths['base'], 'grf_mean_function.npy'), info['grf_mean_function'])

    # Build model
    print("\n--- Building Model ---")
    input_dim = 1 if use_resultant else 3

    # Use transformed sequence lengths if transformations are applied
    transformed_input_len = info['transformed_input_len']
    transformed_output_len = info['transformed_output_len']
    output_dim = info['output_shape'][-1]  # Number of output channels (1 for GRF)

    # Validate transformation compatibility
    # Encoder-only architecture cannot expand from fewer input positions to more output positions
    if transformed_output_len > transformed_input_len:
        raise ValueError(
            f"Invalid transformation combination: output ({transformed_output_len}) > input ({transformed_input_len}).\n"
            f"The encoder-only architecture cannot expand from {transformed_input_len} to {transformed_output_len} positions.\n"
            f"Valid combinations:\n"
            f"  - raw/raw (500→500)\n"
            f"  - bspline/bspline (n_basis→n_basis)\n"
            f"  - fpc/fpc (n_components→n_components)\n"
            f"  - raw/bspline or raw/fpc (compress output only)\n"
            f"Try: --input-transform {args.input_transform} --output-transform {args.input_transform}"
        )

    print(f"Raw input sequence: {info['acc_seq_len']} samples")
    print(f"Raw output sequence: {info['grf_seq_len']} samples")
    if args.input_transform != 'raw' or args.output_transform != 'raw':
        print(f"Transformed input: {transformed_input_len} features")
        print(f"Transformed output: {transformed_output_len} features")

    model = SignalTransformer(
        input_seq_len=transformed_input_len,
        output_seq_len=transformed_output_len,
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout_rate=args.dropout,
        scalar_prediction=scalar_prediction,
    )

    # Build model by calling it with dummy input
    dummy_input = tf.zeros((1, transformed_input_len, input_dim))
    _ = model(dummy_input)

    # Get loss function
    temporal_weights = info.get('temporal_weights') if args.loss in ['weighted', 'reconstruction'] else None

    # Get eigenvalues for eigenvalue_weighted loss (requires FPC output transform)
    eigenvalues = None
    if args.loss == 'eigenvalue_weighted':
        if args.output_transform != 'fpc':
            raise ValueError("eigenvalue_weighted loss requires --output-transform fpc")
        output_transformer = info.get('output_transformer')
        if output_transformer is None or not hasattr(output_transformer, 'get_eigenvalues'):
            raise ValueError("eigenvalue_weighted loss requires FPC output transformer")
        eigenvalues = output_transformer.get_eigenvalues()

    # Get inverse transform components for signal_space loss (requires FPC output transform)
    inverse_transform_components = None
    if args.loss == 'signal_space':
        if args.output_transform != 'fpc':
            raise ValueError("signal_space loss requires --output-transform fpc")
        output_transformer = info.get('output_transformer')
        if output_transformer is None or not hasattr(output_transformer, 'get_inverse_transform_components'):
            raise ValueError("signal_space loss requires FPC output transformer")
        inverse_transform_components = output_transformer.get_inverse_transform_components()

    # Get reconstruction components for reconstruction loss (requires bspline or fpc)
    reconstruction_components = None
    if args.loss == 'reconstruction':
        if args.output_transform == 'raw':
            raise ValueError("reconstruction loss requires --output-transform bspline or fpc")
        output_transformer = info.get('output_transformer')
        if output_transformer is None or not hasattr(output_transformer, 'get_reconstruction_components'):
            raise ValueError("reconstruction loss requires a transformer with get_reconstruction_components()")
        reconstruction_components = output_transformer.get_reconstruction_components()

    loss_fn = get_loss_function(
        args.loss,
        grf_mean_function=info['grf_mean_function'],
        grf_std=float(info['grf_std']),
        sampling_rate=SAMPLING_RATE,
        mse_weight=args.mse_weight,
        jh_weight=args.jh_weight,
        pp_weight=args.pp_weight,
        temporal_weights=temporal_weights,
        lambda_smooth=args.smooth_lambda,
        eigenvalues=eigenvalues,
        inverse_transform_components=inverse_transform_components,
        reconstruction_components=reconstruction_components,
    )
    print(f"Loss function: {args.loss}")
    if args.loss == 'weighted':
        print(f"  Using temporal weights (jerk-based)")
    if args.loss == 'combined':
        print(f"  Weights: MSE={args.mse_weight}, JH={args.jh_weight}, PP={args.pp_weight}")
    if args.loss == 'smooth':
        print(f"  Smoothness lambda: {args.smooth_lambda}")
    if args.loss == 'eigenvalue_weighted':
        print(f"  Using eigenvalue weights from FPC (components weighted by variance explained)")
    if args.loss == 'signal_space':
        print(f"  Computing loss in signal space after inverse FPCA transform")
    if args.loss == 'reconstruction':
        print(f"  Computing loss in signal space using reconstruction matrix")
        if temporal_weights is not None:
            print(f"  Using temporal weights (shape: {temporal_weights.shape}, range: [{temporal_weights.min():.2f}, {temporal_weights.max():.2f}])")
        else:
            print(f"  WARNING: No temporal weights available")

    # Compile model
    if scalar_prediction is not None:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss={
                'curve_output': loss_fn,
                'scalar_output': 'mse',
            },
            loss_weights={
                'curve_output': 1.0,
                'scalar_output': args.scalar_loss_weight,
            },
            metrics={'curve_output': ['mae']},
        )
        print(f"Scalar prediction: {scalar_prediction}")
        print(f"  Scalar loss weight: {args.scalar_loss_weight}")
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=loss_fn,
            metrics=['mae'],
        )

    model.summary()

    # Create callbacks
    callbacks = create_callbacks(paths['checkpoints'], args.patience)

    # Train
    print("\n--- Training ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    model.save(os.path.join(paths['checkpoints'], 'final_model.keras'))
    print(f"\nFinal model saved to {paths['checkpoints']}/final_model.keras")

    # Evaluate on validation set
    print("\n--- Evaluation ---")

    # Get validation data as arrays
    X_val_list, y_val_list = [], []
    for X_batch, y_batch in val_ds:
        X_val_list.append(X_batch.numpy())
        if isinstance(y_batch, dict):
            y_val_list.append(y_batch['curve_output'].numpy())
        else:
            y_val_list.append(y_batch.numpy())
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)

    # Pass ground truth metrics for reference comparison
    results = evaluate_model(
        model, X_val, y_val, loader,
        ground_truth_jh=info.get('val_gt_jump_height'),
        ground_truth_pp=info.get('val_gt_peak_power'),
        body_mass=info.get('val_body_mass'),
        scalar_prediction=scalar_prediction,
    )
    print_evaluation_summary(results)

    # Save evaluation results
    save_results_csv(results, os.path.join(paths['base'], 'evaluation_results.csv'))

    # Generate plots
    print("\n--- Generating Plots ---")

    # Get pre_takeoff_samples for plot alignment
    pre_takeoff_samples = loader.pre_takeoff_samples

    plot_predictions(
        results, n_samples=5,
        sampling_rate=SAMPLING_RATE,
        save_path=os.path.join(paths['figures'], 'prediction_curves.png')
    )

    plot_scatter_metrics(
        results,
        save_path=os.path.join(paths['figures'], 'scatter_metrics.png')
    )

    plot_bland_altman(
        results,
        save_path=os.path.join(paths['figures'], 'bland_altman.png')
    )

    # Plot worst outliers
    plot_outliers(
        results, X_val,
        metric='jump_height',
        n_outliers=5,
        sampling_rate=SAMPLING_RATE,
        pre_takeoff_samples=pre_takeoff_samples,
        data_loader=loader,
        save_path=os.path.join(paths['figures'], 'outliers_jump_height.png')
    )

    plot_outliers(
        results, X_val,
        metric='peak_power',
        n_outliers=5,
        sampling_rate=SAMPLING_RATE,
        pre_takeoff_samples=pre_takeoff_samples,
        data_loader=loader,
        save_path=os.path.join(paths['figures'], 'outliers_peak_power.png')
    )

    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(paths['figures'], 'training_history.png')
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Results saved to: {paths['base']}")
    print("=" * 60)


def plot_training_history(history, save_path: str = None):
    """Plot training and validation loss curves."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax = axes[0]
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE - handle multi-output metric key names
    ax = axes[1]
    if 'mae' in history.history:
        mae_key, val_mae_key = 'mae', 'val_mae'
    elif 'curve_output_mae' in history.history:
        mae_key, val_mae_key = 'curve_output_mae', 'val_curve_output_mae'
    else:
        mae_key = val_mae_key = None

    if mae_key is not None:
        ax.plot(history.history[mae_key], label='Train')
        ax.plot(history.history[val_mae_key], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('Training and Validation MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    return fig


if __name__ == '__main__':
    main()
