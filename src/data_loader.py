"""
Data Loader for Countermovement Jump Data

Loads preprocessed CMJ data from .npz files (created by scripts/prepare_dataset.py)
and prepares it for training the signal-to-signal transformer model.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from typing import Optional

from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.smoothing.validation import (
    SmoothingParameterSearch,
    LinearSmootherGeneralizedCVScorer,
)
from skfda.misc.regularization import L2Regularization
from skfda.misc.operators import LinearDifferentialOperator

from src.transformations import get_transformer, BaseSignalTransformer, IdentityTransformer


# Default paths
DEFAULT_DATA_PATH = str(Path(__file__).parent.parent / "data" / "cmj_dataset.npz")
DEFAULT_SEQ_LEN = 500
SAMPLING_RATE = 250  # Hz (ACC native rate; GRF downsampled from 1000Hz)

# Default durations in milliseconds
DEFAULT_PRE_TAKEOFF_MS = 2000  # 2 seconds before takeoff (500 samples at 250 Hz)
DEFAULT_POST_TAKEOFF_MS = 0    # No extension after takeoff by default


class CMJDataLoader:
    """
    Data loader for countermovement jump accelerometer and GRF data.

    Loads .npz files (created by scripts/prepare_dataset.py) and preprocesses
    signals for the transformer model.

    Args:
        data_path: Path to cmj_dataset.npz file
        pre_takeoff_ms: Duration before takeoff in milliseconds (default 2000ms)
        post_takeoff_ms: Duration after takeoff for ACC input only (default 0ms)
        use_resultant: If True, compute resultant acceleration; else use triaxial
        input_transform: Transform type for input ('raw', 'bspline', 'fpc')
        output_transform: Transform type for output ('raw', 'bspline', 'fpc')
        n_basis: Number of B-spline basis functions
        n_components: Number of FPC components
        variance_threshold: Cumulative variance threshold for FPC selection
        bspline_lambda: B-spline smoothing parameter
        use_varimax: Whether to apply varimax rotation to FPCs
        fpc_smooth_lambda: Pre-FPCA smoothing parameter (None = no smoothing)
        fpc_n_basis_smooth: Number of basis functions for pre-FPCA smoothing
        scalar_prediction: Type of scalar prediction ('jump_height' or None).
            When enabled, datasets include scalar targets alongside curve targets.
        scalar_only: If True, datasets contain only scalar targets (no curve).
        use_bspline_reference: If True, compute and store B-spline reconstruction
            of GRF as a consistent reference for evaluation. This ensures fair
            comparison across different output transforms (B-spline vs FPC) by
            evaluating both against the same smoothed ground truth.
        smooth_signals: If True, apply universal B-spline smoothing with GCV
            to all signals before further analysis. This ensures fair comparison
            across representations by starting from the same smooth functional
            representation.
        smooth_n_basis: Number of B-spline basis functions for smoothing (default 50).

    Note:
        ACC input length = pre_takeoff_ms + post_takeoff_ms
        GRF output length = pre_takeoff_ms only (GRF is 0 during flight)
    """

    def __init__(
        self,
        data_path: str = DEFAULT_DATA_PATH,
        pre_takeoff_ms: int = DEFAULT_PRE_TAKEOFF_MS,
        post_takeoff_ms: int = DEFAULT_POST_TAKEOFF_MS,
        use_resultant: bool = True,
        input_transform: str = 'raw',
        output_transform: str = 'raw',
        n_basis: int = 30,
        n_components: int = 15,
        variance_threshold: float = 0.99,
        bspline_lambda: float = 1e-4,
        use_varimax: bool = True,
        fpc_smooth_lambda: float = None,
        fpc_n_basis_smooth: int = 50,
        score_scale: float = 1.0,
        use_custom_fpca: bool = False,
        simple_normalization: bool = False,
        scalar_prediction: str = None,
        scalar_only: bool = False,
        use_bspline_reference: bool = False,
        smooth_signals: bool = True,
        smooth_n_basis: int = 50,
    ):
        self.data_path = data_path
        self.pre_takeoff_ms = pre_takeoff_ms
        self.post_takeoff_ms = post_takeoff_ms
        self.use_resultant = use_resultant

        # Transformation parameters
        self.input_transform_type = input_transform
        self.output_transform_type = output_transform
        self.n_basis = n_basis
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.bspline_lambda = bspline_lambda
        self.use_varimax = use_varimax
        self.fpc_smooth_lambda = fpc_smooth_lambda
        self.fpc_n_basis_smooth = fpc_n_basis_smooth
        self.score_scale = score_scale
        self.use_custom_fpca = use_custom_fpca
        self.simple_normalization = simple_normalization
        self.scalar_prediction = scalar_prediction
        self.scalar_only = scalar_only
        self.use_bspline_reference = use_bspline_reference
        self.smooth_signals = smooth_signals
        self.smooth_n_basis = smooth_n_basis

        # Universal smoothing state (fitted on training data)
        self._smooth_lambdas_acc: list[float] = []
        self._smooth_lambda_grf: float = None
        self._smooth_basis = None  # B-spline basis object (reused for validation)
        self._smooth_coefficients = None  # (acc_coeffs, grf_coeffs) from latest preprocess

        # Unified pipeline state
        self.normalize_in_coeff_space = False  # True for bspline/fpc representations
        self.grf_coeff_mean = None  # Coefficient/score-space normalization stats
        self.grf_coeff_std = None
        self.acc_coeff_mean = None
        self.acc_coeff_std = None

        # B-spline reference for rigorous evaluation (computed in create_datasets)
        self.bspline_reference_train = None
        self.bspline_reference_val = None

        # Calculate sequence lengths from durations
        self.pre_takeoff_samples = int(pre_takeoff_ms * SAMPLING_RATE / 1000)
        self.post_takeoff_samples = int(post_takeoff_ms * SAMPLING_RATE / 1000)
        self.acc_seq_len = self.pre_takeoff_samples + self.post_takeoff_samples
        self.grf_seq_len = self.pre_takeoff_samples  # GRF only up to takeoff

        # Data storage
        self.acc_data = None
        self.grf_data = None
        self.subject_ids = None

        # Ground truth metrics (from full signal, pre-computed)
        self.ground_truth_jump_height = None
        self.ground_truth_peak_power = None  # Already in W/kg

        # Normalization parameters (mean functions are shape (seq_len, n_channels))
        self.acc_mean_function = None
        self.acc_std = None
        self.grf_mean_function = None
        self.grf_std = None

        # Temporal weights for weighted MSE loss
        self.temporal_weights = None

        # Scalar prediction normalization stats
        self.scalar_mean = None
        self.scalar_std = None

        # Signal transformers (fitted during create_datasets)
        self.input_transformer: Optional[BaseSignalTransformer] = None
        self.output_transformer: Optional[BaseSignalTransformer] = None

    def load_data(self) -> tuple[list, list, np.ndarray]:
        """
        Load data from .npz file (created by scripts/prepare_dataset.py).

        Returns:
            Tuple of (acc_data, grf_data, subject_ids)
        """
        print(f"Loading data from {self.data_path}...")
        data = np.load(self.data_path, allow_pickle=True)

        # ACC signals: object array of (n_timesteps, 3) arrays with takeoff indices
        acc_signals = data['acc_signals']
        acc_takeoff = data['acc_takeoff']
        self.acc_data = [(sig, int(to)) for sig, to in zip(acc_signals, acc_takeoff)]

        # GRF signals: object array of (n_timesteps,) arrays (vertical GRF in BW)
        # Stored at 1000 Hz in .npz; downsample to 250 Hz to match ACC
        self.grf_data = [sig[::4] for sig in data['grf_signals']]

        # Subject IDs (0-indexed, 69 unique participants)
        self.subject_ids = data['subject_ids']

        # Ground truth metrics (already in final units: metres and W/kg)
        self.ground_truth_jump_height = data['jump_height'].astype(np.float64)
        self.ground_truth_peak_power = data['peak_power'].astype(np.float64)

        n_subjects = int(data['n_subjects'])
        print(f"Loaded {len(self.acc_data)} jumps from {n_subjects} subjects")

        return self.acc_data, self.grf_data, self.subject_ids

    def _compute_robust_mean_function(
        self,
        data: np.ndarray,
        clip_threshold: float = 5.0
    ) -> np.ndarray:
        """
        Compute a robust mean function using median and MAD for outlier detection.

        At each time point, values beyond clip_threshold * MAD from the median
        are excluded before computing the mean. This uses robust statistics
        (median and MAD) that are insensitive to extreme outliers.

        The MAD (Median Absolute Deviation) is scaled by 1.4826 to estimate
        the standard deviation for normally distributed data.

        Args:
            data: Array of shape (n_samples, seq_len, n_channels)
            clip_threshold: Number of scaled MADs beyond which to exclude (default 5.0)

        Returns:
            Mean function of shape (seq_len, n_channels)
        """
        n_samples, seq_len, n_channels = data.shape
        mean_function = np.zeros((seq_len, n_channels), dtype=np.float32)

        # MAD scale factor for normal distribution (1/Phi^-1(0.75))
        MAD_SCALE = 1.4826

        for ch in range(n_channels):
            for t in range(seq_len):
                values = data[:, t, ch]

                # Compute robust center and spread
                median = np.median(values)
                mad = np.median(np.abs(values - median))
                robust_std = MAD_SCALE * mad

                # Identify inliers (values within threshold * robust_std of median)
                if robust_std > 1e-8:
                    lower = median - clip_threshold * robust_std
                    upper = median + clip_threshold * robust_std
                    inlier_mask = (values >= lower) & (values <= upper)
                    inliers = values[inlier_mask]

                    # Compute mean of inliers (or median if too few inliers)
                    if len(inliers) > n_samples * 0.5:  # Need at least 50% inliers
                        mean_function[t, ch] = np.mean(inliers)
                    else:
                        # Fall back to median if too many outliers
                        mean_function[t, ch] = median
                else:
                    # No variation - use median
                    mean_function[t, ch] = median

        return mean_function

    def _compute_robust_std(self, data: np.ndarray) -> float:
        """
        Compute a robust estimate of global standard deviation using MAD.

        Uses the Median Absolute Deviation (MAD) scaled to estimate std
        for normally distributed data. This is insensitive to outliers.

        Args:
            data: Array of any shape (will be flattened)

        Returns:
            Robust estimate of standard deviation
        """
        flat = data.flatten()
        median = np.median(flat)
        mad = np.median(np.abs(flat - median))
        # Scale factor for normal distribution
        robust_std = 1.4826 * mad
        return float(robust_std)

    def _smooth_to_functional(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """
        Smooth signals using B-splines with GCV-optimal roughness penalty.

        For each channel, finds the optimal smoothing parameter λ via
        generalised cross-validation (Ramsay & Silverman, 2005), applies
        B-spline smoothing, and evaluates back at the original grid points.

        Also extracts B-spline coefficients for use by downstream transforms
        (B-spline and FPC representations) without a second fitting step.

        Stores the fitted basis for reuse on validation data.

        Args:
            data: Array of shape (n_samples, seq_len, n_channels)

        Returns:
            Tuple of (smoothed array, coefficients array, list of λ per channel)
            - smoothed: shape (n_samples, seq_len, n_channels)
            - coefficients: shape (n_samples, n_basis, n_channels)
        """
        n_samples, seq_len, n_channels = data.shape
        grid_points = np.linspace(0, 1, seq_len)
        smoothed = np.zeros_like(data)
        coefficients = np.zeros((n_samples, self.smooth_n_basis, n_channels), dtype=np.float32)

        # Create B-spline basis (shared across channels)
        basis = BSplineBasis(
            domain_range=(0, 1),
            n_basis=self.smooth_n_basis,
            order=4,  # cubic
        )
        self._smooth_basis = basis

        # Second derivative penalty for roughness
        regularization = L2Regularization(LinearDifferentialOperator(2))

        lambdas = []
        lambda_candidates = np.logspace(-8, 0, 50)

        for ch in range(n_channels):
            fd = FDataGrid(
                data_matrix=data[:, :, ch],
                grid_points=grid_points,
            )

            # Find GCV-optimal λ
            smoother = BasisSmoother(
                basis=basis,
                regularization=regularization,
                smoothing_parameter=1,  # placeholder, overridden by search
            )
            param_search = SmoothingParameterSearch(
                smoother,
                lambda_candidates,
                scoring=LinearSmootherGeneralizedCVScorer(),
            )
            param_search.fit(fd)
            best_lambda = param_search.best_params_['smoothing_parameter']
            lambdas.append(best_lambda)

            # Apply smoothing with best λ, returning basis representation
            best_smoother = BasisSmoother(
                basis=basis,
                regularization=regularization,
                smoothing_parameter=best_lambda,
                return_basis=True,
            )
            fd_basis = best_smoother.fit_transform(fd)

            # Extract coefficients
            coefficients[:, :, ch] = fd_basis.coefficients

            # Evaluate back at original grid points
            evaluated = fd_basis(grid_points)
            if hasattr(evaluated, 'data_matrix'):
                smoothed[:, :, ch] = evaluated.data_matrix.squeeze(axis=-1)
            else:
                smoothed[:, :, ch] = np.asarray(evaluated).squeeze(axis=-1)

        return smoothed, coefficients, lambdas

    def _smooth_with_fitted_lambda(self, data: np.ndarray, lambdas: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """
        Smooth signals using pre-fitted λ values (for validation data).

        Args:
            data: Array of shape (n_samples, seq_len, n_channels)
            lambdas: List of λ values, one per channel

        Returns:
            Tuple of (smoothed array, coefficients array)
            - smoothed: shape (n_samples, seq_len, n_channels)
            - coefficients: shape (n_samples, n_basis, n_channels)
        """
        n_samples, seq_len, n_channels = data.shape
        grid_points = np.linspace(0, 1, seq_len)
        smoothed = np.zeros_like(data)
        coefficients = np.zeros((n_samples, self.smooth_n_basis, n_channels), dtype=np.float32)

        regularization = L2Regularization(LinearDifferentialOperator(2))

        for ch in range(n_channels):
            fd = FDataGrid(
                data_matrix=data[:, :, ch],
                grid_points=grid_points,
            )

            smoother = BasisSmoother(
                basis=self._smooth_basis,
                regularization=regularization,
                smoothing_parameter=lambdas[ch],
                return_basis=True,
            )
            fd_basis = smoother.fit_transform(fd)

            # Extract coefficients
            coefficients[:, :, ch] = fd_basis.coefficients

            # Evaluate back at original grid points
            evaluated = fd_basis(grid_points)
            if hasattr(evaluated, 'data_matrix'):
                smoothed[:, :, ch] = evaluated.data_matrix.squeeze(axis=-1)
            else:
                smoothed[:, :, ch] = np.asarray(evaluated).squeeze(axis=-1)

        return smoothed, coefficients

    def preprocess(
        self,
        acc_data: list = None,
        grf_data: list[np.ndarray] = None,
        fit_normalization: bool = True,
        skip_normalization: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess signals: pad/truncate and normalize using mean function.

        Normalization uses a mean function (average curve) rather than a scalar,
        which is standard for functional data. The mean function is computed
        robustly by clipping outliers at each time point.

        Args:
            acc_data: List of (acc_signal, takeoff_idx) tuples (uses self.acc_data if None)
            grf_data: List of GRF signals (uses self.grf_data if None)
            fit_normalization: Whether to fit normalization parameters
            skip_normalization: If True, skip normalization entirely

        Returns:
            Tuple of preprocessed (acc_array, grf_array)
        """
        if acc_data is None:
            acc_data = self.acc_data
        if grf_data is None:
            grf_data = self.grf_data

        if acc_data is None or grf_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        n_samples = len(acc_data)
        input_dim = 1 if self.use_resultant else 3

        # Initialize arrays with different lengths for ACC and GRF
        acc_array = np.zeros((n_samples, self.acc_seq_len, input_dim), dtype=np.float32)
        grf_array = np.zeros((n_samples, self.grf_seq_len, 1), dtype=np.float32)

        for i, ((acc, takeoff_idx), grf) in enumerate(zip(acc_data, grf_data)):
            # Compute resultant if needed
            if self.use_resultant:
                acc_processed = np.sqrt(np.sum(acc ** 2, axis=1, keepdims=True))
            else:
                acc_processed = acc

            # Align ACC with takeoff at pre_takeoff_samples position
            acc_aligned = self._align_signal_at_takeoff(
                acc_processed, takeoff_idx,
                self.pre_takeoff_samples, self.post_takeoff_samples
            )

            # Align GRF: takeoff is at the end, pad/truncate to grf_seq_len
            grf_aligned = self._align_signal(grf.reshape(-1, 1), self.grf_seq_len)

            acc_array[i] = acc_aligned
            grf_array[i] = grf_aligned

        # Universal B-spline smoothing (before normalization)
        if self.smooth_signals and fit_normalization:
            # Training data: fit GCV-optimal λ and smooth
            acc_array, acc_coeffs, self._smooth_lambdas_acc = self._smooth_to_functional(acc_array)
            grf_array, grf_coeffs, grf_lambdas = self._smooth_to_functional(grf_array)
            self._smooth_lambda_grf = grf_lambdas[0]  # Single GRF channel

            # Store coefficients for downstream use (bspline/fpc representations)
            self._smooth_coefficients = (acc_coeffs, grf_coeffs)

            # Report selected λ values
            channel_names = ['ACC-x', 'ACC-y', 'ACC-z'] if not self.use_resultant else ['ACC-res']
            acc_str = ', '.join(f'{name} λ={lam:.2e}' for name, lam in zip(channel_names, self._smooth_lambdas_acc))
            print(f"GCV-optimal smoothing: {acc_str}, GRF λ={self._smooth_lambda_grf:.2e}")
        elif self.smooth_signals:
            # Validation data: use λ values fitted on training data
            acc_array, acc_coeffs = self._smooth_with_fitted_lambda(acc_array, self._smooth_lambdas_acc)
            grf_array, grf_coeffs = self._smooth_with_fitted_lambda(grf_array, [self._smooth_lambda_grf])

            # Store coefficients for downstream use
            self._smooth_coefficients = (acc_coeffs, grf_coeffs)

        # Normalization options:
        # - simple_normalization: global mean and std (original approach)
        # - default: mean function + MAD-based std (robust but can create extreme values)
        if skip_normalization:
            # Store stats for denormalization during evaluation (even if not normalizing)
            if fit_normalization:
                if self.simple_normalization:
                    # Simple global z-score (scalar mean)
                    self.acc_mean_function = np.full((acc_array.shape[1], acc_array.shape[2]), np.mean(acc_array))
                    self.grf_mean_function = np.full((grf_array.shape[1], grf_array.shape[2]), np.mean(grf_array))
                    self.acc_std = float(np.std(acc_array))
                    self.grf_std = float(np.std(grf_array))
                else:
                    self.acc_mean_function = self._compute_robust_mean_function(acc_array)
                    self.grf_mean_function = self._compute_robust_mean_function(grf_array)
                    self.acc_std = self._compute_robust_std(acc_array)
                    self.grf_std = self._compute_robust_std(grf_array)
            return acc_array, grf_array

        if fit_normalization:
            if self.simple_normalization:
                # Simple global z-score normalization (original approach that worked)
                acc_mean = np.mean(acc_array)
                grf_mean = np.mean(grf_array)
                self.acc_std = float(np.std(acc_array))
                self.grf_std = float(np.std(grf_array))
                # Store mean as a constant array for compatibility with denormalize
                self.acc_mean_function = np.full((acc_array.shape[1], acc_array.shape[2]), acc_mean)
                self.grf_mean_function = np.full((grf_array.shape[1], grf_array.shape[2]), grf_mean)
            else:
                # Compute robust mean functions using median/MAD
                self.acc_mean_function = self._compute_robust_mean_function(acc_array)
                self.grf_mean_function = self._compute_robust_mean_function(grf_array)
                # Compute robust global std for scaling (after centering by mean function)
                acc_centered = acc_array - self.acc_mean_function
                grf_centered = grf_array - self.grf_mean_function
                self.acc_std = self._compute_robust_std(acc_centered)
                self.grf_std = self._compute_robust_std(grf_centered)

        # Center by mean function and scale by global std
        acc_normalized = (acc_array - self.acc_mean_function) / (self.acc_std + 1e-8)
        grf_normalized = (grf_array - self.grf_mean_function) / (self.grf_std + 1e-8)

        return acc_normalized, grf_normalized

    def _align_signal_at_takeoff(
        self,
        signal: np.ndarray,
        takeoff_idx: int,
        pre_samples: int,
        post_samples: int
    ) -> np.ndarray:
        """
        Align signal so that takeoff is at position pre_samples.

        Args:
            signal: Input signal of shape (n_timesteps, n_features)
            takeoff_idx: Index of takeoff in the signal
            pre_samples: Number of samples to include before takeoff
            post_samples: Number of samples to include after takeoff

        Returns:
            Aligned signal of shape (pre_samples + post_samples, n_features)
        """
        n_features = signal.shape[1] if signal.ndim > 1 else 1
        target_len = pre_samples + post_samples
        result = np.zeros((target_len, n_features), dtype=signal.dtype)

        # Calculate source indices
        src_start = max(0, takeoff_idx - pre_samples)
        src_end = min(len(signal), takeoff_idx + post_samples)

        # Calculate destination indices
        dst_start = max(0, pre_samples - takeoff_idx)
        dst_end = dst_start + (src_end - src_start)

        # Copy available data
        result[dst_start:dst_end] = signal[src_start:src_end]

        # Pad at start with first available value if needed
        if dst_start > 0:
            first_val = signal[src_start] if src_start < len(signal) else signal[0]
            result[:dst_start] = first_val

        # Pad at end with last available value if needed
        if dst_end < target_len:
            last_val = signal[src_end - 1] if src_end > 0 else signal[-1]
            result[dst_end:] = last_val

        return result

    def _align_signal(self, signal: np.ndarray, target_len: int) -> np.ndarray:
        """
        Align signal to target length by padding at start or truncating.

        Padding uses the initial value (matching MATLAB aligndata function).

        Args:
            signal: Input signal of shape (n_timesteps, n_features)
            target_len: Target sequence length

        Returns:
            Aligned signal of shape (target_len, n_features)
        """
        current_len = len(signal)
        n_features = signal.shape[1] if signal.ndim > 1 else 1

        if current_len == target_len:
            return signal

        if current_len > target_len:
            # Truncate from start (keep the end which includes takeoff)
            return signal[-target_len:]

        # Pad at start with initial value
        pad_len = target_len - current_len
        initial_value = signal[0]
        padding = np.tile(initial_value, (pad_len, 1)) if signal.ndim > 1 else np.full((pad_len, 1), initial_value)
        return np.vstack([padding, signal])

    def create_datasets(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        batch_size: int = 32,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, dict]:
        """
        Create train and validation TensorFlow datasets.

        Unified pipeline: one B-spline fit, three different extractions.
        - Raw: raw signal → normalize in signal space
        - Smoothed: B-spline fit → evaluate at timepoints → normalize in signal space
        - B-spline: B-spline fit → extract coefficients → normalize in coefficient space
        - FPC: B-spline fit → FPCA on smooth functions → normalize in score space

        Uses participant-level split to prevent data leakage.

        Args:
            test_size: Fraction of subjects for validation
            random_state: Random seed for reproducibility
            batch_size: Batch size for datasets

        Returns:
            Tuple of (train_dataset, val_dataset, info_dict)
        """
        if self.acc_data is None:
            self.load_data()

        # Get unique subjects
        unique_subjects = np.unique(self.subject_ids)
        n_subjects = len(unique_subjects)

        # Split at subject level
        train_subjects, val_subjects = train_test_split(
            unique_subjects,
            test_size=test_size,
            random_state=random_state
        )

        # Create masks for train and validation
        train_mask = np.isin(self.subject_ids, train_subjects)
        val_mask = np.isin(self.subject_ids, val_subjects)

        # Extract data for each split
        train_acc = [self.acc_data[i] for i in range(len(self.acc_data)) if train_mask[i]]
        train_grf = [self.grf_data[i] for i in range(len(self.grf_data)) if train_mask[i]]
        val_acc = [self.acc_data[i] for i in range(len(self.acc_data)) if val_mask[i]]
        val_grf = [self.grf_data[i] for i in range(len(self.grf_data)) if val_mask[i]]

        # Split ground truth metrics
        self.train_gt_jump_height = self.ground_truth_jump_height[train_mask]
        self.train_gt_peak_power = self.ground_truth_peak_power[train_mask]
        self.val_gt_jump_height = self.ground_truth_jump_height[val_mask]
        self.val_gt_peak_power = self.ground_truth_peak_power[val_mask]

        # Determine if we need coefficient-space normalization
        uses_coeff_space = (
            self.input_transform_type in ('bspline', 'fpc')
            or self.output_transform_type in ('bspline', 'fpc')
        )

        if uses_coeff_space:
            # ============================================================
            # UNIFIED PIPELINE: one B-spline fit, multiple extractions
            # ============================================================
            # Step 1: Preprocess without normalization (align + smooth)
            X_train_smooth, y_train_smooth = self.preprocess(
                train_acc, train_grf, fit_normalization=True, skip_normalization=True
            )
            train_acc_coeffs, train_grf_coeffs = self._smooth_coefficients

            # Temporal weights from unnormalized smooth ACC
            self.temporal_weights = self.compute_temporal_weights(X_train_smooth)

            # Validation data (uses fitted λ)
            X_val_smooth, y_val_smooth = self.preprocess(
                val_acc, val_grf, fit_normalization=False, skip_normalization=True
            )
            val_acc_coeffs, val_grf_coeffs = self._smooth_coefficients

            # Step 2: Branch based on representation type
            # --- Input ---
            if self.input_transform_type == 'bspline':
                X_train = self._normalize_coefficients(
                    train_acc_coeffs, fit=True, signal='acc'
                )
                X_val = self._normalize_coefficients(
                    val_acc_coeffs, fit=False, signal='acc'
                )
                # Set up transformer for inverse transform only
                self.input_transformer = self._create_bspline_inverse_transformer(
                    self.acc_seq_len, train_acc_coeffs.shape[2]
                )
                print(f"Input transform: bspline (unified pipeline)")
                print(f"  Coefficients: {X_train.shape} ({self.smooth_n_basis} basis functions)")
            elif self.input_transform_type == 'fpc':
                # FPCA on smooth functions (unnormalized)
                X_train, X_val = self._apply_fpca_unified(
                    X_train_smooth, X_val_smooth, signal='acc'
                )
            else:
                # raw or smoothed: normalize smooth signals in signal space
                self._compute_signal_space_stats(X_train_smooth, 'acc')
                X_train = ((X_train_smooth - self.acc_mean_function)
                           / (self.acc_std + 1e-8)).astype(np.float32)
                X_val = ((X_val_smooth - self.acc_mean_function)
                         / (self.acc_std + 1e-8)).astype(np.float32)
                self.input_transformer = IdentityTransformer()
                self.input_transformer.fit(X_train)

            # --- Output ---
            if self.output_transform_type == 'bspline':
                self.normalize_in_coeff_space = True
                y_train = self._normalize_coefficients(
                    train_grf_coeffs, fit=True, signal='grf'
                )
                y_val = self._normalize_coefficients(
                    val_grf_coeffs, fit=False, signal='grf'
                )
                # Set up transformer for inverse transform only
                self.output_transformer = self._create_bspline_inverse_transformer(
                    self.grf_seq_len, train_grf_coeffs.shape[2]
                )
                # Also store signal-space stats for denormalization path
                self._compute_signal_space_stats(y_train_smooth, 'grf')
                print(f"Output transform: bspline (unified pipeline)")
                print(f"  Coefficients: {y_train.shape} ({self.smooth_n_basis} basis functions)")
            elif self.output_transform_type == 'fpc':
                self.normalize_in_coeff_space = True
                y_train, y_val = self._apply_fpca_unified(
                    y_train_smooth, y_val_smooth, signal='grf'
                )
                # Also store signal-space stats for denormalization path
                self._compute_signal_space_stats(y_train_smooth, 'grf')
            else:
                # raw output with bspline/fpc input
                self._compute_signal_space_stats(y_train_smooth, 'grf')
                y_train = ((y_train_smooth - self.grf_mean_function)
                           / (self.grf_std + 1e-8)).astype(np.float32)
                y_val = ((y_val_smooth - self.grf_mean_function)
                         / (self.grf_std + 1e-8)).astype(np.float32)
                self.output_transformer = IdentityTransformer()
                self.output_transformer.fit(y_train)

            # Compute B-spline reference for rigorous evaluation
            if self.use_bspline_reference:
                # Use the smooth signals directly as reference (they are already
                # the B-spline reconstruction from the unified fit)
                # Normalize to match the signal-space normalization
                self.bspline_reference_train = (
                    (y_train_smooth - self.grf_mean_function) / (self.grf_std + 1e-8)
                ).astype(np.float32)
                self.bspline_reference_val = (
                    (y_val_smooth - self.grf_mean_function) / (self.grf_std + 1e-8)
                ).astype(np.float32)
                print(f"B-spline reference: using unified smooth signals")

        else:
            # ============================================================
            # ORIGINAL PIPELINE: raw or smoothed signal-space processing
            # ============================================================
            # Preprocess training data (fit normalization)
            X_train, y_train = self.preprocess(train_acc, train_grf, fit_normalization=True)

            # Compute temporal weights from unnormalized ACC
            X_train_unnorm = X_train * self.acc_std + self.acc_mean_function
            self.temporal_weights = self.compute_temporal_weights(X_train_unnorm)

            # Preprocess validation data (use fitted normalization)
            X_val, y_val = self.preprocess(val_acc, val_grf, fit_normalization=False)

            # Set identity transformers
            self.input_transformer = IdentityTransformer()
            self.input_transformer.fit(X_train)
            self.output_transformer = IdentityTransformer()
            self.output_transformer.fit(y_train)

            # Compute B-spline reference for rigorous evaluation
            if self.use_bspline_reference:
                from src.transformations import BSplineTransformer
                bspline_ref_transformer = BSplineTransformer(
                    n_basis=self.smooth_n_basis,
                    smoothing_lambda=1e-4,
                )
                bspline_ref_transformer.fit(y_train)
                y_train_coeffs = bspline_ref_transformer.transform(y_train)
                y_val_coeffs = bspline_ref_transformer.transform(y_val)
                self.bspline_reference_train = bspline_ref_transformer.inverse_transform(y_train_coeffs)
                self.bspline_reference_val = bspline_ref_transformer.inverse_transform(y_val_coeffs)
                print(f"B-spline reference computed for rigorous evaluation")

        # Handle scalar prediction targets
        if self.scalar_prediction == 'jump_height' or self.scalar_only:
            # Z-score normalize jump height using training set stats
            self.scalar_mean = float(np.mean(self.train_gt_jump_height))
            self.scalar_std = float(np.std(self.train_gt_jump_height))
            train_scalar = ((self.train_gt_jump_height - self.scalar_mean)
                            / (self.scalar_std + 1e-8)).astype(np.float32)
            val_scalar = ((self.val_gt_jump_height - self.scalar_mean)
                          / (self.scalar_std + 1e-8)).astype(np.float32)

            if self.scalar_only:
                # Scalar-only mode: (X, scalar_targets)
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, train_scalar))
                train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

                val_dataset = tf.data.Dataset.from_tensor_slices((X_val, val_scalar))
                val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

                print(f"Scalar-only mode: predicting jump_height")
                print(f"  JH train mean: {self.scalar_mean:.4f} m, std: {self.scalar_std:.4f} m")
            else:
                # Create multi-output datasets: (X, {'curve_output': y, 'scalar_output': jh})
                train_targets = {'curve_output': y_train, 'scalar_output': train_scalar}
                val_targets = {'curve_output': y_val, 'scalar_output': val_scalar}

                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, train_targets))
                train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

                val_dataset = tf.data.Dataset.from_tensor_slices((X_val, val_targets))
                val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

                print(f"Scalar prediction: {self.scalar_prediction}")
                print(f"  JH train mean: {self.scalar_mean:.4f} m, std: {self.scalar_std:.4f} m")
        else:
            # Create standard single-output datasets
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Determine transformed sequence lengths
        transformed_input_len = X_train.shape[1]
        transformed_output_len = y_train.shape[1]

        info = {
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'n_train_subjects': len(train_subjects),
            'n_val_subjects': len(val_subjects),
            'input_shape': X_train.shape[1:],
            'output_shape': y_train.shape[1:],
            'acc_seq_len': self.acc_seq_len,
            'grf_seq_len': self.grf_seq_len,
            'transformed_input_len': transformed_input_len,
            'transformed_output_len': transformed_output_len,
            'pre_takeoff_ms': self.pre_takeoff_ms,
            'post_takeoff_ms': self.post_takeoff_ms,
            'acc_mean_function': self.acc_mean_function,
            'acc_std': self.acc_std,
            'grf_mean_function': self.grf_mean_function,
            'grf_std': self.grf_std,
            # Ground truth metrics for validation set (from full signal)
            'val_gt_jump_height': self.val_gt_jump_height,
            'val_gt_peak_power': self.val_gt_peak_power,
            # Temporal weights for weighted MSE loss
            'temporal_weights': self.temporal_weights,
            # Transformation info
            'input_transform': self.input_transform_type,
            'output_transform': self.output_transform_type,
            'input_transformer': self.input_transformer,
            'output_transformer': self.output_transformer,
            'skip_normalization': False,  # Currently always False (pre-normalization used)
            # Scalar prediction info
            'scalar_prediction': self.scalar_prediction,
            'scalar_mean': self.scalar_mean,
            'scalar_std': self.scalar_std,
            # B-spline reference for rigorous evaluation
            'use_bspline_reference': self.use_bspline_reference,
            'bspline_reference_val': self.bspline_reference_val,
        }

        print(f"Train: {info['n_train_samples']} samples from {info['n_train_subjects']} subjects")
        print(f"Val: {info['n_val_samples']} samples from {info['n_val_subjects']} subjects")

        return train_dataset, val_dataset, info

    def denormalize_grf(self, grf_normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized GRF back to body weight units.

        Reverses the mean function centering and std scaling applied during
        preprocessing.

        Args:
            grf_normalized: Normalized GRF of shape (n_samples, seq_len, n_channels)

        Returns:
            GRF in body weight units
        """
        return grf_normalized * self.grf_std + self.grf_mean_function

    def denormalize_scalar(self, scalar_normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized scalar predictions back to original units.

        Args:
            scalar_normalized: Z-score normalized scalar values

        Returns:
            Scalar values in original units (e.g., meters for jump height)
        """
        if self.scalar_mean is None or self.scalar_std is None:
            raise RuntimeError("Scalar normalization stats not computed. "
                               "Call create_datasets() with scalar_prediction enabled first.")
        return scalar_normalized * self.scalar_std + self.scalar_mean

    def denormalize_output(self, y: np.ndarray) -> np.ndarray:
        """
        Denormalize model output back to body weight units.

        For bspline/fpc: denormalize in coefficient/score space, then inverse
        transform to signal space, giving BW signals.

        For raw/smoothed: standard signal-space denormalization.

        Args:
            y: Normalized model output (n_samples, n_features, n_channels)

        Returns:
            GRF signals in body weight units (n_samples, seq_len, n_channels)
        """
        if self.normalize_in_coeff_space:
            # Denormalize in coefficient/score space
            y_denorm = y * self.grf_coeff_std + self.grf_coeff_mean
            # Inverse transform to signal space (gives BW signals)
            return self.output_transformer.inverse_transform(y_denorm)
        else:
            return self.denormalize_grf(y)

    def _compute_signal_space_stats(self, data: np.ndarray, signal: str) -> None:
        """
        Compute and store signal-space normalization stats.

        Respects the simple_normalization flag to ensure consistent stats
        regardless of which pipeline path is used.

        Args:
            data: Unnormalized signal array (n_samples, seq_len, n_channels)
            signal: 'acc' or 'grf'
        """
        if self.simple_normalization:
            mean_val = np.mean(data)
            mean_function = np.full(data.shape[1:], mean_val, dtype=np.float32)
            std_val = float(np.std(data))
        else:
            mean_function = self._compute_robust_mean_function(data)
            centered = data - mean_function
            std_val = self._compute_robust_std(centered)

        if signal == 'grf':
            self.grf_mean_function = mean_function
            self.grf_std = std_val
        else:
            self.acc_mean_function = mean_function
            self.acc_std = std_val

    def _normalize_coefficients(
        self,
        coefficients: np.ndarray,
        fit: bool,
        signal: str,
    ) -> np.ndarray:
        """
        Z-score normalize B-spline coefficients.

        Args:
            coefficients: Shape (n_samples, n_basis, n_channels)
            fit: If True, compute and store normalization stats
            signal: 'acc' or 'grf' — determines which stats to use/store

        Returns:
            Normalized coefficients of same shape
        """
        if fit:
            mean = np.mean(coefficients, axis=0)  # (n_basis, n_channels)
            centered = coefficients - mean
            std = float(np.std(centered))
            if signal == 'grf':
                self.grf_coeff_mean = mean
                self.grf_coeff_std = std
            else:
                self.acc_coeff_mean = mean
                self.acc_coeff_std = std
        else:
            if signal == 'grf':
                mean, std = self.grf_coeff_mean, self.grf_coeff_std
            else:
                mean, std = self.acc_coeff_mean, self.acc_coeff_std

        return ((coefficients - mean) / (std + 1e-8)).astype(np.float32)

    def _create_bspline_inverse_transformer(
        self,
        seq_len: int,
        n_channels: int,
    ) -> 'BSplineTransformer':
        """
        Create a BSplineTransformer configured for inverse transform only.

        Uses the basis from the unified smoothing fit (self._smooth_basis).

        Args:
            seq_len: Signal sequence length (for evaluation grid)
            n_channels: Number of signal channels

        Returns:
            Configured BSplineTransformer
        """
        from src.transformations import BSplineTransformer
        transformer = BSplineTransformer(
            n_basis=self.smooth_n_basis,
        )
        # Manually configure for inverse transform without fitting
        transformer._seq_len = seq_len
        transformer._n_channels = n_channels
        transformer._time_points = np.linspace(0, 1, seq_len)
        transformer._basis = self._smooth_basis
        return transformer

    def _apply_fpca_unified(
        self,
        train_smooth: np.ndarray,
        val_smooth: np.ndarray,
        signal: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply FPCA to smooth (unnormalized) signals, then normalize scores.

        This is the unified pipeline path for FPC representations: FPCA
        operates on the smooth functional representation from the shared
        B-spline fit, not on normalized signals.

        Args:
            train_smooth: Training smooth signals (n_samples, seq_len, n_channels)
            val_smooth: Validation smooth signals
            signal: 'acc' or 'grf'

        Returns:
            Tuple of (normalized train scores, normalized val scores)
        """
        # Create and fit FPCA transformer on smooth (unnormalized) signals
        transformer = get_transformer(
            'fpc',
            n_components=self.n_components,
            variance_threshold=self.variance_threshold,
            use_varimax=self.use_varimax,
            score_scale=self.score_scale,
        )
        transformer.fit(train_smooth)

        # Transform to scores
        train_scores = transformer.transform(train_smooth)
        val_scores = transformer.transform(val_smooth)

        # Store transformer
        if signal == 'grf':
            self.output_transformer = transformer
        else:
            self.input_transformer = transformer

        # Normalize scores in score space
        train_normalized = self._normalize_coefficients(train_scores, fit=True, signal=signal)
        val_normalized = self._normalize_coefficients(val_scores, fit=False, signal=signal)

        print(f"{'Output' if signal == 'grf' else 'Input'} transform: fpc (unified pipeline)")
        print(f"  Scores: {train_normalized.shape} ({transformer.n_features} components)")

        return train_normalized, val_normalized

    def compute_temporal_weights(
        self,
        acc_array: np.ndarray,
        min_weight: float = 0.1,
        smooth_window: int = 5,
        weight_type: str = 'jerk',
        propulsion_ms: float = 400.0,
        propulsion_weight: float = 5.0,
    ) -> np.ndarray:
        """
        Compute temporal weights for loss computation.

        Args:
            acc_array: Preprocessed ACC data of shape (n_samples, seq_len, n_features)
            min_weight: Minimum weight to avoid ignoring any region entirely
            smooth_window: Window size for smoothing the weight profile
            weight_type: 'jerk' for jerk-based, 'propulsion' for propulsion-phase emphasis
            propulsion_ms: Duration of propulsion phase in ms (from end of signal)
            propulsion_weight: Weight multiplier for propulsion phase

        Returns:
            Temporal weights of shape (seq_len,), normalized to mean 1.0
        """
        n_samples, seq_len, n_features = acc_array.shape

        if weight_type == 'propulsion':
            # Propulsion-phase weighting: emphasize the last portion of the signal
            # where jump height is determined
            propulsion_samples = int(propulsion_ms * SAMPLING_RATE / 1000)
            propulsion_samples = min(propulsion_samples, seq_len)

            weights = np.ones(seq_len, dtype=np.float32)
            weights[-propulsion_samples:] = propulsion_weight

            # Smooth transition (ramp over 50ms)
            ramp_samples = int(50 * SAMPLING_RATE / 1000)
            if ramp_samples > 0 and propulsion_samples < seq_len:
                ramp_start = seq_len - propulsion_samples - ramp_samples
                ramp_end = seq_len - propulsion_samples
                if ramp_start >= 0:
                    ramp = np.linspace(1.0, propulsion_weight, ramp_samples + 1)[:-1]
                    weights[ramp_start:ramp_end] = ramp

        else:  # 'jerk' - original jerk-based weighting
            # Compute second derivative (jerk) along time axis for each sample
            d2_acc = np.diff(acc_array, n=2, axis=1)  # shape: (n_samples, seq_len-2, n_features)

            # Take absolute value (magnitude of change matters, not direction)
            d2_acc = np.abs(d2_acc)

            # Average across features (if triaxial)
            d2_acc = np.mean(d2_acc, axis=-1)  # shape: (n_samples, seq_len-2)

            # Average across all samples to get global weight profile
            weights = np.mean(d2_acc, axis=0)  # shape: (seq_len-2,)

            # Pad to match original sequence length (diff reduces by 2)
            weights = np.concatenate([[weights[0]], weights, [weights[-1]]])

            # Apply smoothing to avoid spiky gradients
            if smooth_window > 1:
                kernel = np.ones(smooth_window) / smooth_window
                weights = np.convolve(weights, kernel, mode='same')

        # Normalize to [0, 1] range first, then apply floor as fraction of max
        weights = weights / (np.max(weights) + 1e-8)
        weights = np.maximum(weights, min_weight)

        # Normalize so weights have mean 1.0 (preserves loss magnitude)
        weights = weights / np.mean(weights)

        return weights.astype(np.float32)

    def _apply_transformations(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply FDA transformations to input and output signals.

        Creates transformers, fits them on training data, and applies
        to both training and validation data.

        Args:
            X_train: Training input signals (n_samples, seq_len, n_channels)
            y_train: Training output signals (n_samples, seq_len, 1)
            X_val: Validation input signals
            y_val: Validation output signals

        Returns:
            Tuple of transformed (X_train, y_train, X_val, y_val)
        """
        # Choose transformer factory based on use_custom_fpca
        if self.use_custom_fpca:
            from src.transformations_custom import get_custom_transformer as transformer_factory
            print("  Using custom FPCA implementation (discrete dot products)")
        else:
            transformer_factory = get_transformer

        # Create input transformer
        self.input_transformer = transformer_factory(
            self.input_transform_type,
            n_basis=self.n_basis,
            n_components=self.n_components,
            variance_threshold=self.variance_threshold,
            bspline_lambda=self.bspline_lambda,
            use_varimax=self.use_varimax,
            **({"score_scale": self.score_scale} if not self.use_custom_fpca else {}),
        )

        # Create output transformer
        self.output_transformer = transformer_factory(
            self.output_transform_type,
            n_basis=self.n_basis,
            n_components=self.n_components,
            variance_threshold=self.variance_threshold,
            bspline_lambda=self.bspline_lambda,
            use_varimax=self.use_varimax,
            **({"score_scale": self.score_scale} if not self.use_custom_fpca else {}),
        )

        # Fit transformers on training data
        self.input_transformer.fit(X_train)
        self.output_transformer.fit(y_train)

        # Transform training data
        X_train_transformed = self.input_transformer.transform(X_train)
        y_train_transformed = self.output_transformer.transform(y_train)

        # Transform validation data (using fitted transformers)
        X_val_transformed = self.input_transformer.transform(X_val)
        y_val_transformed = self.output_transformer.transform(y_val)

        # Log transformation info
        if self.input_transform_type != 'raw':
            print(f"Input transform: {self.input_transform_type}")
            print(f"  Original shape: {X_train.shape} -> Transformed shape: {X_train_transformed.shape}")
            print(f"  Features: {self.input_transformer.n_features}")

        if self.output_transform_type != 'raw':
            print(f"Output transform: {self.output_transform_type}")
            print(f"  Original shape: {y_train.shape} -> Transformed shape: {y_train_transformed.shape}")
            print(f"  Features: {self.output_transformer.n_features}")

        return (
            X_train_transformed.astype(np.float32),
            y_train_transformed.astype(np.float32),
            X_val_transformed.astype(np.float32),
            y_val_transformed.astype(np.float32),
        )

    def inverse_transform_output(self, y_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform output from coefficient space back to signal space.

        Args:
            y_transformed: Transformed output (n_samples, n_features, n_channels)

        Returns:
            Reconstructed signals (n_samples, seq_len, n_channels)
        """
        if self.output_transformer is None:
            raise RuntimeError("Output transformer not fitted. Call create_datasets() first.")
        return self.output_transformer.inverse_transform(y_transformed)

    def inverse_transform_input(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform input from coefficient space back to signal space.

        Args:
            X_transformed: Transformed input (n_samples, n_features, n_channels)

        Returns:
            Reconstructed signals (n_samples, seq_len, n_channels)
        """
        if self.input_transformer is None:
            raise RuntimeError("Input transformer not fitted. Call create_datasets() first.")
        return self.input_transformer.inverse_transform(X_transformed)

    def get_summary_stats(self) -> dict:
        """Get summary statistics of loaded data."""
        if self.acc_data is None:
            return {}

        lengths = [len(acc) if not isinstance(acc, tuple) else len(acc[0])
                   for acc in self.acc_data]

        return {
            'n_samples': len(self.acc_data),
            'n_subjects': len(np.unique(self.subject_ids)),
            'signal_length_min': min(lengths),
            'signal_length_max': max(lengths),
            'signal_length_mean': np.mean(lengths),
            'signal_length_std': np.std(lengths),
        }


def load_cmj_data(
    data_path: str = DEFAULT_DATA_PATH,
    use_resultant: bool = True,
    test_size: float = 0.2,
    batch_size: int = 32,
    random_state: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, CMJDataLoader]:
    """
    Convenience function to load CMJ data for training.

    Args:
        data_path: Path to MATLAB data file
        use_resultant: Whether to compute resultant acceleration
        test_size: Fraction for validation
        batch_size: Batch size
        random_state: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset, data_loader)
    """
    loader = CMJDataLoader(data_path=data_path, use_resultant=use_resultant)
    loader.load_data()

    train_ds, val_ds, info = loader.create_datasets(
        test_size=test_size,
        batch_size=batch_size,
        random_state=random_state,
    )

    return train_ds, val_ds, loader
