"""
Data Loader for Countermovement Jump Data

Loads and preprocesses accelerometer and GRF data from MATLAB files
for training the signal-to-signal transformer model.
"""

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import tensorflow as tf
from typing import Optional

from .transformations import get_transformer, BaseSignalTransformer


# Default paths
DEFAULT_DATA_PATH = "/Users/markgewhite/ARCHIVE/Data/Processed/All/processedjumpdata.mat"
DEFAULT_SEQ_LEN = 500
SAMPLING_RATE = 250  # Hz (ACC native rate; GRF downsampled from 1000Hz)

# Default durations in milliseconds
DEFAULT_PRE_TAKEOFF_MS = 2000  # 2 seconds before takeoff (500 samples at 250 Hz)
DEFAULT_POST_TAKEOFF_MS = 0    # No extension after takeoff by default


class CMJDataLoader:
    """
    Data loader for countermovement jump accelerometer and GRF data.

    Loads MATLAB files and preprocesses signals for the transformer model.

    Args:
        data_path: Path to processedjumpdata.mat file
        pre_takeoff_ms: Duration before takeoff in milliseconds (default 2000ms)
        post_takeoff_ms: Duration after takeoff for ACC input only (default 0ms)
        use_resultant: If True, compute resultant acceleration; else use triaxial
        sensor_idx: Sensor index (0=lower back, 1=upper back, etc.) - 0-indexed
        grf_plate_idx: Force plate index (2=combined plates) - 0-indexed
        input_transform: Transform type for input ('raw', 'bspline', 'fpc')
        output_transform: Transform type for output ('raw', 'bspline', 'fpc')
        n_basis: Number of B-spline basis functions
        n_components: Number of FPC components
        variance_threshold: Cumulative variance threshold for FPC selection
        bspline_lambda: B-spline smoothing parameter
        use_varimax: Whether to apply varimax rotation to FPCs
        fpc_smooth_lambda: Pre-FPCA smoothing parameter (None = no smoothing)
        fpc_n_basis_smooth: Number of basis functions for pre-FPCA smoothing
        acc_max_threshold: Maximum allowable ACC value in g (samples with higher
            values are excluded as sensor artifacts). Default 100g removes only
            catastrophically corrupted samples while preserving legitimate high-impact data.

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
        sensor_idx: int = 0,
        grf_plate_idx: int = 2,
        input_transform: str = 'raw',
        output_transform: str = 'raw',
        n_basis: int = 30,
        n_components: int = 15,
        variance_threshold: float = 0.99,
        bspline_lambda: float = 1e-4,
        use_varimax: bool = True,
        fpc_smooth_lambda: float = None,
        fpc_n_basis_smooth: int = 50,
        acc_max_threshold: float = 100.0,
    ):
        self.data_path = data_path
        self.pre_takeoff_ms = pre_takeoff_ms
        self.post_takeoff_ms = post_takeoff_ms
        self.use_resultant = use_resultant
        self.sensor_idx = sensor_idx
        self.grf_plate_idx = grf_plate_idx

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
        self.acc_max_threshold = acc_max_threshold

        # Calculate sequence lengths from durations
        self.pre_takeoff_samples = int(pre_takeoff_ms * SAMPLING_RATE / 1000)
        self.post_takeoff_samples = int(post_takeoff_ms * SAMPLING_RATE / 1000)
        self.acc_seq_len = self.pre_takeoff_samples + self.post_takeoff_samples
        self.grf_seq_len = self.pre_takeoff_samples  # GRF only up to takeoff

        # Data storage
        self.acc_data = None
        self.grf_data = None
        self.subject_ids = None
        self.jump_indices = None

        # Ground truth metrics (from full signal, pre-computed in MATLAB)
        self.ground_truth_jump_height = None
        self.ground_truth_peak_power = None
        self.body_mass = None  # For converting peak power to W/kg

        # Normalization parameters (mean functions are shape (seq_len, n_channels))
        self.acc_mean_function = None
        self.acc_std = None
        self.grf_mean_function = None
        self.grf_std = None

        # Temporal weights for weighted MSE loss
        self.temporal_weights = None

        # Signal transformers (fitted during create_datasets)
        self.input_transformer: Optional[BaseSignalTransformer] = None
        self.output_transformer: Optional[BaseSignalTransformer] = None

    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and extract data from MATLAB file.

        Returns:
            Tuple of (acc_data, grf_data, subject_ids)
        """
        print(f"Loading data from {self.data_path}...")
        mat_data = loadmat(self.data_path, squeeze_me=False, struct_as_record=False)

        # Extract key variables
        acc_struct = mat_data['acc'][0, 0]
        grf_struct = mat_data['grf'][0, 0]
        bwall = mat_data['bwall']
        n_subjects = int(mat_data['nSubjects'][0, 0])
        n_jumps_per_subject = mat_data['nJumpsPerSubject'].flatten()

        # Extract pre-computed jump performance metrics (from full signal)
        jumpperf = mat_data['jumpperf'][0, 0]
        gt_jump_height = jumpperf.height  # shape (n_subjects, max_jumps)
        gt_peak_power = jumpperf.peakPower  # shape (n_subjects, max_jumps), in Watts

        # Get takeoff indices - ACC and GRF have different sampling rates
        # ACC: 250 Hz, GRF: 1000 Hz (4:1 ratio)
        acc_takeoff_indices = acc_struct.takeoff
        grf_takeoff_indices = grf_struct.takeoff

        # Extract raw data cells
        # acc.raw shape: (n_subjects, n_jumps, n_sensors) - each element is (n_timesteps, 3)
        # grf.raw shape: (n_subjects, n_jumps, n_plates) - each element is (n_timesteps, 3)
        acc_raw = acc_struct.raw
        grf_raw = grf_struct.raw

        # Collect valid jumps
        acc_list = []
        grf_list = []
        subject_id_list = []
        jump_idx_list = []
        gt_jh_list = []  # Ground truth jump height
        gt_pp_list = []  # Ground truth peak power (Watts)
        body_mass_list = []  # Body mass (kg)
        n_excluded_outliers = 0  # Track excluded samples for logging

        for subj_idx in range(n_subjects):
            n_jumps = int(n_jumps_per_subject[subj_idx])

            for jump_idx in range(n_jumps):
                # Get takeoff indices for both signals (different sampling rates)
                acc_takeoff = self._get_takeoff_index(acc_takeoff_indices, subj_idx, jump_idx)
                grf_takeoff = self._get_takeoff_index(grf_takeoff_indices, subj_idx, jump_idx)
                if acc_takeoff is None or acc_takeoff <= 0:
                    continue
                if grf_takeoff is None or grf_takeoff <= 0:
                    continue

                # Extract accelerometer data (sensor-specific)
                acc_signal = self._extract_acc_signal(acc_raw, subj_idx, jump_idx)
                if acc_signal is None:
                    continue

                # Check for outlier ACC values (sensor artifacts)
                if self.acc_max_threshold is not None:
                    acc_resultant = np.sqrt(np.sum(acc_signal ** 2, axis=1))
                    if np.max(acc_resultant) > self.acc_max_threshold:
                        n_excluded_outliers += 1
                        continue

                # Extract GRF data (vertical component from specified plate)
                grf_signal = self._extract_grf_signal(grf_raw, subj_idx, jump_idx)
                if grf_signal is None:
                    continue

                # Get body weight for normalization
                body_weight = bwall[subj_idx, jump_idx]
                if body_weight <= 0:
                    continue

                # Extract signal windows relative to takeoff
                acc_takeoff = int(acc_takeoff)
                grf_takeoff = int(grf_takeoff)

                # ACC: include post_takeoff_samples after takeoff (for flight/landing)
                acc_end = acc_takeoff + self.post_takeoff_samples
                acc_signal = acc_signal[:acc_end]

                # GRF: truncate at takeoff only (GRF is 0 during flight)
                grf_signal = grf_signal[:grf_takeoff]

                # Skip if signals are too short (need enough pre-takeoff data)
                if acc_takeoff < 100:
                    continue

                # Downsample GRF from 1000Hz to 250Hz to match ACC sampling rate
                # Take every 4th sample
                grf_signal = grf_signal[::4]

                # Normalize GRF by body weight (convert to BW units)
                grf_signal = grf_signal / body_weight

                # Get ground truth metrics (from full signal)
                jh = gt_jump_height[subj_idx, jump_idx]
                pp = gt_peak_power[subj_idx, jump_idx]
                mass = body_weight / 9.812  # Convert N to kg

                acc_list.append((acc_signal, acc_takeoff))  # Store takeoff index with signal
                grf_list.append(grf_signal)
                subject_id_list.append(subj_idx)
                jump_idx_list.append(jump_idx)
                gt_jh_list.append(float(jh))
                gt_pp_list.append(float(pp))
                body_mass_list.append(float(mass))

        print(f"Extracted {len(acc_list)} valid jumps from {n_subjects} subjects")
        if n_excluded_outliers > 0:
            print(f"  Excluded {n_excluded_outliers} samples with ACC > {self.acc_max_threshold}g (sensor artifacts)")

        self.acc_data = acc_list
        self.grf_data = grf_list
        self.subject_ids = np.array(subject_id_list)
        self.jump_indices = np.array(jump_idx_list)
        self.ground_truth_jump_height = np.array(gt_jh_list)
        self.ground_truth_peak_power = np.array(gt_pp_list)
        self.body_mass = np.array(body_mass_list)

        return self.acc_data, self.grf_data, self.subject_ids

    def _get_takeoff_index(self, takeoff_indices: np.ndarray, subj_idx: int, jump_idx: int) -> Optional[int]:
        """Extract takeoff index from array."""
        try:
            takeoff = takeoff_indices[subj_idx, jump_idx]
            if isinstance(takeoff, np.ndarray):
                takeoff = takeoff.flat[0]
            return int(takeoff) if not np.isnan(takeoff) else None
        except (IndexError, TypeError, ValueError):
            return None

    def _extract_acc_signal(self, acc_raw: np.ndarray, subj_idx: int, jump_idx: int) -> Optional[np.ndarray]:
        """Extract accelerometer signal for a specific sensor."""
        try:
            # MATLAB cell array: acc.raw{subject, jump, sensor}
            acc_cell = acc_raw[subj_idx, jump_idx, self.sensor_idx]

            # Handle nested cell/array structure
            if isinstance(acc_cell, np.ndarray):
                if acc_cell.ndim == 0:
                    acc_cell = acc_cell.item()
                elif acc_cell.size == 1:
                    acc_cell = acc_cell.flat[0]

            signal = np.array(acc_cell, dtype=np.float32)

            # Ensure shape is (n_timesteps, 3) for triaxial
            if signal.ndim == 1:
                return None  # Need 3D accelerometer data
            if signal.shape[1] != 3:
                signal = signal.T  # Transpose if needed

            return signal
        except (IndexError, TypeError, ValueError):
            return None

    def _extract_grf_signal(self, grf_raw: np.ndarray, subj_idx: int, jump_idx: int) -> Optional[np.ndarray]:
        """Extract vertical GRF signal from specified force plate."""
        try:
            # grf.raw[subject, jump, plate] - each element is (n_timesteps, 3)
            # Column 2 is vertical GRF
            grf_cell = grf_raw[subj_idx, jump_idx, self.grf_plate_idx]

            # Handle nested cell/array structure
            if isinstance(grf_cell, np.ndarray):
                if grf_cell.ndim == 0:
                    grf_cell = grf_cell.item()

            signal = np.array(grf_cell, dtype=np.float32)

            # Extract vertical component (column index 2)
            if signal.ndim == 2 and signal.shape[1] >= 3:
                signal = signal[:, 2]  # Vertical GRF
            elif signal.ndim == 1:
                pass  # Already 1D
            else:
                return None

            return signal
        except (IndexError, TypeError, ValueError):
            return None

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

        # Functional normalization: center by mean function, scale by global std
        # Mean function and std computed robustly using median/MAD
        if skip_normalization:
            # Store stats for denormalization during evaluation (even if not normalizing)
            if fit_normalization:
                self.acc_mean_function = self._compute_robust_mean_function(acc_array)
                self.grf_mean_function = self._compute_robust_mean_function(grf_array)
                self.acc_std = self._compute_robust_std(acc_array)
                self.grf_std = self._compute_robust_std(grf_array)
            return acc_array, grf_array

        if fit_normalization:
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
        self.train_body_mass = self.body_mass[train_mask]
        self.val_gt_jump_height = self.ground_truth_jump_height[val_mask]
        self.val_gt_peak_power = self.ground_truth_peak_power[val_mask]
        self.val_body_mass = self.body_mass[val_mask]

        # Preprocess training data (fit normalization)
        X_train, y_train = self.preprocess(train_acc, train_grf, fit_normalization=True)

        # Compute temporal weights from training ACC (before normalization for interpretability)
        # Re-preprocess without normalization to get raw aligned signals for weight computation
        X_train_raw, _ = self.preprocess(train_acc, train_grf, fit_normalization=False)
        X_train_raw = X_train_raw * self.acc_std + self.acc_mean_function  # Denormalize
        self.temporal_weights = self.compute_temporal_weights(X_train_raw)

        # Preprocess validation data (use fitted normalization)
        X_val, y_val = self.preprocess(val_acc, val_grf, fit_normalization=False)

        # Apply FDA transformations
        X_train, y_train, X_val, y_val = self._apply_transformations(
            X_train, y_train, X_val, y_val
        )

        # Create TensorFlow datasets
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
            'val_body_mass': self.val_body_mass,
            # Temporal weights for weighted MSE loss
            'temporal_weights': self.temporal_weights,
            # Transformation info
            'input_transform': self.input_transform_type,
            'output_transform': self.output_transform_type,
            'input_transformer': self.input_transformer,
            'output_transformer': self.output_transformer,
            'skip_normalization': False,  # Currently always False (pre-normalization used)
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

    def compute_temporal_weights(
        self,
        acc_array: np.ndarray,
        min_weight: float = 0.1,
        smooth_window: int = 5,
    ) -> np.ndarray:
        """
        Compute temporal weights from the second derivative of ACC data.

        Weights are higher in regions of rapid acceleration change (jerk),
        emphasizing countermovement and propulsion phases over quiet standing.

        Args:
            acc_array: Preprocessed ACC data of shape (n_samples, seq_len, n_features)
            min_weight: Minimum weight to avoid ignoring any region entirely
            smooth_window: Window size for smoothing the weight profile

        Returns:
            Temporal weights of shape (seq_len,), normalized to mean 1.0
        """
        n_samples, seq_len, n_features = acc_array.shape

        # Compute second derivative (jerk) along time axis for each sample
        # Using central differences: d2x/dt2 ≈ x[i+1] - 2*x[i] + x[i-1]
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

        # Apply minimum weight floor
        weights = np.maximum(weights, min_weight)

        # Normalize so mean weight = 1.0
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
        # Create input transformer
        self.input_transformer = get_transformer(
            self.input_transform_type,
            n_basis=self.n_basis,
            n_components=self.n_components,
            variance_threshold=self.variance_threshold,
            bspline_lambda=self.bspline_lambda,
            use_varimax=self.use_varimax,
            fpc_smooth_lambda=self.fpc_smooth_lambda,
            fpc_n_basis_smooth=self.fpc_n_basis_smooth,
        )

        # Create output transformer
        self.output_transformer = get_transformer(
            self.output_transform_type,
            n_basis=self.n_basis,
            n_components=self.n_components,
            variance_threshold=self.variance_threshold,
            bspline_lambda=self.bspline_lambda,
            use_varimax=self.use_varimax,
            fpc_smooth_lambda=self.fpc_smooth_lambda,
            fpc_n_basis_smooth=self.fpc_n_basis_smooth,
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

        lengths = [len(acc) for acc in self.acc_data]

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
