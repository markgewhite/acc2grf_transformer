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


# Default paths
DEFAULT_DATA_PATH = "/Users/markgewhite/ARCHIVE/Data/Processed/All/processedjumpdata.mat"
DEFAULT_SEQ_LEN = 500
SAMPLING_RATE = 250  # Hz (ACC native rate; GRF downsampled from 1000Hz)


class CMJDataLoader:
    """
    Data loader for countermovement jump accelerometer and GRF data.

    Loads MATLAB files and preprocesses signals for the transformer model.

    Args:
        data_path: Path to processedjumpdata.mat file
        seq_len: Target sequence length (signals padded/truncated to this)
        use_resultant: If True, compute resultant acceleration; else use triaxial
        sensor_idx: Sensor index (0=lower back, 1=upper back, etc.) - 0-indexed
        grf_plate_idx: Force plate index (2=combined plates) - 0-indexed
    """

    def __init__(
        self,
        data_path: str = DEFAULT_DATA_PATH,
        seq_len: int = DEFAULT_SEQ_LEN,
        use_resultant: bool = True,
        sensor_idx: int = 0,
        grf_plate_idx: int = 2,
    ):
        self.data_path = data_path
        self.seq_len = seq_len
        self.use_resultant = use_resultant
        self.sensor_idx = sensor_idx
        self.grf_plate_idx = grf_plate_idx

        # Data storage
        self.acc_data = None
        self.grf_data = None
        self.subject_ids = None
        self.jump_indices = None

        # Normalization parameters
        self.acc_mean = None
        self.acc_std = None
        self.grf_mean = None
        self.grf_std = None

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

                # Extract GRF data (vertical component from specified plate)
                grf_signal = self._extract_grf_signal(grf_raw, subj_idx, jump_idx)
                if grf_signal is None:
                    continue

                # Get body weight for normalization
                body_weight = bwall[subj_idx, jump_idx]
                if body_weight <= 0:
                    continue

                # Truncate each signal at its respective takeoff index
                acc_takeoff = int(acc_takeoff)
                grf_takeoff = int(grf_takeoff)
                acc_signal = acc_signal[:acc_takeoff]
                grf_signal = grf_signal[:grf_takeoff]

                # Skip if signals are too short
                if len(acc_signal) < 100:
                    continue

                # Downsample GRF from 1000Hz to 250Hz to match ACC sampling rate
                # Take every 4th sample
                grf_signal = grf_signal[::4]

                # Normalize GRF by body weight (convert to BW units)
                grf_signal = grf_signal / body_weight

                acc_list.append(acc_signal)
                grf_list.append(grf_signal)
                subject_id_list.append(subj_idx)
                jump_idx_list.append(jump_idx)

        print(f"Extracted {len(acc_list)} valid jumps from {n_subjects} subjects")

        self.acc_data = acc_list
        self.grf_data = grf_list
        self.subject_ids = np.array(subject_id_list)
        self.jump_indices = np.array(jump_idx_list)

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

    def preprocess(
        self,
        acc_data: list[np.ndarray] = None,
        grf_data: list[np.ndarray] = None,
        fit_normalization: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess signals: pad/truncate and normalize.

        Args:
            acc_data: List of accelerometer signals (uses self.acc_data if None)
            grf_data: List of GRF signals (uses self.grf_data if None)
            fit_normalization: Whether to fit normalization parameters

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

        # Initialize arrays
        acc_array = np.zeros((n_samples, self.seq_len, input_dim), dtype=np.float32)
        grf_array = np.zeros((n_samples, self.seq_len, 1), dtype=np.float32)

        for i, (acc, grf) in enumerate(zip(acc_data, grf_data)):
            # Compute resultant if needed
            if self.use_resultant:
                acc_processed = np.sqrt(np.sum(acc ** 2, axis=1, keepdims=True))
            else:
                acc_processed = acc

            # Align signals: pad at start with initial value, truncate if too long
            acc_aligned = self._align_signal(acc_processed, self.seq_len)
            grf_aligned = self._align_signal(grf.reshape(-1, 1), self.seq_len)

            acc_array[i] = acc_aligned
            grf_array[i] = grf_aligned

        # Z-score normalization
        if fit_normalization:
            self.acc_mean = np.mean(acc_array)
            self.acc_std = np.std(acc_array)
            self.grf_mean = np.mean(grf_array)
            self.grf_std = np.std(grf_array)

        acc_normalized = (acc_array - self.acc_mean) / (self.acc_std + 1e-8)
        grf_normalized = (grf_array - self.grf_mean) / (self.grf_std + 1e-8)

        return acc_normalized, grf_normalized

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

        # Preprocess training data (fit normalization)
        X_train, y_train = self.preprocess(train_acc, train_grf, fit_normalization=True)

        # Preprocess validation data (use fitted normalization)
        X_val, y_val = self.preprocess(val_acc, val_grf, fit_normalization=False)

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        info = {
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'n_train_subjects': len(train_subjects),
            'n_val_subjects': len(val_subjects),
            'input_shape': X_train.shape[1:],
            'output_shape': y_train.shape[1:],
            'acc_mean': self.acc_mean,
            'acc_std': self.acc_std,
            'grf_mean': self.grf_mean,
            'grf_std': self.grf_std,
        }

        print(f"Train: {info['n_train_samples']} samples from {info['n_train_subjects']} subjects")
        print(f"Val: {info['n_val_samples']} samples from {info['n_val_subjects']} subjects")

        return train_dataset, val_dataset, info

    def denormalize_grf(self, grf_normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized GRF back to body weight units.

        Args:
            grf_normalized: Z-score normalized GRF

        Returns:
            GRF in body weight units
        """
        return grf_normalized * self.grf_std + self.grf_mean

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
