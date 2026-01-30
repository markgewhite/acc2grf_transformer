"""
Signal Transformations for Functional Data Analysis

Provides transformations between raw signals and functional representations
(B-splines, Functional PCA) for input/output compression and smoothing.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from scipy import linalg

# scikit-fda imports for functional data analysis
from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis, FDataBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.dim_reduction import FPCA


class BaseSignalTransformer(ABC):
    """
    Abstract base class for signal transformations.

    All transformers follow the same interface to allow polymorphic use
    in the data pipeline.
    """

    @abstractmethod
    def fit(self, signals: np.ndarray) -> 'BaseSignalTransformer':
        """
        Fit the transformer to training data.

        Args:
            signals: Training signals of shape (n_samples, seq_len, n_channels)

        Returns:
            self
        """
        pass

    @abstractmethod
    def transform(self, signals: np.ndarray) -> np.ndarray:
        """
        Transform signals to coefficient representation.

        Args:
            signals: Signals of shape (n_samples, seq_len, n_channels)

        Returns:
            Coefficients of shape (n_samples, n_features, n_channels)
        """
        pass

    @abstractmethod
    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Reconstruct signals from coefficients.

        Args:
            coefficients: Coefficients of shape (n_samples, n_features, n_channels)

        Returns:
            Reconstructed signals of shape (n_samples, seq_len, n_channels)
        """
        pass

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of features in the transformed representation."""
        pass

    def fit_transform(self, signals: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(signals).transform(signals)


class IdentityTransformer(BaseSignalTransformer):
    """
    Identity transformer - passes signals through unchanged.

    Used for 'raw' input/output mode to maintain consistent interface.
    """

    def __init__(self):
        self._seq_len = None
        self._n_channels = None

    def fit(self, signals: np.ndarray) -> 'IdentityTransformer':
        """Store signal dimensions."""
        self._seq_len = signals.shape[1]
        self._n_channels = signals.shape[2]
        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """Return signals unchanged."""
        return signals

    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        """Return coefficients unchanged (they are the signals)."""
        return coefficients

    @property
    def n_features(self) -> int:
        """Returns sequence length since signals pass through unchanged."""
        return self._seq_len


class BSplineTransformer(BaseSignalTransformer):
    """
    B-spline basis representation with smoothing using scikit-fda.

    Represents signals as linear combinations of B-spline basis functions,
    with optional roughness penalty for smoothing.

    Args:
        n_basis: Number of B-spline basis functions (default: 30)
        degree: B-spline degree (default: 3, cubic splines)
        smoothing_lambda: Roughness penalty weight (default: 1e-4)
            Higher values produce smoother fits.
    """

    def __init__(
        self,
        n_basis: int = 30,
        degree: int = 3,
        smoothing_lambda: float = 1e-4,
    ):
        self.n_basis = n_basis
        self.degree = degree
        self.smoothing_lambda = smoothing_lambda

        # Fitted parameters
        self._seq_len = None
        self._n_channels = None
        self._time_points = None
        self._basis = None  # scikit-fda BSplineBasis

    def fit(self, signals: np.ndarray) -> 'BSplineTransformer':
        """
        Fit B-spline basis to training signals.

        Args:
            signals: Training signals of shape (n_samples, seq_len, n_channels)

        Returns:
            self
        """
        n_samples, seq_len, n_channels = signals.shape
        self._seq_len = seq_len
        self._n_channels = n_channels

        # Create time points (normalized to [0, 1])
        self._time_points = np.linspace(0, 1, seq_len)

        # Create B-spline basis using scikit-fda (order = degree + 1)
        self._basis = BSplineBasis(
            domain_range=(0, 1),
            n_basis=self.n_basis,
            order=self.degree + 1
        )

        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """
        Transform signals to B-spline coefficients.

        Args:
            signals: Signals of shape (n_samples, seq_len, n_channels)

        Returns:
            Coefficients of shape (n_samples, n_basis, n_channels)
        """
        if self._basis is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        n_samples = signals.shape[0]
        coefficients = np.zeros((n_samples, self.n_basis, self._n_channels))

        for ch in range(self._n_channels):
            # Create FDataGrid for this channel
            fd = FDataGrid(signals[:, :, ch], self._time_points)

            # Apply basis smoothing with regularization
            smoother = BasisSmoother(
                basis=self._basis,
                smoothing_parameter=self.smoothing_lambda,
                return_basis=True
            )
            fd_basis = smoother.fit_transform(fd)

            # Extract coefficients
            coefficients[:, :, ch] = fd_basis.coefficients

        return coefficients

    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Reconstruct signals from B-spline coefficients.

        Args:
            coefficients: Coefficients of shape (n_samples, n_basis, n_channels)

        Returns:
            Reconstructed signals of shape (n_samples, seq_len, n_channels)
        """
        if self._basis is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        n_samples = coefficients.shape[0]
        signals = np.zeros((n_samples, self._seq_len, self._n_channels))

        for ch in range(self._n_channels):
            # Create FDataBasis from coefficients
            fd_basis = FDataBasis(self._basis, coefficients[:, :, ch])

            # Evaluate at time points
            evaluated = fd_basis(self._time_points)
            # Handle both FDataGrid and numpy array returns
            if hasattr(evaluated, 'data_matrix'):
                signals[:, :, ch] = evaluated.data_matrix.squeeze(axis=-1)
            else:
                # Direct numpy array return (newer scikit-fda versions)
                signals[:, :, ch] = np.asarray(evaluated).squeeze(axis=-1)

        return signals

    @property
    def n_features(self) -> int:
        """Number of B-spline basis functions."""
        return self.n_basis

    def get_reconstruction_error(self, signals: np.ndarray) -> dict:
        """
        Compute reconstruction error statistics.

        Args:
            signals: Original signals of shape (n_samples, seq_len, n_channels)

        Returns:
            Dictionary with RMSE, MAE, and max error
        """
        coefficients = self.transform(signals)
        reconstructed = self.inverse_transform(coefficients)

        errors = signals - reconstructed

        return {
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'mae': np.mean(np.abs(errors)),
            'max_error': np.max(np.abs(errors)),
        }

    def get_reconstruction_components(self) -> dict:
        """
        Get components needed for signal-space loss computation.

        Returns:
            Dictionary with:
                - reconstruction_matrix: (seq_len, n_basis) basis evaluation matrix
                - mean_function: None (B-splines don't use mean centering)
        """
        if self._basis is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        # Evaluate basis functions at all time points
        # This gives us the matrix B where signal = B @ coefficients
        from skfda.representation.basis import FDataBasis
        import numpy as np

        # Create identity coefficients to evaluate each basis function
        identity = np.eye(self.n_basis)
        fd_basis = FDataBasis(self._basis, identity)

        # Evaluate at time points: result is (n_basis, seq_len, 1)
        evaluated = fd_basis(self._time_points)
        if hasattr(evaluated, 'data_matrix'):
            basis_matrix = evaluated.data_matrix.squeeze(axis=-1)  # (n_basis, seq_len)
        else:
            basis_matrix = np.asarray(evaluated).squeeze(axis=-1)

        # Transpose to (seq_len, n_basis) for reconstruction: signal = B @ coeffs
        basis_matrix = basis_matrix.T

        return {
            'reconstruction_matrix': basis_matrix,
            'mean_function': None,
        }


def get_transformer(
    transform_type: str,
    n_basis: int = 30,
    n_components: int = 15,
    variance_threshold: float = 0.99,
    bspline_lambda: float = 1e-4,
    use_varimax: bool = True,
    fpc_smooth_lambda: float = None,
    fpc_n_basis_smooth: int = 50,
    standardize_scores: bool = False,
    score_scale: float = 1.0,
) -> BaseSignalTransformer:
    """
    Factory function to create signal transformers.

    Args:
        transform_type: One of 'raw', 'bspline', 'fpc'
        n_basis: Number of B-spline basis functions
        n_components: Number of FPC components (if not using variance threshold)
        variance_threshold: Cumulative variance threshold for automatic FPC selection
        bspline_lambda: B-spline smoothing penalty
        use_varimax: Whether to apply varimax rotation to FPCs
        fpc_smooth_lambda: Pre-FPCA smoothing parameter (None = no smoothing)
        fpc_n_basis_smooth: Number of basis functions for pre-FPCA smoothing
        score_scale: Scale factor for FPC scores (1.0 = no scaling, 'auto' = sqrt(n_timepoints))

    Returns:
        Configured transformer instance
    """
    if transform_type == 'raw':
        return IdentityTransformer()
    elif transform_type == 'bspline':
        return BSplineTransformer(
            n_basis=n_basis,
            smoothing_lambda=bspline_lambda,
        )
    elif transform_type == 'fpc':
        return FPCATransformer(
            n_components=n_components,
            variance_threshold=variance_threshold,
            use_varimax=use_varimax,
            smooth_lambda=fpc_smooth_lambda,
            n_basis_smooth=fpc_n_basis_smooth,
            standardize_scores=standardize_scores,
            score_scale=score_scale,
        )
    else:
        raise ValueError(f"Unknown transform type: {transform_type}. "
                        f"Choose from: raw, bspline, fpc")


def varimax_rotation(
    loadings: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply varimax rotation to redistribute variance across components.

    Varimax rotation is an orthogonal rotation that maximizes the variance
    of the squared loadings within each factor, making each factor more
    interpretable by having high loadings on few variables.

    Args:
        loadings: Loading matrix of shape (n_variables, n_factors)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (rotated_loadings, rotation_matrix)
    """
    n_vars, n_factors = loadings.shape

    # Initialize rotation matrix as identity
    rotation = np.eye(n_factors)
    rotated = loadings.copy()

    for iteration in range(max_iter):
        old_rotated = rotated.copy()

        for i in range(n_factors - 1):
            for j in range(i + 1, n_factors):
                # Extract the two columns
                x = rotated[:, i]
                y = rotated[:, j]

                # Compute rotation angle that maximizes varimax criterion
                u = x ** 2 - y ** 2
                v = 2 * x * y
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u ** 2 - v ** 2)
                D = 2 * np.sum(u * v)

                # Optimal angle
                num = D - 2 * A * B / n_vars
                den = C - (A ** 2 - B ** 2) / n_vars

                phi = 0.25 * np.arctan2(num, den)

                # Apply rotation
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)

                rotated[:, i] = x * cos_phi + y * sin_phi
                rotated[:, j] = -x * sin_phi + y * cos_phi

                # Update rotation matrix
                rot_ij = np.eye(n_factors)
                rot_ij[i, i] = cos_phi
                rot_ij[j, j] = cos_phi
                rot_ij[i, j] = sin_phi
                rot_ij[j, i] = -sin_phi
                rotation = rotation @ rot_ij

        # Check convergence
        diff = np.max(np.abs(rotated - old_rotated))
        if diff < tol:
            break

    return rotated, rotation


class FPCATransformer(BaseSignalTransformer):
    """
    Functional PCA transformer with optional varimax rotation and pre-smoothing.

    Performs PCA on functional data to extract principal modes of variation
    (eigenfunctions). Signals are represented as linear combinations of
    these eigenfunctions via their scores.

    Uses scikit-fda for FPCA computation with optional pre-smoothing
    (Ramsay & Silverman approach).

    Args:
        n_components: Number of components to retain (if variance_threshold is None)
        variance_threshold: Cumulative variance threshold for automatic selection
            If provided, overrides n_components
        use_varimax: Whether to apply varimax rotation to redistribute variance
        varimax_max_iter: Maximum iterations for varimax algorithm
        smooth_lambda: Smoothing parameter for pre-FPCA B-spline smoothing (default: None)
            If None, no pre-smoothing is applied
        n_basis_smooth: Number of basis functions for pre-FPCA smoothing (default: 50)
        score_scale: Scale factor for FPC scores (default: 1.0)
            Set to 'auto' or sqrt(n_timepoints) to match discrete normalization convention.
            scikit-fda uses L² normalization which produces scores ~sqrt(N) smaller than
            discrete normalization. Scaling by sqrt(N) recovers the original magnitude.
    """

    def __init__(
        self,
        n_components: int = 15,
        variance_threshold: float = None,
        use_varimax: bool = True,
        varimax_max_iter: int = 100,
        smooth_lambda: float = None,
        n_basis_smooth: int = 50,
        standardize_scores: bool = False,
        score_scale: float = 1.0,
    ):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.use_varimax = use_varimax
        self.varimax_max_iter = varimax_max_iter
        self.smooth_lambda = smooth_lambda
        self.n_basis_smooth = n_basis_smooth
        self.standardize_scores = standardize_scores
        self.score_scale = score_scale

        # Fitted parameters (per channel)
        self._seq_len = None
        self._n_channels = None
        self._time_points = None
        self._fpca_objects = None  # List of fitted FPCA objects per channel
        self._actual_n_components = None  # List of int per channel
        self._rotation_matrices = None  # List of rotation matrices if varimax applied
        self._cumulative_variance = None  # List of cumulative variance explained
        self._eigenvalues = None  # List of eigenvalues (variance explained) per channel
        self._score_std = None  # List of score std per channel (for standardization)
        self._effective_score_scale = None  # Computed during fit if 'auto'

    def _smooth_signals(self, signals: np.ndarray) -> np.ndarray:
        """
        Apply B-spline smoothing to signals before FPCA (Ramsay-Silverman approach).

        Args:
            signals: Signals of shape (n_samples, seq_len, n_channels)

        Returns:
            Smoothed signals of same shape
        """
        basis = BSplineBasis(
            domain_range=(0, 1),
            n_basis=self.n_basis_smooth,
            order=4  # Cubic splines
        )
        smoothed = np.zeros_like(signals)

        for ch in range(signals.shape[2]):
            fd = FDataGrid(signals[:, :, ch], self._time_points)
            smoother = BasisSmoother(
                basis=basis,
                smoothing_parameter=self.smooth_lambda
            )
            fd_smooth = smoother.fit_transform(fd)
            smoothed[:, :, ch] = fd_smooth.data_matrix.squeeze(axis=-1)

        return smoothed

    def fit(self, signals: np.ndarray) -> 'FPCATransformer':
        """
        Fit FPCA to training signals using scikit-fda.

        Args:
            signals: Training signals of shape (n_samples, seq_len, n_channels)

        Returns:
            self
        """
        n_samples, seq_len, n_channels = signals.shape
        self._seq_len = seq_len
        self._n_channels = n_channels
        self._time_points = np.linspace(0, 1, seq_len)

        # Apply pre-smoothing if requested (Ramsay-Silverman approach)
        if self.smooth_lambda is not None:
            signals = self._smooth_signals(signals)

        # Fit FPCA for each channel separately using scikit-fda
        self._fpca_objects = []
        self._actual_n_components = []
        self._rotation_matrices = []
        self._cumulative_variance = []
        self._eigenvalues = []

        for ch in range(n_channels):
            # Create FDataGrid for this channel
            fd = FDataGrid(signals[:, :, ch], self._time_points)

            # Determine max components to extract
            n_comp_max = min(self.n_components, n_samples - 1, seq_len)
            if self.variance_threshold is not None:
                # Extract more components to allow variance-based selection
                n_comp_max = min(n_samples - 1, seq_len)

            # Fit scikit-fda FPCA (centering=True handles mean internally)
            fpca = FPCA(n_components=n_comp_max, centering=True)
            fpca.fit(fd)

            # Get explained variance ratio
            explained_var_ratio = fpca.explained_variance_ratio_
            cum_var = np.cumsum(explained_var_ratio)

            # Determine number of components to retain
            if self.variance_threshold is not None:
                n_comp = np.searchsorted(cum_var, self.variance_threshold) + 1
                n_comp = min(n_comp, n_comp_max)
            else:
                n_comp = min(self.n_components, n_comp_max)

            cum_var = cum_var[:n_comp]

            # Handle varimax rotation (applied to scores post-transform)
            rotation_matrix = None
            if self.use_varimax and n_comp > 1:
                # Get scores for varimax computation
                scores = fpca.transform(fd)[:, :n_comp]
                # Compute loadings from eigenfunctions for varimax
                eigenfuncs = fpca.components_.data_matrix[:n_comp, :, 0].T
                eigenvalues = fpca.explained_variance_[:n_comp]
                loadings = eigenfuncs * np.sqrt(eigenvalues)
                _, rotation_matrix = varimax_rotation(
                    loadings, max_iter=self.varimax_max_iter
                )

            self._fpca_objects.append(fpca)
            self._actual_n_components.append(n_comp)
            self._rotation_matrices.append(rotation_matrix)
            self._cumulative_variance.append(cum_var)
            # Store individual eigenvalues (variance explained per component)
            self._eigenvalues.append(fpca.explained_variance_ratio_[:n_comp])

        # Compute score std for standardization (if enabled)
        if self.standardize_scores:
            self._score_std = []
            for ch in range(n_channels):
                fpca = self._fpca_objects[ch]
                n_comp_ch = self._actual_n_components[ch]
                fd = FDataGrid(signals[:, :, ch], self._time_points)
                ch_scores = fpca.transform(fd)[:, :n_comp_ch]
                if self._rotation_matrices[ch] is not None:
                    ch_scores = ch_scores @ self._rotation_matrices[ch]
                # Compute std per component, with floor to avoid division by zero
                score_std = np.std(ch_scores, axis=0)
                score_std = np.maximum(score_std, 1e-8)
                self._score_std.append(score_std)

        # Compute effective score scale
        # scikit-fda uses L² normalization, which produces scores ~sqrt(N) smaller
        # than discrete normalization. Setting score_scale='auto' or sqrt(N) recovers
        # the discrete normalization magnitude.
        if self.score_scale == 'auto':
            self._effective_score_scale = np.sqrt(seq_len)
            print(f"  Auto score_scale: sqrt({seq_len}) = {self._effective_score_scale:.2f}")
        else:
            self._effective_score_scale = float(self.score_scale)
            if self._effective_score_scale != 1.0:
                print(f"  Score scale: {self._effective_score_scale:.2f}")

        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """
        Transform signals to FPC scores using scikit-fda.

        Args:
            signals: Signals of shape (n_samples, seq_len, n_channels)

        Returns:
            Scores of shape (n_samples, max_n_components, n_channels)
        """
        if self._fpca_objects is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        n_samples, seq_len, n_channels = signals.shape

        # Get maximum number of components across channels
        max_n_comp = max(self._actual_n_components)

        # Transform each channel using scikit-fda
        scores = np.zeros((n_samples, max_n_comp, n_channels))

        for ch in range(n_channels):
            fpca = self._fpca_objects[ch]
            n_comp_ch = self._actual_n_components[ch]

            # Create FDataGrid for this channel
            fd = FDataGrid(signals[:, :, ch], self._time_points)

            # Transform using scikit-fda (returns numpy array)
            ch_scores = fpca.transform(fd)[:, :n_comp_ch]  # (n_samples, n_comp_ch)

            # Apply varimax rotation if fitted
            if self._rotation_matrices[ch] is not None:
                ch_scores = ch_scores @ self._rotation_matrices[ch]

            # Standardize scores if enabled
            if self.standardize_scores and self._score_std is not None:
                ch_scores = ch_scores / self._score_std[ch]

            scores[:, :n_comp_ch, ch] = ch_scores

        # Apply score scaling to match discrete normalization convention
        if self._effective_score_scale != 1.0:
            scores = scores * self._effective_score_scale

        return scores

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Reconstruct signals from FPC scores using scikit-fda.

        Args:
            scores: Scores of shape (n_samples, n_components, n_channels)

        Returns:
            Reconstructed signals of shape (n_samples, seq_len, n_channels)
        """
        if self._fpca_objects is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        n_samples = scores.shape[0]

        # Reverse score scaling first (before any other operations)
        if self._effective_score_scale != 1.0:
            scores = scores / self._effective_score_scale

        # Reconstruct each channel using scikit-fda
        signals = np.zeros((n_samples, self._seq_len, self._n_channels))

        for ch in range(self._n_channels):
            fpca = self._fpca_objects[ch]
            n_comp_ch = self._actual_n_components[ch]
            n_comp_fpca = fpca.n_components  # Total components in FPCA object

            # Extract scores for this channel
            ch_scores = scores[:, :n_comp_ch, ch]  # (n_samples, n_comp_ch)

            # Reverse standardization if applied
            if self.standardize_scores and self._score_std is not None:
                ch_scores = ch_scores * self._score_std[ch]

            # Reverse varimax rotation if applied
            if self._rotation_matrices[ch] is not None:
                # Rotation matrix is orthogonal, so inverse = transpose
                ch_scores = ch_scores @ self._rotation_matrices[ch].T

            # Pad scores to match FPCA's n_components if needed
            if n_comp_ch < n_comp_fpca:
                padded_scores = np.zeros((n_samples, n_comp_fpca))
                padded_scores[:, :n_comp_ch] = ch_scores
                ch_scores = padded_scores

            # Inverse transform using scikit-fda (returns FDataGrid)
            fd_reconstructed = fpca.inverse_transform(ch_scores)
            signals[:, :, ch] = fd_reconstructed.data_matrix.squeeze(axis=-1)

        return signals

    @property
    def n_features(self) -> int:
        """Number of FPC components (maximum across channels)."""
        if self._actual_n_components is not None:
            return max(self._actual_n_components)
        return self.n_components

    def get_variance_explained(self) -> list[np.ndarray]:
        """
        Get cumulative variance explained for each channel.

        Returns:
            List of cumulative variance arrays, one per channel
        """
        if self._cumulative_variance is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")
        return self._cumulative_variance

    def get_eigenvalues(self) -> np.ndarray:
        """
        Get eigenvalues (variance explained ratio) for each component.

        Returns:
            Array of shape (n_components, n_channels) with variance explained
            by each component. For single-channel data, shape is (n_components, 1).
        """
        if self._eigenvalues is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        # Stack eigenvalues from all channels
        # Each channel may have different number of components, so pad to max
        max_comp = max(len(ev) for ev in self._eigenvalues)
        n_channels = len(self._eigenvalues)

        result = np.zeros((max_comp, n_channels), dtype=np.float32)
        for ch, ev in enumerate(self._eigenvalues):
            result[:len(ev), ch] = ev

        return result

    def get_eigenfunctions(self, rotated: bool = True) -> list[np.ndarray]:
        """
        Get eigenfunctions (principal components) for each channel.

        Args:
            rotated: If True and varimax was applied, return rotated eigenfunctions
                that correspond to the rotated scores. Default True.

        Returns:
            List of eigenfunction arrays of shape (seq_len, n_components)
        """
        if self._fpca_objects is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        eigenfunctions = []
        for ch, fpca in enumerate(self._fpca_objects):
            n_comp = self._actual_n_components[ch]
            # Extract eigenfunctions from scikit-fda components
            # components_.data_matrix has shape (n_components, n_points, 1)
            eigenfuncs = fpca.components_.data_matrix[:n_comp, :, 0].T  # (seq_len, n_comp)

            # Apply varimax rotation to eigenfunctions if requested
            # This ensures eigenfunctions match the rotated scores
            if rotated and self._rotation_matrices[ch] is not None:
                eigenfuncs = eigenfuncs @ self._rotation_matrices[ch]

            eigenfunctions.append(eigenfuncs)

        return eigenfunctions

    def get_inverse_transform_components(self) -> dict:
        """
        Get all components needed for TensorFlow-based inverse transform.

        Returns a dictionary with:
        - eigenfunctions: list of arrays (seq_len, n_components) per channel
        - mean_functions: list of arrays (seq_len,) per channel
        - rotation_matrices: list of arrays or None per channel (for varimax)
        - score_std: list of arrays or None per channel (for standardization)
        - n_components: list of int per channel

        These can be used to implement inverse FPCA in TensorFlow for
        gradient-based optimization.
        """
        if self._fpca_objects is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        eigenfunctions = []
        mean_functions = []

        for ch, fpca in enumerate(self._fpca_objects):
            n_comp = self._actual_n_components[ch]

            # Extract eigenfunctions
            eigenfuncs = fpca.components_.data_matrix[:n_comp, :, 0].T  # (seq_len, n_comp)
            eigenfunctions.append(eigenfuncs.astype(np.float32))

            # Extract mean function (the function subtracted during centering)
            mean_func = fpca.mean_.data_matrix[0, :, 0]  # (seq_len,)
            mean_functions.append(mean_func.astype(np.float32))

        return {
            'eigenfunctions': eigenfunctions,
            'mean_functions': mean_functions,
            'rotation_matrices': self._rotation_matrices,
            'score_std': self._score_std if self.standardize_scores else None,
            'n_components': self._actual_n_components,
            'seq_len': self._seq_len,
            'n_channels': self._n_channels,
        }

    def get_reconstruction_error(self, signals: np.ndarray) -> dict:
        """
        Compute reconstruction error statistics.

        Args:
            signals: Original signals of shape (n_samples, seq_len, n_channels)

        Returns:
            Dictionary with RMSE, MAE, max error, and variance explained
        """
        scores = self.transform(signals)
        reconstructed = self.inverse_transform(scores)

        errors = signals - reconstructed

        return {
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'mae': np.mean(np.abs(errors)),
            'max_error': np.max(np.abs(errors)),
            'variance_explained': [cv[-1] if len(cv) > 0 else 0 for cv in self._cumulative_variance],
        }

    def get_time_points(self) -> np.ndarray:
        """Get the time points used for functional data representation."""
        if self._time_points is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")
        return self._time_points

    def get_reconstruction_components(self) -> dict:
        """
        Get components needed for signal-space loss computation.

        Returns components in a simplified format for the ReconstructionLoss class.
        Only supports single-channel data (typical for GRF).

        Returns:
            Dictionary with:
                - reconstruction_matrix: (seq_len, n_components) eigenfunctions
                - mean_function: (seq_len,) mean function
        """
        if self._fpca_objects is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        if self._n_channels != 1:
            raise ValueError("get_reconstruction_components only supports single-channel data")

        fpca = self._fpca_objects[0]
        n_comp = self._actual_n_components[0]

        # Eigenfunctions: (seq_len, n_components)
        eigenfuncs = fpca.components_.data_matrix[:n_comp, :, 0].T

        # Mean function: (seq_len,)
        mean_func = fpca.mean_.data_matrix[0, :, 0]

        return {
            'reconstruction_matrix': eigenfuncs.astype(np.float32),
            'mean_function': mean_func.astype(np.float32),
        }


def learn_fpc_projection_matrix(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Learn projection matrix from ACC FPC scores to GRF FPC scores using Ridge regression.

    This is the practical approach when eigenfunction inner products don't work
    (which happens when input and output signals have fundamentally different
    temporal characteristics).

    Args:
        X_train: Training input scores of shape (n_samples, n_input_features)
            For triaxial ACC with 15 FPCs: shape is (n_samples, 45)
        y_train: Training output scores of shape (n_samples, n_output_features)
            For GRF with 15 FPCs: shape is (n_samples, 15)
        alpha: Ridge regularization strength (default: 1.0)

    Returns:
        P: Learned projection matrix of shape (n_input_features, n_output_features)

    Example:
        Predicted GRF scores = X @ P
    """
    from sklearn.linear_model import Ridge

    # Flatten if needed
    if X_train.ndim == 3:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if y_train.ndim == 3:
        y_train = y_train.reshape(y_train.shape[0], -1)

    # Learn projection via Ridge regression
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X_train, y_train)

    # Return coefficients as projection matrix
    return model.coef_.T.astype(np.float32)  # Shape: (n_input, n_output)


def compute_fpc_projection_matrix(
    input_transformer: FPCATransformer,
    output_transformer: FPCATransformer,
    time_points: np.ndarray = None,
) -> tuple[np.ndarray, float]:
    """
    Compute functional projection matrix P from input (ACC) FPCs to output (GRF) FPCs.

    This implements the MATLAB approach where projection coefficients are computed
    as the inner product (overlap integral) between input and output eigenfunctions:

        P(i,j) = ∫[φ_input_i(t) × φ_output_j(t)] dt / ∫[φ_output_j(t)²] dt

    For orthonormal eigenfunctions (L² normalized), the denominator is 1.

    The rescale factor accounts for magnitude differences between input and output
    mean functions:

        rescale = sqrt(∫(mean_output)² dt / ∫(mean_input)² dt)

    Args:
        input_transformer: Fitted FPCATransformer for input signals (e.g., ACC)
        output_transformer: Fitted FPCATransformer for output signals (e.g., GRF)
        time_points: Optional time points for integration. If None, uses the
            time points from output_transformer.

    Returns:
        P: Projection matrix of shape (n_input_features, n_output_components)
            For triaxial ACC (3 channels × 15 FPCs) → single GRF (15 FPCs),
            P has shape (45, 15). Each column j represents how input FPCs
            project onto output FPC j.
        rescale: Magnitude rescaling factor (scalar)

    Example:
        Predicted GRF scores = rescale * (ACC_scores @ P)

    Note:
        The projection matrix provides a physics-informed linear baseline.
        Rows are ordered by channel: for 3-channel input with 15 FPCs each,
        rows 0-14 are channel 0, rows 15-29 are channel 1, rows 30-44 are channel 2.
    """
    # Validate transformers are fitted
    if input_transformer._fpca_objects is None:
        raise RuntimeError("Input transformer not fitted")
    if output_transformer._fpca_objects is None:
        raise RuntimeError("Output transformer not fitted")

    # Get time points for integration
    if time_points is None:
        time_points = output_transformer.get_time_points()

    # Get eigenfunctions from both transformers
    # Returns list of arrays, each (seq_len, n_components)
    input_eigenfuncs = input_transformer.get_eigenfunctions()
    output_eigenfuncs = output_transformer.get_eigenfunctions()

    # Get mean functions for rescaling
    input_components = input_transformer.get_inverse_transform_components()
    output_components = output_transformer.get_inverse_transform_components()
    input_means = input_components['mean_functions']  # List of (seq_len,) per channel
    output_means = output_components['mean_functions']

    # Determine dimensions
    n_input_channels = len(input_eigenfuncs)
    n_output_channels = len(output_eigenfuncs)
    n_input_components_per_channel = [ef.shape[1] for ef in input_eigenfuncs]
    n_output_components = output_eigenfuncs[0].shape[1]  # Assume single output channel

    # Total input features (flattened across channels)
    total_input_features = sum(n_input_components_per_channel)

    # Initialize projection matrix
    # Shape: (total_input_features, n_output_components)
    P = np.zeros((total_input_features, n_output_components), dtype=np.float32)

    # Compute inner products via trapezoidal integration
    # For each output FPC j, compute how each input FPC i projects onto it
    row_offset = 0
    for ch in range(n_input_channels):
        n_comp_ch = n_input_components_per_channel[ch]
        input_ef = input_eigenfuncs[ch]  # (seq_len, n_comp_ch)

        for out_ch in range(n_output_channels):
            output_ef = output_eigenfuncs[out_ch]  # (seq_len, n_output_components)

            for i in range(n_comp_ch):
                for j in range(n_output_components):
                    # Compute inner product: ∫[φ_in_i(t) × φ_out_j(t)] dt
                    inner_product = np.trapz(
                        input_ef[:, i] * output_ef[:, j],
                        time_points
                    )

                    # For L²-normalized eigenfunctions, denominator is 1
                    # But compute it anyway for robustness
                    norm_sq = np.trapz(output_ef[:, j] ** 2, time_points)

                    if norm_sq > 1e-10:
                        P[row_offset + i, j] = inner_product / norm_sq
                    else:
                        P[row_offset + i, j] = inner_product

        row_offset += n_comp_ch

    # Compute rescale factor from mean functions
    # rescale = sqrt(∫(mean_output)² / ∫(mean_input)²)
    # For multi-channel input, use the combined magnitude
    output_mean_sq = 0.0
    for mean_func in output_means:
        output_mean_sq += np.trapz(mean_func ** 2, time_points)

    input_mean_sq = 0.0
    for mean_func in input_means:
        input_mean_sq += np.trapz(mean_func ** 2, time_points)

    if input_mean_sq > 1e-10:
        rescale = np.sqrt(output_mean_sq / input_mean_sq)
    else:
        rescale = 1.0

    return P, float(rescale)
