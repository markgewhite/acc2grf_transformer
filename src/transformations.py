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


def get_transformer(
    transform_type: str,
    n_basis: int = 30,
    n_components: int = 15,
    variance_threshold: float = 0.99,
    bspline_lambda: float = 1e-4,
    use_varimax: bool = True,
    fpc_smooth_lambda: float = None,
    fpc_n_basis_smooth: int = 50,
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
    """

    def __init__(
        self,
        n_components: int = 15,
        variance_threshold: float = None,
        use_varimax: bool = True,
        varimax_max_iter: int = 100,
        smooth_lambda: float = None,
        n_basis_smooth: int = 50,
    ):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.use_varimax = use_varimax
        self.varimax_max_iter = varimax_max_iter
        self.smooth_lambda = smooth_lambda
        self.n_basis_smooth = n_basis_smooth

        # Fitted parameters (per channel)
        self._seq_len = None
        self._n_channels = None
        self._time_points = None
        self._fpca_objects = None  # List of fitted FPCA objects per channel
        self._actual_n_components = None  # List of int per channel
        self._rotation_matrices = None  # List of rotation matrices if varimax applied
        self._cumulative_variance = None  # List of cumulative variance explained

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

            scores[:, :n_comp_ch, ch] = ch_scores

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

        # Reconstruct each channel using scikit-fda
        signals = np.zeros((n_samples, self._seq_len, self._n_channels))

        for ch in range(self._n_channels):
            fpca = self._fpca_objects[ch]
            n_comp_ch = self._actual_n_components[ch]
            n_comp_fpca = fpca.n_components  # Total components in FPCA object

            # Extract scores for this channel
            ch_scores = scores[:, :n_comp_ch, ch]  # (n_samples, n_comp_ch)

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

    def get_eigenfunctions(self) -> list[np.ndarray]:
        """
        Get eigenfunctions (principal components) for each channel.

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
            eigenfunctions.append(eigenfuncs)

        return eigenfunctions

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
