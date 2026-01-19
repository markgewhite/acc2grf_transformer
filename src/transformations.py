"""
Signal Transformations for Functional Data Analysis

Provides transformations between raw signals and functional representations
(B-splines, Functional PCA) for input/output compression and smoothing.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from scipy import interpolate
from scipy import linalg


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
    B-spline basis representation with smoothing.

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
        self._knots = None
        self._basis_matrix = None  # B: (seq_len, n_basis)
        self._penalty_matrix = None  # P: (n_basis, n_basis) roughness penalty
        self._solve_matrix = None  # (B'B + lambda*P)^-1 @ B' for fitting

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

        # Create knot sequence for B-splines
        # Interior knots: n_basis - degree - 1 knots equally spaced
        n_interior_knots = self.n_basis - self.degree - 1
        if n_interior_knots < 0:
            raise ValueError(f"n_basis ({self.n_basis}) must be > degree ({self.degree})")

        interior_knots = np.linspace(0, 1, n_interior_knots + 2)[1:-1]

        # Add boundary knots (clamped splines)
        self._knots = np.concatenate([
            np.zeros(self.degree + 1),
            interior_knots,
            np.ones(self.degree + 1)
        ])

        # Build basis matrix B: evaluate each basis function at each time point
        self._basis_matrix = self._build_basis_matrix()

        # Build roughness penalty matrix P: integral of (B'')^2
        self._penalty_matrix = self._build_penalty_matrix()

        # Precompute solve matrix for efficient fitting
        # c = argmin ||Bc - y||^2 + lambda * c'Pc
        # Solution: c = (B'B + lambda*P)^-1 @ B' @ y
        BtB = self._basis_matrix.T @ self._basis_matrix
        regularized = BtB + self.smoothing_lambda * self._penalty_matrix
        self._solve_matrix = linalg.solve(regularized, self._basis_matrix.T)

        return self

    def _build_basis_matrix(self) -> np.ndarray:
        """Build B-spline basis matrix (seq_len, n_basis)."""
        basis = np.zeros((self._seq_len, self.n_basis))

        for i in range(self.n_basis):
            # Create unit coefficient vector for basis function i
            coeffs = np.zeros(self.n_basis)
            coeffs[i] = 1.0

            # Create B-spline and evaluate
            spline = interpolate.BSpline(self._knots, coeffs, self.degree)
            basis[:, i] = spline(self._time_points)

        return basis

    def _build_penalty_matrix(self) -> np.ndarray:
        """
        Build roughness penalty matrix via second derivative integral.

        P[i,j] = integral of B_i''(t) * B_j''(t) dt

        Uses numerical integration over fine grid.
        """
        # Evaluate second derivatives on fine grid
        n_eval = 1000
        t_fine = np.linspace(0, 1, n_eval)

        # Build second derivative basis matrix
        basis_d2 = np.zeros((n_eval, self.n_basis))
        for i in range(self.n_basis):
            coeffs = np.zeros(self.n_basis)
            coeffs[i] = 1.0
            spline = interpolate.BSpline(self._knots, coeffs, self.degree)
            # Second derivative
            spline_d2 = spline.derivative(2)
            basis_d2[:, i] = spline_d2(t_fine)

        # Penalty matrix: integral of B''_i * B''_j
        # Numerical integration via trapezoidal rule
        dt = 1.0 / (n_eval - 1)
        penalty = basis_d2.T @ basis_d2 * dt

        return penalty

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """
        Transform signals to B-spline coefficients.

        Args:
            signals: Signals of shape (n_samples, seq_len, n_channels)

        Returns:
            Coefficients of shape (n_samples, n_basis, n_channels)
        """
        if self._solve_matrix is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        n_samples, seq_len, n_channels = signals.shape

        # Solve for coefficients for each sample and channel
        coefficients = np.zeros((n_samples, self.n_basis, n_channels))

        for i in range(n_samples):
            for j in range(n_channels):
                # c = solve_matrix @ y
                coefficients[i, :, j] = self._solve_matrix @ signals[i, :, j]

        return coefficients

    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Reconstruct signals from B-spline coefficients.

        Args:
            coefficients: Coefficients of shape (n_samples, n_basis, n_channels)

        Returns:
            Reconstructed signals of shape (n_samples, seq_len, n_channels)
        """
        if self._basis_matrix is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        n_samples, _, n_channels = coefficients.shape

        # Reconstruct: y = B @ c
        signals = np.zeros((n_samples, self._seq_len, n_channels))

        for i in range(n_samples):
            for j in range(n_channels):
                signals[i, :, j] = self._basis_matrix @ coefficients[i, :, j]

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
        # FPCATransformer will be added in Phase 3
        return FPCATransformer(
            n_components=n_components,
            variance_threshold=variance_threshold,
            use_varimax=use_varimax,
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
    Functional PCA transformer with optional varimax rotation.

    Performs PCA on functional data to extract principal modes of variation
    (eigenfunctions). Signals are represented as linear combinations of
    these eigenfunctions via their scores.

    Args:
        n_components: Number of components to retain (if variance_threshold is None)
        variance_threshold: Cumulative variance threshold for automatic selection
            If provided, overrides n_components
        use_varimax: Whether to apply varimax rotation to redistribute variance
        varimax_max_iter: Maximum iterations for varimax algorithm
    """

    def __init__(
        self,
        n_components: int = 15,
        variance_threshold: float = None,
        use_varimax: bool = True,
        varimax_max_iter: int = 100,
    ):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.use_varimax = use_varimax
        self.varimax_max_iter = varimax_max_iter

        # Fitted parameters (per channel)
        self._seq_len = None
        self._n_channels = None
        self._mean_function = None  # (seq_len, n_channels)
        self._eigenfunctions = None  # List of (seq_len, n_components) per channel
        self._eigenvalues = None  # List of (n_components,) per channel
        self._actual_n_components = None  # List of int per channel
        self._rotation_matrices = None  # List of rotation matrices if varimax applied
        self._cumulative_variance = None  # List of cumulative variance explained

    def fit(self, signals: np.ndarray) -> 'FPCATransformer':
        """
        Fit FPCA to training signals.

        Args:
            signals: Training signals of shape (n_samples, seq_len, n_channels)

        Returns:
            self
        """
        n_samples, seq_len, n_channels = signals.shape
        self._seq_len = seq_len
        self._n_channels = n_channels

        # Compute mean function per channel
        self._mean_function = np.mean(signals, axis=0)  # (seq_len, n_channels)

        # Center signals
        centered = signals - self._mean_function

        # Fit FPCA for each channel separately
        self._eigenfunctions = []
        self._eigenvalues = []
        self._actual_n_components = []
        self._rotation_matrices = []
        self._cumulative_variance = []

        for ch in range(n_channels):
            # Extract channel data: (n_samples, seq_len)
            X_ch = centered[:, :, ch]

            # Compute covariance matrix: (seq_len, seq_len)
            # Using sample covariance
            cov_matrix = X_ch.T @ X_ch / (n_samples - 1)

            # Eigendecomposition
            eigenvalues, eigenvectors = linalg.eigh(cov_matrix)

            # Sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Ensure non-negative eigenvalues (numerical stability)
            eigenvalues = np.maximum(eigenvalues, 0)

            # Cumulative variance explained
            total_var = np.sum(eigenvalues)
            if total_var > 0:
                cum_var = np.cumsum(eigenvalues) / total_var
            else:
                cum_var = np.ones(len(eigenvalues))

            # Determine number of components
            if self.variance_threshold is not None:
                # Select components by variance threshold
                n_comp = np.searchsorted(cum_var, self.variance_threshold) + 1
                n_comp = min(n_comp, len(eigenvalues), seq_len)
            else:
                n_comp = min(self.n_components, len(eigenvalues), seq_len)

            # Truncate to selected components
            eigenvalues = eigenvalues[:n_comp]
            eigenvectors = eigenvectors[:, :n_comp]  # (seq_len, n_comp)
            cum_var = cum_var[:n_comp]

            # Apply varimax rotation if requested
            rotation_matrix = None
            if self.use_varimax and n_comp > 1:
                # Scale eigenvectors by sqrt of eigenvalues to create loadings
                # Loadings = eigenvectors * sqrt(eigenvalues)
                loadings = eigenvectors * np.sqrt(eigenvalues)
                rotated_loadings, rotation_matrix = varimax_rotation(
                    loadings, max_iter=self.varimax_max_iter
                )
                # Apply rotation directly to eigenvectors to preserve orthonormality
                # Since rotation_matrix is orthogonal, V @ R is still orthonormal
                eigenvectors = eigenvectors @ rotation_matrix
                # Recompute eigenvalues as variance explained in each rotated direction
                scores = X_ch @ eigenvectors
                eigenvalues = np.var(scores, axis=0, ddof=1)

            self._eigenfunctions.append(eigenvectors)
            self._eigenvalues.append(eigenvalues)
            self._actual_n_components.append(n_comp)
            self._rotation_matrices.append(rotation_matrix)
            self._cumulative_variance.append(cum_var)

        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """
        Transform signals to FPC scores.

        Args:
            signals: Signals of shape (n_samples, seq_len, n_channels)

        Returns:
            Scores of shape (n_samples, max_n_components, n_channels)
        """
        if self._eigenfunctions is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        n_samples, seq_len, n_channels = signals.shape

        # Center signals
        centered = signals - self._mean_function

        # Get maximum number of components across channels
        max_n_comp = max(self._actual_n_components)

        # Transform each channel
        scores = np.zeros((n_samples, max_n_comp, n_channels))

        for ch in range(n_channels):
            X_ch = centered[:, :, ch]  # (n_samples, seq_len)
            eigenfuncs = self._eigenfunctions[ch]  # (seq_len, n_comp_ch)
            n_comp_ch = self._actual_n_components[ch]

            # Project onto eigenfunctions: scores = X @ eigenfunctions
            ch_scores = X_ch @ eigenfuncs  # (n_samples, n_comp_ch)
            scores[:, :n_comp_ch, ch] = ch_scores

        return scores

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Reconstruct signals from FPC scores.

        Args:
            scores: Scores of shape (n_samples, n_components, n_channels)

        Returns:
            Reconstructed signals of shape (n_samples, seq_len, n_channels)
        """
        if self._eigenfunctions is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        n_samples = scores.shape[0]

        # Reconstruct each channel
        signals = np.zeros((n_samples, self._seq_len, self._n_channels))

        for ch in range(self._n_channels):
            eigenfuncs = self._eigenfunctions[ch]  # (seq_len, n_comp_ch)
            n_comp_ch = self._actual_n_components[ch]

            # Reconstruct: X = scores @ eigenfunctions.T + mean
            ch_scores = scores[:, :n_comp_ch, ch]  # (n_samples, n_comp_ch)
            reconstructed = ch_scores @ eigenfuncs.T  # (n_samples, seq_len)
            signals[:, :, ch] = reconstructed + self._mean_function[:, ch]

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
        if self._eigenfunctions is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")
        return self._eigenfunctions

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
