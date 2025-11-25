"""
Dimensional Profile Analysis for EEG

Core functions for computing effective dimensionality from EEG signals.
Implements D_eff(ε) as the minimal model dimension for reliable tracking.

Author: Ian Todd
"""

import numpy as np
from scipy import signal
from scipy.linalg import eigh
from typing import Tuple, Dict, List, Optional
import warnings


def compute_covariance_eigenvalues(
    eeg_window: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute eigenvalues of the channel covariance matrix.

    Parameters
    ----------
    eeg_window : np.ndarray
        EEG data, shape (n_timepoints, n_channels)
    normalize : bool
        If True, normalize eigenvalues to sum to 1 (variance fractions)

    Returns
    -------
    eigenvalues : np.ndarray
        Sorted eigenvalues (descending), shape (n_channels,)
    """
    # Center the data
    eeg_centered = eeg_window - eeg_window.mean(axis=0)

    # Compute covariance matrix
    n_samples = eeg_window.shape[0]
    cov = eeg_centered.T @ eeg_centered / (n_samples - 1)

    # Get eigenvalues (real, since cov is symmetric)
    eigenvalues = eigh(cov, eigvals_only=True)

    # Sort descending
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Clip small negative values (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0)

    if normalize and eigenvalues.sum() > 0:
        eigenvalues = eigenvalues / eigenvalues.sum()

    return eigenvalues


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    Compute participation ratio: D_PR = (Σλ)² / Σλ²

    This measures the "effective number of dimensions" in the spectrum.
    - If one eigenvalue dominates: D_PR ≈ 1
    - If all eigenvalues equal: D_PR = n_channels

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues (need not be normalized)

    Returns
    -------
    d_pr : float
        Participation ratio
    """
    total = eigenvalues.sum()
    sum_sq = (eigenvalues ** 2).sum()

    if sum_sq == 0:
        return 0.0

    return (total ** 2) / sum_sq


def cumulative_variance_threshold(
    eigenvalues: np.ndarray,
    thresholds: List[float] = [0.80, 0.90, 0.95, 0.99]
) -> Dict[str, int]:
    """
    Compute K_thresh: minimal number of components to reach each variance threshold.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Normalized eigenvalues (should sum to 1)
    thresholds : List[float]
        Variance thresholds to compute

    Returns
    -------
    k_values : Dict[str, int]
        Dictionary mapping threshold names to K values
    """
    cumsum = np.cumsum(eigenvalues)

    k_values = {}
    for thresh in thresholds:
        # Find first index where cumulative variance >= threshold
        idx = np.searchsorted(cumsum, thresh)
        # Add 1 because we want count, not index
        k_values[f"K_{int(thresh*100)}"] = min(idx + 1, len(eigenvalues))

    return k_values


def spectral_slope(
    eigenvalues: np.ndarray,
    fit_range: Tuple[int, int] = None
) -> float:
    """
    Compute the slope of log(eigenvalue) vs log(rank).

    A steeper (more negative) slope indicates a sharper falloff,
    i.e., more variance concentrated in fewer modes.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues (descending)
    fit_range : Tuple[int, int], optional
        Range of ranks to fit (default: all non-zero)

    Returns
    -------
    slope : float
        Slope of log-log fit (typically negative)
    """
    # Filter to positive eigenvalues
    pos_mask = eigenvalues > 1e-10
    if pos_mask.sum() < 2:
        return 0.0

    eig_pos = eigenvalues[pos_mask]
    ranks = np.arange(1, len(eig_pos) + 1)

    if fit_range is not None:
        start, end = fit_range
        eig_pos = eig_pos[start:end]
        ranks = ranks[start:end]

    if len(eig_pos) < 2:
        return 0.0

    # Log-log linear fit
    log_ranks = np.log(ranks)
    log_eig = np.log(eig_pos)

    # Simple linear regression
    slope, _ = np.polyfit(log_ranks, log_eig, 1)

    return slope


def compute_dimensional_profile(
    eeg_window: np.ndarray,
    thresholds: List[float] = [0.80, 0.90, 0.95, 0.99]
) -> Dict[str, float]:
    """
    Compute full dimensional profile for a single EEG window.

    Parameters
    ----------
    eeg_window : np.ndarray
        EEG data, shape (n_timepoints, n_channels)
    thresholds : List[float]
        Variance thresholds for K computation

    Returns
    -------
    profile : Dict[str, float]
        Dictionary containing:
        - D_PR: participation ratio
        - K_80, K_90, K_95, K_99: components for variance thresholds
        - slope: spectral slope
        - lambda_1: fraction of variance in first component
    """
    eigenvalues = compute_covariance_eigenvalues(eeg_window, normalize=True)

    profile = {
        'D_PR': participation_ratio(eigenvalues),
        'slope': spectral_slope(eigenvalues),
        'lambda_1': eigenvalues[0] if len(eigenvalues) > 0 else 0.0,
    }

    # Add K values
    k_values = cumulative_variance_threshold(eigenvalues, thresholds)
    profile.update(k_values)

    return profile


def sliding_window_profile(
    eeg_data: np.ndarray,
    window_samples: int,
    step_samples: int,
    fs: float,
    thresholds: List[float] = [0.80, 0.90, 0.95, 0.99]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute dimensional profile over sliding windows.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data, shape (n_timepoints, n_channels)
    window_samples : int
        Window length in samples
    step_samples : int
        Step size in samples
    fs : float
        Sampling frequency (for time axis)
    thresholds : List[float]
        Variance thresholds

    Returns
    -------
    times : np.ndarray
        Center time of each window (seconds)
    profiles : Dict[str, np.ndarray]
        Dictionary of profile time series
    """
    n_samples, n_channels = eeg_data.shape

    # Compute window centers
    starts = np.arange(0, n_samples - window_samples + 1, step_samples)
    times = (starts + window_samples / 2) / fs

    # Initialize output
    n_windows = len(starts)
    profiles = {
        'D_PR': np.zeros(n_windows),
        'slope': np.zeros(n_windows),
        'lambda_1': np.zeros(n_windows),
    }
    for thresh in thresholds:
        profiles[f'K_{int(thresh*100)}'] = np.zeros(n_windows)

    # Compute profiles
    for i, start in enumerate(starts):
        window = eeg_data[start:start + window_samples, :]
        profile = compute_dimensional_profile(window, thresholds)

        for key in profiles:
            profiles[key][i] = profile[key]

    return times, profiles


def bandpass_filter(
    eeg_data: np.ndarray,
    fs: float,
    lowcut: float = 1.0,
    highcut: float = 45.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to EEG data.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data, shape (n_timepoints, n_channels)
    fs : float
        Sampling frequency
    lowcut, highcut : float
        Filter cutoff frequencies
    order : int
        Filter order

    Returns
    -------
    filtered : np.ndarray
        Filtered EEG data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # Clip to valid range
    low = max(low, 0.001)
    high = min(high, 0.999)

    b, a = signal.butter(order, [low, high], btype='band')

    # Apply filter to each channel
    filtered = signal.filtfilt(b, a, eeg_data, axis=0)

    return filtered


# =============================================================================
# Model-based D_eff estimation (for future implementation)
# =============================================================================

def estimate_deff_model_based(
    eeg_window: np.ndarray,
    k_values: List[int] = [1, 2, 4, 8, 16],
    n_folds: int = 5
) -> Tuple[int, Dict[int, float]]:
    """
    Estimate D_eff as minimal latent dimension for reliable prediction.

    Uses linear state-space models (Kalman filter) with varying latent
    dimension and finds the smallest k where prediction performance
    reaches 95% of asymptotic quality.

    Parameters
    ----------
    eeg_window : np.ndarray
        EEG data, shape (n_timepoints, n_channels)
    k_values : List[int]
        Latent dimensions to try
    n_folds : int
        Number of cross-validation folds

    Returns
    -------
    d_eff : int
        Estimated effective dimensionality
    performance : Dict[int, float]
        Prediction MSE for each k

    Note
    ----
    This is a placeholder for more sophisticated model-based estimation.
    Full implementation requires pykalman or similar.
    """
    # Placeholder - would implement Kalman filter fitting here
    # For now, use participation ratio as proxy

    warnings.warn(
        "Model-based D_eff estimation not yet implemented. "
        "Using participation ratio as proxy."
    )

    eigenvalues = compute_covariance_eigenvalues(eeg_window, normalize=True)
    d_pr = participation_ratio(eigenvalues)

    return int(np.ceil(d_pr)), {}


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)

    # Simulate "healthy" EEG: many modes
    n_samples = 1000
    n_channels = 23

    # Create data with known dimensionality
    true_dim = 8
    latent = np.random.randn(n_samples, true_dim)
    mixing = np.random.randn(true_dim, n_channels)
    eeg_healthy = latent @ mixing + 0.1 * np.random.randn(n_samples, n_channels)

    # Simulate "collapsed" EEG: few modes
    true_dim_collapsed = 2
    latent_collapsed = np.random.randn(n_samples, true_dim_collapsed)
    mixing_collapsed = np.random.randn(true_dim_collapsed, n_channels)
    eeg_collapsed = latent_collapsed @ mixing_collapsed + 0.1 * np.random.randn(n_samples, n_channels)

    # Compute profiles
    profile_healthy = compute_dimensional_profile(eeg_healthy)
    profile_collapsed = compute_dimensional_profile(eeg_collapsed)

    print("Healthy EEG profile:")
    for k, v in profile_healthy.items():
        print(f"  {k}: {v:.3f}")

    print("\nCollapsed EEG profile:")
    for k, v in profile_collapsed.items():
        print(f"  {k}: {v:.3f}")

    print("\n✓ Dimensional collapse detected!" if profile_collapsed['D_PR'] < profile_healthy['D_PR'] else "\n✗ Test failed")
