"""
Robustness Analysis for Seizure Dimensionality

Addresses reviewer concerns:
1. Autocorrelation - analyze with non-overlapping windows
2. Transition zone - analyze 30-60 min pre-ictal gradient
3. Eigenvector localization - where is coordination happening?
4. Rolling variance - practical warning signal
5. Surrogate testing - phase shuffling to confirm dynamical coupling

Author: Ian Todd
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.linalg import eigh
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from dimensional_profile import (
    compute_covariance_eigenvalues,
    compute_dimensional_profile,
    participation_ratio,
    bandpass_filter
)


# =============================================================================
# 1. NON-OVERLAPPING WINDOWS (Autocorrelation fix)
# =============================================================================

def analyze_non_overlapping(
    profiles_df: pd.DataFrame,
    step_ratio: int = 10
) -> Dict[str, any]:
    """
    Resample to non-overlapping windows to correct for autocorrelation.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        Full results with overlapping windows
    step_ratio : int
        Take every Nth window (10 = non-overlapping for 10s window, 1s step)

    Returns
    -------
    results : dict
        Statistics computed on non-overlapping subset
    """
    # Subsample
    df_nonoverlap = profiles_df.iloc[::step_ratio].copy()

    interictal = df_nonoverlap[df_nonoverlap['label'] == 0]
    preictal = df_nonoverlap[df_nonoverlap['label'] == 1]
    ictal = df_nonoverlap[df_nonoverlap['label'] == 2]

    results = {
        'n_interictal': len(interictal),
        'n_preictal': len(preictal),
        'n_ictal': len(ictal),
    }

    # T-tests on non-overlapping data
    for metric in ['D_PR', 'K_95', 'lambda_1']:
        if len(preictal) > 10 and len(interictal) > 10:
            t_stat, p_val = stats.ttest_ind(
                interictal[metric].dropna(),
                preictal[metric].dropna()
            )
            results[f'{metric}_t'] = t_stat
            results[f'{metric}_p'] = p_val
            results[f'{metric}_inter_mean'] = interictal[metric].mean()
            results[f'{metric}_pre_mean'] = preictal[metric].mean()
            results[f'{metric}_inter_var'] = interictal[metric].var()
            results[f'{metric}_pre_var'] = preictal[metric].var()

    # Variance ratio (Levene test)
    for metric in ['D_PR', 'lambda_1']:
        if len(preictal) > 10 and len(interictal) > 10:
            stat, p = stats.levene(
                interictal[metric].dropna(),
                preictal[metric].dropna()
            )
            results[f'{metric}_levene_F'] = stat
            results[f'{metric}_levene_p'] = p
            results[f'{metric}_var_ratio'] = (
                interictal[metric].var() / preictal[metric].var()
            )

    return results


# =============================================================================
# 2. TRANSITION ZONE ANALYSIS (30-60 min gradient)
# =============================================================================

def analyze_transition_zone(
    profiles_df: pd.DataFrame,
    seizure_onsets: List[float],
    time_bins: List[Tuple[int, int]] = [(-60, -45), (-45, -30), (-30, -15), (-15, 0)]
) -> pd.DataFrame:
    """
    Analyze the transition zone to look for gradients approaching seizure.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        Full results
    seizure_onsets : List[float]
        Seizure onset times in seconds
    time_bins : List[Tuple[int, int]]
        Time bins in minutes relative to seizure

    Returns
    -------
    gradient_df : pd.DataFrame
        Metrics by time bin
    """
    results = []

    for start_min, end_min in time_bins:
        bin_data = []

        for onset in seizure_onsets:
            start_sec = onset + start_min * 60
            end_sec = onset + end_min * 60

            mask = (profiles_df['time'] >= start_sec) & (profiles_df['time'] < end_sec)
            subset = profiles_df[mask]

            if len(subset) > 0:
                bin_data.append(subset)

        if bin_data:
            combined = pd.concat(bin_data)

            results.append({
                'bin': f'{start_min} to {end_min}',
                'n_windows': len(combined),
                'D_PR_mean': combined['D_PR'].mean(),
                'D_PR_std': combined['D_PR'].std(),
                'D_PR_var': combined['D_PR'].var(),
                'lambda_1_mean': combined['lambda_1'].mean(),
                'lambda_1_var': combined['lambda_1'].var(),
                'K_95_mean': combined['K_95'].mean(),
            })

    return pd.DataFrame(results)


# =============================================================================
# 3. EIGENVECTOR LOCALIZATION (Spatial analysis)
# =============================================================================

def compute_eigenvector_loadings(
    eeg_window: np.ndarray,
    channel_names: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvector loadings for spatial localization.

    Parameters
    ----------
    eeg_window : np.ndarray
        EEG data, shape (n_timepoints, n_channels)
    channel_names : List[str], optional
        Channel names for labeling

    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues (descending)
    eigenvectors : np.ndarray
        Eigenvectors, shape (n_channels, n_channels)
        Column i is eigenvector for eigenvalue i
    """
    # Center
    eeg_centered = eeg_window - eeg_window.mean(axis=0)

    # Covariance
    n_samples = eeg_window.shape[0]
    cov = eeg_centered.T @ eeg_centered / (n_samples - 1)

    # Eigendecomposition
    eigenvalues, eigenvectors = eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def compare_pc1_localization(
    eeg_interictal: np.ndarray,
    eeg_preictal: np.ndarray,
    channel_names: List[str]
) -> Dict[str, np.ndarray]:
    """
    Compare PC1 loadings between interictal and preictal.

    Parameters
    ----------
    eeg_interictal : np.ndarray
        Interictal EEG segment
    eeg_preictal : np.ndarray
        Preictal EEG segment
    channel_names : List[str]
        Channel names

    Returns
    -------
    results : dict
        PC1 loadings for each state
    """
    _, evec_inter = compute_eigenvector_loadings(eeg_interictal)
    _, evec_pre = compute_eigenvector_loadings(eeg_preictal)

    # PC1 is first column (highest eigenvalue)
    pc1_inter = np.abs(evec_inter[:, 0])  # Absolute value for loading magnitude
    pc1_pre = np.abs(evec_pre[:, 0])

    # Normalize to sum to 1
    pc1_inter = pc1_inter / pc1_inter.sum()
    pc1_pre = pc1_pre / pc1_pre.sum()

    # Compute localization index (inverse participation ratio of loadings)
    # Higher = more localized to fewer channels
    loc_inter = (pc1_inter ** 2).sum()  # 1/N if uniform, 1 if single channel
    loc_pre = (pc1_pre ** 2).sum()

    return {
        'pc1_interictal': pc1_inter,
        'pc1_preictal': pc1_pre,
        'localization_interictal': loc_inter,
        'localization_preictal': loc_pre,
        'channel_names': channel_names,
    }


# =============================================================================
# 4. ROLLING VARIANCE (Warning signal)
# =============================================================================

def compute_rolling_variance(
    profiles_df: pd.DataFrame,
    window_minutes: float = 5.0,
    step_minutes: float = 1.0,
    metric: str = 'D_PR'
) -> pd.DataFrame:
    """
    Compute rolling variance of dimensional metrics as a warning signal.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        Full results with 'time' and metric columns
    window_minutes : float
        Rolling window size in minutes
    step_minutes : float
        Step size in minutes
    metric : str
        Which metric to compute variance of

    Returns
    -------
    rolling_df : pd.DataFrame
        Time series of rolling variance
    """
    # Convert to seconds
    window_sec = window_minutes * 60
    step_sec = step_minutes * 60

    # Sort by time
    df = profiles_df.sort_values('time').copy()

    times = df['time'].values
    values = df[metric].values

    # Compute rolling variance
    results = []
    t = times.min() + window_sec / 2

    while t < times.max() - window_sec / 2:
        mask = (times >= t - window_sec / 2) & (times < t + window_sec / 2)
        window_vals = values[mask]

        if len(window_vals) > 10:
            results.append({
                'time': t,
                f'{metric}_var': np.var(window_vals),
                f'{metric}_mean': np.mean(window_vals),
                f'{metric}_std': np.std(window_vals),
                'n_windows': len(window_vals),
            })

        t += step_sec

    return pd.DataFrame(results)


def detect_variance_drops(
    rolling_df: pd.DataFrame,
    metric: str = 'D_PR',
    threshold_percentile: float = 20
) -> pd.DataFrame:
    """
    Detect periods of abnormally low variance (potential warning).

    Parameters
    ----------
    rolling_df : pd.DataFrame
        Output from compute_rolling_variance
    metric : str
        Metric name
    threshold_percentile : float
        Percentile below which variance is considered "low"

    Returns
    -------
    alerts : pd.DataFrame
        Times when variance dropped below threshold
    """
    var_col = f'{metric}_var'
    threshold = np.percentile(rolling_df[var_col], threshold_percentile)

    alerts = rolling_df[rolling_df[var_col] < threshold].copy()
    alerts['threshold'] = threshold

    return alerts


# =============================================================================
# 5. SURROGATE DATA TEST (Phase shuffling)
# =============================================================================

def phase_shuffle(signal: np.ndarray) -> np.ndarray:
    """
    Phase shuffle a signal while preserving power spectrum.

    Parameters
    ----------
    signal : np.ndarray
        1D time series

    Returns
    -------
    shuffled : np.ndarray
        Phase-shuffled signal with same power spectrum
    """
    # FFT
    fft = np.fft.rfft(signal)

    # Random phases (but keep DC and Nyquist real)
    random_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, len(fft)))
    random_phases[0] = 1  # DC component stays real
    if len(signal) % 2 == 0:
        random_phases[-1] = 1  # Nyquist stays real

    # Apply random phases
    fft_shuffled = fft * random_phases

    # Inverse FFT
    shuffled = np.fft.irfft(fft_shuffled, n=len(signal))

    return shuffled


def create_surrogate_eeg(eeg_data: np.ndarray) -> np.ndarray:
    """
    Create surrogate EEG by phase-shuffling each channel independently.

    This preserves power spectrum but destroys phase relationships
    (functional connectivity).

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data, shape (n_timepoints, n_channels)

    Returns
    -------
    surrogate : np.ndarray
        Surrogate data with same shape
    """
    surrogate = np.zeros_like(eeg_data)

    for ch in range(eeg_data.shape[1]):
        surrogate[:, ch] = phase_shuffle(eeg_data[:, ch])

    return surrogate


def surrogate_test(
    eeg_interictal: np.ndarray,
    eeg_preictal: np.ndarray,
    n_surrogates: int = 100
) -> Dict[str, any]:
    """
    Test whether D_PR difference is due to phase relationships.

    If the pre-ictal "locking" is due to dynamical coupling (not just
    spectral power), it should disappear in phase-shuffled surrogates.

    Parameters
    ----------
    eeg_interictal : np.ndarray
        Interictal EEG segment
    eeg_preictal : np.ndarray
        Preictal EEG segment
    n_surrogates : int
        Number of surrogate datasets

    Returns
    -------
    results : dict
        Real vs surrogate statistics
    """
    # Real data
    profile_inter = compute_dimensional_profile(eeg_interictal)
    profile_pre = compute_dimensional_profile(eeg_preictal)

    real_dpr_diff = profile_pre['D_PR'] - profile_inter['D_PR']
    real_var_ratio = (
        np.var(compute_covariance_eigenvalues(eeg_interictal)) /
        np.var(compute_covariance_eigenvalues(eeg_preictal))
    )

    # Surrogate data
    surrogate_dpr_diffs = []
    surrogate_var_ratios = []

    for _ in tqdm(range(n_surrogates), desc="Surrogate testing"):
        surr_inter = create_surrogate_eeg(eeg_interictal)
        surr_pre = create_surrogate_eeg(eeg_preictal)

        prof_surr_inter = compute_dimensional_profile(surr_inter)
        prof_surr_pre = compute_dimensional_profile(surr_pre)

        surrogate_dpr_diffs.append(prof_surr_pre['D_PR'] - prof_surr_inter['D_PR'])

        var_inter = np.var(compute_covariance_eigenvalues(surr_inter))
        var_pre = np.var(compute_covariance_eigenvalues(surr_pre))
        if var_pre > 0:
            surrogate_var_ratios.append(var_inter / var_pre)

    surrogate_dpr_diffs = np.array(surrogate_dpr_diffs)
    surrogate_var_ratios = np.array(surrogate_var_ratios)

    # P-values (one-tailed)
    p_dpr = (surrogate_dpr_diffs >= real_dpr_diff).mean()
    p_var = (surrogate_var_ratios >= real_var_ratio).mean()

    return {
        'real_dpr_diff': real_dpr_diff,
        'surrogate_dpr_diff_mean': surrogate_dpr_diffs.mean(),
        'surrogate_dpr_diff_std': surrogate_dpr_diffs.std(),
        'p_dpr': p_dpr,
        'real_var_ratio': real_var_ratio,
        'surrogate_var_ratio_mean': surrogate_var_ratios.mean(),
        'surrogate_var_ratio_std': surrogate_var_ratios.std(),
        'p_var_ratio': p_var,
    }


# =============================================================================
# MAIN: Run all robustness analyses
# =============================================================================

def run_robustness_analyses(
    profiles_path: str,
    output_dir: str = 'results'
):
    """Run all robustness analyses and save results."""

    print("="*70)
    print("ROBUSTNESS ANALYSES")
    print("="*70)

    # Load data
    profiles_df = pd.read_csv(profiles_path)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 1. Non-overlapping windows
    print("\n1. NON-OVERLAPPING WINDOWS (Autocorrelation correction)")
    print("-"*50)
    nonoverlap_results = analyze_non_overlapping(profiles_df, step_ratio=10)

    print(f"  N (non-overlapping): interictal={nonoverlap_results['n_interictal']}, "
          f"preictal={nonoverlap_results['n_preictal']}")

    for metric in ['D_PR', 'lambda_1']:
        if f'{metric}_p' in nonoverlap_results:
            print(f"  {metric}: inter={nonoverlap_results[f'{metric}_inter_mean']:.3f}, "
                  f"pre={nonoverlap_results[f'{metric}_pre_mean']:.3f}, "
                  f"p={nonoverlap_results[f'{metric}_p']:.2e}")
            print(f"    Variance ratio: {nonoverlap_results[f'{metric}_var_ratio']:.2f}x, "
                  f"Levene p={nonoverlap_results[f'{metric}_levene_p']:.2e}")

    # 2. Transition zone gradient
    print("\n2. TRANSITION ZONE GRADIENT")
    print("-"*50)

    # Find seizure onsets from ictal windows
    seizure_files = profiles_df[profiles_df['label'] == 2]['file'].unique()
    seizure_onsets = []
    for sf in seizure_files:
        file_data = profiles_df[profiles_df['file'] == sf]
        ictal_times = file_data[file_data['label'] == 2]['time']
        if len(ictal_times) > 0:
            seizure_onsets.append(ictal_times.min())

    if seizure_onsets:
        gradient_df = analyze_transition_zone(profiles_df, seizure_onsets)
        print(gradient_df.to_string(index=False))
        gradient_df.to_csv(output_path / 'transition_zone_gradient.csv', index=False)

    # 3. Rolling variance
    print("\n3. ROLLING VARIANCE (Warning signal)")
    print("-"*50)

    # Compute for each file with seizures
    for sf in seizure_files[:3]:  # First 3 seizure files
        file_data = profiles_df[profiles_df['file'] == sf].copy()
        if len(file_data) > 100:
            rolling_df = compute_rolling_variance(file_data, window_minutes=5)

            # Check variance around seizure
            ictal_time = file_data[file_data['label'] == 2]['time'].min()
            pre_seizure = rolling_df[rolling_df['time'] < ictal_time].tail(10)
            baseline = rolling_df[rolling_df['time'] < ictal_time - 1800].head(10)

            if len(pre_seizure) > 0 and len(baseline) > 0:
                print(f"  {sf}:")
                print(f"    Baseline D_PR variance: {baseline['D_PR_var'].mean():.3f}")
                print(f"    Pre-seizure D_PR variance: {pre_seizure['D_PR_var'].mean():.3f}")
                print(f"    Ratio: {baseline['D_PR_var'].mean() / pre_seizure['D_PR_var'].mean():.2f}x")

    print("\n" + "="*70)
    print("Robustness analyses complete")
    print("="*70)

    return nonoverlap_results


if __name__ == "__main__":
    results_path = Path(__file__).parent.parent / 'results' / 'chb01_profiles.csv'

    if not results_path.exists():
        print(f"Results not found at {results_path}")
        print("Run chb_mit_analysis.py first")
        exit(1)

    run_robustness_analyses(str(results_path))
