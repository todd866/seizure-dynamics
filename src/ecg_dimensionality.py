"""
ECG Dimensionality Analysis

Applies the same dimensional profile analysis to multi-lead ECG data
to test whether cardiac "dimensionality" follows similar patterns to EEG:
- Healthy: moderate dimensionality, high variability (good HRV)
- Pathological: either rigid (low D, poor HRV) or chaotic (high D, arrhythmia)

Uses PTB-XL database (12-lead ECG, 21,837 records, various conditions)

Author: Ian Todd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import wfdb
import ast


def bandpass_filter_ecg(data: np.ndarray, fs: float,
                        low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Bandpass filter for ECG (different band than EEG)."""
    nyq = fs / 2
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(4, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, data, axis=0)


def compute_dimensional_profile(data_window: np.ndarray) -> Dict[str, float]:
    """
    Compute dimensional profile metrics for a multi-lead ECG window.
    Same metrics as EEG analysis for direct comparison.
    """
    # Center data
    data_centered = data_window - data_window.mean(axis=0)
    n_samples, n_channels = data_centered.shape

    # Covariance matrix
    cov = data_centered.T @ data_centered / (n_samples - 1)

    # Eigenvalue decomposition
    eigenvalues = eigh(cov, eigvals_only=True)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Numerical stability

    # Normalize
    total_var = eigenvalues.sum()
    eigenvalues_norm = eigenvalues / total_var

    # Participation ratio
    D_PR = (total_var ** 2) / (eigenvalues ** 2).sum()

    # K_theta metrics
    cumsum = np.cumsum(eigenvalues_norm)
    K_80 = np.searchsorted(cumsum, 0.80) + 1
    K_90 = np.searchsorted(cumsum, 0.90) + 1
    K_95 = np.searchsorted(cumsum, 0.95) + 1
    K_99 = np.searchsorted(cumsum, 0.99) + 1

    # Spectral slope
    ranks = np.arange(1, len(eigenvalues) + 1)
    log_ranks = np.log(ranks)
    log_eigs = np.log(eigenvalues_norm)
    slope, intercept = np.polyfit(log_ranks, log_eigs, 1)

    return {
        'D_PR': D_PR,
        'K_80': K_80,
        'K_90': K_90,
        'K_95': K_95,
        'K_99': K_99,
        'slope': slope,
        'lambda_1': eigenvalues_norm[0],
        'n_channels': n_channels
    }


def load_ptbxl_record(record_path: str) -> Tuple[np.ndarray, float]:
    """Load a PTB-XL record and return data + sampling frequency."""
    record = wfdb.rdrecord(record_path)
    data = record.p_signal  # (n_samples, 12 leads)
    fs = record.fs
    return data, fs


def analyze_ptbxl_dataset(
    data_dir: Path,
    metadata_path: Path,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
    max_records: int = None
) -> pd.DataFrame:
    """
    Analyze PTB-XL dataset for dimensional profiles.

    Parameters
    ----------
    data_dir : Path
        Path to PTB-XL records
    metadata_path : Path
        Path to ptbxl_database.csv
    window_sec : float
        Window size in seconds
    step_sec : float
        Step size in seconds
    max_records : int, optional
        Limit number of records (for testing)

    Returns
    -------
    results_df : pd.DataFrame
        Dimensional profiles with diagnostic labels
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Parse the scp_codes column (it's a string representation of a dict)
    metadata['scp_codes_dict'] = metadata['scp_codes'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else {}
    )

    # Simplify to major categories
    def get_primary_diagnosis(scp_dict):
        if not scp_dict:
            return 'UNKNOWN'
        # Get the code with highest confidence
        primary = max(scp_dict, key=scp_dict.get)
        return primary

    metadata['primary_dx'] = metadata['scp_codes_dict'].apply(get_primary_diagnosis)

    if max_records:
        metadata = metadata.head(max_records)

    results = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Analyzing ECG"):
        try:
            # Construct record path
            record_name = row['filename_hr'].replace('.hea', '')
            record_path = data_dir / record_name

            if not (data_dir / f"{record_name}.hea").exists():
                continue

            # Load record
            data, fs = load_ptbxl_record(str(record_path))

            # Filter
            data_filt = bandpass_filter_ecg(data, fs)

            window_samples = int(window_sec * fs)
            step_samples = int(step_sec * fs)

            # Slide through record
            n_samples = len(data_filt)
            window_profiles = []

            for start in range(0, n_samples - window_samples, step_samples):
                window = data_filt[start:start + window_samples]
                profile = compute_dimensional_profile(window)
                window_profiles.append(profile)

            if not window_profiles:
                continue

            # Aggregate: mean and variance of each metric
            profile_df = pd.DataFrame(window_profiles)

            result = {
                'ecg_id': row['ecg_id'],
                'patient_id': row['patient_id'],
                'age': row['age'],
                'sex': row['sex'],
                'primary_dx': row['primary_dx'],
                # Mean dimensional metrics
                'D_PR_mean': profile_df['D_PR'].mean(),
                'D_PR_var': profile_df['D_PR'].var(),
                'lambda_1_mean': profile_df['lambda_1'].mean(),
                'lambda_1_var': profile_df['lambda_1'].var(),
                'slope_mean': profile_df['slope'].mean(),
                'slope_var': profile_df['slope'].var(),
                'K_95_mean': profile_df['K_95'].mean(),
                'K_95_var': profile_df['K_95'].var(),
                'n_windows': len(window_profiles)
            }
            results.append(result)

        except Exception as e:
            continue

    return pd.DataFrame(results)


def plot_ecg_dimensionality_by_diagnosis(results_df: pd.DataFrame, output_path: str = None):
    """
    Plot dimensional metrics grouped by diagnosis category.
    """
    # Group diagnoses into major categories
    def categorize_dx(dx):
        if dx in ['NORM']:
            return 'Normal'
        elif dx in ['MI', 'IMI', 'AMI', 'LMI', 'PMI']:
            return 'Myocardial Infarction'
        elif dx in ['AFIB', 'AFLT']:
            return 'Atrial Fibrillation'
        elif dx in ['STACH', 'SBRAD']:
            return 'Rate Abnormality'
        elif dx in ['LVH', 'RVH']:
            return 'Hypertrophy'
        elif dx in ['LBBB', 'RBBB', 'IVCD']:
            return 'Conduction Block'
        else:
            return 'Other'

    results_df['dx_category'] = results_df['primary_dx'].apply(categorize_dx)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    categories = ['Normal', 'Myocardial Infarction', 'Atrial Fibrillation',
                  'Conduction Block', 'Hypertrophy', 'Other']
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

    # (a) D_PR by category
    ax = axes[0, 0]
    cat_data = [results_df[results_df['dx_category'] == cat]['D_PR_mean'].dropna()
                for cat in categories if cat in results_df['dx_category'].values]
    cat_labels = [cat for cat in categories if cat in results_df['dx_category'].values]
    ax.boxplot(cat_data, labels=cat_labels)
    ax.set_ylabel('Mean $D_{PR}$')
    ax.set_title('(a) Participation Ratio by Diagnosis')
    ax.tick_params(axis='x', rotation=45)

    # (b) D_PR variance (our "CRV" - Cardiac Rate Variability of dimensionality)
    ax = axes[0, 1]
    cat_data = [results_df[results_df['dx_category'] == cat]['D_PR_var'].dropna()
                for cat in categories if cat in results_df['dx_category'].values]
    ax.boxplot(cat_data, labels=cat_labels)
    ax.set_ylabel('Variance of $D_{PR}$')
    ax.set_title('(b) Dimensional Variability by Diagnosis')
    ax.tick_params(axis='x', rotation=45)

    # (c) Lambda_1 by category
    ax = axes[1, 0]
    cat_data = [results_df[results_df['dx_category'] == cat]['lambda_1_mean'].dropna()
                for cat in categories if cat in results_df['dx_category'].values]
    ax.boxplot(cat_data, labels=cat_labels)
    ax.set_ylabel('Mean $\\lambda_1$')
    ax.set_title('(c) Dominant Mode Fraction by Diagnosis')
    ax.tick_params(axis='x', rotation=45)

    # (d) Slope by category
    ax = axes[1, 1]
    cat_data = [results_df[results_df['dx_category'] == cat]['slope_mean'].dropna()
                for cat in categories if cat in results_df['dx_category'].values]
    ax.boxplot(cat_data, labels=cat_labels)
    ax.set_ylabel('Mean Spectral Slope')
    ax.set_title('(d) Eigenvalue Spectrum Slope by Diagnosis')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def compare_eeg_ecg_dimensionality(
    eeg_results: pd.DataFrame,
    ecg_results: pd.DataFrame,
    output_path: str = None
):
    """
    Side-by-side comparison of EEG and ECG dimensional profiles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # EEG: by seizure state
    ax = axes[0]
    for label, name, color in [(0, 'Interictal', 'blue'),
                                (1, 'Pre-ictal', 'orange'),
                                (2, 'Ictal', 'red')]:
        data = eeg_results[eeg_results['label'] == label]['D_PR']
        ax.hist(data, bins=50, alpha=0.5, label=name, color=color, density=True)
    ax.set_xlabel('$D_{PR}$')
    ax.set_ylabel('Density')
    ax.set_title('EEG: Dimensionality by Seizure State')
    ax.legend()

    # ECG: by diagnosis
    ax = axes[1]
    ecg_results['dx_category'] = ecg_results['primary_dx'].apply(
        lambda x: 'Normal' if x == 'NORM' else 'Pathological'
    )
    for cat, color in [('Normal', 'green'), ('Pathological', 'red')]:
        data = ecg_results[ecg_results['dx_category'] == cat]['D_PR_mean']
        ax.hist(data, bins=50, alpha=0.5, label=cat, color=color, density=True)
    ax.set_xlabel('Mean $D_{PR}$')
    ax.set_ylabel('Density')
    ax.set_title('ECG: Dimensionality by Diagnosis')
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent

    # PTB-XL paths
    ptbxl_dir = base_dir / 'data' / 'physionet.org' / 'files' / 'ptb-xl' / '1.0.3'
    metadata_path = ptbxl_dir / 'ptbxl_database.csv'

    if not metadata_path.exists():
        print(f"PTB-XL metadata not found at {metadata_path}")
        print("Please download from: https://physionet.org/content/ptb-xl/1.0.3/")
        exit(1)

    print("="*70)
    print("ECG DIMENSIONALITY ANALYSIS (PTB-XL)")
    print("="*70)

    # Quick test with first 100 records
    print("\nAnalyzing first 100 records (test run)...")
    results = analyze_ptbxl_dataset(
        ptbxl_dir,
        metadata_path,
        max_records=100
    )

    print(f"\nAnalyzed {len(results)} records")
    print(f"\nDiagnosis distribution:")
    print(results['primary_dx'].value_counts().head(10))

    print(f"\nDimensional metrics by diagnosis:")
    for dx in results['primary_dx'].unique()[:5]:
        subset = results[results['primary_dx'] == dx]
        print(f"\n{dx} (n={len(subset)}):")
        print(f"  D_PR: {subset['D_PR_mean'].mean():.2f} ± {subset['D_PR_mean'].std():.2f}")
        print(f"  D_PR var: {subset['D_PR_var'].mean():.4f}")
        print(f"  λ₁: {subset['lambda_1_mean'].mean():.3f}")

    # Save results
    results_path = base_dir / 'results' / 'ptbxl_dimensionality.csv'
    results.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")

    # Plot
    figures_dir = base_dir / 'paper' / 'figures'
    plot_ecg_dimensionality_by_diagnosis(
        results,
        figures_dir / 'fig_ecg_dimensionality.pdf'
    )
