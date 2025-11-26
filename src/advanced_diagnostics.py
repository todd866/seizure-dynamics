"""
Advanced Diagnostics: Spatial Locking & Surrogate Testing

Implements advanced validation analyses for the seizure dimensionality paper:
1. Spatial Entropy: Quantifies if the dominant mode (PC1) becomes spatially
   "locked" (low entropy) in the pre-ictal state.
2. Surrogate Testing: Uses phase-shuffling to test if D_PR effects are due
   to dynamical coupling vs. simple spectral power changes.

Author: Ian Todd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.linalg import eigh

from dimensional_profile import (
    compute_dimensional_profile,
    bandpass_filter
)

# Try to import EEG loader (mne or pyedflib)
try:
    import mne
    mne.set_log_level('ERROR')
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


def load_data_segment(file_path: Path, start_sec: float, duration_sec: float):
    """Robustly load a segment of EEG data."""
    if HAS_MNE:
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        sfreq = raw.info['sfreq']

        start_idx = int(start_sec * sfreq)
        stop_idx = int((start_sec + duration_sec) * sfreq)

        data, times = raw[0:len(raw.ch_names), start_idx:stop_idx]
        return data.T, sfreq, raw.ch_names
    else:
        raise ImportError("Please install 'mne' to run advanced diagnostics.")


def compute_eigenvector_entropy(eigenvectors: np.ndarray) -> float:
    """
    Compute Shannon entropy of the dominant eigenvector (PC1) weights.

    H = - sum(w_i * log(w_i))
    where w_i are the squared, normalized loadings of PC1.

    Low Entropy -> Spatially Locked (Variance driven by few channels)
    High Entropy -> Distributed (Variance driven by global network)
    """
    # Get PC1 (eigenvector corresponding to largest eigenvalue)
    # eigh returns ascending eigenvalues, so PC1 is the last column
    pc1 = eigenvectors[:, -1]

    # Normalize loadings to get a probability-like distribution
    weights = pc1 ** 2
    weights = weights / np.sum(weights)

    # Compute Entropy (add epsilon to avoid log(0))
    entropy = -np.sum(weights * np.log(weights + 1e-12))
    return entropy


def phase_shuffle_surrogate(data: np.ndarray) -> np.ndarray:
    """
    Create phase-shuffled surrogate data.
    Preserves power spectrum of each channel exactly.
    Destroys phase relationships (coupling) between channels.
    """
    n_samples, n_channels = data.shape
    surrogate = np.zeros_like(data)

    for ch in range(n_channels):
        x = data[:, ch]
        xf = np.fft.rfft(x)

        # Generate random phases
        random_phases = np.exp(1j * np.random.uniform(0, 2*np.pi, len(xf)))
        random_phases[0] = 1.0  # Keep DC unchanged
        if n_samples % 2 == 0:
            random_phases[-1] = 1.0  # Keep Nyquist real

        # Apply phase shift
        xf_shuffled = xf * random_phases

        # Inverse FFT
        surrogate[:, ch] = np.fft.irfft(xf_shuffled, n=n_samples)

    return surrogate


def run_spatial_locking_analysis(
    profiles_df: pd.DataFrame,
    data_dir: Path,
    output_dir: Path = None,
    n_windows: int = 200
):
    """
    Compare Spatial Entropy between Interictal and Pre-ictal states.
    """
    print("\n" + "="*60)
    print("RUNNING SPATIAL LOCKING ANALYSIS (Eigenvector Entropy)")
    print("="*60)

    results = []

    for label, state_name in [(0, 'Interictal'), (1, 'Pre-ictal')]:
        subset = profiles_df[profiles_df['label'] == label]
        if subset.empty:
            continue

        # Sample random windows
        sample_indices = np.random.choice(
            subset.index,
            size=min(n_windows, len(subset)),
            replace=False
        )
        sampled_rows = subset.loc[sample_indices]

        for _, row in tqdm(sampled_rows.iterrows(), total=len(sampled_rows),
                          desc=f"Analyzing {state_name}"):
            try:
                file_path = data_dir / row['file']
                if not file_path.exists():
                    continue

                # Load 10s window
                data, fs, ch_names = load_data_segment(file_path, row['time'], 10.0)

                # Filter
                data_filt = bandpass_filter(data, fs, 1, 45)

                # Covariance & Eigenvectors
                data_centered = data_filt - data_filt.mean(axis=0)
                cov = np.cov(data_centered, rowvar=False)
                evals, evecs = eigh(cov)

                # Compute Entropy of PC1
                h = compute_eigenvector_entropy(evecs)

                results.append({
                    'State': state_name,
                    'Spatial_Entropy': h,
                    'PC1_Max_Loading': np.max(np.abs(evecs[:, -1])),
                    'File': row['file']
                })

            except Exception as e:
                pass

    res_df = pd.DataFrame(results)

    if not res_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Entropy Distribution
        for state, color in [('Interictal', 'blue'), ('Pre-ictal', 'orange')]:
            data = res_df[res_df['State'] == state]['Spatial_Entropy']
            axes[0].hist(data, bins=30, alpha=0.5, label=state, color=color)
        axes[0].set_xlabel('Spatial Entropy of PC1')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Spatial Entropy Distribution\n(Lower = More Locked)')
        axes[0].legend()

        # 2. Max Loading Distribution
        for state, color in [('Interictal', 'blue'), ('Pre-ictal', 'orange')]:
            data = res_df[res_df['State'] == state]['PC1_Max_Loading']
            axes[1].hist(data, bins=30, alpha=0.5, label=state, color=color)
        axes[1].set_xlabel('Max Single-Channel Loading (PC1)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Max Loading Distribution\n(Higher = Single Channel Dominance)')
        axes[1].legend()

        plt.tight_layout()

        if output_dir:
            output_dir = Path(output_dir)
            plt.savefig(output_dir / 'fig8_spatial_locking.pdf', dpi=300, bbox_inches='tight')
            print(f"Saved: {output_dir / 'fig8_spatial_locking.pdf'}")

        # Stats
        mean_inter = res_df[res_df['State']=='Interictal']['Spatial_Entropy'].mean()
        mean_pre = res_df[res_df['State']=='Pre-ictal']['Spatial_Entropy'].mean()
        print(f"\nMean Spatial Entropy:")
        print(f"  Interictal: {mean_inter:.4f}")
        print(f"  Pre-ictal:  {mean_pre:.4f}")
        print(f"  Difference: {mean_pre - mean_inter:.4f}")
        if mean_pre < mean_inter:
            print("  (Confirmed: Pre-ictal state is more spatially localized)")
        else:
            print("  (Result: No evidence of increased spatial locking)")

    return res_df


def run_surrogate_test(
    profiles_df: pd.DataFrame,
    data_dir: Path,
    output_dir: Path = None,
    n_samples: int = 50,
    n_surrogates_per_sample: int = 20
):
    """
    Run Surrogate Data Testing.

    Hypothesis: If D_PR effects are due to phase coupling, they should vanish
    in phase-shuffled surrogates.
    """
    print("\n" + "="*60)
    print("RUNNING SURROGATE DATA TEST (Phase Shuffling)")
    print("="*60)

    surrogate_results = []

    # Focus on Pre-ictal state
    subset = profiles_df[profiles_df['label'] == 1]
    if subset.empty:
        print("No pre-ictal data found.")
        return None

    sample_indices = np.random.choice(
        subset.index,
        size=min(n_samples, len(subset)),
        replace=False
    )
    sampled_rows = subset.loc[sample_indices]

    for _, row in tqdm(sampled_rows.iterrows(), total=len(sampled_rows),
                      desc="Surrogate Testing"):
        try:
            file_path = data_dir / row['file']
            if not file_path.exists():
                continue

            # Load real data
            data, fs, _ = load_data_segment(file_path, row['time'], 10.0)
            data_filt = bandpass_filter(data, fs, 1, 45)

            # Real Profile
            real_profile = compute_dimensional_profile(data_filt)
            real_dpr = real_profile['D_PR']

            # Generate Surrogates
            surr_dprs = []
            for _ in range(n_surrogates_per_sample):
                data_surr = phase_shuffle_surrogate(data_filt)
                surr_profile = compute_dimensional_profile(data_surr)
                surr_dprs.append(surr_profile['D_PR'])

            mean_surr_dpr = np.mean(surr_dprs)

            surrogate_results.append({
                'Real_D_PR': real_dpr,
                'Surrogate_Mean_D_PR': mean_surr_dpr,
                'Difference': real_dpr - mean_surr_dpr,
                'Z_Score': (real_dpr - mean_surr_dpr) / (np.std(surr_dprs) + 1e-6)
            })

        except Exception as e:
            pass

    res_df = pd.DataFrame(surrogate_results)

    if not res_df.empty:
        plt.figure(figsize=(8, 6))

        # Plot Real vs Surrogate D_PR
        plt.scatter(res_df['Real_D_PR'], res_df['Surrogate_Mean_D_PR'],
                   alpha=0.6, c='purple', s=50)

        # Plot unity line
        min_val = min(res_df['Real_D_PR'].min(), res_df['Surrogate_Mean_D_PR'].min())
        max_val = max(res_df['Real_D_PR'].max(), res_df['Surrogate_Mean_D_PR'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.75,
                label="No Coupling Effect")

        plt.xlabel('Real Data $D_{PR}$')
        plt.ylabel('Phase-Shuffled Surrogate $D_{PR}$')
        plt.title('Impact of Phase Coupling on Dimensionality')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_dir:
            output_dir = Path(output_dir)
            plt.savefig(output_dir / 'fig9_surrogate_test.pdf', dpi=300, bbox_inches='tight')
            print(f"Saved: {output_dir / 'fig9_surrogate_test.pdf'}")

        # Statistical Interpretation
        mean_z = res_df['Z_Score'].mean()
        print(f"\nSurrogate Test Results:")
        print(f"  Mean Z-Score (Real vs Surrogate): {mean_z:.2f}")
        if mean_z < -1.96:
            print("  CONCLUSION: Real data has significantly LOWER dimensionality than expected.")
            print("  (Strong phase-locking/synchronization reduces the dimension)")
        elif mean_z > 1.96:
            print("  CONCLUSION: Real data has significantly HIGHER dimensionality than expected.")
        else:
            print("  CONCLUSION: Dimensionality largely explained by power spectrum.")

    return res_df


if __name__ == "__main__":
    # Configuration
    base_dir = Path(__file__).parent.parent
    results_file = base_dir / 'results' / 'chb01_profiles.csv'
    data_dir = base_dir / 'data' / 'physionet.org' / 'files' / 'chbmit' / '1.0.0' / 'chb01'
    figures_dir = base_dir / 'paper' / 'figures'

    if results_file.exists() and data_dir.exists():
        df = pd.read_csv(results_file)

        # Run analyses
        spatial_df = run_spatial_locking_analysis(df, data_dir, figures_dir, n_windows=100)
        surrogate_df = run_surrogate_test(df, data_dir, figures_dir, n_samples=30)
    else:
        print(f"Error: Could not find results file ({results_file}) or data dir ({data_dir})")
        print("Please adjust the paths.")
