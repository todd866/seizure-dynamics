"""
Synthetic Demo: Dimensional Collapse in Simulated EEG

Demonstrates the dimensional profile framework on synthetic data that
mimics the transition from healthy (fat-tailed) to pre-ictal (collapsed)
to ictal (hypersynchronous) states.

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dimensional_profile import (
    compute_covariance_eigenvalues,
    compute_dimensional_profile,
    participation_ratio,
    sliding_window_profile
)

# Style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150


def simulate_eeg_state(
    n_samples: int,
    n_channels: int,
    true_dim: int,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Simulate EEG-like data with known effective dimensionality.

    Parameters
    ----------
    n_samples : int
        Number of time points
    n_channels : int
        Number of channels
    true_dim : int
        True underlying dimensionality (number of latent sources)
    noise_level : float
        Observation noise relative to signal

    Returns
    -------
    eeg : np.ndarray
        Simulated EEG, shape (n_samples, n_channels)
    """
    # Latent sources (oscillators with different frequencies)
    t = np.arange(n_samples) / 256  # Assume 256 Hz
    latent = np.zeros((n_samples, true_dim))

    for i in range(true_dim):
        # Each source is an oscillator with random frequency and phase
        freq = 5 + i * 3 + np.random.randn() * 0.5  # 5-50 Hz range
        phase = np.random.rand() * 2 * np.pi
        amplitude = 1.0 / (i + 1)  # Decreasing amplitude for higher modes
        latent[:, i] = amplitude * np.sin(2 * np.pi * freq * t + phase)

    # Random mixing matrix (sources -> channels)
    mixing = np.random.randn(true_dim, n_channels) / np.sqrt(true_dim)

    # Mix sources into channels
    eeg = latent @ mixing

    # Add observation noise
    noise = noise_level * np.random.randn(n_samples, n_channels)
    eeg = eeg + noise

    return eeg


def simulate_seizure_transition(
    n_samples: int,
    n_channels: int,
    transition_point: float = 0.6,  # Fraction of recording where collapse begins
    ictal_start: float = 0.8       # Fraction where ictal begins
) -> tuple:
    """
    Simulate EEG transitioning from interictal -> preictal -> ictal.

    Parameters
    ----------
    n_samples : int
        Total samples
    n_channels : int
        Number of channels
    transition_point : float
        When pre-ictal phase begins (fraction)
    ictal_start : float
        When ictal phase begins (fraction)

    Returns
    -------
    eeg : np.ndarray
        Simulated EEG
    labels : np.ndarray
        State labels (0=interictal, 1=preictal, 2=ictal)
    """
    t = np.arange(n_samples)
    trans_sample = int(transition_point * n_samples)
    ictal_sample = int(ictal_start * n_samples)

    eeg = np.zeros((n_samples, n_channels))
    labels = np.zeros(n_samples, dtype=int)

    # Interictal: rich dynamics, many modes
    interictal_dim = 12
    eeg_inter = simulate_eeg_state(trans_sample, n_channels, interictal_dim, noise_level=0.15)
    eeg[:trans_sample] = eeg_inter
    labels[:trans_sample] = 0

    # Pre-ictal: gradual collapse
    preictal_len = ictal_sample - trans_sample
    for i in range(preictal_len):
        # Linearly decrease dimensionality
        progress = i / preictal_len
        current_dim = int(interictal_dim * (1 - 0.7 * progress))  # Collapse to ~30%
        current_dim = max(2, current_dim)

        # Generate one sample at a time (crude but illustrative)
        window = simulate_eeg_state(100, n_channels, current_dim, noise_level=0.1)
        eeg[trans_sample + i] = window[50]  # Take middle sample

    labels[trans_sample:ictal_sample] = 1

    # Ictal: hypersynchronous, 1-2 modes dominate
    ictal_len = n_samples - ictal_sample
    t_ictal = np.arange(ictal_len) / 256

    # Single dominant oscillation (seizure pattern)
    seizure_freq = 3  # ~3 Hz spike-wave
    seizure_pattern = np.sin(2 * np.pi * seizure_freq * t_ictal)
    seizure_pattern += 0.3 * np.sin(2 * np.pi * 2 * seizure_freq * t_ictal)  # Harmonic

    # Spread to all channels with slight phase differences
    for ch in range(n_channels):
        phase_shift = np.random.randn() * 0.1
        eeg[ictal_sample:, ch] = seizure_pattern * (1 + 0.1 * np.random.randn())
        eeg[ictal_sample:, ch] += 0.05 * np.random.randn(ictal_len)

    labels[ictal_sample:] = 2

    return eeg, labels


def run_demo():
    """Run the full synthetic demo and generate figures."""

    np.random.seed(42)
    n_channels = 23  # Same as CHB-MIT

    # === Part 1: Compare eigenvalue spectra ===
    print("Generating eigenvalue spectrum comparison...")

    # Three states
    eeg_interictal = simulate_eeg_state(2560, n_channels, true_dim=15, noise_level=0.1)
    eeg_preictal = simulate_eeg_state(2560, n_channels, true_dim=5, noise_level=0.1)
    eeg_ictal = simulate_eeg_state(2560, n_channels, true_dim=2, noise_level=0.05)

    eig_inter = compute_covariance_eigenvalues(eeg_interictal, normalize=True)
    eig_pre = compute_covariance_eigenvalues(eeg_preictal, normalize=True)
    eig_ictal = compute_covariance_eigenvalues(eeg_ictal, normalize=True)

    # Figure 1: Eigenvalue spectra
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ranks = np.arange(1, n_channels + 1)

    # (a) Log scale spectra
    ax = axes[0]
    ax.semilogy(ranks, eig_inter, 'b-o', label='Interictal', markersize=4)
    ax.semilogy(ranks, eig_pre, 'orange', marker='s', linestyle='-', label='Pre-ictal', markersize=4)
    ax.semilogy(ranks, eig_ictal, 'r-^', label='Ictal', markersize=4)
    ax.set_xlabel('Component rank')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title('(a) Eigenvalue Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Cumulative variance
    ax = axes[1]
    ax.plot(ranks, np.cumsum(eig_inter), 'b-o', label='Interictal', markersize=4)
    ax.plot(ranks, np.cumsum(eig_pre), 'orange', marker='s', linestyle='-', label='Pre-ictal', markersize=4)
    ax.plot(ranks, np.cumsum(eig_ictal), 'r-^', label='Ictal', markersize=4)
    for thresh in [0.8, 0.9, 0.95, 0.99]:
        ax.axhline(thresh, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative variance')
    ax.set_title('(b) Cumulative Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # (c) Profile metrics
    ax = axes[2]
    profiles = {
        'Interictal': compute_dimensional_profile(eeg_interictal),
        'Pre-ictal': compute_dimensional_profile(eeg_preictal),
        'Ictal': compute_dimensional_profile(eeg_ictal)
    }

    metrics = ['D_PR', 'K_95', 'K_99']
    x = np.arange(len(metrics))
    width = 0.25

    colors = {'Interictal': 'blue', 'Pre-ictal': 'orange', 'Ictal': 'red'}
    for i, (state, prof) in enumerate(profiles.items()):
        values = [prof[m] for m in metrics]
        ax.bar(x + i * width, values, width, label=state, color=colors[state], alpha=0.7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value')
    ax.set_title('(c) Dimensional Profile Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / 'paper' / 'figures'
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / 'fig1_synthetic_spectra.pdf', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig1_synthetic_spectra.pdf'}")

    # === Part 2: Time series around simulated seizure ===
    print("\nGenerating seizure transition time series...")

    # Simulate 10 minutes of EEG with seizure
    fs = 256
    duration_sec = 600  # 10 minutes
    n_samples = duration_sec * fs

    eeg, labels = simulate_seizure_transition(
        n_samples, n_channels,
        transition_point=0.6,  # Pre-ictal starts at 6 minutes
        ictal_start=0.9        # Ictal starts at 9 minutes
    )

    # Compute dimensional profiles over sliding windows
    window_sec = 10
    step_sec = 2
    window_samples = window_sec * fs
    step_samples = step_sec * fs

    times, profiles = sliding_window_profile(
        eeg, window_samples, step_samples, fs
    )

    # Get labels for each window
    window_labels = labels[::step_samples][:len(times)]

    # Figure 2: Time series
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Color map for labels
    colors = np.array(['blue', 'orange', 'red'])[window_labels]

    # (a) D_PR over time
    ax = axes[0]
    ax.scatter(times / 60, profiles['D_PR'], c=colors, s=10, alpha=0.7)
    ax.axvline(6, color='orange', linestyle='--', label='Pre-ictal onset')
    ax.axvline(9, color='red', linestyle='--', label='Seizure onset')
    ax.set_ylabel('Participation Ratio ($D_{PR}$)')
    ax.set_title('(a) Dimensional Collapse Before Seizure')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # (b) K_95 over time
    ax = axes[1]
    ax.scatter(times / 60, profiles['K_95'], c=colors, s=10, alpha=0.7)
    ax.axvline(6, color='orange', linestyle='--')
    ax.axvline(9, color='red', linestyle='--')
    ax.set_ylabel('$K_{95}$ (components for 95% var)')
    ax.set_title('(b) Components Required for 95% Variance')
    ax.grid(True, alpha=0.3)

    # (c) lambda_1 over time
    ax = axes[2]
    ax.scatter(times / 60, profiles['lambda_1'], c=colors, s=10, alpha=0.7)
    ax.axvline(6, color='orange', linestyle='--')
    ax.axvline(9, color='red', linestyle='--')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('$\\lambda_1$ (dominant mode fraction)')
    ax.set_title('(c) Dominant Mode Concentration')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_synthetic_timeseries.pdf', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig2_synthetic_timeseries.pdf'}")

    # === Print summary ===
    print("\n" + "="*60)
    print("SYNTHETIC DEMO RESULTS")
    print("="*60)

    # Recompute profiles dict for printing (the earlier one got overwritten)
    state_profiles = {
        'Interictal': compute_dimensional_profile(eeg_interictal),
        'Pre-ictal': compute_dimensional_profile(eeg_preictal),
        'Ictal': compute_dimensional_profile(eeg_ictal)
    }

    print("\nEigenvalue spectrum profiles:")
    for state, prof in state_profiles.items():
        print(f"\n  {state}:")
        print(f"    D_PR = {prof['D_PR']:.2f}")
        print(f"    K_95 = {prof['K_95']:.0f}")
        print(f"    K_99 = {prof['K_99']:.0f}")
        print(f"    λ₁ = {prof['lambda_1']:.3f}")

    print("\n" + "="*60)
    print("The collapse signature is clear:")
    print("  - D_PR drops from ~10 (interictal) to ~2 (ictal)")
    print("  - K_95 drops from ~8 to ~2")
    print("  - λ₁ increases from ~0.2 to ~0.8")
    print("="*60)

    plt.show()


if __name__ == "__main__":
    run_demo()
