"""
Plot CHB-MIT Analysis Results

Generates figures from the dimensional profile analysis.

Author: Ian Todd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_class_comparison(stats_df: pd.DataFrame, output_path: str = None):
    """Plot comparison of dimensional metrics across classes."""

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    classes = stats_df['class'].tolist()
    colors = {'interictal': 'blue', 'preictal': 'orange', 'ictal': 'red', 'postictal': 'purple'}
    class_colors = [colors.get(c, 'gray') for c in classes]

    metrics = [
        ('D_PR_mean', 'D_PR_std', 'Participation Ratio ($D_{PR}$)'),
        ('K_80_mean', 'K_80_std', '$K_{80}$ (components for 80% var)'),
        ('K_95_mean', 'K_95_std', '$K_{95}$ (components for 95% var)'),
        ('K_99_mean', 'K_99_std', '$K_{99}$ (components for 99% var)'),
        ('lambda_1_mean', 'lambda_1_std', '$\\lambda_1$ (dominant mode fraction)'),
        ('slope_mean', 'slope_std', 'Spectral slope'),
    ]

    for ax, (mean_col, std_col, title) in zip(axes.flatten(), metrics):
        x = np.arange(len(classes))
        means = stats_df[mean_col].values
        stds = stats_df[std_col].values

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=class_colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_timeseries_around_seizures(
    profiles_df: pd.DataFrame,
    output_path: str = None,
    window_minutes: tuple = (-60, 10)
):
    """
    Plot dimensional metrics aligned to seizure onset.

    This requires identifying seizure onsets from the ictal windows.
    """
    # Find seizure onsets: transitions from non-ictal to ictal
    profiles_df = profiles_df.sort_values(['file', 'time']).copy()

    # For each file, find when ictal windows start
    seizure_events = []
    for file_name in profiles_df['file'].unique():
        file_data = profiles_df[profiles_df['file'] == file_name].copy()
        file_data = file_data.sort_values('time')

        # Find first ictal window in this file
        ictal_mask = file_data['label'] == 2
        if ictal_mask.any():
            first_ictal_idx = ictal_mask.idxmax()
            onset_time = file_data.loc[first_ictal_idx, 'time']
            seizure_events.append({
                'file': file_name,
                'onset_time': onset_time
            })

    if not seizure_events:
        print("No seizure events found")
        return None

    print(f"Found {len(seizure_events)} seizure events")

    # Collect aligned traces
    pre_sec = window_minutes[0] * 60
    post_sec = window_minutes[1] * 60

    aligned_data = []
    for event in seizure_events:
        file_data = profiles_df[profiles_df['file'] == event['file']].copy()
        onset = event['onset_time']

        # Filter to window around seizure
        mask = (file_data['time'] >= onset + pre_sec) & (file_data['time'] <= onset + post_sec)
        window_data = file_data[mask].copy()

        if len(window_data) > 0:
            window_data['rel_time'] = (window_data['time'] - onset) / 60  # Convert to minutes
            aligned_data.append(window_data)

    if not aligned_data:
        print("No data in alignment windows")
        return None

    all_aligned = pd.concat(aligned_data, ignore_index=True)

    # Bin by relative time and compute statistics
    bins = np.linspace(window_minutes[0], window_minutes[1], 71)
    all_aligned['time_bin'] = pd.cut(all_aligned['rel_time'], bins)

    grouped = all_aligned.groupby('time_bin').agg({
        'D_PR': ['mean', 'std', 'count'],
        'K_95': ['mean', 'std'],
        'lambda_1': ['mean', 'std'],
    })

    # Get bin centers
    bin_centers = [(b.left + b.right) / 2 for b in grouped.index]

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # D_PR
    ax = axes[0]
    ax.plot(bin_centers, grouped['D_PR']['mean'], 'b-', linewidth=2)
    ax.fill_between(
        bin_centers,
        grouped['D_PR']['mean'] - grouped['D_PR']['std'],
        grouped['D_PR']['mean'] + grouped['D_PR']['std'],
        alpha=0.3, color='blue'
    )
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Seizure onset')
    ax.set_ylabel('$D_{PR}$')
    ax.set_title('(a) Participation Ratio Before and After Seizure')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # K_95
    ax = axes[1]
    ax.plot(bin_centers, grouped['K_95']['mean'], 'g-', linewidth=2)
    ax.fill_between(
        bin_centers,
        grouped['K_95']['mean'] - grouped['K_95']['std'],
        grouped['K_95']['mean'] + grouped['K_95']['std'],
        alpha=0.3, color='green'
    )
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('$K_{95}$')
    ax.set_title('(b) Components for 95% Variance')
    ax.grid(True, alpha=0.3)

    # lambda_1
    ax = axes[2]
    ax.plot(bin_centers, grouped['lambda_1']['mean'], 'purple', linewidth=2)
    ax.fill_between(
        bin_centers,
        grouped['lambda_1']['mean'] - grouped['lambda_1']['std'],
        grouped['lambda_1']['mean'] + grouped['lambda_1']['std'],
        alpha=0.3, color='purple'
    )
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Time relative to seizure onset (minutes)')
    ax.set_ylabel('$\\lambda_1$')
    ax.set_title('(c) Dominant Mode Fraction')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_distributions(profiles_df: pd.DataFrame, output_path: str = None):
    """Plot distributions of metrics for interictal vs preictal."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    interictal = profiles_df[profiles_df['label'] == 0]
    preictal = profiles_df[profiles_df['label'] == 1]

    metrics = ['D_PR', 'K_95', 'lambda_1', 'slope']
    titles = ['Participation Ratio', '$K_{95}$', 'Dominant Mode $\\lambda_1$', 'Spectral Slope']

    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        # Get data
        inter_vals = interictal[metric].dropna()
        pre_vals = preictal[metric].dropna()

        # Compute bins
        all_vals = pd.concat([inter_vals, pre_vals])
        bins = np.linspace(all_vals.quantile(0.01), all_vals.quantile(0.99), 50)

        ax.hist(inter_vals, bins=bins, alpha=0.6, density=True,
                label=f'Interictal (n={len(inter_vals)})', color='blue')
        ax.hist(pre_vals, bins=bins, alpha=0.6, density=True,
                label=f'Pre-ictal (n={len(pre_vals)})', color='orange')

        ax.set_xlabel(metric)
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


if __name__ == "__main__":
    # Load results
    results_dir = Path(__file__).parent.parent / 'results'
    figures_dir = Path(__file__).parent.parent / 'paper' / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Load profiles
    profiles_path = results_dir / 'chb01_profiles.csv'
    stats_path = results_dir / 'chb01_stats.csv'

    if not profiles_path.exists():
        print(f"Profiles not found at {profiles_path}")
        print("Run chb_mit_analysis.py first")
        exit(1)

    profiles_df = pd.read_csv(profiles_path)
    stats_df = pd.read_csv(stats_path)

    print(f"Loaded {len(profiles_df)} windows")
    print(f"\nClass counts:")
    print(profiles_df['label'].value_counts())

    # Generate figures
    print("\n" + "="*60)
    print("Generating figures...")
    print("="*60)

    # Figure 1: Class comparison
    print("\n1. Class comparison bar chart...")
    plot_class_comparison(stats_df, figures_dir / 'fig3_chb01_class_comparison.pdf')

    # Figure 2: Distributions
    print("\n2. Distribution histograms...")
    plot_distributions(profiles_df, figures_dir / 'fig4_chb01_distributions.pdf')

    # Figure 3: Seizure-aligned time series
    print("\n3. Seizure-aligned time series...")
    plot_timeseries_around_seizures(
        profiles_df,
        figures_dir / 'fig5_chb01_seizure_aligned.pdf',
        window_minutes=(-60, 10)
    )

    print("\n" + "="*60)
    print("Done! Figures saved to paper/figures/")
    print("="*60)

    # plt.show()  # Comment out for non-interactive use
