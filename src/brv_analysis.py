"""
Brain Rate Variability (BRV) Analysis

Implements concrete BRV index and temporal classification to address
reviewer feedback:
1. Define BRV_DPR(t) = variance of D_PR over rolling window
2. Show BRV time course around seizures
3. Test if temporal features improve classification AUC

Author: Ian Todd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def compute_brv_timeseries(
    profiles_df: pd.DataFrame,
    window_minutes: float = 5.0,
    step_minutes: float = 1.0,
    metrics: List[str] = ['D_PR', 'lambda_1', 'K_95']
) -> pd.DataFrame:
    """
    Compute Brain Rate Variability (variance of dimensional metrics over time).

    Parameters
    ----------
    profiles_df : pd.DataFrame
        Analysis results with time and metric columns
    window_minutes : float
        Rolling window size
    step_minutes : float
        Step size
    metrics : List[str]
        Which metrics to compute variance of

    Returns
    -------
    brv_df : pd.DataFrame
        BRV time series
    """
    window_sec = window_minutes * 60
    step_sec = step_minutes * 60

    df = profiles_df.sort_values('time').copy()
    times = df['time'].values

    results = []
    t = times.min() + window_sec / 2

    while t < times.max() - window_sec / 2:
        mask = (times >= t - window_sec / 2) & (times < t + window_sec / 2)

        if mask.sum() > 10:
            row = {'time': t, 'n_windows': mask.sum()}

            for metric in metrics:
                vals = df.loc[mask, metric].values
                row[f'BRV_{metric}'] = np.var(vals)
                row[f'mean_{metric}'] = np.mean(vals)

            # Get majority label in this window
            labels = df.loc[mask, 'label'].values
            row['label'] = int(np.median(labels[labels >= 0])) if (labels >= 0).any() else -1

            results.append(row)

        t += step_sec

    return pd.DataFrame(results)


def plot_brv_around_seizures(
    profiles_df: pd.DataFrame,
    output_path: str = None,
    window_minutes: Tuple[float, float] = (-60, 10)
):
    """
    Plot BRV time course aligned to seizure onset.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        Analysis results
    output_path : str, optional
        Path to save figure
    window_minutes : Tuple[float, float]
        Time window around seizure (minutes before, after)
    """
    # Find seizure onsets
    seizure_files = profiles_df[profiles_df['label'] == 2]['file'].unique()
    seizure_onsets = []

    for sf in seizure_files:
        file_data = profiles_df[profiles_df['file'] == sf]
        ictal_times = file_data[file_data['label'] == 2]['time']
        if len(ictal_times) > 0:
            seizure_onsets.append((sf, ictal_times.min()))

    print(f"Found {len(seizure_onsets)} seizures")

    # Compute BRV for each file and align to seizure
    aligned_brv = []

    for file_name, onset in tqdm(seizure_onsets, desc="Computing BRV"):
        file_data = profiles_df[profiles_df['file'] == file_name].copy()

        if len(file_data) < 100:
            continue

        # Compute BRV for this file
        brv_df = compute_brv_timeseries(file_data, window_minutes=5, step_minutes=1)

        if len(brv_df) == 0:
            continue

        # Align to seizure onset
        brv_df['rel_time'] = (brv_df['time'] - onset) / 60  # Convert to minutes

        # Filter to window around seizure
        pre_min, post_min = window_minutes
        mask = (brv_df['rel_time'] >= pre_min) & (brv_df['rel_time'] <= post_min)
        aligned_brv.append(brv_df[mask])

    if not aligned_brv:
        print("No aligned BRV data")
        return None

    all_aligned = pd.concat(aligned_brv, ignore_index=True)

    # Bin and average
    bins = np.linspace(window_minutes[0], window_minutes[1], 71)
    all_aligned['time_bin'] = pd.cut(all_aligned['rel_time'], bins)

    grouped = all_aligned.groupby('time_bin', observed=True).agg({
        'BRV_D_PR': ['mean', 'std', 'count'],
        'BRV_lambda_1': ['mean', 'std'],
        'mean_D_PR': ['mean', 'std'],
    })

    bin_centers = [(b.left + b.right) / 2 for b in grouped.index]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Mean D_PR
    ax = axes[0, 0]
    ax.plot(bin_centers, grouped['mean_D_PR']['mean'], 'b-', linewidth=2)
    ax.fill_between(bin_centers,
                    grouped['mean_D_PR']['mean'] - grouped['mean_D_PR']['std'],
                    grouped['mean_D_PR']['mean'] + grouped['mean_D_PR']['std'],
                    alpha=0.3, color='blue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Seizure onset')
    ax.set_ylabel('Mean $D_{PR}$')
    ax.set_title('(a) Mean Participation Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) BRV_D_PR (variance of D_PR)
    ax = axes[0, 1]
    ax.plot(bin_centers, grouped['BRV_D_PR']['mean'], 'g-', linewidth=2)
    ax.fill_between(bin_centers,
                    np.maximum(grouped['BRV_D_PR']['mean'] - grouped['BRV_D_PR']['std'], 0),
                    grouped['BRV_D_PR']['mean'] + grouped['BRV_D_PR']['std'],
                    alpha=0.3, color='green')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('$BRV_{D_{PR}}$ (variance)')
    ax.set_title('(b) Brain Rate Variability: $D_{PR}$')
    ax.grid(True, alpha=0.3)

    # (c) BRV_lambda_1
    ax = axes[1, 0]
    ax.plot(bin_centers, grouped['BRV_lambda_1']['mean'], 'purple', linewidth=2)
    ax.fill_between(bin_centers,
                    np.maximum(grouped['BRV_lambda_1']['mean'] - grouped['BRV_lambda_1']['std'], 0),
                    grouped['BRV_lambda_1']['mean'] + grouped['BRV_lambda_1']['std'],
                    alpha=0.3, color='purple')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Time relative to seizure (minutes)')
    ax.set_ylabel('$BRV_{\\lambda_1}$ (variance)')
    ax.set_title('(c) Brain Rate Variability: $\\lambda_1$')
    ax.grid(True, alpha=0.3)

    # (d) Summary statistics
    ax = axes[1, 1]

    # Compute baseline vs pre-seizure BRV
    baseline = all_aligned[all_aligned['rel_time'] < -30]
    pre_seizure = all_aligned[(all_aligned['rel_time'] >= -30) & (all_aligned['rel_time'] < 0)]

    if len(baseline) > 0 and len(pre_seizure) > 0:
        stats_text = f"""BRV Comparison (5-min windows)

Baseline (< -30 min):
  BRV_DPR = {baseline['BRV_D_PR'].mean():.3f} ± {baseline['BRV_D_PR'].std():.3f}
  BRV_λ₁ = {baseline['BRV_lambda_1'].mean():.5f} ± {baseline['BRV_lambda_1'].std():.5f}

Pre-seizure (-30 to 0 min):
  BRV_DPR = {pre_seizure['BRV_D_PR'].mean():.3f} ± {pre_seizure['BRV_D_PR'].std():.3f}
  BRV_λ₁ = {pre_seizure['BRV_lambda_1'].mean():.5f} ± {pre_seizure['BRV_lambda_1'].std():.5f}

Ratio (baseline / pre-seizure):
  BRV_DPR: {baseline['BRV_D_PR'].mean() / pre_seizure['BRV_D_PR'].mean():.2f}x
  BRV_λ₁: {baseline['BRV_lambda_1'].mean() / pre_seizure['BRV_lambda_1'].mean():.2f}x
"""
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.axis('off')
    ax.set_title('(d) Summary Statistics')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def temporal_classification(
    profiles_df: pd.DataFrame,
    history_minutes: float = 5.0
) -> Dict[str, float]:
    """
    Test if adding temporal features (variance over recent history)
    improves classification AUC.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        Analysis results
    history_minutes : float
        How far back to compute variance

    Returns
    -------
    results : dict
        Classification results for different feature sets
    """
    # Filter to interictal (0) and preictal (1)
    df = profiles_df[profiles_df['label'].isin([0, 1])].copy()
    df = df.sort_values(['file', 'time'])

    # Compute rolling variance features
    history_sec = history_minutes * 60

    print("Computing temporal features...")
    brv_dpr = []
    brv_lambda1 = []

    for file_name in tqdm(df['file'].unique()):
        file_data = df[df['file'] == file_name].sort_values('time')

        for idx in file_data.index:
            t = file_data.loc[idx, 'time']
            mask = (file_data['time'] >= t - history_sec) & (file_data['time'] < t)
            history = file_data.loc[mask]

            if len(history) > 5:
                brv_dpr.append(history['D_PR'].var())
                brv_lambda1.append(history['lambda_1'].var())
            else:
                brv_dpr.append(np.nan)
                brv_lambda1.append(np.nan)

    df['BRV_D_PR'] = brv_dpr
    df['BRV_lambda_1'] = brv_lambda1

    # Remove rows with NaN
    df_valid = df.dropna(subset=['BRV_D_PR', 'BRV_lambda_1'])

    print(f"Valid samples: {len(df_valid)}")

    y = df_valid['label'].values
    scaler = StandardScaler()

    results = {}

    # (1) Baseline: single-window features only
    X_baseline = df_valid[['D_PR', 'K_95', 'lambda_1', 'slope']].values
    X_baseline = scaler.fit_transform(X_baseline)

    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    scores = cross_val_score(clf, X_baseline, y, cv=5, scoring='roc_auc')
    results['baseline_AUC'] = scores.mean()
    results['baseline_AUC_std'] = scores.std()

    print(f"Baseline (single-window): AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    # (2) With BRV features
    X_with_brv = df_valid[['D_PR', 'K_95', 'lambda_1', 'slope', 'BRV_D_PR', 'BRV_lambda_1']].values
    X_with_brv = scaler.fit_transform(X_with_brv)

    scores = cross_val_score(clf, X_with_brv, y, cv=5, scoring='roc_auc')
    results['with_brv_AUC'] = scores.mean()
    results['with_brv_AUC_std'] = scores.std()

    print(f"With BRV features: AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    # (3) BRV features only
    X_brv_only = df_valid[['BRV_D_PR', 'BRV_lambda_1']].values
    X_brv_only = scaler.fit_transform(X_brv_only)

    scores = cross_val_score(clf, X_brv_only, y, cv=5, scoring='roc_auc')
    results['brv_only_AUC'] = scores.mean()
    results['brv_only_AUC_std'] = scores.std()

    print(f"BRV only: AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    return results


if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / 'results'
    figures_dir = Path(__file__).parent.parent / 'paper' / 'figures'

    profiles_path = results_dir / 'chb01_profiles.csv'

    if not profiles_path.exists():
        print(f"Profiles not found at {profiles_path}")
        exit(1)

    profiles_df = pd.read_csv(profiles_path)

    print("="*70)
    print("BRAIN RATE VARIABILITY ANALYSIS")
    print("="*70)

    # 1. Plot BRV time course around seizures
    print("\n1. BRV time course around seizures...")
    plot_brv_around_seizures(
        profiles_df,
        figures_dir / 'fig7_brv_timecourse.pdf'
    )

    # 2. Temporal classification
    print("\n2. Temporal classification (does adding BRV improve AUC?)...")
    clf_results = temporal_classification(profiles_df, history_minutes=5)

    print("\n" + "="*70)
    print("CLASSIFICATION SUMMARY")
    print("="*70)
    print(f"Baseline (D_PR, K_95, λ₁, slope):  AUC = {clf_results['baseline_AUC']:.3f} ± {clf_results['baseline_AUC_std']:.3f}")
    print(f"With BRV features:                  AUC = {clf_results['with_brv_AUC']:.3f} ± {clf_results['with_brv_AUC_std']:.3f}")
    print(f"BRV only (variance features):       AUC = {clf_results['brv_only_AUC']:.3f} ± {clf_results['brv_only_AUC_std']:.3f}")

    improvement = clf_results['with_brv_AUC'] - clf_results['baseline_AUC']
    print(f"\nImprovement from adding BRV: {improvement:+.3f}")
