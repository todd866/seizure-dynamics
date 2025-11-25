"""
Visualization for Seizure Dimensional Collapse

Generate figures showing:
1. Eigenvalue spectra (fat tail vs collapsed)
2. D_eff time series around seizures
3. Class distributions (interictal vs preictal)
4. ROC curves for classification

Author: Ian Todd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Style settings
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150


def plot_eigenvalue_spectrum(
    eigenvalues_healthy: np.ndarray,
    eigenvalues_collapsed: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Eigenvalue Spectrum"
) -> plt.Axes:
    """
    Plot eigenvalue spectra showing fat tail vs collapsed.

    Parameters
    ----------
    eigenvalues_healthy : np.ndarray
        Eigenvalues from healthy/interictal state
    eigenvalues_collapsed : np.ndarray
        Eigenvalues from collapsed/preictal state
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title

    Returns
    -------
    ax : plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ranks = np.arange(1, len(eigenvalues_healthy) + 1)

    ax.semilogy(ranks, eigenvalues_healthy, 'b-', linewidth=2,
                label='Interictal (fat tail)', alpha=0.8)
    ax.semilogy(ranks, eigenvalues_collapsed, 'r-', linewidth=2,
                label='Pre-ictal (collapsed)', alpha=0.8)

    ax.set_xlabel('Component rank')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_cumulative_variance(
    eigenvalues_healthy: np.ndarray,
    eigenvalues_collapsed: np.ndarray,
    ax: Optional[plt.Axes] = None,
    thresholds: List[float] = [0.80, 0.90, 0.95, 0.99]
) -> plt.Axes:
    """
    Plot cumulative variance curves.

    Parameters
    ----------
    eigenvalues_healthy : np.ndarray
        Normalized eigenvalues from healthy state
    eigenvalues_collapsed : np.ndarray
        Normalized eigenvalues from collapsed state
    ax : plt.Axes, optional
        Axes to plot on
    thresholds : List[float]
        Threshold lines to draw

    Returns
    -------
    ax : plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ranks = np.arange(1, len(eigenvalues_healthy) + 1)

    cum_healthy = np.cumsum(eigenvalues_healthy)
    cum_collapsed = np.cumsum(eigenvalues_collapsed)

    ax.plot(ranks, cum_healthy, 'b-', linewidth=2,
            label='Interictal', alpha=0.8)
    ax.plot(ranks, cum_collapsed, 'r-', linewidth=2,
            label='Pre-ictal', alpha=0.8)

    # Threshold lines
    for thresh in thresholds:
        ax.axhline(thresh, color='gray', linestyle='--', alpha=0.5)
        ax.text(len(ranks) * 0.85, thresh + 0.01, f'{int(thresh*100)}%',
                fontsize=8, color='gray')

    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative variance explained')
    ax.set_title('Dimensional Profile: Components to Threshold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    return ax


def plot_deff_timeseries(
    times: np.ndarray,
    d_pr: np.ndarray,
    labels: np.ndarray,
    seizure_times: Optional[List[Tuple[float, float]]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Participation Ratio Over Time"
) -> plt.Axes:
    """
    Plot D_eff (participation ratio) time series around seizures.

    Parameters
    ----------
    times : np.ndarray
        Window center times (seconds or minutes)
    d_pr : np.ndarray
        Participation ratio values
    labels : np.ndarray
        Window labels (0=interictal, 1=preictal, 2=ictal)
    seizure_times : List[Tuple[float, float]], optional
        Seizure onset/offset times for shading
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title

    Returns
    -------
    ax : plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    # Color by label
    colors = {0: 'blue', 1: 'orange', 2: 'red', 3: 'purple', -1: 'gray'}
    color_array = [colors.get(l, 'gray') for l in labels]

    ax.scatter(times, d_pr, c=color_array, s=2, alpha=0.5)

    # Shade seizure periods
    if seizure_times:
        for onset, offset in seizure_times:
            ax.axvspan(onset, offset, color='red', alpha=0.2)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Participation Ratio (D_PR)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='blue', label='Interictal', alpha=0.7),
        mpatches.Patch(facecolor='orange', label='Pre-ictal', alpha=0.7),
        mpatches.Patch(facecolor='red', label='Ictal', alpha=0.7),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    return ax


def plot_class_distributions(
    results: pd.DataFrame,
    metric: str = 'D_PR',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot distribution of a metric for interictal vs preictal.

    Parameters
    ----------
    results : pd.DataFrame
        Results with 'label' and metric columns
    metric : str
        Which metric to plot
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    ax : plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    interictal = results[results['label'] == 0][metric].dropna()
    preictal = results[results['label'] == 1][metric].dropna()

    bins = np.linspace(
        min(interictal.min(), preictal.min()) if len(preictal) > 0 else interictal.min(),
        max(interictal.max(), preictal.max()) if len(preictal) > 0 else interictal.max(),
        50
    )

    ax.hist(interictal, bins=bins, alpha=0.6, label='Interictal',
            density=True, color='blue')
    if len(preictal) > 0:
        ax.hist(preictal, bins=bins, alpha=0.6, label='Pre-ictal',
                density=True, color='orange')

    ax.set_xlabel(metric)
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {metric}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_multi_metric_comparison(
    results: pd.DataFrame,
    metrics: List[str] = ['D_PR', 'K_95', 'slope', 'lambda_1'],
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a multi-panel figure comparing metrics across classes.

    Parameters
    ----------
    results : pd.DataFrame
        Results DataFrame
    metrics : List[str]
        Metrics to plot
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    fig : plt.Figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for i, metric in enumerate(metrics[:4]):
        plot_class_distributions(results, metric, ax=axes[i])

    plt.tight_layout()
    return fig


def plot_seizure_aligned_average(
    results: pd.DataFrame,
    seizure_onsets: List[float],
    metric: str = 'D_PR',
    window_minutes: Tuple[float, float] = (-60, 10),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot seizure-aligned average of a metric.

    Parameters
    ----------
    results : pd.DataFrame
        Results with 'time' and metric columns
    seizure_onsets : List[float]
        Seizure onset times (seconds)
    metric : str
        Which metric to plot
    window_minutes : Tuple[float, float]
        Time window around seizure (minutes before, after)
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    ax : plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    pre_min, post_min = window_minutes
    pre_sec, post_sec = pre_min * 60, post_min * 60

    # Collect all seizure-aligned traces
    traces = []
    for onset in seizure_onsets:
        mask = (results['time'] >= onset + pre_sec) & (results['time'] <= onset + post_sec)
        subset = results[mask].copy()
        if len(subset) > 0:
            subset['rel_time'] = (subset['time'] - onset) / 60  # Convert to minutes
            traces.append(subset[['rel_time', metric]])

    if not traces:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        return ax

    # Combine and bin
    all_traces = pd.concat(traces)

    # Bin by relative time
    bins = np.linspace(pre_min, post_min, 71)  # 1-minute bins
    all_traces['bin'] = pd.cut(all_traces['rel_time'], bins)
    grouped = all_traces.groupby('bin')[metric].agg(['mean', 'std', 'count'])

    # Plot
    bin_centers = [(b.left + b.right) / 2 for b in grouped.index]
    ax.plot(bin_centers, grouped['mean'], 'b-', linewidth=2)
    ax.fill_between(
        bin_centers,
        grouped['mean'] - grouped['std'],
        grouped['mean'] + grouped['std'],
        alpha=0.3
    )

    ax.axvline(0, color='red', linestyle='--', label='Seizure onset')
    ax.set_xlabel('Time relative to seizure (minutes)')
    ax.set_ylabel(metric)
    ax.set_title(f'Seizure-Aligned Average: {metric}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def create_summary_figure(
    results: pd.DataFrame,
    eigenvalues_interictal: np.ndarray,
    eigenvalues_preictal: np.ndarray,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create the main summary figure for the paper.

    Parameters
    ----------
    results : pd.DataFrame
        Analysis results
    eigenvalues_interictal : np.ndarray
        Example interictal eigenvalue spectrum
    eigenvalues_preictal : np.ndarray
        Example pre-ictal eigenvalue spectrum
    output_path : str, optional
        Path to save figure

    Returns
    -------
    fig : plt.Figure
    """
    fig = plt.figure(figsize=(12, 10))

    # Layout: 2x2 + bottom row
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])

    # (a) Eigenvalue spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    plot_eigenvalue_spectrum(eigenvalues_interictal, eigenvalues_preictal, ax=ax1)
    ax1.set_title('(a) Eigenvalue Spectrum: Fat Tail vs Collapse')

    # (b) Cumulative variance
    ax2 = fig.add_subplot(gs[0, 1])
    plot_cumulative_variance(eigenvalues_interictal, eigenvalues_preictal, ax=ax2)
    ax2.set_title('(b) Cumulative Variance: Components to Threshold')

    # (c) D_PR distribution
    ax3 = fig.add_subplot(gs[1, 0])
    plot_class_distributions(results, 'D_PR', ax=ax3)
    ax3.set_title('(c) Participation Ratio Distribution')

    # (d) K_95 distribution
    ax4 = fig.add_subplot(gs[1, 1])
    plot_class_distributions(results, 'K_95', ax=ax4)
    ax4.set_title('(d) Components for 95% Variance')

    # (e) Conceptual schematic (bottom spanning)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.text(0.5, 0.5,
             'Interictal: High D_eff, fat-tailed spectrum, rich dynamics\n'
             '↓\n'
             'Pre-ictal: D_eff collapse, steep spectrum, few modes dominate\n'
             '↓\n'
             'Ictal: Minimal D_eff, hypersynchrony, dimensional collapse complete',
             ha='center', va='center', fontsize=11,
             transform=ax5.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.axis('off')
    ax5.set_title('(e) The Collapse Cascade')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    return fig


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)

    n_channels = 23

    # Simulate interictal: many modes
    eig_interictal = np.exp(-0.15 * np.arange(n_channels))
    eig_interictal = eig_interictal / eig_interictal.sum()

    # Simulate pre-ictal: few modes dominate
    eig_preictal = np.exp(-0.4 * np.arange(n_channels))
    eig_preictal = eig_preictal / eig_preictal.sum()

    # Create dummy results
    n_windows = 1000
    results = pd.DataFrame({
        'time': np.arange(n_windows) * 10,
        'label': np.random.choice([0, 1], size=n_windows, p=[0.8, 0.2]),
        'D_PR': np.where(
            np.random.choice([0, 1], size=n_windows, p=[0.8, 0.2]) == 0,
            np.random.normal(8, 1.5, n_windows),
            np.random.normal(5, 1.2, n_windows)
        ),
        'K_95': np.where(
            np.random.choice([0, 1], size=n_windows, p=[0.8, 0.2]) == 0,
            np.random.normal(12, 2, n_windows),
            np.random.normal(7, 1.5, n_windows)
        ),
    })

    # Create summary figure
    fig = create_summary_figure(
        results,
        eig_interictal,
        eig_preictal,
        output_path='figures/fig1_dimensional_collapse.pdf'
    )

    plt.show()
