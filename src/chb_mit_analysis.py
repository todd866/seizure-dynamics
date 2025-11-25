"""
CHB-MIT Dimensional Profile Analysis

Main experiment script for analyzing dimensional collapse in seizure data.

Author: Ian Todd
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm

# Try to import EEG libraries
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    warnings.warn("mne not installed. Install with: pip install mne")

try:
    import pyedflib
    HAS_PYEDFLIB = True
except ImportError:
    HAS_PYEDFLIB = False

from dimensional_profile import (
    compute_dimensional_profile,
    sliding_window_profile,
    bandpass_filter
)
from download_chb_mit import parse_summary_file


def load_edf_file(filepath: str) -> Tuple[np.ndarray, float, List[str]]:
    """
    Load EDF file and return data, sampling rate, channel names.

    Parameters
    ----------
    filepath : str
        Path to EDF file

    Returns
    -------
    data : np.ndarray
        EEG data, shape (n_samples, n_channels)
    fs : float
        Sampling frequency
    channels : List[str]
        Channel names
    """
    if HAS_MNE:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        data = raw.get_data().T  # (n_samples, n_channels)
        fs = raw.info['sfreq']
        channels = raw.ch_names
        return data, fs, channels

    elif HAS_PYEDFLIB:
        f = pyedflib.EdfReader(filepath)
        n_channels = f.signals_in_file
        channels = f.getSignalLabels()
        fs = f.getSampleFrequency(0)

        n_samples = f.getNSamples()[0]
        data = np.zeros((n_samples, n_channels))

        for i in range(n_channels):
            data[:, i] = f.readSignal(i)

        f.close()
        return data, fs, channels

    else:
        raise ImportError("Need either mne or pyedflib to load EDF files")


def classify_windows(
    window_times: np.ndarray,
    seizure_times: List[Tuple[float, float]],
    preictal_duration: float = 1800.0,  # 30 minutes
    postictal_duration: float = 300.0,   # 5 minutes
    min_interictal_gap: float = 3600.0   # 1 hour from any seizure
) -> np.ndarray:
    """
    Classify windows as interictal, preictal, ictal, or postictal.

    Parameters
    ----------
    window_times : np.ndarray
        Center times of windows (seconds)
    seizure_times : List[Tuple[float, float]]
        List of (onset, offset) times for seizures
    preictal_duration : float
        How long before seizure counts as preictal
    postictal_duration : float
        How long after seizure counts as postictal
    min_interictal_gap : float
        Minimum distance from any seizure for interictal

    Returns
    -------
    labels : np.ndarray
        Labels: 0=interictal, 1=preictal, 2=ictal, 3=postictal, -1=excluded
    """
    labels = np.zeros(len(window_times), dtype=int)

    for i, t in enumerate(window_times):
        # Check each seizure
        for onset, offset in seizure_times:
            # Ictal
            if onset <= t <= offset:
                labels[i] = 2
                break

            # Preictal
            if onset - preictal_duration <= t < onset:
                labels[i] = 1
                break

            # Postictal
            if offset < t <= offset + postictal_duration:
                labels[i] = 3
                break

        # If not classified yet, check if far enough from all seizures
        if labels[i] == 0:
            min_dist = float('inf')
            for onset, offset in seizure_times:
                dist = min(abs(t - onset), abs(t - offset))
                min_dist = min(min_dist, dist)

            if min_dist < min_interictal_gap:
                labels[i] = -1  # Too close, exclude

    return labels


def analyze_subject(
    subject_dir: str,
    window_sec: float = 10.0,
    step_sec: float = 1.0,
    preictal_min: float = 30.0
) -> pd.DataFrame:
    """
    Analyze all recordings for a single subject.

    Parameters
    ----------
    subject_dir : str
        Path to subject directory (e.g., data/chb-mit/chb01)
    window_sec : float
        Window length in seconds
    step_sec : float
        Step size in seconds
    preictal_min : float
        Pre-ictal period in minutes

    Returns
    -------
    results : pd.DataFrame
        DataFrame with columns:
        - time: window center time (absolute, from recording start)
        - file: source file
        - label: 0=interictal, 1=preictal, 2=ictal, 3=postictal
        - D_PR, K_80, K_90, K_95, K_99, slope, lambda_1
    """
    subject_path = Path(subject_dir)
    subject_id = subject_path.name

    # Parse summary file
    summary_path = subject_path / f"{subject_id}-summary.txt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    info = parse_summary_file(str(summary_path))

    print(f"Subject {subject_id}: {len(info['files'])} files, {len(info['seizures'])} seizures")

    all_results = []

    for edf_file in tqdm(info['files'], desc=f"Processing {subject_id}"):
        edf_path = subject_path / edf_file
        if not edf_path.exists():
            print(f"  Skipping {edf_file} (not found)")
            continue

        try:
            # Load data
            data, fs, channels = load_edf_file(str(edf_path))

            # Filter
            data_filtered = bandpass_filter(data, fs, lowcut=1.0, highcut=45.0)

            # Get seizures in this file
            file_seizures = [
                (start, end) for f, start, end in info['seizures']
                if f == edf_file
            ]

            # Compute dimensional profiles
            window_samples = int(window_sec * fs)
            step_samples = int(step_sec * fs)

            times, profiles = sliding_window_profile(
                data_filtered,
                window_samples=window_samples,
                step_samples=step_samples,
                fs=fs
            )

            # Classify windows
            labels = classify_windows(
                times,
                file_seizures,
                preictal_duration=preictal_min * 60
            )

            # Build results dataframe for this file
            file_results = pd.DataFrame({
                'time': times,
                'file': edf_file,
                'label': labels,
                **profiles
            })

            all_results.append(file_results)

        except Exception as e:
            print(f"  Error processing {edf_file}: {e}")
            continue

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def compute_class_statistics(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for each window class.

    Parameters
    ----------
    results : pd.DataFrame
        Output from analyze_subject

    Returns
    -------
    stats : pd.DataFrame
        Mean and std of each metric for interictal vs preictal
    """
    label_names = {0: 'interictal', 1: 'preictal', 2: 'ictal', 3: 'postictal'}
    metrics = ['D_PR', 'K_80', 'K_90', 'K_95', 'K_99', 'slope', 'lambda_1']

    stats = []
    for label_val, label_name in label_names.items():
        subset = results[results['label'] == label_val]
        if len(subset) == 0:
            continue

        row = {'class': label_name, 'n_windows': len(subset)}
        for metric in metrics:
            row[f'{metric}_mean'] = subset[metric].mean()
            row[f'{metric}_std'] = subset[metric].std()

        stats.append(row)

    return pd.DataFrame(stats)


def run_classification_experiment(
    results: pd.DataFrame,
    features: List[str] = ['D_PR', 'K_95', 'slope', 'lambda_1']
) -> Dict[str, float]:
    """
    Train a simple classifier to distinguish preictal from interictal.

    Parameters
    ----------
    results : pd.DataFrame
        Output from analyze_subject
    features : List[str]
        Features to use

    Returns
    -------
    metrics : Dict[str, float]
        Classification metrics (accuracy, AUC, etc.)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    # Filter to interictal (0) and preictal (1) only
    subset = results[results['label'].isin([0, 1])].copy()

    if len(subset) < 100:
        return {'error': 'Insufficient data'}

    X = subset[features].values
    y = subset['label'].values

    # Handle NaN/inf
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) < 100:
        return {'error': 'Insufficient valid data'}

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train classifier
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)

    # Cross-validation
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc')

    return {
        'auc_mean': cv_scores.mean(),
        'auc_std': cv_scores.std(),
        'n_interictal': (y == 0).sum(),
        'n_preictal': (y == 1).sum(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze dimensional collapse in CHB-MIT data"
    )
    parser.add_argument(
        "--data-dir",
        default="data/chb-mit",
        help="Path to CHB-MIT data"
    )
    parser.add_argument(
        "--subject",
        default="chb01",
        help="Subject ID to analyze"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    subject_dir = Path(args.data_dir) / args.subject

    if not subject_dir.exists():
        print(f"Subject directory not found: {subject_dir}")
        print("Run download_chb_mit.py first")
        exit(1)

    # Run analysis
    print(f"\nAnalyzing {args.subject}...")
    results = analyze_subject(str(subject_dir))

    if results.empty:
        print("No results generated")
        exit(1)

    # Save results
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)

    results.to_csv(output_path / f"{args.subject}_profiles.csv", index=False)
    print(f"Saved profiles to {output_path / f'{args.subject}_profiles.csv'}")

    # Compute statistics
    stats = compute_class_statistics(results)
    print("\nClass statistics:")
    print(stats.to_string(index=False))

    stats.to_csv(output_path / f"{args.subject}_stats.csv", index=False)

    # Run classification
    print("\nRunning classification experiment...")
    clf_results = run_classification_experiment(results)
    print(f"Preictal vs Interictal AUC: {clf_results.get('auc_mean', 'N/A'):.3f} "
          f"(+/- {clf_results.get('auc_std', 'N/A'):.3f})")
