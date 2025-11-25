"""
Download CHB-MIT Scalp EEG Database from PhysioNet

The CHB-MIT database contains 24 pediatric patients with intractable seizures.
- 23-channel scalp EEG
- 256 Hz sampling rate
- 163 annotated seizures
- ~844 hours of recordings

Source: https://physionet.org/content/chbmit/1.0.0/

Author: Ian Todd
"""

import os
import subprocess
from pathlib import Path
from typing import Optional
import argparse


def download_chb_mit(
    output_dir: str = "data/chb-mit",
    subjects: Optional[list] = None,
    dry_run: bool = False
) -> None:
    """
    Download CHB-MIT database using wget.

    Parameters
    ----------
    output_dir : str
        Directory to save data
    subjects : list, optional
        Specific subjects to download (e.g., ['chb01', 'chb02'])
        If None, downloads all subjects
    dry_run : bool
        If True, just print what would be downloaded
    """
    base_url = "https://physionet.org/files/chbmit/1.0.0/"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if subjects is None:
        # Download everything
        cmd = [
            "wget", "-r", "-N", "-c", "-np",
            "--directory-prefix", str(output_path),
            base_url
        ]
        print(f"Downloading full CHB-MIT database to {output_path}")
        print("This is ~4GB and may take a while...")
    else:
        # Download specific subjects
        for subj in subjects:
            subj_url = f"{base_url}{subj}/"
            cmd = [
                "wget", "-r", "-N", "-c", "-np",
                "--directory-prefix", str(output_path),
                subj_url
            ]
            print(f"Downloading {subj}...")

            if not dry_run:
                subprocess.run(cmd)

        return

    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
    else:
        subprocess.run(cmd)


def download_with_wfdb(
    output_dir: str = "data/chb-mit",
    subject: str = "chb01"
) -> None:
    """
    Alternative download using wfdb library.

    This downloads individual records more cleanly but requires
    knowing the record names.

    Parameters
    ----------
    output_dir : str
        Directory to save data
    subject : str
        Subject ID (e.g., 'chb01')
    """
    try:
        import wfdb
    except ImportError:
        print("wfdb not installed. Run: pip install wfdb")
        return

    output_path = Path(output_dir) / subject
    output_path.mkdir(parents=True, exist_ok=True)

    # CHB-MIT record naming convention: chbXX_YY
    # We need to fetch the summary file first to get record names
    print(f"Downloading {subject} using wfdb...")

    # Download the summary file
    summary_url = f"chbmit/{subject}/{subject}-summary.txt"
    try:
        wfdb.io.dl_files(
            "chbmit",
            str(output_path),
            [f"{subject}/{subject}-summary.txt"]
        )
        print(f"Downloaded summary for {subject}")
    except Exception as e:
        print(f"Error downloading summary: {e}")


def parse_summary_file(summary_path: str) -> dict:
    """
    Parse CHB-MIT summary file to extract seizure annotations.

    Parameters
    ----------
    summary_path : str
        Path to the *-summary.txt file

    Returns
    -------
    info : dict
        Dictionary containing:
        - 'files': list of EDF files
        - 'seizures': list of (file, start_sec, end_sec) tuples
    """
    info = {'files': [], 'seizures': []}

    with open(summary_path, 'r') as f:
        lines = f.readlines()

    current_file = None
    n_seizures = 0
    seizure_starts = []
    seizure_ends = []

    for line in lines:
        line = line.strip()

        if line.startswith("File Name:"):
            # Save previous file info
            if current_file and n_seizures > 0:
                for start, end in zip(seizure_starts, seizure_ends):
                    info['seizures'].append((current_file, start, end))

            # Start new file
            current_file = line.split(":")[1].strip()
            info['files'].append(current_file)
            n_seizures = 0
            seizure_starts = []
            seizure_ends = []

        elif line.startswith("Number of Seizures"):
            n_seizures = int(line.split(":")[1].strip())

        elif "Seizure Start Time" in line or "Seizure  Start Time" in line:
            # Handle both "Seizure Start" and "Seizure  Start" (double space)
            time_str = line.split(":")[1].strip().replace(" seconds", "")
            seizure_starts.append(int(time_str))

        elif "Seizure End Time" in line or "Seizure  End Time" in line:
            time_str = line.split(":")[1].strip().replace(" seconds", "")
            seizure_ends.append(int(time_str))

    # Don't forget the last file
    if current_file and n_seizures > 0:
        for start, end in zip(seizure_starts, seizure_ends):
            info['seizures'].append((current_file, start, end))

    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download CHB-MIT EEG database"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/chb-mit",
        help="Output directory"
    )
    parser.add_argument(
        "--subjects", "-s",
        nargs="+",
        default=None,
        help="Specific subjects to download (e.g., chb01 chb02)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just print what would be downloaded"
    )

    args = parser.parse_args()

    # For quick testing, just download first 2 subjects
    if args.subjects is None:
        print("Tip: For quick testing, use --subjects chb01 chb02")
        print("Full database is ~4GB")

    download_chb_mit(
        output_dir=args.output,
        subjects=args.subjects,
        dry_run=args.dry_run
    )
