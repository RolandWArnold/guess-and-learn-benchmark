#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyzes a directory of G&L experiment results to provide precise statistics
on the number of completed experiments and the total number of data points
(learning steps) generated.

Usage:
    python calculate_experiment_stats.py /path/to/your/results
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm

def analyze_results_directory(results_dir: Path):
    """
    Scans a directory for JSON result files and calculates statistics.

    Args:
        results_dir: The path to the directory containing the results.
    """
    if not results_dir.is_dir():
        print(f"❌ Error: Directory not found at '{results_dir.resolve()}'")
        return

    print(f"🔍 Scanning for experiment files in '{results_dir.resolve()}'...")

    # Use rglob to find all matching files recursively
    json_files = list(results_dir.rglob("*_results.json"))

    if not json_files:
        print("❌ Error: No '*_results.json' files found in the specified directory.")
        return

    # Initialize counters
    valid_experiment_count = 0
    total_data_points = 0
    skipped_files = 0

    # Use tqdm for a progress bar
    pbar = tqdm(json_files, desc="Analyzing files", unit="file")
    for filepath in pbar:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # The key metric is the 'error_history' array.
            # Its length represents the number of learning steps (data points).
            error_history = data.get("error_history")

            if isinstance(error_history, list):
                valid_experiment_count += 1
                total_data_points += len(error_history)
            else:
                # The file is valid JSON but lacks the necessary key.
                print(f"\n⚠️ Warning: Skipped '{filepath.name}' (missing or invalid 'error_history' key).")
                skipped_files += 1

        except (json.JSONDecodeError, IOError) as e:
            # The file is corrupted, unreadable, or not valid JSON.
            print(f"\n⚠️ Warning: Skipped '{filepath.name}' (could not parse: {e}).")
            skipped_files += 1
            continue

    # --- Print the final report ---
    print("\n" + "="*40)
    print("📊 Experiment Statistics Report")
    print("="*40)
    print(f"Total Valid Experiments Found:    {valid_experiment_count:,}")
    print(f"Total Data Points (Learning Steps): {total_data_points:,}")
    if skipped_files > 0:
        print(f"Files Skipped (corrupt/malformed): {skipped_files}")
    print("="*40)
    print("\nThese are the precise numbers to use in your paper.")


def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Calculate precise statistics from G&L experiment result files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to the root directory containing the JSON result files."
    )
    args = parser.parse_args()

    analyze_results_directory(args.results_dir)

if __name__ == "__main__":
    main()