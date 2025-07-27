#!/usr/bin/env python
"""
Regenerates a single G&L error curve plot from a specified _results.json file.

This script is useful for tweaking plot aesthetics or recreating a single
figure without needing to rerun the original experiment.
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

# Use a non-interactive backend to prevent plots from showing up in a window
matplotlib.use("Agg")


def replot_experiment(json_path: Path):
    """
    Loads a results.json file and saves a new plot based on its data.
    """
    if not json_path.is_file() or not json_path.name.endswith("_results.json"):
        print(f"Error: Invalid input file. Please provide a path to a '_results.json' file.")
        return

    print(f"Loading data from: {json_path.name}")
    with open(json_path, "r") as f:
        data = json.load(f)

    # --- Extract necessary data from the JSON file ---
    error_history = data.get("error_history")
    params = data.get("params")

    if not error_history or not params:
        print("Error: JSON file is missing 'error_history' or 'params' key.")
        return

    # --- Create the plot ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(error_history, linewidth=2)

    # Construct a title from the parameters
    title = f"G&L Error Curve: {params.get('model', 'N/A').upper()} on " f"{params.get('dataset', 'N/A').upper()} ({params.get('strategy', 'N/A').capitalize()})"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Number of Labeled Samples", fontsize=12)
    ax.set_ylabel("Cumulative Errors", fontsize=12)
    ax.grid(True, alpha=0.5, linestyle="--")

    # Add an annotation for the final error count
    if error_history:
        final_error = error_history[-1]
        num_samples = len(error_history)
        ax.annotate(
            f"Final Error: {final_error}", xy=(num_samples - 1, final_error), xytext=(-80, 20), textcoords="offset points", arrowprops=dict(arrowstyle="->", color="red", alpha=0.7), ha="center"
        )

    # --- Save the new plot ---
    # Construct the output filename by replacing the suffix
    plot_path = json_path.with_name(json_path.stem.replace("_results", "") + "_plot.png")

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Successfully regenerated plot and saved to: {plot_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate a G&L plot from a results.json file.")
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to the input _results.json file.",
    )
    args = parser.parse_args()

    replot_experiment(args.json_file)


if __name__ == "__main__":
    main()
