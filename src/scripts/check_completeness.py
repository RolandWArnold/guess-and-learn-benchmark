#!/usr/bin/env python
"""
Scans the results directory to verify the completeness of the G&L benchmark suite.

This script checks for a complete set of random seeds (0-4) for all 80 valid
experiment combinations. It then generates and prints the precise `run_all.py`
commands needed to run only the missing experiments.
"""
import argparse
from pathlib import Path

# --- Define the complete, valid experiment grid ---

# Based on the final, correct logic for the benchmark
DATASETS = ["mnist", "ag_news"]
STRATEGIES = ["random", "entropy", "margin", "confidence", "least_confidence"]
SEEDS = list(range(5))

# Valid pairings for each dataset
VALID_PAIRS = {
    "mnist": [
        ("knn", "G&L-SO"),
        ("knn", "G&L-SB_50"),
        ("perceptron", "G&L-SO"),
        ("perceptron", "G&L-SB_50"),
        ("cnn", "G&L-SO"),
        ("cnn", "G&L-SB_50"),
        ("resnet50", "G&L-PO"),
        ("resnet50", "G&L-PB_50"),
        ("vit-b-16", "G&L-PO"),
        ("vit-b-16", "G&L-PB_50"),
    ],
    "ag_news": [("text-knn", "G&L-SO"), ("text-knn", "G&L-SB_50"), ("text-perceptron", "G&L-SO"), ("text-perceptron", "G&L-SB_50"), ("bert-base", "G&L-PO"), ("bert-base", "G&L-PB_50")],
}

# Define all tracks for robust parsing
ALL_TRACKS = {"G&L-SO", "G&L-PO", "G&L-SB_50", "G&L-PB_50"}


def generate_expected_filenames(subset_size: int) -> set:
    """Generates the set of all expected filenames for the s300 subset."""
    expected = set()
    for dataset, pairs in VALID_PAIRS.items():
        for model, track in pairs:
            for strategy in STRATEGIES:
                for seed in SEEDS:
                    stem = f"{dataset}_{model}_{strategy}_{track}_" f"seed{seed}_s{subset_size}"
                    expected.add(f"{stem}_results.json")
    return expected


def main():
    parser = argparse.ArgumentParser(description="Check for missing G&L experiments.")
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to the directory containing the JSON result files.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=300,
        help="The subset size to check for (e.g., 300).",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"Error: Directory not found at '{args.results_dir}'")
        return

    # --- Find missing files ---
    expected_files = generate_expected_filenames(args.subset)
    existing_files = {f.name for f in args.results_dir.glob(f"*_s{args.subset}_results.json")}
    missing_files = sorted(list(expected_files - existing_files))

    if not missing_files:
        print(f"✅ All {len(expected_files)} experiments for the s{args.subset} subset are complete!")
        return

    print(f"Found {len(missing_files)} missing experiments for the s{args.subset} subset.")

    # --- Group missing files to generate commands ---
    commands_to_run = {}
    for filename in missing_files:
        try:
            stem = filename.removesuffix("_results.json")

            # 1. Parse from the end (structured part)
            rest, subset_str = stem.rsplit("_s", 1)
            rest, seed_str = rest.rsplit("_seed", 1)
            seed = int(seed_str)

            # 2. Find the track
            track = next((t for t in ALL_TRACKS if rest.endswith(f"_{t}")), None)
            if not track:
                continue
            rest = rest.removesuffix(f"_{track}")

            # 3. Find the strategy (longest match first for 'least_confidence')
            strategy = next((s for s in sorted(STRATEGIES, key=len, reverse=True) if rest.endswith(f"_{s}")), None)
            if not strategy:
                continue
            rest = rest.removesuffix(f"_{strategy}")

            # 4. What's left is "dataset_model"
            dataset, model = None, None
            if rest.startswith("ag_news_"):
                dataset = "ag_news"
                model = rest.removeprefix("ag_news_")
            else:
                # Handle single-word datasets
                parts = rest.split("_", 1)
                if len(parts) == 2 and parts[0] in DATASETS:
                    dataset, model = parts

            if not all([dataset, model]):
                continue

            # Group experiments to minimize the number of commands
            key = (dataset, model, track)
            if key not in commands_to_run:
                commands_to_run[key] = {"strategies": set(), "seeds": set()}

            commands_to_run[key]["strategies"].add(strategy)
            commands_to_run[key]["seeds"].add(seed)

        except (ValueError, IndexError):
            print(f"  - Warning: Could not parse filename, skipping: {filename}")
            continue

    # --- Print the commands ---
    print("\n--- To run the missing experiments, execute the following commands: ---\n")
    for (dataset, model, track), params in commands_to_run.items():
        strategies_str = ",".join(sorted(list(params["strategies"])))
        seeds_str = " ".join(map(str, sorted(list(params["seeds"]))))

        command = (
            f"./run_all.py \\\n"
            f"    --datasets {dataset} \\\n"
            f"    --models {model} \\\n"
            f"    --strategies {strategies_str} \\\n"
            f"    --tracks '{track}' \\\n"
            f"    --subset {args.subset} \\\n"
            f"    --seeds {seeds_str} \\\n"
            f"    --workers 8\n"
        )
        print(command)


if __name__ == "__main__":
    main()
