#!/usr/bin/env python3
import os
import sys
import subprocess

# --- Configuration ---
# Mimics ${OUTDIR:-./runs}
OUT_DIR = os.environ.get("OUTDIR", "./runs")
# Mimics ${DEVICES:-cpu}
DEVICES = os.environ.get("DEVICES", "cpu")

def run(specific_args):
    """
    Helper to run src/run_all.py with common and specific arguments.
    """
    # Base command
    cmd = [
        sys.executable, "src/run_all.py",
        *specific_args,
        "--output_dir", OUT_DIR,
        "--devices", DEVICES
    ]

    # Print command for logging (similar to 'echo + ...')
    # We join with spaces for readability, though the actual execution is safer
    print(f"+ {' '.join(cmd)}")

    # Run command. check=True halts execution if the command fails (mimics set -e)
    subprocess.run(cmd, check=True)

def main():
    print("=== Running ONLY k-Center Greedy Experiments ===")

    # 1. AG News — BERT-BASE (PO & PB_50)
    print("Starting AG News (BERT)...")
    run([
        "--datasets", "ag_news",
        "--models", "bert-base",
        "--strategies", "k_center_greedy",
        "--tracks", "G&L-PO", "G&L-PB_50",  # Python handles the '&' safely automatically
        "--seeds", "0", "1", "2", "3", "4",
        "--subset", "300"
    ])

    # 2. AG News — Small Models (SO & SB_50)
    print("Starting AG News (Small Models)...")
    run([
        "--datasets", "ag_news",
        "--models", "text-knn", "text-perceptron",
        "--strategies", "k_center_greedy",
        "--tracks", "G&L-SO", "G&L-SB_50",
        "--seeds", "0", "1", "2", "3", "4",
        "--subset", "300"
    ])

    # 3. MNIST — CNN (SO & All Batch Sizes)
    print("Starting MNIST (CNN)...")
    run([
        "--datasets", "mnist",
        "--models", "cnn",
        "--strategies", "k_center_greedy",
        "--tracks", "G&L-SO", "G&L-SB_10", "G&L-SB_50", "G&L-SB_200",
        "--seeds", "0", "1", "2", "3", "4",
        "--subset", "300"
    ])

    # 4. MNIST — Small Models (SO & SB_50)
    print("Starting MNIST (Small Models)...")
    run([
        "--datasets", "mnist",
        "--models", "knn", "perceptron",
        "--strategies", "k_center_greedy",
        "--tracks", "G&L-SO", "G&L-SB_50",
        "--seeds", "0", "1", "2", "3", "4",
        "--subset", "300"
    ])

    # 5. MNIST — Pretrained (PO & PB_50)
    print("Starting MNIST (Pretrained)...")
    run([
        "--datasets", "mnist",
        "--models", "resnet50", "vit-b-16",
        "--strategies", "k_center_greedy",
        "--tracks", "G&L-PO", "G&L-PB_50",
        "--seeds", "0", "1", "2", "3", "4",
        "--subset", "300"
    ])

    print("✅ All k-Center Greedy experiments completed.")

if __name__ == "__main__":
    main()
