import argparse
import torch

torch.backends.cudnn.enabled = True
torch.backends.nnpack.enabled = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False

import numpy as np
import random
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from guess_and_learn.datasets import get_data_for_protocol
from guess_and_learn.models import get_model
from guess_and_learn.strategies import get_strategy
from guess_and_learn.protocol import GnlProtocol, save_results


def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"Using device: {device}")

    # Load data
    print(f"Loading dataset: {args.dataset}")
    X_pool, Y_pool = get_data_for_protocol(args.dataset)
    print(f"Pool size: {len(X_pool)}")

    # Get model, strategy
    print(f"Loading model: {args.model}")
    model = get_model(args.model, args.dataset, device)

    print(f"Using strategy: {args.strategy}")
    strategy = get_strategy(args.strategy)

    # Define track configuration
    track_config = {"track": args.track, "K": args.k_batch, "lr": args.lr, "epochs_per_update": args.epochs_per_update, "train_batch_size": args.train_batch_size}
    print(f"Track configuration: {track_config}")

    # Initialize and run protocol
    protocol = GnlProtocol(model, strategy, X_pool, Y_pool, track_config)
    error_history = protocol.run()

    # Save results
    params = vars(args)
    save_results(error_history, params, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Guess-and-Learn Benchmark")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "fashion-mnist", "cifar10", "svhn", "ag_news"])
    parser.add_argument("--model", type=str, required=True, choices=["knn", "perceptron", "cnn", "resnet50", "vit-b-16", "bert-base"])
    parser.add_argument("--strategy", type=str, required=True, choices=["random", "confidence", "least_confidence", "margin", "entropy"])
    parser.add_argument("--track", type=str, required=True, choices=["G&L-SO", "G&L-PO", "G&L-SB", "G&L-PB"])

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--k_batch", type=int, default=1, help="Batch size K for SB and PB tracks")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for model updates")
    parser.add_argument("--epochs_per_update", type=int, default=5, help="Number of epochs for batch updates")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training within an update step")

    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")

    args = parser.parse_args()

    # Some basic validation
    if args.track in ["G&L-SB", "G&L-PB"] and args.k_batch <= 1:
        raise ValueError("K must be > 1 for batch tracks (G&L-SB, G&L-PB)")
    if args.model == "bert-base" and args.dataset != "ag_news":
        raise ValueError("BERT model is only compatible with AG News dataset.")

    main(args)
