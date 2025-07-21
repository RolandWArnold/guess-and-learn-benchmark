#!/usr/bin/env python

import argparse
import itertools
import multiprocessing as mp
import os
import random
import re
import sys

import numpy as np
import torch

from guess_and_learn.datasets import get_data_for_protocol
from guess_and_learn.models import get_model
from guess_and_learn.strategies import get_strategy
from guess_and_learn.protocol import GnlProtocol, save_results


# ────────────────────────────────────────────────────────────────────
# 1.  Per-experiment worker
# ────────────────────────────────────────────────────────────────────
def run_single_experiment(exp):
    (seed, dataset, model_name, strategy_name, track, K, device, reset_weights, subset_cap, output_dir) = exp

    # deterministic RNG — reproducible per run
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── 1. load FULL pool (keeps correct num_classes) ───────────────
    X_full, Y_full = get_data_for_protocol(dataset)

    # ── 2. instantiate model & strategy before any sub-sampling ─────
    model = get_model(model_name, dataset, device, track_config={"track": track, "K": K})
    strategy = get_strategy(strategy_name)

    # track-specific hyper-params (add more if needed)
    track_cfg = {"track": track, "K": K, "reset_weights": reset_weights}
    if model_name == "cnn":
        track_cfg.update({"lr": 0.01, "epochs_per_update": 5, "train_batch_size": 32})

    # ── 3. optional sub-sampling AFTER model init ───────────────────
    if subset_cap and len(Y_full) > subset_cap:
        sel = np.random.choice(len(Y_full), subset_cap, replace=False)
        if isinstance(X_full, list):
            X = [X_full[i] for i in sel]
        else:
            X = X_full[sel]
        Y = Y_full[sel]
    else:
        X, Y = X_full, Y_full

    # ── 4. run protocol ─────────────────────────────────────────────
    proto = GnlProtocol(model, strategy, X, Y, track_cfg)
    results = proto.run()  # (error_history, labeled_indices, is_error)

    params = {"seed": seed, "dataset": dataset, "model": model_name, "strategy": strategy_name, "track": track}
    save_results(*results, params, output_dir, model, X, Y)


# ────────────────────────────────────────────────────────────────────
# 2.  Grid generator
# ────────────────────────────────────────────────────────────────────
DEFAULT_DATASETS = ["mnist", "fashion-mnist", "cifar10", "svhn", "ag_news"]
DEFAULT_MODELS_V = ["knn", "perceptron", "cnn", "resnet50", "vit-b-16"]
DEFAULT_MODELS_T = ["text-knn", "text-perceptron", "bert-base"]
DEFAULT_STRATEGIES = ["random", "confidence", "least_confidence", "margin", "entropy"]
DEFAULT_TRACKS = ["G&L-SO", "G&L-SB_50", "G&L-PO", "G&L-PB_50"]


def expand_grid(args):
    # consolidate CLI flags (or fall back to defaults)
    datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
    all_models = args.models.split(",") if args.models else (DEFAULT_MODELS_V + DEFAULT_MODELS_T)
    strategies = args.strategies.split(",") if args.strategies else DEFAULT_STRATEGIES
    tracks = args.tracks.split(",") if args.tracks else DEFAULT_TRACKS
    seeds = args.seeds

    # split model list so AG News gets only text models
    models_v = [m for m in all_models if not m.startswith("text-")]
    models_t = [m for m in all_models if m.startswith("text-") or m == "bert-base"]

    # optional per-dataset subset map (csv: ds:N,ds2:M …)
    subset_map = {}
    if args.subset_map:
        try:
            subset_map = {k: int(v) for k, v in (pair.split(":") for pair in args.subset_map.split(","))}
        except ValueError:
            print("⚠️  Malformed --subset_map string; ignoring.", file=sys.stderr)

    device = torch.device(args.devices)
    grid = []

    for seed, ds, strat, track in itertools.product(seeds, datasets, strategies, tracks):
        # derive K from track suffix (e.g. SB_50 → K=50)
        m = re.search(r"(\d+)$", track)
        K = int(m.group(1)) if m else 1

        # choose candidate models for this dataset
        cand_models = models_t if ds == "ag_news" else models_v

        for mdl in cand_models:
            subset_cap = subset_map.get(ds, args.subset)  # per-dataset override
            grid.append((seed, ds, mdl, strat, track, K, device, args.reset_weights, subset_cap, args.output_dir))
    return grid


# ────────────────────────────────────────────────────────────────────
# 3.  Main entry-point
# ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run full default grid (ignores other selection flags)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    parser.add_argument("--devices", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--reset-weights", action="store_true", help="Re-initialise model weights before each batch update")

    # flexible selection flags
    parser.add_argument("--datasets", type=str, help="CSV list")
    parser.add_argument("--models", type=str, help="CSV list (prefix text-* for AG News)")
    parser.add_argument("--strategies", type=str, help="CSV list")
    parser.add_argument("--tracks", type=str, help="CSV list")
    parser.add_argument("--subset", type=int, help="Global pool cap (overridden by --subset_map)")
    parser.add_argument("--subset_map", type=str, help="csv like ds1:N,ds2:M to override subset per dataset")
    parser.add_argument("--output_dir", type=str, default="results", help="Where to dump JSON/PNG/PT files")
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()), help="# processes for multiprocessing pool")
    args = parser.parse_args()

    # --all overrides everything else
    if args.all:
        args.datasets = ",".join(DEFAULT_DATASETS)
        args.models = ",".join(DEFAULT_MODELS_V + DEFAULT_MODELS_T)
        args.strategies = ",".join(DEFAULT_STRATEGIES)
        args.tracks = ",".join(DEFAULT_TRACKS)

    grid = expand_grid(args)
    if not grid:
        print("No experiments selected – check your flag combination.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Launching {len(grid)} experiments → {args.output_dir}")

    with mp.Pool(processes=args.workers) as pool:
        pool.map(run_single_experiment, grid)


if __name__ == "__main__":
    main()
