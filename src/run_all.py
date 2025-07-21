#!/usr/bin/env python
"""
Launch grid of Guess-and-Learn experiments with automatic
 * idempotent skipping (local or S3)
 * optional S3 download of finished artefacts

Environment knob
----------------
export RESULTS_S3_PREFIX=s3://my-bucket/gnl-runs   # enable remote cache
# or leave unset to stay purely on local disk.
"""
from __future__ import annotations
import os, sys, re, random, argparse, itertools, multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch, datasets  # 3rd-party

from guess_and_learn.datasets import get_data_for_protocol
from guess_and_learn.models import get_model
from guess_and_learn.strategies import get_strategy
from guess_and_learn.protocol import GnlProtocol, save_results
from guess_and_learn.io_utils import s3_enabled, s3_exists, s3_download, s3_upload

datasets.config.DOWNLOAD_MODE = datasets.DownloadMode.REUSE_CACHE_IF_EXISTS


# ────────────────────────────────────────────────────────────────────
# 1.  Per-experiment worker
# ────────────────────────────────────────────────────────────────────
def _exp_id(seed: int, dataset: str, model: str, strategy: str, track: str) -> str:
    return f"{dataset}_{model}_{strategy}_{track}_seed{seed}"


def run_single_experiment(exp_tuple):
    (seed, dataset, model_name, strategy_name, track, K, device, reset_weights, subset_cap, output_dir) = exp_tuple
    start = time.time()

    exp_tag = _exp_id(seed, dataset, model_name, strategy_name, track)
    results_p = Path(output_dir) / f"{exp_tag}_results.json"

    # --------------------------------------------------------------- #
    # 0.  Skip / restore if artefact exists                           #
    # --------------------------------------------------------------- #
    if results_p.exists():
        print(f"[SKIP-local] {exp_tag}")
        return
    if s3_enabled() and s3_exists(results_p):
        print(f"[SKIP-s3]   {exp_tag}  (downloading JSON)")
        s3_download(results_p)  # makes later plotting easy
        return

    print(f"[{os.getpid()}] {dataset} {model_name} {strategy_name} {track} seed={seed} …")

    # deterministic RNG — reproducible per run
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. load FULL pool (keeps correct num_classes)
    X_full, Y_full = get_data_for_protocol(dataset)

    # 2. instantiate model & strategy before any sub-sampling
    model = get_model(model_name, dataset, device, track_config={"track": track, "K": K, "seed": seed})
    strategy = get_strategy(strategy_name)
    track_cfg = {"track": track, "K": K, "reset_weights": reset_weights, "seed": seed}
    if model_name == "cnn":
        track_cfg.update(lr=0.01, epochs_per_update=5, train_batch_size=32)

    # 3. optional sub-sampling AFTER model init
    if subset_cap and len(Y_full) > subset_cap:
        sel = np.random.choice(len(Y_full), subset_cap, replace=False)
        X = [X_full[i] for i in sel] if isinstance(X_full, list) else X_full[sel]
        Y = Y_full[sel]
    else:
        X, Y = X_full, Y_full

    # 4. run protocol
    proto = GnlProtocol(model, strategy, X, Y, track_cfg)
    error_hist, lab_ix, is_err = proto.run()

    params = dict(seed=seed, dataset=dataset, model=model_name, strategy=strategy_name, track=track)
    duration = time.time() - start

    # persist locally
    save_results(duration, error_hist, lab_ix, is_err, params, output_dir, model, X_pool=X, Y_pool=Y)

    # mirror to S3 (if enabled)
    if s3_enabled():
        for ext in ("_results.json", "_plot.png", "_features.pt", "_labels.pt"):
            p = Path(output_dir) / f"{exp_tag}{ext}"
            s3_upload(p, quiet=True)


# ────────────────────────────────────────────────────────────────────
# 2.  Grid generator (unchanged except small tidy-ups)
# ────────────────────────────────────────────────────────────────────
DEFAULT_DATASETS = ["mnist", "fashion-mnist", "cifar10", "svhn", "ag_news"]
DEFAULT_MODELS_V = ["knn", "perceptron", "cnn", "resnet50", "vit-b-16"]
DEFAULT_MODELS_T = ["text-knn", "text-perceptron", "bert-base"]
DEFAULT_STRATEGIES = ["random", "confidence", "least_confidence", "margin", "entropy"]
DEFAULT_TRACKS = ["G&L-SO", "G&L-SB_50", "G&L-PO", "G&L-PB_50"]


def expand_grid(args) -> list[tuple]:
    datasets_ = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
    all_models = args.models.split(",") if args.models else (DEFAULT_MODELS_V + DEFAULT_MODELS_T)
    strategies = args.strategies.split(",") if args.strategies else DEFAULT_STRATEGIES
    tracks = args.tracks.split(",") if args.tracks else DEFAULT_TRACKS
    seeds = args.seeds

    models_v = [m for m in all_models if not m.startswith("text-")]
    models_t = [m for m in all_models if m.startswith("text-") or m == "bert-base"]

    # dataset-specific pool caps (csv: ds:N,ds2:M)
    subset_map = {}
    if args.subset_map:
        try:
            subset_map = {k: int(v) for k, v in (p.split(":") for p in args.subset_map.split(","))}
        except ValueError:
            print("⚠️  Malformed --subset_map; ignoring.", file=sys.stderr)

    device = torch.device(args.devices)
    jobs = []

    for seed, ds, strat, track in itertools.product(seeds, datasets_, strategies, tracks):
        K = int(re.search(r"(\d+)$", track).group(1)) if re.search(r"(\d+)$", track) else 1
        mdl_list = models_t if ds == "ag_news" else models_v
        for mdl in mdl_list:
            cap = subset_map.get(ds, args.subset)
            jobs.append((seed, ds, mdl, strat, track, K, device, args.reset_weights, cap, args.output_dir))
    return jobs


# ────────────────────────────────────────────────────────────────────
# 3.  Main entry-point
# ────────────────────────────────────────────────────────────────────
def main():
    # ensure safe mp start for PyTorch
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Run the full default grid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    default_device = "mps" if torch.backends.mps.is_available() else "cpu"
    ap.add_argument("--devices", default=default_device)
    ap.add_argument("--reset-weights", action="store_true")

    # flexible selection flags
    ap.add_argument("--datasets")
    ap.add_argument("--models")
    ap.add_argument("--strategies")
    ap.add_argument("--tracks")
    ap.add_argument("--subset", type=int)
    ap.add_argument("--subset_map")
    ap.add_argument("--output_dir", default="results")
    ap.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    # optional sharding for multinode
    ap.add_argument("--shard", nargs=2, type=int, metavar=("k", "n"))
    args = ap.parse_args()

    if args.all:
        args.datasets = ",".join(DEFAULT_DATASETS)
        args.models = ",".join(DEFAULT_MODELS_V + DEFAULT_MODELS_T)
        args.strategies = ",".join(DEFAULT_STRATEGIES)
        args.tracks = ",".join(DEFAULT_TRACKS)

    grid = expand_grid(args)
    if args.shard:
        k, n = args.shard
        grid = [job for idx, job in enumerate(grid) if idx % n == k]

    if not grid:
        print("Nothing to run — check flags.", file=sys.stderr)
        sys.exit(1)

    Path(args.output_dir).mkdir(exist_ok=True)
    print(f"Launching {len(grid)} experiments  →  {args.output_dir}")

    with mp.Pool(args.workers) as pool:
        pool.map(run_single_experiment, grid, chunksize=1)


if __name__ == "__main__":
    main()
