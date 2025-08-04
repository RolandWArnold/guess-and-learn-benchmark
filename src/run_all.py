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
from guess_and_learn.models import ExperimentConfig, get_model
from guess_and_learn.strategies import get_strategy
from guess_and_learn.protocol import GnlProtocol, save_results
from guess_and_learn.io_utils import s3_enabled, s3_exists, s3_download, s3_upload

from PIL import Image, UnidentifiedImageError

datasets.config.DOWNLOAD_MODE = datasets.DownloadMode.REUSE_CACHE_IF_EXISTS

torch.backends.cudnn.enabled = True
torch.backends.nnpack.enabled = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False


def run_single_experiment(exp_config: ExperimentConfig):
    start = time.time()

    exp_tag = exp_config.exp_id()

    # --------------------------------------------------------------- #
    # 0.  Skip / restore if artefact exists                           #
    # --------------------------------------------------------------- #

    # ─── Treat the plot as the final artifact ──────────────────────────
    plot_path = exp_config.output_dir / f"{exp_tag}_plot.png"

    # helper to delete any stray files
    def _cleanup():
        for ext in ("_results.json", "_plot.png", "_labels.pt", "_features.pt"):
            p = exp_config.output_dir / f"{exp_tag}{ext}"
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    # 0a.  If we have a valid, fully‐written plot, assume everything else is done
    if plot_path.exists():
        try:
            with Image.open(plot_path) as img:
                img.verify()  # throws if truncated or invalid
            print(f"[SKIP]    {exp_tag} (plot OK)")
            return
        except (IOError, UnidentifiedImageError):
            print(f"[WARN]    {exp_tag} plot corrupted—cleaning up and re-running")
            _cleanup()

    # 0b.  Otherwise, if using S3, try the same there
    if s3_enabled() and s3_exists(plot_path):
        # download, then verify
        s3_download(plot_path)
        try:
            with Image.open(plot_path) as img:
                img.verify()
            print(f"[SKIP-s3] {exp_tag} (plot OK in S3)")
            return
        except (IOError, UnidentifiedImageError):
            print(f"[WARN]    {exp_tag} plot corrupted in S3—cleaning up and re-running")
            _cleanup()

    print(f"[{os.getpid()}] {exp_config.dataset} {exp_config.model} {exp_config.strategy} {exp_config.track} seed={exp_config.seed} …")

    # deterministic RNG — reproducible per run
    random.seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    torch.manual_seed(exp_config.seed)

    # 1. load FULL pool
    X_full, Y_full = get_data_for_protocol(exp_config.dataset)

    # 2. instantiate model & strategy before any sub-sampling
    model = get_model(exp_config)
    strategy = get_strategy(exp_config.strategy)

    if exp_config.subset and len(Y_full) > exp_config.subset:
        sel = np.random.choice(len(Y_full), exp_config.subset, replace=False)
        X = [X_full[i] for i in sel] if isinstance(X_full, list) else X_full[sel]
        Y = Y_full[sel]
    else:
        X, Y = X_full, Y_full

    # 3. run protocol
    proto = GnlProtocol(model, strategy, X, Y, exp_config)
    error_hist, lab_ix, is_err = proto.run()

    params = dict(seed=exp_config.seed, dataset=exp_config.dataset, model=exp_config.model, strategy=exp_config.strategy, track=exp_config.track, subset=exp_config.subset)
    duration = time.time() - start

    # persist locally
    save_results(duration, error_hist, lab_ix, is_err, params, exp_config.output_dir, model, X_pool=X, Y_pool=Y)

    # mirror to S3 (if enabled)
    if s3_enabled():
        for ext in ("_results.json", "_plot.png", "_features.pt", "_labels.pt"):
            p = exp_config.output_dir / f"{exp_tag}{ext}"
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

    models_t = [m for m in all_models if m.startswith("text-") or m == "bert-base"]
    models_v = [m for m in all_models if m not in models_t]

    # dataset-specific pool caps (csv: ds:N,ds2:M)
    subset_map = {}
    if args.subset_map:
        try:
            subset_map = {k: int(v) for k, v in (p.split(":") for p in args.subset_map.split(","))}
        except ValueError:
            print("⚠️  Malformed --subset_map; ignoring.", file=sys.stderr)

    pretrained_models = {"resnet50", "vit-b-16", "bert-base"}
    device = torch.device(args.devices)
    output_dir = Path(args.output_dir)
    jobs = []

    for seed, ds, strat, track_name in itertools.product(seeds, datasets_, strategies, tracks):
        K = int(re.search(r"(\d+)$", track_name).group(1)) if re.search(r"(\d+)$", track_name) else 1
        mdl_list = models_t if ds == "ag_news" else models_v
        for mdl in mdl_list:
            is_pretrained_model = mdl in pretrained_models
            is_pretrained_track = track_name.startswith("G&L-P")

            if is_pretrained_model and not is_pretrained_track:
                continue
            if not is_pretrained_model and is_pretrained_track:
                continue

            # Create the single, comprehensive config object
            exp_config = ExperimentConfig(
                dataset=ds,
                model=mdl,
                strategy=strat,
                track=track_name,
                seed=seed,
                subset=subset_map.get(ds, args.subset),
                device=device,
                output_dir=output_dir,
                reset_weights=args.reset_weights,
                K=K,
                is_online=track_name.startswith(("G&L-SO", "G&L-PO")),
                is_pretrained_track=is_pretrained_track,
            )
            if mdl == "cnn":
                exp_config.hyperparams.update(lr=0.01, epochs_per_update=5, train_batch_size=32)

            jobs.append(exp_config)
    return jobs


# ────────────────────────────────────────────────────────────────────
# 3.  Main entry-point
# ────────────────────────────────────────────────────────────────────
def main():
    # ensure safe mp start for PyTorch
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Run the full default grid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
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
