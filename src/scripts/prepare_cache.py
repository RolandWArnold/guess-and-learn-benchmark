#!/usr/bin/env python
"""
prepare_cache.py  ·  G&L replication toolkit
--------------------------------------------------
Downloads **all** datasets and pretrained model weights once so that
subsequent runs can execute fully offline (or behind a restrictive
fire-wall / CI container with no network access).

It honours the environment variable  HF_CACHE_DIR - if you set it to a
project-local folder the entire Hugging-Face cache will live there.
Otherwise we fall back to the user-level default ~/.cache/huggingface.

Vision datasets land in <data_dir>/ (default ./data). TorchVision
silently skips a download when the archive files already exist.
"""

import argparse
import os
from pathlib import Path

import torchvision
import torchvision.transforms as T
from datasets import load_dataset, DownloadMode
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------
DEF_DATASETS_VISION = ["mnist", "fashion-mnist", "cifar10", "svhn"]
DEF_MODELS_HF = [
    "bert-base-uncased",  # text
    "google/vit-base-patch16-224-in21k",  # vision transformer
]

# TorchVision weight enum requires model instantiation
TV_MODEL_CALLS = [
    lambda: torchvision.models.resnet50(weights="IMAGENET1K_V1"),
]

TRANSFORM = T.ToTensor()  # simple to-tensor for dataset download

# ---------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------


def prime_torchvision_dataset(name: str, root: Path):
    """Download a TorchVision dataset if not already cached."""
    name = name.lower()
    if name == "mnist":
        torchvision.datasets.MNIST(root, train=True, download=True, transform=TRANSFORM)
        torchvision.datasets.MNIST(root, train=False, download=True, transform=TRANSFORM)
    elif name == "fashion-mnist":
        torchvision.datasets.FashionMNIST(root, train=True, download=True, transform=TRANSFORM)
        torchvision.datasets.FashionMNIST(root, train=False, download=True, transform=TRANSFORM)
    elif name == "cifar10":
        torchvision.datasets.CIFAR10(root, train=True, download=True, transform=TRANSFORM)
        torchvision.datasets.CIFAR10(root, train=False, download=True, transform=TRANSFORM)
    elif name == "svhn":
        torchvision.datasets.SVHN(root, split="train", download=True, transform=TRANSFORM)
        torchvision.datasets.SVHN(root, split="test", download=True, transform=TRANSFORM)
    else:
        raise ValueError(f"Unknown TorchVision dataset: {name}")


def prime_hf_dataset(cache: Path):
    """Ensure AG News is cached under *cache* directory."""
    print("↳ Downloading AG News dataset …")
    load_dataset(
        "ag_news",
        cache_dir=str(cache),
        download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
    )


def prime_transformer(name: str, cache: Path):
    """Download a Hugging-Face model & tokenizer."""
    print(f"↳ Downloading {name} weights …")
    model = AutoModel.from_pretrained(name, cache_dir=str(cache))

    # Text models → prime tokenizer
    if "bert" in name or "gpt" in name or "roberta" in name:
        AutoTokenizer.from_pretrained(name, cache_dir=str(cache))
    # Vision models → prime image processor
    else:
        AutoImageProcessor.from_pretrained(name, cache_dir=str(cache))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="Vision dataset root dir (default ./data)")
    parser.add_argument("--hf_cache", type=str, default=os.getenv("HF_CACHE_DIR", "~/.cache/huggingface"), help="Hugging-Face cache dir (env HF_CACHE_DIR overrides)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    cache_dir = Path(args.hf_cache).expanduser()

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Priming TorchVision datasets ===")
    for ds in DEF_DATASETS_VISION:
        print(f"• {ds}")
        prime_torchvision_dataset(ds, data_dir)

    print("\n=== Priming Hugging-Face dataset ===")
    prime_hf_dataset(cache_dir)

    print("\n=== Priming Transformer checkpoints ===")
    for model in DEF_MODELS_HF:
        prime_transformer(model, cache_dir)

    print("\n=== Priming TorchVision model weights ===")
    for fn in TV_MODEL_CALLS:
        fn()  # downloads weights if missing

    print("\n✔ All artefacts cached. You can now set HF_HUB_OFFLINE=1 and run offline.")


if __name__ == "__main__":
    main()
