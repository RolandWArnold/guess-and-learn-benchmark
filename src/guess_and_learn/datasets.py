"""
Dataset helpers for Guess-and-Learn
July 2025 – offline-first edition
"""

from __future__ import annotations
import os
from pathlib import Path

import torch, torchvision
import torchvision.transforms as transforms
import datasets
from datasets import load_dataset

torchvision.datasets.MNIST.mirrors = [
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
]

# --------------------------------------------------------------------- #
#  Shared HuggingFace cache                                             #
# --------------------------------------------------------------------- #
HF_CACHE = Path(os.getenv("HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface")))
HF_CACHE.mkdir(parents=True, exist_ok=True)
datasets.config.DOWNLOAD_MODE = datasets.DownloadMode.REUSE_CACHE_IF_EXISTS


def get_dataset(name: str, data_dir: str = "./data"):
    to_tensor = transforms.ToTensor()

    if name.lower() == "mnist":
        tx = transforms.Compose([to_tensor])
        return (
            torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=tx),
            torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=tx),
        )

    elif name.lower() == "fashion-mnist":
        tx = transforms.Compose([to_tensor])
        return (
            torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=tx),
            torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=tx),
        )

    elif name.lower() == "cifar10":
        tx = transforms.Compose([to_tensor])
        return (
            torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=tx),
            torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=tx),
        )

    elif name.lower() == "svhn":
        tx = transforms.Compose([transforms.Grayscale(1), to_tensor])
        return (
            torchvision.datasets.SVHN(data_dir, split="train", download=True, transform=tx),
            torchvision.datasets.SVHN(data_dir, split="test", download=True, transform=tx),
        )

    elif name.lower() == "ag_news":
        ds = load_dataset("ag_news", cache_dir=HF_CACHE)  # ← cache_dir added
        return ds["train"], ds["test"]

    else:
        raise ValueError(f"Unknown dataset '{name}'")


def get_data_for_protocol(dataset_name: str):
    """
    Prepares a dataset for the G&L protocol by combining train and test,
    and returning features and labels as tensors.
    For simplicity, we use the test set for many vision tasks as the pool.
    """
    if dataset_name.lower() in ["mnist", "fashion-mnist", "cifar10", "svhn"]:
        _, test = get_dataset(dataset_name)
        X = torch.stack([test[i][0] for i in range(len(test))])
        Y = torch.tensor([test[i][1] for i in range(len(test))])
        return X, Y

    elif dataset_name.lower() == "ag_news":
        _, test = get_dataset("ag_news")
        return test["text"], torch.tensor(test["label"])

    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")
