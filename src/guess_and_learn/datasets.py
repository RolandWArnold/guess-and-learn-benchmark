import torch
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import TensorDataset


def get_dataset(name: str, data_dir: str = "./data"):
    """
    Loads the requested dataset and returns (train_set, test_set).

    New spec (July 2025):
      • No dataset-specific normalisation.
      • Vision images are left in raw [0,1] space; resizing happens later.
    """

    # Common “to-tensor” transform
    to_tensor = transforms.ToTensor()

    if name.lower() == "mnist":
        transform = transforms.Compose([to_tensor])  # 1×28×28
        train_set = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        return train_set, test_set

    elif name.lower() == "fashion-mnist":
        transform = transforms.Compose([to_tensor])
        train_set = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
        return train_set, test_set

    elif name.lower() == "cifar10":
        transform = transforms.Compose([to_tensor])  # 3×32×32
        train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        return train_set, test_set

    elif name.lower() == "svhn":
        # grayscale conversion remains (paper robustness test)
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                to_tensor,
            ]
        )
        train_set = torchvision.datasets.SVHN(data_dir, split="train", download=True, transform=transform)
        test_set = torchvision.datasets.SVHN(data_dir, split="test", download=True, transform=transform)
        return train_set, test_set

    elif name.lower() == "ag_news":
        # This is a special case. We'll use Hugging Face datasets and return TensorDatasets
        # Note: For G&L, we typically use the test set as the pool to label
        dataset = load_dataset("ag_news")  # For simplicity, we'll use the test set as the pool for G&L
        # In a real scenario, you'd tokenize and embed this. Here we just return it.
        # The model wrapper will need to handle tokenization.
        return dataset["train"], dataset["test"]

    else:
        raise ValueError(f"Dataset '{name}' not supported.")


def get_data_for_protocol(dataset_name: str):
    """
    Prepares a dataset for the G&L protocol by combining train and test,
    and returning features and labels as tensors.
    For simplicity, we use the test set for many vision tasks as the pool.
    """
    if dataset_name.lower() in ["mnist", "fashion-mnist", "cifar10", "svhn"]:
        _, test_set = get_dataset(dataset_name)

        # Use the test set as the pool for G&L
        X = torch.stack([test_set[i][0] for i in range(len(test_set))])
        Y = torch.tensor([test_set[i][1] for i in range(len(test_set))])

        return X, Y
    elif dataset_name.lower() == "ag_news":
        # For AG News, the model will handle tokenization.
        # We return the raw text and labels from the test set.
        _, test_set = get_dataset("ag_news")
        X = test_set["text"]
        Y = torch.tensor(test_set["label"])
        return X, Y
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported for protocol.")
