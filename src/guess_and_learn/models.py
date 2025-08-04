from __future__ import annotations

from dataclasses import dataclass, field
import contextlib, os, copy
import operator
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import TensorDataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .datasets import get_data_for_protocol


# --------------------------------------------------------------------- #
#  Global cache directory (shared with prepare_cache.py)                #
# --------------------------------------------------------------------- #
HF_CACHE = Path(os.getenv("HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface")))
HF_CACHE.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    """A structured container for all experiment parameters."""

    # Core experiment identifiers
    dataset: str
    model: str
    strategy: str
    track: str
    seed: int
    subset: int | None

    # Global run settings
    device: torch.device
    output_dir: Path
    reset_weights: bool = False

    # Derived track properties (parsed once)
    K: int = 1
    is_online: bool = False
    is_pretrained_track: bool = False

    # Model-specific hyperparameters
    hyperparams: dict = field(default_factory=dict)

    def exp_id(self) -> str:
        """Generates the unique filename base for this experiment."""
        base_name = f"{self.dataset}_{self.model}_{self.strategy}_{self.track}_seed{self.seed}"
        if self.subset:
            base_name += f"_s{self.subset}"
        return base_name

    def get(self, key, default=None):
        """Provides dict-like .get() for easy hyperparameter access."""
        return self.hyperparams.get(key, default)


# --- Base Model Wrapper ------------------------------------------------------
class GnlModel:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def update(self, X_labeled, Y_labeled, exp_config: ExperimentConfig):
        raise NotImplementedError

    def extract_features(self, X):
        if isinstance(X, list):
            raise NotImplementedError("Sub-class must implement text feature extraction.")
        return X.reshape(X.shape[0], -1).to(self.device)

    def reset(self):
        raise NotImplementedError("This model does not support --reset-weights.")


# --- k-NN Model --------------------------------------------------------------
class KnnModel(GnlModel):
    def __init__(self, n_neighbors=7, num_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.is_fitted = False
        self.num_classes = num_classes

    def _flat(self, X):
        if torch.is_tensor(X):
            return X.reshape(X.shape[0], -1).cpu().numpy()
        raise TypeError("KnnModel expects a numeric torch.Tensor; received non-tensor input.")

    def predict(self, X):
        if not self.is_fitted:
            if self.num_classes is None:
                raise ValueError("num_classes not set for cold-start prediction.")
            return torch.randint(0, self.num_classes, (X.shape[0],), dtype=torch.long, device=self.device)
        return torch.tensor(self.model.predict(self._flat(X)), dtype=torch.long, device=self.device)

    def predict_proba(self, X):
        if not self.is_fitted:
            if self.num_classes is None:
                raise ValueError("num_classes not set for cold-start proba.")
            return torch.ones(X.shape[0], self.num_classes, dtype=torch.float32, device=self.device) / self.num_classes
        proba = self.model.predict_proba(self._flat(X))
        return torch.tensor(proba, dtype=torch.float32, device=self.device)

    def update(self, X_labeled, Y_labeled, exp_config: ExperimentConfig):
        k = min(self.n_neighbors, len(X_labeled))
        if k != self.n_neighbors:
            self.model = KNeighborsClassifier(n_neighbors=k)
            self.n_neighbors = k
        X_flat = self._flat(X_labeled)
        Y_np = Y_labeled.cpu().numpy()
        if self.num_classes is None:
            self.num_classes = len(np.unique(Y_np))
        self.model.fit(X_flat, Y_np)
        self.is_fitted = True

    def reset(self):
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.is_fitted = False


# --- True Multi-class Perceptron --------------------------------------------
class PerceptronModel(GnlModel):
    def __init__(self, input_dim, num_classes, lr=0.1, device=None):
        super().__init__(device)
        self.lr = lr
        self.model = nn.Linear(input_dim, num_classes, bias=True).to(device)
        nn.init.xavier_uniform_(self.model.weight)
        nn.init.zeros_(self.model.bias)

    def predict(self, X):
        return torch.argmax(self.predict_proba(X), dim=1)

    def predict_proba(self, X):
        X_flat = X.reshape(X.shape[0], -1).to(self.device)
        with torch.no_grad():
            logits = self.model(X_flat)
        return torch.softmax(logits, dim=1)

    def _perceptron_step(self, x_vec, y_true):
        logits = self.model(x_vec.unsqueeze(0))
        y_pred = torch.argmax(logits, dim=1).item()
        if y_pred != y_true:
            self.model.weight.data[y_true] += self.lr * x_vec
            self.model.weight.data[y_pred] -= self.lr * x_vec
            self.model.bias.data[y_true] += self.lr
            self.model.bias.data[y_pred] -= self.lr

    def update(self, X_labeled, Y_labeled, exp_config: ExperimentConfig):
        # Correctly uses the boolean flag from the config object
        idxs = [-1] if exp_config.is_online else range(len(X_labeled))
        for i in idxs:
            x = X_labeled[i].reshape(-1).to(self.device)
            y = Y_labeled[i].item()
            self._perceptron_step(x, y)

    def reset(self):
        nn.init.xavier_uniform_(self.model.weight)
        nn.init.zeros_(self.model.bias)


# --- Simple CNN Model -------------------------------------------
class SimpleCnn(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, input_size=32):
        super(SimpleCnn, self).__init__()
        self.input_size = input_size
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            lin_in = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lin_in, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def reinit(self):
        def _w(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_w)


class CnnModel(GnlModel):
    def __init__(self, in_channels, num_classes, input_size=32, lr=0.01, device=None):
        super().__init__(device)
        self.model = SimpleCnn(in_channels, num_classes, input_size).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def predict(self, X):
        return torch.argmax(self.predict_proba(X), dim=1)

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device))
        return torch.softmax(logits, dim=1)

    def update(self, X_labeled, Y_labeled, exp_config: ExperimentConfig):
        self.model.train()
        epochs = exp_config.get("epochs_per_update", 5)
        batch_size = exp_config.get("train_batch_size", 32)
        loader = torch.utils.data.DataLoader(TensorDataset(X_labeled, Y_labeled), batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(data), target)
                loss.backward()
                self.optimizer.step()

    def extract_features(self, X):
        with torch.no_grad():
            return self.model.features(X.to(self.device)).flatten(1)

    def reset(self):
        """Re-initialise weights *and* rebuild the optimiser in a model-agnostic way."""

        # 1) Random-restart the network weights
        self.model.reinit()

        # 2) Recreate the optimiser with the same class & hyper-parameters
        opt_cls = self.optimizer.__class__  # e.g. torch.optim.SGD / AdamW
        opt_defaults = self.optimizer.defaults.copy()  # lr, momentum, betas, etc.
        opt_defaults.pop("params", None)
        self.optimizer = opt_cls(self.model.parameters(), **opt_defaults)


# --- TEXT-SPECIFIC MODELS (AG News) ---
class TextPerceptronModel(GnlModel):
    def __init__(self, input_dim, num_classes, vectorizer, lr=0.1, device=None):
        super().__init__(device)
        self.vec = vectorizer
        self.lr = lr
        self.num_classes = num_classes
        self.lin = nn.Linear(input_dim, num_classes, bias=True).to(device)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def _featurise(self, texts):
        X = self.vec.transform(texts).toarray()
        return torch.tensor(X, dtype=torch.float32, device=self.device)

    def predict(self, X):
        with torch.no_grad():
            logits = self.lin(self._featurise(X))
        return torch.argmax(torch.softmax(logits, dim=1), dim=1)

    def predict_proba(self, X):
        with torch.no_grad():
            logits = self.lin(self._featurise(X))
        return torch.softmax(logits, dim=1)

    def update(self, X_labeled, Y_labeled, exp_config: ExperimentConfig):
        # Correctly uses the boolean flag from the config object
        if exp_config.is_online:
            X_labeled, Y_labeled = X_labeled[-1:], Y_labeled[-1:]

        X_vec = self._featurise(X_labeled)
        for x_vec, y_true in zip(X_vec, Y_labeled):
            y_true = int(y_true)
            logits = self.lin(x_vec.unsqueeze(0))
            y_pred = int(torch.argmax(logits))
            if y_pred != y_true:
                self.lin.weight.data[y_true] += self.lr * x_vec
                self.lin.weight.data[y_pred] -= self.lr * x_vec
                self.lin.bias.data[y_true] += self.lr
                self.lin.bias.data[y_pred] -= self.lr

    def extract_features(self, X):
        X_tfidf = self.vec.transform(X).toarray()
        return torch.tensor(X_tfidf, dtype=torch.float32, device=self.device)

    def reset(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)


class TextKnnModel(KnnModel):
    def __init__(self, n_neighbors, num_classes, vectorizer, **kwargs):
        super().__init__(n_neighbors=n_neighbors, num_classes=num_classes, **kwargs)
        self.vec = vectorizer

    def _flat(self, X):
        if isinstance(X, list):
            return self.vec.transform(X).toarray()
        return super()._flat(X)

    def update(self, X_labeled, Y_labeled, exp_config: ExperimentConfig):
        super().update(X_labeled, Y_labeled, exp_config)

    def extract_features(self, X):
        X_tfidf = self.vec.transform(X).toarray()
        return torch.tensor(X_tfidf, dtype=torch.float32, device=self.device)


# --- Pretrained Model Wrapper -----------------------------------------------
class PretrainedModelWrapper(GnlModel):
    def __init__(self, model_name, num_classes, exp_config: ExperimentConfig, device=None):
        super().__init__(device)
        self.model_name = model_name
        self.tokenizer = None
        self.num_classes = num_classes
        self.exp_config = exp_config
        self.feature_extractor = None
        self.vision_normalizer = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        if "bert" in model_name:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes, cache_dir=HF_CACHE).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE)
            self.feature_extractor = getattr(self.model, "bert", getattr(self.model, "roberta", None))
        elif "resnet" in model_name:
            self.model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            self.model = self.model.to(device)
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        elif "vit" in model_name:
            self.model = AutoModel.from_pretrained(model_name, cache_dir=HF_CACHE).to(device)
            self.classifier = nn.Linear(self.model.config.hidden_size, num_classes).to(device)
        else:
            raise ValueError(f"Model name {model_name} not supported")

        self._initial_state = {
            "model": copy.deepcopy(self.model.state_dict()),
            "classifier": copy.deepcopy(getattr(self, "classifier", nn.Identity()).state_dict()),
        }
        self._configure_for_track()

    def _prep_vision(self, x_batch: torch.Tensor) -> torch.Tensor:
        # Move to GPU once at the beginning and ensure data is in [0, 1] range before processing
        x_batch = x_batch.to(self.device).clamp(0.0, 1.0)
        if x_batch.shape[-1] != 224:
            x_batch = T.functional.resize(x_batch, size=[224, 224], antialias=True)
        if x_batch.shape[1] == 1:
            x_batch = x_batch.expand(-1, 3, -1, -1)
        return self.vision_normalizer(x_batch)

    def _configure_for_track(self):
        # Correctly uses the boolean flag from the config object
        is_po_track = self.exp_config.is_pretrained_track and self.exp_config.is_online

        for p in self.model.parameters():
            p.requires_grad = True
        if hasattr(self, "classifier"):
            for p in self.classifier.parameters():
                p.requires_grad = True

        if is_po_track:
            for p in self.model.parameters():
                p.requires_grad = False
            # Unfreeze the final layer(s)
            if "resnet" in self.model_name:
                for p in self.model.fc.parameters():
                    p.requires_grad = True
            elif "bert" in self.model_name:
                for p in self.model.classifier.parameters():
                    p.requires_grad = True
            elif "vit" in self.model_name:
                for p in self.classifier.parameters():
                    p.requires_grad = True

        params = [p for p in self.model.parameters() if p.requires_grad]
        if hasattr(self, "classifier"):
            params.extend([p for p in self.classifier.parameters() if p.requires_grad])

        lr = self.exp_config.get("lr", 2e-5)
        if "resnet" in self.model_name and is_po_track:
            self.optimizer = optim.SGD(params, lr=self.exp_config.get("lr", 0.01))
        else:
            self.optimizer = optim.AdamW(params, lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def predict(self, X):
        return torch.argmax(self.predict_proba(X), dim=1)

    def predict_proba(self, X):
        self.model.eval()
        if hasattr(self, "classifier"):
            self.classifier.eval()
        cast = self.device.type in ("cuda", "mps")
        with torch.no_grad(), torch.autocast(device_type=self.device.type) if cast else contextlib.nullcontext():
            if self.tokenizer:
                texts = X if isinstance(X, list) else [str(i) for i in X]
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                logits = self.model(**inputs).logits
            elif "vit" in self.model_name:
                outputs = self.model(pixel_values=self._prep_vision(X))
                logits = self.classifier(outputs.last_hidden_state[:, 0])
            else:
                logits = self.model(self._prep_vision(X))
        return torch.softmax(logits, dim=1)

    def update(self, X_labeled, Y_labeled, exp_config: ExperimentConfig):
        self.model.train()
        if hasattr(self, "classifier"):
            self.classifier.train()

        if exp_config.is_online:
            if self.tokenizer:
                data = [str(X_labeled[-1])]
            else:
                data = X_labeled[-1:].to(self.device)
            target = Y_labeled[-1:].to(self.device)
            self.optimizer.zero_grad()
            cast = self.device.type in ("cuda", "mps")
            with torch.autocast(device_type=self.device.type) if cast else contextlib.nullcontext():
                if self.tokenizer:
                    inputs = self.tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                    logits = self.model(**inputs).logits
                elif "vit" in self.model_name:
                    pv = self._prep_vision(data)
                    logits = self.classifier(self.model(pixel_values=pv).last_hidden_state[:, 0])
                else:
                    logits = self.model(self._prep_vision(data))
                loss = self.loss_fn(logits, target)
            loss.backward()
            self.optimizer.step()
        else:  # Batch update
            epochs = exp_config.get("epochs_per_update", 3)
            batch_size = exp_config.get("train_batch_size", 16)
            idx_dataset = TensorDataset(torch.arange(len(X_labeled)))
            loader = torch.utils.data.DataLoader(idx_dataset, batch_size=batch_size, shuffle=True)
            for _ in range(epochs):
                for (idx_batch,) in loader:
                    if self.tokenizer:
                        data_slice = operator.itemgetter(*idx_batch.tolist())(X_labeled)
                        data = list(data_slice) if isinstance(data_slice, tuple) else [data_slice]
                    else:
                        data = X_labeled[idx_batch].to(self.device)
                    target = Y_labeled[idx_batch].to(self.device)
                    self.optimizer.zero_grad()
                    cast = self.device.type in ("cuda", "mps")
                    with torch.autocast(device_type=self.device.type) if cast else contextlib.nullcontext():
                        if self.tokenizer:
                            inputs = self.tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                            logits = self.model(**inputs).logits
                        elif "vit" in self.model_name:
                            pv = self._prep_vision(data)
                            logits = self.classifier(self.model(pixel_values=pv).last_hidden_state[:, 0])
                        else:
                            logits = self.model(self._prep_vision(data))
                        loss = self.loss_fn(logits, target)
                    loss.backward()
                    self.optimizer.step()

    def extract_features(self, X):
        with torch.no_grad():
            if self.tokenizer:
                texts = X if isinstance(X, list) else [str(i) for i in X]
                inp = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                hidden = self.feature_extractor(**inp).last_hidden_state[:, 0]
                return hidden
            elif "vit" in self.model_name:
                return self.model(pixel_values=self._prep_vision(X)).last_hidden_state[:, 0]
            else:
                x = self._prep_vision(X)
                feats = self.feature_extractor(x)
                return torch.flatten(feats, 1)

    def reset(self):
        self.model.load_state_dict(self._initial_state["model"])
        if hasattr(self, "classifier"):
            self.classifier.load_state_dict(self._initial_state["classifier"])
        self._configure_for_track()


# --- Factory -----------------------------------------------------------------
def get_model(exp_config: ExperimentConfig):
    """Factory function to instantiate a model based on the ExperimentConfig."""
    name = exp_config.model
    dataset_name = exp_config.dataset
    device = exp_config.device
    X, Y = get_data_for_protocol(dataset_name)
    num_classes = len(torch.unique(Y)) if torch.is_tensor(Y) else len(set(Y))
    input_shape = X.shape[1:] if torch.is_tensor(X) else None

    if name in ["knn", "text-knn"]:
        if dataset_name.lower() == "ag_news":
            print("Fitting TF-IDF vectorizer for k-NN on AG News...")
            vectorizer = TfidfVectorizer(max_features=20000)
            vectorizer.fit(X)
            return TextKnnModel(n_neighbors=7, num_classes=num_classes, vectorizer=vectorizer)
        return KnnModel(n_neighbors=7, num_classes=num_classes)

    elif name in ["perceptron", "text-perceptron"]:
        lr = exp_config.get("lr", 0.1)
        if dataset_name.lower() == "ag_news":
            print("Fitting TF-IDF vectorizer for Perceptron on AG News...")
            vectorizer = TfidfVectorizer(max_features=20000)
            vectorizer.fit(X)
            input_dim = len(vectorizer.get_feature_names_out())
            return TextPerceptronModel(input_dim, num_classes, vectorizer, lr=lr, device=device)
        if input_shape is None:
           raise ValueError("Cannot determine input shape for perceptron")
        return PerceptronModel(np.prod(input_shape), num_classes, lr=lr, device=device)

    elif name == "cnn":
        in_channels = input_shape[0] if len(input_shape) == 3 else 1
        lr = exp_config.get("lr", 0.01)
        return CnnModel(in_channels, num_classes, input_shape[1], lr=lr, device=device)

    elif name in ["resnet50", "vit-b-16", "bert-base"]:
        model_map = {
            "resnet50": "resnet50",
            "vit-b-16": "google/vit-base-patch16-224-in21k",
            "bert-base": "bert-base-uncased",
        }
        return PretrainedModelWrapper(model_map[name], num_classes, exp_config, device=device)

    else:
        raise ValueError(f"Model '{name}' not supported.")
