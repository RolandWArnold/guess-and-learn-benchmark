from .datasets import get_data_for_protocol
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torchvision
from torch.utils.data import TensorDataset


# --- Base Model Wrapper ------------------------------------------------------
class GnlModel:
    def __init__(self, device="cpu"):
        self.device = device

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def update(self, X_labeled, Y_labeled, track_config):
        raise NotImplementedError

    # common hook used by save_results()
    def extract_features(self, X):
        if isinstance(X, list):
            raise NotImplementedError("Sub-class must implement text feature extraction.")
        return X.reshape(X.shape[0], -1).to(self.device)


# --- k-NN Model --------------------------------------------------------------
class KnnModel(GnlModel):
    """
    Only change: accept num_classes at construction so early uniform
    predictions match non-10-class datasets (e.g. AG News).
    """

    def __init__(self, n_neighbors=7, num_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.is_fitted = False
        self.num_classes = num_classes  # <- injected from factory

    def _flat(self, X):
        if torch.is_tensor(X):
            return X.reshape(X.shape[0], -1).cpu().numpy()
        raise TypeError("KnnModel expects a numeric torch.Tensor; received non-tensor input.")

    def predict(self, X):
        if not self.is_fitted:
            if self.num_classes is None:
                raise ValueError("num_classes not set for cold-start prediction.")
            return torch.randint(0, self.num_classes, (X.shape[0],), dtype=torch.long)
        return torch.tensor(self.model.predict(self._flat(X)), dtype=torch.long)

    def predict_proba(self, X):
        if not self.is_fitted:
            if self.num_classes is None:
                raise ValueError("num_classes not set for cold-start proba.")
            return torch.ones(X.shape[0], self.num_classes) / self.num_classes
        proba = self.model.predict_proba(self._flat(X))
        return torch.tensor(proba, dtype=torch.float32)

    def update(self, X_labeled, Y_labeled, track_config):
        X_flat = self._flat(X_labeled)
        Y_np = Y_labeled.cpu().numpy()
        if self.num_classes is None:
            self.num_classes = len(np.unique(Y_np))
        self.model.fit(X_flat, Y_np)
        self.is_fitted = True


# --- True Multi-class Perceptron --------------------------------------------
class PerceptronModel(GnlModel):
    """
    Replaces logistic-regression training with classic perceptron updates.
    Forward path & probabilities stay identical.
    """

    def __init__(self, input_dim, num_classes, lr=1.0, device="cpu"):
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
        """Single-example perceptron weight update."""
        logits = self.model(x_vec.unsqueeze(0))
        y_pred = torch.argmax(logits, dim=1).item()
        if y_pred != y_true:
            self.model.weight.data[y_true] += self.lr * x_vec
            self.model.weight.data[y_pred] -= self.lr * x_vec
            self.model.bias.data[y_true] += self.lr
            self.model.bias.data[y_pred] -= self.lr

    def update(self, X_labeled, Y_labeled, track_config):
        online = track_config["track"].startswith("G&L-SO")
        idxs = [-1] if online else range(len(X_labeled))
        for i in idxs:
            x = X_labeled[i].reshape(-1).to(self.device)
            y = Y_labeled[i].item()
            self._perceptron_step(x, y)


# --- Simple CNN Model (unchanged) -------------------------------------------
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


class CnnModel(GnlModel):
    # (identical except feature extractor override)
    def __init__(self, in_channels, num_classes, input_size=32, lr=0.01, device="cpu"):
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

    def update(self, X_labeled, Y_labeled, track_config):
        self.model.train()
        epochs = track_config.get("epochs_per_update", 5)
        batch_size = track_config.get("train_batch_size", 32)
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


# --- Pretrained Model Wrapper -----------------------------------------------
class PretrainedModelWrapper(GnlModel):
    def __init__(self, model_name, num_classes, track_config, device="cpu"):
        super().__init__(device)
        self.model_name = model_name
        self.tokenizer = None
        self.num_classes = num_classes
        self.track_config = track_config

        # Model loading unchanged ...
        if "bert" in model_name:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif "resnet" in model_name:
            self.model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            self.model = self.model.to(device)
        elif "vit" in model_name:
            self.model = AutoModel.from_pretrained(model_name)
            self.classifier = nn.Linear(self.model.config.hidden_size, num_classes).to(device)
            self.model = self.model.to(device)
        else:
            raise ValueError(f"Model name {model_name} not supported")

        self._configure_for_track()

    def _configure_for_track(self):
        # recognise suffixes like G&L-PO_1
        is_po_track = self.track_config["track"].startswith("G&L-PO")

        for p in self.model.parameters():
            p.requires_grad = True
        if hasattr(self, "classifier"):
            for p in self.classifier.parameters():
                p.requires_grad = True

        if is_po_track:
            for p in self.model.parameters():
                p.requires_grad = False
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

        if "resnet" in self.model_name and is_po_track:
            self.optimizer = optim.SGD(params, lr=self.track_config.get("lr", 0.01))
        else:
            self.optimizer = optim.AdamW(params, lr=self.track_config.get("lr", 2e-5))

        self.loss_fn = nn.CrossEntropyLoss()

    def predict(self, X):
        return torch.argmax(self.predict_proba(X), dim=1)

    def predict_proba(self, X):
        self.model.eval()
        if hasattr(self, "classifier"):
            self.classifier.eval()

        with torch.no_grad():
            # Text (BERT)
            if self.tokenizer:
                texts = X if isinstance(X, list) else X.tolist()
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                logits = self.model(**inputs).logits

            # Vision Transformer
            elif "vit" in self.model_name:
                outputs = self.model(pixel_values=X.to(self.device))
                logits = self.classifier(outputs.last_hidden_state[:, 0])

            # ResNet-50
            else:
                logits = self.model(X.to(self.device))

        return torch.softmax(logits, dim=1)

    # unchanged except ‘startswith’ for suffix-robust track detection
    def update(self, X_labeled, Y_labeled, track_config):
        self.model.train()
        if hasattr(self, "classifier"):
            self.classifier.train()

        # recognise e.g. G&L-SO_50 or G&L-PO_1
        track_str = track_config["track"]
        is_online_track = track_str.startswith("G&L-SO") or track_str.startswith("G&L-PO")

        if is_online_track:
            # --- Online update (single most recent sample) ------------------
            if self.tokenizer:  # text
                data = [X_labeled[-1]]
                target = Y_labeled[-1:].to(self.device)
            else:  # image
                data = X_labeled[-1:].to(self.device)
                target = Y_labeled[-1:].to(self.device)

            self.optimizer.zero_grad()
            if self.tokenizer:
                inputs = self.tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                logits = self.model(**inputs).logits
            elif "vit" in self.model_name:
                logits = self.classifier(self.model(pixel_values=data).last_hidden_state[:, 0])
            else:  # ResNet
                logits = self.model(data)

            loss = self.loss_fn(logits, target)
            loss.backward()
            self.optimizer.step()

        else:
            # --- Batch update (PB / SB) ------------------------------------
            epochs = track_config.get("epochs_per_update", 3)
            batch_size = track_config.get("train_batch_size", 16)

            if self.tokenizer:  # text batches
                for _ in range(epochs):
                    for i in range(0, len(X_labeled), batch_size):
                        texts = X_labeled[i : i + batch_size]
                        labels = Y_labeled[i : i + batch_size].to(self.device)
                        inputs = self.tokenizer(
                            texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512,
                        ).to(self.device)

                        self.optimizer.zero_grad()
                        logits = self.model(**inputs).logits
                        loss = self.loss_fn(logits, labels)
                        loss.backward()
                        self.optimizer.step()
            else:  # image batches
                loader = torch.utils.data.DataLoader(
                    TensorDataset(X_labeled, Y_labeled),
                    batch_size=batch_size,
                    shuffle=True,
                )
                for _ in range(epochs):
                    for data, target in loader:
                        data, target = data.to(self.device), target.to(self.device)
                        self.optimizer.zero_grad()
                        if "vit" in self.model_name:
                            logits = self.classifier(self.model(pixel_values=data).last_hidden_state[:, 0])
                        else:
                            logits = self.model(data)
                        loss = self.loss_fn(logits, target)
                        loss.backward()
                        self.optimizer.step()

    # feature extractor for save_results()
    def extract_features(self, X):
        with torch.no_grad():
            if self.tokenizer:  # BERT CLS embedding
                texts = X if isinstance(X, list) else X.tolist()
                inp = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                backbone = getattr(self.model, "bert", None) or getattr(self.model, "roberta", None) or getattr(self.model, "backbone", None)
                hidden = backbone(**inp).last_hidden_state[:, 0]
                return hidden
            elif "vit" in self.model_name:
                return self.model(pixel_values=X.to(self.device)).last_hidden_state[:, 0]
            else:  # ResNet
                x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(X.to(self.device)))))
                feats = self.model.avgpool(self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(x)))))
                return torch.flatten(feats, 1)


# --- Factory -----------------------------------------------------------------
def get_model(name, dataset_name, device, track_config=None):
    X, Y = get_data_for_protocol(dataset_name)
    num_classes = len(torch.unique(Y)) if torch.is_tensor(Y) else len(set(Y))

    if torch.is_tensor(X):
        input_shape = X.shape[1:]
    else:
        input_shape = None

    if name == "knn":
        return KnnModel(n_neighbors=7, num_classes=num_classes)

    elif name == "perceptron":
        if input_shape is None:
            raise ValueError("Cannot determine input shape for perceptron")
        return PerceptronModel(np.prod(input_shape), num_classes, lr=1.0, device=device)

    elif name == "cnn":
        if input_shape is None:
            raise ValueError("Cannot determine input shape for CNN")
        in_channels = input_shape[0] if len(input_shape) == 3 else 1
        return CnnModel(in_channels, num_classes, input_shape[1], lr=0.01, device=device)

    elif name in ["resnet50", "vit-b-16", "bert-base"]:
        if track_config is None:
            raise ValueError("track_config must be provided for pretrained model")
        model_map = {
            "resnet50": "resnet50",
            "vit-b-16": "google/vit-base-patch16-224-in21k",
            "bert-base": "bert-base-uncased",
        }
        return PretrainedModelWrapper(model_map[name], num_classes, track_config, device=device)

    else:
        raise ValueError(f"Model '{name}' not supported.")
