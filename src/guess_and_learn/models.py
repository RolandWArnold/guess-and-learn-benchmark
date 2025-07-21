from .datasets import get_data_for_protocol
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torchvision
from torch.utils.data import TensorDataset
import copy


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

    def reset(self):
        raise NotImplementedError("This model does not support --reset-weights.")


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
            return torch.randint(0, self.num_classes, (X.shape[0],), dtype=torch.long, device=self.device)
        return torch.tensor(self.model.predict(self._flat(X)), dtype=torch.long, device=self.device)

    def predict_proba(self, X):
        if not self.is_fitted:
            if self.num_classes is None:
                raise ValueError("num_classes not set for cold-start proba.")
            return torch.ones(X.shape[0], self.num_classes, dtype=torch.float32, device=self.device) / self.num_classes
        proba = self.model.predict_proba(self._flat(X))
        return torch.tensor(proba, dtype=torch.float32, device=self.device)

    def update(self, X_labeled, Y_labeled, track_config):
        # ── guard: k cannot exceed available samples ────────────────────────────
        k = min(self.n_neighbors, len(X_labeled))
        if k != self.n_neighbors:  # rebuild with smaller k
            self.model = KNeighborsClassifier(n_neighbors=k)
            self.n_neighbors = k  # keep attribute in sync
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
    """
    Replaces logistic-regression training with classic perceptron updates.
    Forward path & probabilities stay identical.
    """

    # default η = 0.1 as per paper; overridable via track_config
    def __init__(self, input_dim, num_classes, lr=0.1, device="cpu"):
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

    def reset(self):
        """Re-initialise weights *and* rebuild the optimiser in a model-agnostic way."""
        # 1) Random-restart the network weights
        self.model.reinit()

        # 2) Recreate the optimiser with the same class & hyper-parameters
        opt_cls = self.optimizer.__class__  # e.g. torch.optim.SGD / AdamW
        opt_defaults = self.optimizer.defaults.copy()  # lr, momentum, betas, etc.

        # ‘params’ is injected by torch into each param_group – remove if present
        opt_defaults.pop("params", None)

        self.optimizer = opt_cls(self.model.parameters(), **opt_defaults)


# ---------------------------------------------------------------------#
#  TEXT-SPECIFIC MODELS (AG News)                                      #
# ---------------------------------------------------------------------#


class TextPerceptronModel(GnlModel):
    """Perceptron using a PRE-FITTED TF-IDF vectoriser."""

    # FIX: Init now accepts a pre-fitted vectorizer and input_dim
    def __init__(self, input_dim, num_classes, vectorizer, lr=0.1, device="cpu"):
        super().__init__(device)
        self.vec = vectorizer  # Use the passed, pre-fitted vectorizer
        self.lr = lr
        self.num_classes = num_classes
        self.lin = nn.Linear(input_dim, num_classes, bias=True).to(device)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    # ----- helpers ----------------------------------------------------
    def _featurise(self, texts):
        # .transform is correct here because the vectorizer is already fitted
        X = self.vec.transform(texts).toarray()
        return torch.tensor(X, dtype=torch.float32, device=self.device)

    # ----- API --------------------------------------------------------
    def predict(self, X):
        # FIX: No need for a `self.lin is None` check anymore
        with torch.no_grad():
            logits = self.lin(self._featurise(X))
        return torch.argmax(torch.softmax(logits, dim=1), dim=1)

    def predict_proba(self, X):
        # FIX: No need for a `self.lin is None` check anymore
        with torch.no_grad():
            logits = self.lin(self._featurise(X))
        return torch.softmax(logits, dim=1)

    def update(self, X_labeled, Y_labeled, track_config):
        """
        FIX: The update logic is now simplified. It no longer fits the
        vectorizer or initializes the model. It only performs weight updates.
        """
        online = track_config["track"].startswith("G&L-SO")
        if online:
            # For online, only train on the most recent sample
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
    """k-NN over PRE-FITTED TF-IDF features for raw-string datasets."""

    # FIX: Init now accepts a pre-fitted vectorizer
    def __init__(self, n_neighbors, num_classes, vectorizer, **kwargs):
        super().__init__(n_neighbors=n_neighbors, num_classes=num_classes, **kwargs)
        self.vec = vectorizer  # Use the passed, pre-fitted vectorizer

    def _flat(self, X):
        if isinstance(X, list):
            # .transform is correct here because the vectorizer is already fitted
            return self.vec.transform(X).toarray()
        return super()._flat(X)

    def predict(self, X):
        n = len(X) if isinstance(X, list) else X.shape[0]
        if not self.is_fitted:
            if self.num_classes is None:
                raise ValueError("num_classes not set for cold-start prediction.")
            return torch.randint(0, self.num_classes, (n,), dtype=torch.long, device=self.device)
        return torch.tensor(self.model.predict(self._flat(X)), dtype=torch.long, device=self.device)

    def predict_proba(self, X):
        n = len(X) if isinstance(X, list) else X.shape[0]
        if not self.is_fitted:
            if self.num_classes is None:
                raise ValueError("num_classes not set for cold-start proba.")
            return torch.ones(n, self.num_classes, dtype=torch.float32, device=self.device) / self.num_classes
        proba = self.model.predict_proba(self._flat(X))
        return torch.tensor(proba, dtype=torch.float32, device=self.device)

    def update(self, X_labeled, Y_labeled, track_config):
        """
        FIX: The update logic is now simplified. It no longer fits the
        vectorizer. It just calls the parent update method.
        """
        # The incorrect `self.vec.fit(X_labeled)` is removed.
        super().update(X_labeled, Y_labeled, track_config)

    def extract_features(self, X):
        """
        FIX: Consistently uses the pre-fitted vectorizer to transform data.
        """
        X_tfidf = self.vec.transform(X).toarray()
        return torch.tensor(X_tfidf, dtype=torch.float32, device=self.device)


# --- Pretrained Model Wrapper -----------------------------------------------
class PretrainedModelWrapper(GnlModel):
    def __init__(self, model_name, num_classes, track_config, device="cpu"):
        super().__init__(device)
        self.model_name = model_name
        self.tokenizer = None
        self.num_classes = num_classes
        self.track_config = track_config
        self.feature_extractor = None  # For robust feature extraction

        # ---------------------------------------------------------------
        # 1. Load model / tokenizer / classifier
        # ---------------------------------------------------------------
        if "bert" in model_name:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.feature_extractor = getattr(self.model, "bert", getattr(self.model, "roberta", None))

        elif "resnet" in model_name:
            self.model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            self.model = self.model.to(device)
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

        elif "vit" in model_name:
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.classifier = nn.Linear(self.model.config.hidden_size, num_classes).to(device)

        else:
            raise ValueError(f"Model name {model_name} not supported")

        # ---------------------------------------------------------------
        # 2. Take a deep snapshot *before* any fine-tuning so we can
        #    restore it in reset()
        # ---------------------------------------------------------------
        self._initial_state = {
            "model": copy.deepcopy(self.model.state_dict()),
            "classifier": copy.deepcopy(getattr(self, "classifier", nn.Identity()).state_dict()),
        }

        # ---------------------------------------------------------------
        # 3. Configure trainability & optimiser
        # ---------------------------------------------------------------
        self._configure_for_track()

        # ---------------------------------------------------------------
        # 4. Vision preprocessing pipeline
        # ---------------------------------------------------------------
        self._vision_tx = T.Compose(
            [
                T.Resize(224),
                T.Lambda(lambda img: img.repeat(3, 1, 1) if img.shape[0] == 1 else img),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet μ/σ
            ]
        )

    def _prep_vision(self, x_batch: torch.Tensor) -> torch.Tensor:
        x_batch = x_batch.clamp(0.0, 1.0)
        if x_batch.shape[-1] != 224:
            x_batch = torch.stack([self._vision_tx(img) for img in x_batch])
        return x_batch.to(self.device)

    def _configure_for_track(self):
        is_po_track = self.track_config["track"].startswith("G&L-PO")

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
        params = list({id(p): p for p in params}.values())

        # Set optimizer
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
            if self.tokenizer:  # Text (BERT)
                texts = X if isinstance(X, list) else [str(i) for i in X]
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                logits = self.model(**inputs).logits
            elif "vit" in self.model_name:  # Vision Transformer
                outputs = self.model(pixel_values=self._prep_vision(X))
                logits = self.classifier(outputs.last_hidden_state[:, 0])
            else:  # ResNet-50
                logits = self.model(self._prep_vision(X))
        return torch.softmax(logits, dim=1)

    def update(self, X_labeled, Y_labeled, track_config):
        self.model.train()
        if hasattr(self, "classifier"):
            self.classifier.train()

        track_str = track_config["track"]
        is_online_track = track_str.startswith(("G&L-SO", "G&L-PO"))

        if is_online_track:
            # Online update: only the newest sample
            data = [X_labeled[-1]] if self.tokenizer else X_labeled[-1:].to(self.device)
            target = Y_labeled[-1:].to(self.device)
            self.optimizer.zero_grad()
            # Forward pass
            if self.tokenizer:
                inputs = self.tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                logits = self.model(**inputs).logits
            elif "vit" in self.model_name:
                pv = self._prep_vision(data)
                logits = self.classifier(self.model(pixel_values=pv).last_hidden_state[:, 0])
            else:  # ResNet
                logits = self.model(self._prep_vision(data))
            # Backward pass
            loss = self.loss_fn(logits, target)
            loss.backward()
            self.optimizer.step()
        else:
            # Batch update: SB / PB every K samples
            epochs = track_config.get("epochs_per_update", 3)
            batch_size = track_config.get("train_batch_size", 16)

            # Use DataLoader for both text and image batching
            dataset = TensorDataset(torch.arange(len(X_labeled)))  # Use indices for text
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for _ in range(epochs):
                for (idx_batch,) in loader:
                    if self.tokenizer:  # Text
                        data = [X_labeled[i] for i in idx_batch]
                        target = Y_labeled[idx_batch].to(self.device)
                        inputs = self.tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                        logits = self.model(**inputs).logits
                    else:  # Image
                        data = X_labeled[idx_batch].to(self.device)
                        target = Y_labeled[idx_batch].to(self.device)
                        if "vit" in self.model_name:
                            pv = self._prep_vision(data)
                            logits = self.classifier(self.model(pixel_values=pv).last_hidden_state[:, 0])
                        else:  # ResNet
                            logits = self.model(self._prep_vision(data))

                    self.optimizer.zero_grad()
                    loss = self.loss_fn(logits, target)
                    loss.backward()
                    self.optimizer.step()

    def extract_features(self, X):
        with torch.no_grad():
            if self.tokenizer:  # BERT CLS embedding
                texts = X if isinstance(X, list) else [str(i) for i in X]
                inp = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                # Use the pre-defined feature_extractor (the bert backbone)
                hidden = self.feature_extractor(**inp).last_hidden_state[:, 0]
                return hidden
            elif "vit" in self.model_name:  # ViT
                return self.model(pixel_values=self._prep_vision(X)).last_hidden_state[:, 0]
            else:  # ResNet-50
                # FIX: Use the robust feature_extractor created in __init__
                x = self._prep_vision(X)
                feats = self.feature_extractor(x)
                return torch.flatten(feats, 1)

    def reset(self):
        # 1. Restore weights
        self.model.load_state_dict(self._initial_state["model"])
        if hasattr(self, "classifier"):
            self.classifier.load_state_dict(self._initial_state["classifier"])

        # 2. Re-configure trainability & create a fresh optimiser
        self._configure_for_track()


# --- Factory -----------------------------------------------------------------
def get_model(name, dataset_name, device, track_config=None):
    X, Y = get_data_for_protocol(dataset_name)
    num_classes = len(torch.unique(Y)) if torch.is_tensor(Y) else len(set(Y))

    if torch.is_tensor(X):
        input_shape = X.shape[1:]
    else:  # Text data
        input_shape = None

    # --- Classical Models --------------------------------------------------
    if name == "knn":
        if dataset_name.lower() == "ag_news":
            # FIX: Pre-fit the vectorizer on the entire unlabeled pool
            print("Fitting TF-IDF vectorizer for k-NN on AG News...")
            vectorizer = TfidfVectorizer(max_features=20000)
            vectorizer.fit(X)
            return TextKnnModel(n_neighbors=7, num_classes=num_classes, vectorizer=vectorizer)
        return KnnModel(n_neighbors=7, num_classes=num_classes)

    elif name == "perceptron":
        lr = track_config.get("lr", 0.1) if track_config else 0.1
        if dataset_name.lower() == "ag_news":
            # FIX: Pre-fit the vectorizer on the entire unlabeled pool
            print("Fitting TF-IDF vectorizer for Perceptron on AG News...")
            vectorizer = TfidfVectorizer(max_features=20000)
            vectorizer.fit(X)
            input_dim = len(vectorizer.get_feature_names_out())
            return TextPerceptronModel(input_dim, num_classes, vectorizer, lr=lr, device=device)

        if input_shape is None:
            raise ValueError("Cannot determine input shape for perceptron")
        return PerceptronModel(np.prod(input_shape), num_classes, lr=lr, device=device)

    # --- Deep Models -------------------------------------------------------
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
