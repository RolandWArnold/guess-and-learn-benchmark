from .datasets import get_data_for_protocol
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torchvision
from torch.utils.data import TensorDataset

# --- Base Model Wrapper ---
class GnlModel:
    def __init__(self, device='cpu'):
        self.device = device

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def update(self, X_labeled, Y_labeled, track_config):
        raise NotImplementedError

# --- k-NN Model ---
class KnnModel(GnlModel):
    def __init__(self, n_neighbors=7, **kwargs):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.is_fitted = False
        self.num_classes = None

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1).cpu().numpy()
        if not self.is_fitted:
            # Random guess if not fitted
            if self.num_classes is None:
                return torch.zeros(X.shape[0], dtype=torch.long)
            return torch.randint(0, self.num_classes, (X.shape[0],), dtype=torch.long)
        return torch.tensor(self.model.predict(X_flat), dtype=torch.long)

    def predict_proba(self, X):
        X_flat = X.reshape(X.shape[0], -1).cpu().numpy()
        if not self.is_fitted:
            # Uniform probability if not fitted
            if self.num_classes is None:
                self.num_classes = 10  # Default assumption
            return torch.ones(X.shape[0], self.num_classes) / self.num_classes
        proba = self.model.predict_proba(X_flat)
        return torch.tensor(proba, dtype=torch.float32)

    def update(self, X_labeled, Y_labeled, track_config):
        # k-NN is non-parametric, "update" means refitting on all available data ("memory append")
        X_flat = X_labeled.reshape(X_labeled.shape[0], -1).cpu().numpy()
        Y_np = Y_labeled.cpu().numpy()

        # Set num_classes if not already set
        if self.num_classes is None:
            self.num_classes = len(np.unique(Y_np))

        self.model.fit(X_flat, Y_np)
        self.is_fitted = True

# --- Perceptron Model ---
class PerceptronModel(GnlModel):
    def __init__(self, input_dim, num_classes, lr=0.1, device='cpu'):
        super().__init__(device)
        self.lr = lr
        self.model = nn.Linear(input_dim, num_classes).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize with small random weights for better initial predictions
        nn.init.xavier_uniform_(self.model.weight)
        nn.init.zeros_(self.model.bias)

    def predict(self, X):
        return torch.argmax(self.predict_proba(X), dim=1)

    def predict_proba(self, X):
        X_flat = X.reshape(X.shape[0], -1).to(self.device)
        with torch.no_grad():
            logits = self.model(X_flat)
        return torch.softmax(logits, dim=1)

    def update(self, X_labeled, Y_labeled, track_config):
        self.model.train()

        if track_config['track'] == 'G&L-SO':
            # Online update: train only on the most recent point
            X_flat = X_labeled[-1:].reshape(1, -1).to(self.device)
            Y_target = Y_labeled[-1:].to(self.device)
        else:
            # Batch update: train on all labeled data
            X_flat = X_labeled.reshape(X_labeled.shape[0], -1).to(self.device)
            Y_target = Y_labeled.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(X_flat)
        loss = self.loss_fn(output, Y_target)
        loss.backward()
        self.optimizer.step()

# --- Simple CNN Model ---
class SimpleCnn(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, input_size=32):
        super(SimpleCnn, self).__init__()
        self.input_size = input_size

        # Architecture matches paper: Conv-ReLU-Pool ×3
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

        # Calculate correct input size for linear layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, input_size, input_size)
            linear_input_size = self.features(dummy_input).flatten(1).shape[1]

        # MLP with 512 units and dropout 0.5
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CnnModel(GnlModel):
    def __init__(self, in_channels, num_classes, input_size=32, lr=0.01, device='cpu'):
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
        # Batch update for SB track
        self.model.train()
        epochs = track_config.get('epochs_per_update', 5)
        batch_size = track_config.get('train_batch_size', 32)

        train_dataset = TensorDataset(X_labeled, Y_labeled)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()

# --- Pretrained Model Wrapper ---
class PretrainedModelWrapper(GnlModel):
    def __init__(self, model_name, num_classes, track_config, device='cpu'):
        super().__init__(device)
        self.model_name = model_name
        self.tokenizer = None
        self.num_classes = num_classes
        self.track_config = track_config

        # --- Model Loading ---
        if 'bert' in model_name:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif 'resnet' in model_name:
            self.model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            self.model = self.model.to(device)
        elif 'vit' in model_name:
            self.model = AutoModel.from_pretrained(model_name)
            self.classifier = nn.Linear(self.model.config.hidden_size, num_classes).to(device)
            self.model = self.model.to(device)
        else:
            raise ValueError(f"Model name {model_name} not supported")

        # Configure model parameters and optimizer in __init__ based on track ---
        self._configure_for_track()

    def _configure_for_track(self):
        """Set requires_grad and initialize the optimizer based on the G&L track."""
        is_po_track = self.track_config['track'] == 'G&L-PO'

        # Default: all params are trainable (for PB tracks)
        for param in self.model.parameters():
            param.requires_grad = True
        if hasattr(self, 'classifier'):
            for param in self.classifier.parameters():
                param.requires_grad = True

        # If PO track, freeze the backbone
        if is_po_track:
            for param in self.model.parameters():
                param.requires_grad = False

            if 'resnet' in self.model_name:
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            elif 'bert' in self.model_name:
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
            elif 'vit' in self.model_name:
                # The base model is frozen, only the custom classifier is trained
                for param in self.classifier.parameters():
                    param.requires_grad = True

        # Collect all trainable parameters
        params_to_update = [p for p in self.model.parameters() if p.requires_grad]
        if hasattr(self, 'classifier'):
            params_to_update.extend([p for p in self.classifier.parameters() if p.requires_grad])

        # Use correct optimizer and LR based on paper's spec
        if 'resnet' in self.model_name and is_po_track:
            # Paper specifies SGD for ResNet-50 head online training
            self.optimizer = optim.SGD(params_to_update, lr=self.track_config.get('lr', 0.01))
        else:
            # Paper specifies AdamW for ViT fine-tuning
            self.optimizer = optim.AdamW(params_to_update, lr=self.track_config.get('lr', 2e-5))

        self.loss_fn = nn.CrossEntropyLoss()

    def predict(self, X):
        return torch.argmax(self.predict_proba(X), dim=1)

    def predict_proba(self, X):
        self.model.eval()
        if hasattr(self, 'classifier'):
            self.classifier.eval()

        with torch.no_grad():
            if self.tokenizer:
                if isinstance(X, list): texts = X
                else: texts = X.tolist() if hasattr(X, 'tolist') else [str(x) for x in X]
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                logits = self.model(**inputs).logits
            elif 'vit' in self.model_name:
                outputs = self.model(pixel_values=X.to(self.device))
                logits = self.classifier(outputs.last_hidden_state[:, 0])
            else:
                logits = self.model(X.to(self.device))
        return torch.softmax(logits, dim=1)

    def update(self, X_labeled, Y_labeled, track_config):
        self.model.train()
        if hasattr(self, 'classifier'):
            self.classifier.train()

        # Differentiate between Online (PO) and Batch (PB) updates
        is_online_track = track_config['track'] in ['G&L-SO', 'G&L-PO']

        if is_online_track:
            # --- Online Update (for PO tracks) ---
            # Get the single most recent sample
            if self.tokenizer:
                data = [X_labeled[-1]]
                target = Y_labeled[-1:].to(self.device)
            else:
                data = X_labeled[-1:].to(self.device)
                target = Y_labeled[-1:].to(self.device)

            self.optimizer.zero_grad()
            if self.tokenizer:
                inputs = self.tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                logits = self.model(**inputs).logits
            elif 'vit' in self.model_name:
                outputs = self.model(pixel_values=data)
                logits = self.classifier(outputs.last_hidden_state[:, 0])
            else:
                logits = self.model(data)

            loss = self.loss_fn(logits, target)
            loss.backward()
            self.optimizer.step()
        else:
            # --- Batch Update (for PB tracks) ---
            epochs = track_config.get('epochs_per_update', 3)
            batch_size = track_config.get('train_batch_size', 16)

            if self.tokenizer:
                # Handle text data
                for epoch in range(epochs):
                    for i in range(0, len(X_labeled), batch_size):
                        batch_texts = X_labeled[i:i+batch_size]
                        batch_labels = Y_labeled[i:i+batch_size].to(self.device)
                        inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

                        self.optimizer.zero_grad()
                        logits = self.model(**inputs).logits
                        loss = self.loss_fn(logits, batch_labels)
                        loss.backward()
                        self.optimizer.step()
            else:
                # Handle image data
                train_dataset = TensorDataset(X_labeled, Y_labeled)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                for epoch in range(epochs):
                    for data, target in train_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        self.optimizer.zero_grad()
                        if 'vit' in self.model_name:
                            outputs = self.model(pixel_values=data)
                            logits = self.classifier(outputs.last_hidden_state[:, 0])
                        else:
                            logits = self.model(data)
                        loss = self.loss_fn(logits, target)
                        loss.backward()
                        self.optimizer.step()

def get_model(name, dataset_name, device, track_config=None):
    # INFO: track_config is required for pretrained models to configure them correctly per paper spec
    X, Y = get_data_for_protocol(dataset_name)

    if torch.is_tensor(X):
        input_shape = X.shape[1:]
    else: # Text data
        input_shape = None

    if torch.is_tensor(Y):
        num_classes = len(torch.unique(Y))
    else: # Text data
        num_classes = len(set(Y))

    if name == 'knn':
        model = KnnModel(n_neighbors=7)
        model.num_classes = num_classes
        return model
    elif name == 'perceptron':
        if input_shape is None: raise ValueError("Cannot determine input shape for perceptron")
        input_dim = np.prod(input_shape)
        return PerceptronModel(input_dim, num_classes, lr=0.1, device=device)
    elif name == 'cnn':
        if input_shape is None: raise ValueError("Cannot determine input shape for CNN")
        # Handle both grayscale (1) and color (3)
        in_channels = input_shape[0] if len(input_shape) == 3 else 1
        input_size = input_shape[1] if len(input_shape) >= 2 else 28
        return CnnModel(in_channels, num_classes, input_size, lr=0.01, device=device)
    elif name in ['resnet50', 'vit-b-16', 'bert-base']:
        if track_config is None:
            raise ValueError(f"track_config must be provided for pretrained model '{name}'")

        model_map = {
            'resnet50': 'resnet50',
            'vit-b-16': 'google/vit-base-patch16-224-in21k',
            'bert-base': 'bert-base-uncased'
        }
        return PretrainedModelWrapper(model_map[name], num_classes, track_config, device=device)
    else:
        raise ValueError(f"Model '{name}' not supported.")