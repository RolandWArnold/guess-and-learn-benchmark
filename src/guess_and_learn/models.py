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

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1).cpu().numpy()
        if not self.is_fitted:
            # Random guess if not fitted
            return torch.randint(0, 10, (X.shape[0],))
        return torch.tensor(self.model.predict(X_flat))

    def predict_proba(self, X):
        X_flat = X.reshape(X.shape[0], -1).cpu().numpy()
        if not self.is_fitted:
            # Uniform probability if not fitted
            return torch.ones(X.shape[0], 10) / 10
        return torch.tensor(self.model.predict_proba(X_flat), dtype=torch.float32)

    def update(self, X_labeled, Y_labeled, track_config):
        # k-NN is non-parametric, "update" means refitting on all available data
        X_flat = X_labeled.reshape(X_labeled.shape[0], -1).cpu().numpy()
        Y_np = Y_labeled.cpu().numpy()
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

    def predict(self, X):
        return torch.argmax(self.predict_proba(X), dim=1)

    def predict_proba(self, X):
        X_flat = X.reshape(X.shape[0], -1).to(self.device)
        with torch.no_grad():
            logits = self.model(X_flat)
        return torch.softmax(logits, dim=1)

    def update(self, X_labeled, Y_labeled, track_config):
        # Online update: train for one epoch on the newly added point
        self.model.train()
        X_flat = X_labeled[-1:].reshape(1, -1).to(self.device)
        Y_target = Y_labeled[-1:].to(self.device)

        self.optimizer.zero_grad()
        output = self.model(X_flat)
        loss = self.loss_fn(output, Y_target)
        loss.backward()
        self.optimizer.step()

# --- Simple CNN Model ---
class SimpleCnn(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(SimpleCnn, self).__init__()
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
        # Adjust the input features to the linear layer based on input size
        # For 32x32 input (CIFAR/SVHN), it becomes 128 * 4 * 4
        # For 28x28 input (MNIST), it becomes 128 * 3 * 3 after padding adjustments
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), # Assuming 32x32 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CnnModel(GnlModel):
    def __init__(self, in_channels, num_classes, lr=0.01, device='cpu'):
        super().__init__(device)
        self.model = SimpleCnn(in_channels, num_classes).to(device)
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
        # Batch update
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
    def __init__(self, model_name, num_classes, device='cpu'):
        super().__init__(device)
        self.model_name = model_name
        self.tokenizer = None
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
            # This is a bit simplified; real ViT fine-tuning requires a custom head
            self.classifier = nn.Linear(self.model.config.hidden_size, num_classes).to(device)
            self.model = self.model.to(device)
        else:
            raise ValueError(f"Model name {model_name} not supported")

    def _get_params_to_update(self, track_config):
        if 'bert' in self.model_name:
            return self.model.parameters() # fine-tune all
        elif 'resnet' in self.model.__class__.__name__.lower():
            if track_config['track'] == 'G&L-PO':
                return self.model.fc.parameters()
            else: # G&L-PB
                return list(self.model.layer4.parameters()) + list(self.model.fc.parameters())
        elif 'vit' in self.model_name:
             if track_config['track'] == 'G&L-PO':
                return self.classifier.parameters()
             else: # G&L-PB
                return list(self.model.parameters()) + list(self.classifier.parameters())

    def predict(self, X):
        return torch.argmax(self.predict_proba(X), dim=1)

    def predict_proba(self, X):
        self.model.eval()
        if hasattr(self, 'classifier'):
            self.classifier.eval()

        with torch.no_grad():
            if self.tokenizer:
                # Handle text data
                inputs = self.tokenizer(X, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
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

        # Freeze/unfreeze layers based on track
        if 'resnet' in self.model.__class__.__name__.lower() and track_config['track'] == 'G&L-PO':
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

        params_to_update = self._get_params_to_update(track_config)
        optimizer = optim.AdamW(params_to_update, lr=track_config.get('lr', 2e-5))
        loss_fn = nn.CrossEntropyLoss()

        epochs = track_config.get('epochs_per_update', 3)
        batch_size = track_config.get('train_batch_size', 16)

        if self.tokenizer:
            # For text, we can't just use TensorDataset
            # This is a simplification. A real implementation would use a custom Dataset class.
            train_loader = torch.utils.data.DataLoader(list(zip(X_labeled, Y_labeled)), batch_size=batch_size, shuffle=True)
        else:
            train_dataset = TensorDataset(X_labeled, Y_labeled)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for data, target in train_loader:
                if self.tokenizer:
                    target = target.to(self.device)
                    inputs = self.tokenizer(list(data), return_tensors="pt", padding=True, truncation=True).to(self.device)
                    logits = self.model(**inputs).logits
                else:
                    data, target = data.to(self.device), target.to(self.device)
                    if 'vit' in self.model_name:
                        outputs = self.model(pixel_values=data)
                        logits = self.classifier(outputs.last_hidden_state[:, 0])
                    else:
                        logits = self.model(data)

                loss = loss_fn(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

def get_model(name, dataset_name, device):
    X, Y = get_data_for_protocol(dataset_name)
    input_shape = X.shape[1:]  # Remove batch dimension
    num_classes = len(torch.unique(Y))

    if name == 'knn':
        return KnnModel()
    elif name == 'perceptron':
        input_dim = np.prod(input_shape)
        return PerceptronModel(input_dim, num_classes, device=device)
    elif name == 'cnn':
        in_channels = input_shape[0] if len(input_shape) == 3 else 1
        return CnnModel(in_channels, num_classes, device=device)
    elif name == 'resnet50':
        return PretrainedModelWrapper('resnet50', num_classes, device=device)
    elif name == 'vit-b-16':
        return PretrainedModelWrapper('google/vit-base-patch16-224-in21k', num_classes, device=device)
    elif name == 'bert-base':
        return PretrainedModelWrapper('bert-base-uncased', num_classes, device=device)
    else:
        raise ValueError(f"Model '{name}' not supported.")