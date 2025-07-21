import torch
import torch.nn.functional as F
import numpy as np

class AcquisitionStrategy:
    def select(self, model, unlabeled_indices, X_pool):
        raise NotImplementedError

class RandomStrategy(AcquisitionStrategy):
    def select(self, model, unlabeled_indices, X_pool):
        return np.random.choice(unlabeled_indices)

class ConfidenceStrategy(AcquisitionStrategy):
    def select(self, model, unlabeled_indices, X_pool):
        with torch.no_grad():
            outputs = model.predict_proba(X_pool[unlabeled_indices])
            confidence, _ = torch.max(outputs, dim=1)
            best_idx = torch.argmax(confidence).item()
            return unlabeled_indices[best_idx]

class LeastConfidenceStrategy(AcquisitionStrategy):
    def select(self, model, unlabeled_indices, X_pool):
        with torch.no_grad():
            outputs = model.predict_proba(X_pool[unlabeled_indices])
            confidence, _ = torch.max(outputs, dim=1)
            worst_idx = torch.argmin(confidence).item()
            return unlabeled_indices[worst_idx]

class MarginStrategy(AcquisitionStrategy):
    def select(self, model, unlabeled_indices, X_pool):
        with torch.no_grad():
            outputs = model.predict_proba(X_pool[unlabeled_indices])
            sorted_probs, _ = torch.sort(outputs, dim=1, descending=True)
            margins = sorted_probs[:, 0] - sorted_probs[:, 1]
            best_idx = torch.argmin(margins).item()
            return unlabeled_indices[best_idx]

class EntropyStrategy(AcquisitionStrategy):
    def select(self, model, unlabeled_indices, X_pool):
        with torch.no_grad():
            outputs = model.predict_proba(X_pool[unlabeled_indices])
            # Add a small epsilon to prevent log(0)
            entropy = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1)
            best_idx = torch.argmax(entropy).item()
            return unlabeled_indices[best_idx]

def get_strategy(name: str):
    strategies = {
        'random': RandomStrategy,
        'confidence': ConfidenceStrategy,
        'least_confidence': LeastConfidenceStrategy,
        'margin': MarginStrategy,
        'entropy': EntropyStrategy
    }
    if name.lower() in strategies:
        return strategies[name.lower()]()
    else:
        raise ValueError(f"Strategy '{name}' not supported.")