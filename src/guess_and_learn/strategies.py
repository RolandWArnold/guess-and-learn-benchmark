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
    """Select the example with HIGHEST confidence (easy-first)"""
    def select(self, model, unlabeled_indices, X_pool):
        with torch.no_grad():
            if isinstance(X_pool, list):
                # Handle text data
                X_batch = [X_pool[i] for i in unlabeled_indices]
            else:
                X_batch = X_pool[unlabeled_indices]

            outputs = model.predict_proba(X_batch)
            confidence, _ = torch.max(outputs, dim=1)
            best_idx = torch.argmax(confidence).item()
            return unlabeled_indices[best_idx]

class LeastConfidenceStrategy(AcquisitionStrategy):
    """Select the example with LOWEST confidence (uncertainty sampling)"""
    def select(self, model, unlabeled_indices, X_pool):
        with torch.no_grad():
            if isinstance(X_pool, list):
                # Handle text data
                X_batch = [X_pool[i] for i in unlabeled_indices]
            else:
                X_batch = X_pool[unlabeled_indices]

            outputs = model.predict_proba(X_batch)
            confidence, _ = torch.max(outputs, dim=1)
            worst_idx = torch.argmin(confidence).item()
            return unlabeled_indices[worst_idx]

class MarginStrategy(AcquisitionStrategy):
    """Select the example with SMALLEST margin (difference between top 2 predictions)"""
    def select(self, model, unlabeled_indices, X_pool):
        with torch.no_grad():
            if isinstance(X_pool, list):
                # Handle text data
                X_batch = [X_pool[i] for i in unlabeled_indices]
            else:
                X_batch = X_pool[unlabeled_indices]

            outputs = model.predict_proba(X_batch)
            sorted_probs, _ = torch.sort(outputs, dim=1, descending=True)
            margins = sorted_probs[:, 0] - sorted_probs[:, 1]
            best_idx = torch.argmin(margins).item()
            return unlabeled_indices[best_idx]

class EntropyStrategy(AcquisitionStrategy):
    """Select the example with HIGHEST entropy (most uncertain)"""
    def select(self, model, unlabeled_indices, X_pool):
        with torch.no_grad():
            if isinstance(X_pool, list):
                # Handle text data
                X_batch = [X_pool[i] for i in unlabeled_indices]
            else:
                X_batch = X_pool[unlabeled_indices]

            outputs = model.predict_proba(X_batch)
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