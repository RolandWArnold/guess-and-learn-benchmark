import torch
import numpy as np
from tqdm import tqdm


class AcquisitionStrategy:
    def select(self, model, unlabeled_indices, X_pool, batch_size=256):
        raise NotImplementedError


class RandomStrategy(AcquisitionStrategy):
    def select(self, model, unlabeled_indices, X_pool, batch_size=256):
        return np.random.choice(unlabeled_indices)


class ConfidenceStrategy(AcquisitionStrategy):
    """Select the example with HIGHEST confidence (easy-first)"""

    def select(self, model, unlabeled_indices, X_pool, batch_size=256):
        best_confidence = -1.0
        best_idx = -1

        # Use tqdm for a progress bar during selection
        pbar = tqdm(range(0, len(unlabeled_indices), batch_size), desc="Confidence-Select", leave=False)

        with torch.no_grad():
            for i in pbar:
                batch_indices = unlabeled_indices[i : i + batch_size]
                if isinstance(X_pool, list):
                    X_batch = [X_pool[j] for j in batch_indices]
                else:
                    X_batch = X_pool[batch_indices]

                outputs = model.predict_proba(X_batch)
                confidence, _ = torch.max(outputs, dim=1)

                batch_best_confidence, batch_best_local_idx = torch.max(confidence, dim=0)

                if batch_best_confidence > best_confidence:
                    best_confidence = batch_best_confidence.item()
                    best_idx = batch_indices[batch_best_local_idx.item()]

        return best_idx


class LeastConfidenceStrategy(AcquisitionStrategy):
    """Select the example with LOWEST confidence (uncertainty sampling)"""

    def select(self, model, unlabeled_indices, X_pool, batch_size=256):
        min_confidence = float("inf")
        worst_idx = -1

        pbar = tqdm(range(0, len(unlabeled_indices), batch_size), desc="LeastConfidence-Select", leave=False)

        with torch.no_grad():
            for i in pbar:
                batch_indices = unlabeled_indices[i : i + batch_size]
                if isinstance(X_pool, list):
                    X_batch = [X_pool[j] for j in batch_indices]
                else:
                    X_batch = X_pool[batch_indices]

                outputs = model.predict_proba(X_batch)
                if outputs.shape[1] == 0:
                    continue

                confidence, _ = torch.max(outputs, dim=1)
                batch_min_confidence, batch_worst_local_idx = torch.min(confidence, dim=0)

                if batch_min_confidence < min_confidence:
                    min_confidence = batch_min_confidence.item()
                    worst_idx = batch_indices[batch_worst_local_idx.item()]

        return worst_idx if worst_idx != -1 else np.random.choice(unlabeled_indices)


class MarginStrategy(AcquisitionStrategy):
    """Select the example with SMALLEST margin (difference between top 2 predictions)"""

    def select(self, model, unlabeled_indices, X_pool, batch_size=256):
        min_margin = float("inf")
        best_idx = -1

        pbar = tqdm(range(0, len(unlabeled_indices), batch_size), desc="Margin-Select", leave=False)

        with torch.no_grad():
            for i in pbar:
                batch_indices = unlabeled_indices[i : i + batch_size]
                if isinstance(X_pool, list):
                    X_batch = [X_pool[j] for j in batch_indices]
                else:
                    X_batch = X_pool[batch_indices]

                outputs = model.predict_proba(X_batch)
                if outputs.shape[1] < 2:
                    continue

                sorted_probs, _ = torch.sort(outputs, dim=1, descending=True)
                margins = sorted_probs[:, 0] - sorted_probs[:, 1]

                batch_min_margin, batch_best_local_idx = torch.min(margins, dim=0)

                if batch_min_margin < min_margin:
                    min_margin = batch_min_margin.item()
                    best_idx = batch_indices[batch_best_local_idx.item()]

        return best_idx if best_idx != -1 else np.random.choice(unlabeled_indices)


class EntropyStrategy(AcquisitionStrategy):
    """Select the example with HIGHEST entropy (most uncertain)"""

    def select(self, model, unlabeled_indices, X_pool, batch_size=256):
        max_entropy = -1.0
        best_idx = -1

        pbar = tqdm(range(0, len(unlabeled_indices), batch_size), desc="Entropy-Select", leave=False)

        with torch.no_grad():
            for i in pbar:
                batch_indices = unlabeled_indices[i : i + batch_size]
                if isinstance(X_pool, list):
                    X_batch = [X_pool[j] for j in batch_indices]
                else:
                    X_batch = X_pool[batch_indices]

                outputs = model.predict_proba(X_batch)
                entropy = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1)

                batch_max_entropy, batch_best_local_idx = torch.max(entropy, dim=0)

                if batch_max_entropy > max_entropy:
                    max_entropy = batch_max_entropy.item()
                    best_idx = batch_indices[batch_best_local_idx.item()]

        return best_idx if best_idx != -1 else np.random.choice(unlabeled_indices)


def get_strategy(name: str):
    strategies = {
        "random": RandomStrategy,
        "confidence": ConfidenceStrategy,
        "least_confidence": LeastConfidenceStrategy,
        "margin": MarginStrategy,
        "entropy": EntropyStrategy,
    }
    if name.lower() in strategies:
        return strategies[name.lower()]()
    else:
        raise ValueError(f"Strategy '{name}' not supported.")
