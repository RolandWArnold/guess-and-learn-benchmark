import torch
import numpy as np
from tqdm import tqdm


class AcquisitionStrategy:
    def select(self, model, unlabeled_indices, X_pool, n_to_acquire=1, batch_size=256):
        """
        Selects a batch of samples from the unlabeled pool.

        Args:
            model: The current model instance.
            unlabeled_indices: Indices of the unlabeled samples.
            X_pool: The entire data pool.
            n_to_acquire: The number of samples to select.
            batch_size: The processing batch size for GPU memory efficiency.

        Returns:
            A list of the top `n_to_acquire` indices.
        """
        raise NotImplementedError


class RandomStrategy(AcquisitionStrategy):
    def select(self, model, unlabeled_indices, X_pool, n_to_acquire=1, batch_size=256):
        """Efficiently selects a random subset without replacement."""
        num_to_select = min(n_to_acquire, len(unlabeled_indices))
        return np.random.choice(unlabeled_indices, size=num_to_select, replace=False)


class ConfidenceStrategy(AcquisitionStrategy):
    """Selects the batch of examples with the HIGHEST confidence (easy-first)."""

    def select(self, model, unlabeled_indices, X_pool, n_to_acquire=1, batch_size=256):
        all_scores = []
        all_indices = []

        pbar = tqdm(range(0, len(unlabeled_indices), batch_size), desc="Confidence-Select (Batch)", leave=False)

        with torch.no_grad():
            for i in pbar:
                batch_indices = unlabeled_indices[i : i + batch_size]
                if isinstance(X_pool, list):
                    X_batch = [X_pool[j] for j in batch_indices]
                else:
                    X_batch = X_pool[batch_indices]

                outputs = model.predict_proba(X_batch)
                confidence, _ = torch.max(outputs, dim=1)

                all_scores.append(confidence)
                all_indices.extend(batch_indices)

        all_scores = torch.cat(all_scores)

        # Sort scores in descending order (highest confidence first)
        top_k_indices = torch.topk(all_scores, k=min(n_to_acquire, len(all_scores))).indices

        best_indices = [all_indices[i] for i in top_k_indices]
        return best_indices


class LeastConfidenceStrategy(AcquisitionStrategy):
    """Selects the batch of examples with the LOWEST confidence (uncertainty sampling)."""

    def select(self, model, unlabeled_indices, X_pool, n_to_acquire=1, batch_size=256):
        all_scores = []
        all_indices = []

        pbar = tqdm(range(0, len(unlabeled_indices), batch_size), desc="LeastConfidence-Select (Batch)", leave=False)

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

                all_scores.append(confidence)
                all_indices.extend(batch_indices)

        if not all_scores:
            return RandomStrategy().select(model, unlabeled_indices, X_pool, n_to_acquire, batch_size)

        all_scores = torch.cat(all_scores)

        # Sort scores in ascending order (lowest confidence first) by using largest=False
        top_k_indices = torch.topk(all_scores, k=min(n_to_acquire, len(all_scores)), largest=False).indices

        best_indices = [all_indices[i] for i in top_k_indices]
        return best_indices


class MarginStrategy(AcquisitionStrategy):
    """Selects the batch of examples with the SMALLEST margin."""

    def select(self, model, unlabeled_indices, X_pool, n_to_acquire=1, batch_size=256):
        all_scores = []
        all_indices = []

        pbar = tqdm(range(0, len(unlabeled_indices), batch_size), desc="Margin-Select (Batch)", leave=False)

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

                all_scores.append(margins)
                all_indices.extend(batch_indices)

        if not all_scores:
            return RandomStrategy().select(model, unlabeled_indices, X_pool, n_to_acquire, batch_size)

        all_scores = torch.cat(all_scores)

        # Sort scores in ascending order (smallest margin first) by using largest=False
        top_k_indices = torch.topk(all_scores, k=min(n_to_acquire, len(all_scores)), largest=False).indices

        best_indices = [all_indices[i] for i in top_k_indices]
        return best_indices


class EntropyStrategy(AcquisitionStrategy):
    """Selects the batch of examples with the HIGHEST entropy."""

    def select(self, model, unlabeled_indices, X_pool, n_to_acquire=1, batch_size=256):
        all_scores = []
        all_indices = []

        pbar = tqdm(range(0, len(unlabeled_indices), batch_size), desc="Entropy-Select (Batch)", leave=False)

        with torch.no_grad():
            for i in pbar:
                batch_indices = unlabeled_indices[i : i + batch_size]
                if isinstance(X_pool, list):
                    X_batch = [X_pool[j] for j in batch_indices]
                else:
                    X_batch = X_pool[batch_indices]

                outputs = model.predict_proba(X_batch)
                entropy = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1)

                all_scores.append(entropy)
                all_indices.extend(batch_indices)

        if not all_scores:
            return RandomStrategy().select(model, unlabeled_indices, X_pool, n_to_acquire, batch_size)

        all_scores = torch.cat(all_scores)

        # Sort scores in descending order (highest entropy first)
        top_k_indices = torch.topk(all_scores, k=min(n_to_acquire, len(all_scores))).indices

        best_indices = [all_indices[i] for i in top_k_indices]
        return best_indices


def get_strategy(name: str):
    """Factory function to get a strategy instance by name."""
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
