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


class KCenterGreedyStrategy(AcquisitionStrategy):
    """
    Selects a batch of examples using the k-Center Greedy method (Furthest Point Sampling).
    It aims to minimize the maximum distance between any data point and the set of labeled points.
    In batch mode (n > 1), it iteratively selects points that maximize the distance to the
    union of the already labeled set and the currently selected batch.
    """

    def select(self, model, unlabeled_indices, X_pool, n_to_acquire=1, batch_size=256):
        # 1. Identify labeled vs unlabeled indices within the full pool context
        n_pool = len(X_pool)
        # Note: We assume indices not in 'unlabeled_indices' are labeled.
        labeled_indices = list(set(range(n_pool)) - set(unlabeled_indices))

        # --- Helper: Extract features in batches to prevent GPU OOM ---
        def get_features(indices):
            # Check if the underlying model is a PyTorch model (has .training attribute)
            is_torch = hasattr(model.model, "training")
            feats = []

            # Toggle eval mode only if it's a PyTorch model
            was_training = False
            if is_torch:
                was_training = model.model.training
                model.model.eval()

            with torch.no_grad():
                for i in range(0, len(indices), batch_size):
                    batch_idx = indices[i : i + batch_size]
                    if isinstance(X_pool, list):
                        batch_x = [X_pool[j] for j in batch_idx]
                    else:
                        batch_x = X_pool[batch_idx]

                    # Extract features and move to CPU to save GPU memory
                    f = model.extract_features(batch_x).detach().cpu()
                    feats.append(f)

            # Restore model state only if it was a PyTorch model
            if is_torch and was_training:
                model.model.train()

            if not feats:
                return torch.tensor([])
            return torch.cat(feats, dim=0)
        # -------------------------------------------------------------

        # Get features for the candidate pool (unlabeled)
        unlabeled_feats = get_features(unlabeled_indices)
        n_unlabeled = unlabeled_feats.shape[0]

        # Safety check: if asking for more than we have, return everything
        if n_to_acquire >= n_unlabeled:
            return list(unlabeled_indices)

        # 2. Initialize Minimum Distances
        # We maintain a 'min_dist' for every unlabeled point: dist to the CLOSEST labeled point.

        if not labeled_indices:
            # COLD START: No labeled data yet.
            # Strategy: Pick 1 random point to anchor the diversity, then be greedy.
            # This ensures even the first batch is diverse (Furthest Point Sampling).
            first_idx = np.random.choice(n_unlabeled)
            selected_indices_local = [first_idx]

            # Initialize distances relative to this first random point
            # cdist expects [1, D] and [N, D] -> [1, N]
            initial_feat = unlabeled_feats[first_idx].unsqueeze(0)
            current_min_dists = torch.cdist(initial_feat, unlabeled_feats).squeeze(0) # Shape: [N_unlabeled]

            # Mask the selected point (dist = -1) so it isn't picked again
            current_min_dists[first_idx] = -1.0

            # We have picked 1, reduce remaining quota
            n_needed = n_to_acquire - 1
        else:
            # WARM START: Calculate distances to existing labeled set
            labeled_feats = get_features(labeled_indices)

            # Compute dists [N_unlabeled, N_labeled]
            # Note: If N*M is huge, this can be chunked, but fits in RAM for MNIST/AG News
            dists = torch.cdist(unlabeled_feats, labeled_feats)

            # For each unlabeled point, find distance to the CLOSEST labeled point
            current_min_dists, _ = torch.min(dists, dim=1) # Shape: [N_unlabeled]

            selected_indices_local = []
            n_needed = n_to_acquire

        # 3. Greedy Selection Loop
        # Iteratively pick the point that is *furthest* from the current set (Labeled + Selected)
        for _ in range(n_needed):
            # Find the point with the maximum 'minimum distance'
            idx_in_unlabeled = torch.argmax(current_min_dists).item()
            selected_indices_local.append(idx_in_unlabeled)

            # Update distances for the next iteration (unless we are done)
            if len(selected_indices_local) < n_to_acquire:
                new_feat = unlabeled_feats[idx_in_unlabeled].unsqueeze(0)

                # Calculate distance from all unlabeled points to this NEWLY selected point
                dist_to_new = torch.cdist(unlabeled_feats, new_feat).squeeze(1)

                # Update the running minimum:
                # New min_dist is min(old_min_dist, dist_to_new_point)
                current_min_dists = torch.min(current_min_dists, dist_to_new)

                # Mask the selected point
                current_min_dists[idx_in_unlabeled] = -1.0

        # 4. Map local indices back to global pool indices
        final_indices = [unlabeled_indices[i] for i in selected_indices_local]

        # Ensure we return standard python ints for JSON serialization safety
        return [int(x) for x in final_indices]


def get_strategy(name: str):
    """Factory function to get a strategy instance by name."""
    strategies = {
        "random": RandomStrategy,
        "confidence": ConfidenceStrategy,
        "least_confidence": LeastConfidenceStrategy,
        "margin": MarginStrategy,
        "entropy": EntropyStrategy,
        "k_center_greedy": KCenterGreedyStrategy,
    }
    if name.lower() in strategies:
        return strategies[name.lower()]()
    else:
        raise ValueError(f"Strategy '{name}' not supported.")
