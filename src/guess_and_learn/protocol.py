import re
import numpy as np
from tqdm import tqdm
import json
import os
import matplotlib

# Corrected relative import
from .models import ExperimentConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import random


class GnlProtocol:
    def __init__(self, model, strategy, X_pool, Y_pool, exp_config: ExperimentConfig):
        self.model = model
        self.strategy = strategy
        self.X_pool = X_pool
        self.Y_pool = Y_pool
        self.exp_config = exp_config
        self.n_pool = len(X_pool)

        self.unlabeled_indices = list(range(self.n_pool))
        self.labeled_indices = []
        self.is_error = []

        self.cumulative_errors = 0
        self.error_history = []

        seed = self.exp_config.seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def run(self):
        """Main experiment loop, fully refactored to use ExperimentConfig."""
        pbar = tqdm(total=self.n_pool, desc="G&L Protocol")

        while len(self.unlabeled_indices) > 0:
            # Step 1: Determine acquisition batch size from the config object
            update_cadence = self.exp_config.K
            n_to_acquire = 1 if self.exp_config.is_online else update_cadence
            n_to_acquire = min(n_to_acquire, len(self.unlabeled_indices))

            # Step 2: Select a BATCH of examples
            selected_indices_batch = self.strategy.select(
                self.model, self.unlabeled_indices, self.X_pool, n_to_acquire=n_to_acquire
            )

            # Step 3: Process the acquired batch
            for selected_pool_idx in selected_indices_batch:
                # --- BUG FIX: Ensure index is a standard Python int ---
                selected_pool_idx = int(selected_pool_idx)

                if isinstance(self.X_pool, list):
                    x_t = [self.X_pool[selected_pool_idx]]
                else:
                    x_raw = self.X_pool[selected_pool_idx]
                    x_t = x_raw.unsqueeze(0) if torch.is_tensor(x_raw) else [x_raw]

                y_t_true = self.Y_pool[selected_pool_idx]
                y_t_pred = self.model.predict(x_t)

                pred_val = y_t_pred.item() if torch.is_tensor(y_t_pred) else y_t_pred
                true_val = y_t_true.item() if torch.is_tensor(y_t_true) else y_t_true
                made_error = pred_val != true_val
                if made_error:
                    self.cumulative_errors += 1

                self.error_history.append(self.cumulative_errors)
                self.labeled_indices.append(selected_pool_idx)
                self.is_error.append(made_error)
                self.unlabeled_indices.remove(selected_pool_idx)
                pbar.update(1)

            # Step 4: Check for model update
            should_update = False
            if self.exp_config.is_online:
                should_update = True
            elif len(self.labeled_indices) > 0 and len(self.labeled_indices) % update_cadence == 0:
                should_update = True

            if should_update:
                if not self.exp_config.is_online and self.exp_config.reset_weights:
                    try:
                        self.model.reset()
                    except NotImplementedError:
                        pass

                if isinstance(self.X_pool, list):
                    X_labeled = [self.X_pool[i] for i in self.labeled_indices]
                else:
                    X_labeled = self.X_pool[self.labeled_indices]
                Y_labeled = self.Y_pool[self.labeled_indices]

                pbar.set_description(f"G&L Protocol (Updating model with {len(self.labeled_indices)} labels)")
                self.model.update(X_labeled, Y_labeled, self.exp_config)

            if self.labeled_indices:
                pbar.set_postfix({"errors": self.cumulative_errors, "error_rate": f"{self.cumulative_errors/len(self.labeled_indices):.3f}"})

        pbar.close()
        return self.error_history, self.labeled_indices, self.is_error


def _to_jsonable(obj):
    """Recursively cast NumPy / Torch types to vanilla Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (torch.Tensor,)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def save_results(
    duration: float,
    error_history,
    labeled_indices,
    is_error,
    params,
    output_dir,
    model=None,
    X_pool=None,
    Y_pool=None,
    batch_size: int = 512,
):
    """Persist results, plot, and (optionally) dataset-wide features / labels."""
    if "seed" in params and params["seed"] is not None:
        seed = params["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    subset_str = f"_s{params['subset']}" if params.get("subset") else ""
    fname_base = f"{params['dataset']}_{params['model']}_{params['strategy']}_" f"{params['track']}_seed{params['seed']}{subset_str}"

    results_path = os.path.join(output_dir, f"{fname_base}_results.json")
    with open(results_path, "w") as f:
        json.dump(
            _to_jsonable(
                {
                    "error_history": error_history,
                    "params": params,
                    "labeled_indices": labeled_indices,
                    "is_error": is_error,
                    "final_error_count": error_history[-1] if error_history else 0,
                    "final_error_rate": (error_history[-1] / len(error_history) if error_history else 0),
                    "rng_state": {
                        "python_random": random.getstate(),
                        "numpy_random": np.random.get_state(),
                        "torch_random": torch.random.get_rng_state(),
                    },
                    "duration": duration,
                }
            ),
            f,
            indent=2,
        )

    if X_pool is None or Y_pool is None:
        return

    label_path = os.path.join(output_dir, f"{fname_base}_labels.pt")
    if not os.path.exists(label_path):
        torch.save(Y_pool, label_path)

    feature_path = os.path.join(output_dir, f"{fname_base}_features.pt")
    if os.path.exists(feature_path):
        print(f"Features already exist at {feature_path}, skipping extraction.")
    else:
        if model is not None and hasattr(model, "extract_features"):
            try:
                if isinstance(X_pool, list):
                    feats = model.extract_features(X_pool)
                else:
                    batches = []
                    for i in range(0, len(X_pool), batch_size):
                        xb = X_pool[i : i + batch_size]
                        with torch.no_grad():
                            fb = model.extract_features(xb).cpu()
                        batches.append(fb)
                    feats = torch.cat(batches, dim=0)
                torch.save(feats, feature_path)
                print(f"Saved features to {feature_path}")
            except NotImplementedError:
                pass

        if not os.path.exists(feature_path):
            if not isinstance(X_pool, list) and torch.is_tensor(X_pool):
                raw = X_pool.reshape(X_pool.shape[0], -1).cpu()
                torch.save(raw, feature_path)
                print(f"Saved raw flattened features to {feature_path}")
            else:
                print("Feature dump skipped (no extractor for text model).")

    plt.figure(figsize=(10, 6))
    plt.plot(error_history, linewidth=2)
    plt.title(f"G&L Error Curve: {params['model']} on {params['dataset']} ({params['strategy']})")
    plt.xlabel("Number of Labeled Samples")
    plt.ylabel("Cumulative Errors")
    plt.grid(True, alpha=0.3)
    if error_history:
        plt.annotate(
            f"Final: {error_history[-1]} errors",
            xy=(len(error_history) - 1, error_history[-1]),
            xytext=(len(error_history) * 0.7, error_history[-1] * 1.1),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
        )

    plot_path = os.path.join(output_dir, f"{fname_base}_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Results saved to {results_path}")
    print(f"Plot saved to {plot_path}")