import re
import numpy as np
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import torch
import random


class GnlProtocol:
    def __init__(self, model, strategy, X_pool, Y_pool, track_config):
        self.model = model
        self.strategy = strategy
        self.X_pool = X_pool
        self.Y_pool = Y_pool
        self.track_config = track_config
        self.n_pool = len(X_pool)

        self.unlabeled_indices = list(range(self.n_pool))
        self.labeled_indices = []
        self.is_error = []

        self.cumulative_errors = 0
        self.error_history = []

        seed = track_config.get("seed")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def run(self):
        pbar = tqdm(range(self.n_pool), desc="G&L Protocol")

        for t in pbar:
            # Step 1: Select next example using acquisition strategy
            # The strategy should ALWAYS be applied according to the paper
            selected_pool_idx = self.strategy.select(self.model, self.unlabeled_indices, self.X_pool)

            # Step 2: Make prediction (guess) BEFORE updating model
            if isinstance(self.X_pool, list):
                # Handle text data
                x_t = [self.X_pool[selected_pool_idx]]
            else:
                x_t = self.X_pool[selected_pool_idx].unsqueeze(0)

            y_t_true = self.Y_pool[selected_pool_idx]
            y_t_pred = self.model.predict(x_t)

            # Step 3: Get oracle feedback and update error
            if torch.is_tensor(y_t_pred):
                pred_val = y_t_pred.item()
            else:
                pred_val = y_t_pred

            if torch.is_tensor(y_t_true):
                true_val = y_t_true.item()
            else:
                true_val = y_t_true

            made_error = pred_val != true_val
            if made_error:
                self.cumulative_errors += 1

            # Step 4: Update tracking (do this BEFORE model update)
            self.error_history.append(self.cumulative_errors)
            self.labeled_indices.append(selected_pool_idx)
            self.is_error.append(made_error)
            self.unlabeled_indices.remove(selected_pool_idx)

            # Step 5 – model-update cadence (online vs batch) ------------
            track_str = self.track_config["track"]

            # If K not given explicitly, try to parse numeric suffix, e.g. *_50
            if "K" in self.track_config:
                update_cadence = self.track_config["K"]
            else:
                m = re.search(r"(\d+)$", track_str)  # accepts SB200 or SB_200
                update_cadence = int(m.group(1)) if m else 1

            should_update = False

            if track_str.startswith(("G&L-SO", "G&L-PO")):
                # Online tracks: update after every sample
                should_update = True
            elif track_str.startswith(("G&L-SB", "G&L-PB")):
                # Batch tracks: update every K samples
                if len(self.labeled_indices) % update_cadence == 0:
                    should_update = True

            if should_update:
                # Get labeled data for update
                if isinstance(self.X_pool, list):
                    # Handle text data
                    X_labeled = [self.X_pool[i] for i in self.labeled_indices]
                else:
                    X_labeled = self.X_pool[self.labeled_indices]
                Y_labeled = self.Y_pool[self.labeled_indices]

                pbar.set_description(f"G&L Protocol (Updating model with {len(self.labeled_indices)} labels)")
                self.model.update(X_labeled, Y_labeled, self.track_config)

            pbar.set_postfix({"errors": self.cumulative_errors, "error_rate": f"{self.cumulative_errors/(t+1):.3f}"})

        return self.error_history, self.labeled_indices, self.is_error


def save_results(error_history, labeled_indices, is_error, params, output_dir, model=None, X_pool=None, Y_pool=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename_base = f"{params['dataset']}_{params['model']}_{params['strategy']}_{params['track']}_seed{params['seed']}"

    results_path = os.path.join(output_dir, f"{filename_base}_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "error_history": error_history,
                "params": params,
                "labeled_indices": labeled_indices,
                "is_error": is_error,
                "final_error_count": error_history[-1] if error_history else 0,
                "final_error_rate": error_history[-1] / len(error_history) if error_history else 0,
            },
            f,
            indent=2,
        )

    # Create error curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(error_history, linewidth=2)
    plt.title(f"G&L Error Curve: {params['model']} on {params['dataset']} ({params['strategy']})")
    plt.xlabel("Number of Labeled Samples")
    plt.ylabel("Cumulative Errors")
    plt.grid(True, alpha=0.3)

    # Add final error count annotation
    if error_history:
        plt.annotate(
            f"Final: {error_history[-1]} errors",
            xy=(len(error_history) - 1, error_history[-1]),
            xytext=(len(error_history) * 0.7, error_history[-1] * 1.1),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
        )

    plot_path = os.path.join(output_dir, f"{filename_base}_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Results saved to {results_path}")
    print(f"Plot saved to {plot_path}")
    print(f"Final error count: {error_history[-1] if error_history else 0}")
    print(f"Final error rate: {error_history[-1]/len(error_history) if error_history else 0:.3f}")

    # Save dataset features and labels if provided
    if X_pool is not None and Y_pool is not None:
        label_path = os.path.join(output_dir, f"{params['dataset']}_labels.pt")
        feature_path = os.path.join(output_dir, f"{params['dataset']}_features.pt")

        if not os.path.exists(label_path):
            torch.save(Y_pool, label_path)
            print(f"Saved labels to {label_path}")

        if not os.path.exists(feature_path):
            if model is not None:
                with torch.no_grad():
                    feats = model.extract_features(X_pool)
                torch.save(feats.cpu() if torch.is_tensor(feats) else feats, feature_path)
                print(f"Saved features to {feature_path}")
            elif not isinstance(X_pool, list):  # vision fallback only
                X_flat = X_pool.reshape(X_pool.shape[0], -1) if torch.is_tensor(X_pool) else X_pool
                torch.save(X_flat, feature_path)
                print(f"Saved features to {feature_path}")
            else:
                # raw text + no model → skip feature dump cleanly
                pass
