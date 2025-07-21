import torch
import numpy as np
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
from .datasets import get_data_for_protocol

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

        self.cumulative_errors = 0
        self.error_history = []

    def run(self):
        pbar = tqdm(range(self.n_pool), desc="G&L Protocol")
        for t in pbar:
            # 1. Select
            if not self.labeled_indices or self.track_config['track'] in ['G&L-SO', 'G&L-PO']:
                # For online tracks or the first step, selection happens every time
                selected_pool_idx = self.strategy.select(self.model, self.unlabeled_indices, self.X_pool)
            else:
                # For batch tracks, we can select randomly within the batch to save compute
                # A more sophisticated implementation would still score all points.
                selected_pool_idx = np.random.choice(self.unlabeled_indices)

            # 2. Guess
            x_t = self.X_pool[selected_pool_idx].unsqueeze(0)
            y_t_true = self.Y_pool[selected_pool_idx]
            y_t_pred = self.model.predict(x_t)

            # 3. Evaluate
            if y_t_pred.item() != y_t_true.item():
                self.cumulative_errors += 1

            self.error_history.append(self.cumulative_errors)

            # Move index from unlabeled to labeled
            self.unlabeled_indices.remove(selected_pool_idx)
            self.labeled_indices.append(selected_pool_idx)

            # 4. Update
            update_cadence = self.track_config.get('K', 1)
            if len(self.labeled_indices) % update_cadence == 0:
                X_labeled = self.X_pool[self.labeled_indices]
                Y_labeled = self.Y_pool[self.labeled_indices]
                pbar.set_description(f"G&L Protocol (Updating model with {len(self.labeled_indices)} labels)")
                self.model.update(X_labeled, Y_labeled, self.track_config)

            pbar.set_postfix({'errors': self.cumulative_errors})

        return self.error_history

def save_results(results, params, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a unique filename
    filename_base = f"{params['dataset']}_{params['model']}_{params['strategy']}_{params['track']}_seed{params['seed']}"

    # Save results
    results_path = os.path.join(output_dir, f"{filename_base}_results.json")
    with open(results_path, 'w') as f:
        json.dump({'error_history': results, 'params': params}, f)

    # Save plot
    plt.figure(figsize=(10, 6))
    plt.plot(results)
    plt.title(f"G&L Error Curve for {params['model']} on {params['dataset']} ({params['strategy']})")
    plt.xlabel("Number of Labeled Samples")
    plt.ylabel("Cumulative Errors")
    plt.grid(True)
    plot_path = os.path.join(output_dir, f"{filename_base}_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Results saved to {results_path}")
    print(f"Plot saved to {plot_path}")