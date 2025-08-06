import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rich.console import Console

def plot_vs_optimal(results_dir: str, output_dir: str, experiments_to_plot: dict):
    """
    Plots learning curves for specific experiments against a conceptual optimal.
    """
    console = Console()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    for plot_name, config in experiments_to_plot.items():
        console.print(f"\n[bold]Generating plot: {plot_name}[/bold]")
        plt.figure(figsize=(10, 6))

        all_histories = []
        for exp_pattern in config['patterns']:
            exp_records = []
            for filepath in Path(results_dir).rglob(f"{exp_pattern}*_results.json"):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    history = data['error_history']
                    for step, error_count in enumerate(history):
                        exp_records.append({
                            'pattern': exp_pattern,
                            'step': step + 1,
                            'error_count': error_count,
                            'seed': data['params']['seed']
                        })
                except (KeyError, json.JSONDecodeError):
                    continue
            all_histories.extend(exp_records)

        if not all_histories:
            console.print(f"[yellow]Warning: No data found for plot '{plot_name}'[/yellow]")
            continue

        df = pd.DataFrame(all_histories)

        # Plot learning curves
        sns.lineplot(data=df, x='step', y='error_count', hue='pattern', errorbar='sd', legend='full')

        # Plot conceptual optimal
        plt.axhline(y=config['optimal'], color='red', linestyle='--', linewidth=2, label=f"Conceptual Optimal ({config['optimal']} errors)")

        plt.title(f"Adaptability Gap on {config['title']}", fontsize=16)
        plt.xlabel("Number of Labeled Samples", fontsize=12)
        plt.ylabel("Cumulative Errors", fontsize=12)
        plt.legend()
        plt.tight_layout()

        plot_path = output_path / f"{plot_name}.png"
        plt.savefig(plot_path, dpi=300)
        console.print(f"✅ Saved plot to {plot_path}")
        plt.close()

if __name__ == '__main__':
    # Define the experiments you want to plot here
    EXPERIMENTS = {
        "mnist_adaptability_gap_full": {
            "title": "MNIST (Full Dataset)",
            "patterns": [
                "mnist_vit-b-16_entropy_G&L-PB_50",
                "mnist_cnn_entropy_G&L-SB_50",
                "mnist_perceptron_entropy_G&L-SO",
            ],
            "optimal": 15
        },
        "agnews_adaptability_gap_full": {
            "title": "AG News (Full Dataset)",
            "patterns": [
                "ag_news_bert-base_entropy_G&L-PB_50",
                "ag_news_text-perceptron_entropy_G&L-SO",
            ],
            "optimal": 25
        },
    }

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory with JSON results.")
    parser.add_argument("--output_dir", default="analysis_output_new", help="Directory for output.")
    args = parser.parse_args()

    plot_vs_optimal(args.results_dir, args.output_dir, EXPERIMENTS)