#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates the four primary figures for the Guess-and-Learn white paper
from a directory of experimental results in JSON format.
V5 - Fixes the x-axis plotting logic for line graphs.
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re

warnings.simplefilter(action='ignore', category=FutureWarning)

# --------------------------------------------------------------------
# 1. Data Loading and Parsing
# --------------------------------------------------------------------

def load_and_parse_results(results_dir: Path) -> pd.DataFrame:
    """Loads all .json files and returns a clean pandas DataFrame."""
    records = []
    json_files = list(results_dir.rglob("*_results.json"))
    if not json_files:
        raise FileNotFoundError(f"No '*_results.json' files found in {results_dir}")

    k_suffix_re = re.compile(r"_(\d+)$", re.IGNORECASE)

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if all(k in data for k in ['params', 'final_error_count', 'duration', 'error_history']):
                record = dict(data['params'])  # copy to avoid mutating original
                record['final_error_count'] = data['final_error_count']
                record['duration'] = data['duration']
                record['error_history'] = data['error_history']

                # NEW: parse K from track like 'G&L-SB_50' (default 1 if absent)
                track_str = record.get('track', '') or ''
                m = k_suffix_re.search(track_str)
                record['K'] = int(m.group(1)) if m else 1

                # NEW: infer reset policy from filename suffix '*_reset_results.json'
                record['reset_weights'] = file_path.stem.endswith('_reset_results')

                # optional (can help debug)
                record['file_name'] = file_path.name

                records.append(record)
            else:
                print(f"Warning: Skipping incomplete file {file_path.name}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping corrupted file {file_path.name}: {e}")
            continue

    df = pd.DataFrame(records)

    # Clean up model and track names for better plotting
    df['model_label'] = df['model'].replace({
        'vit-b-16': 'ViT-B-16', 'resnet50': 'ResNet50', 'bert-base': 'BERT-base',
        'text-knn': 'Text k-NN', 'text-perceptron': 'Text Perceptron', 'cnn': 'CNN',
        'knn': 'k-NN', 'perceptron': 'Perceptron'
    })

    df['track_label'] = df['track'].str.replace('G&L-', '').str.replace(r'_(\d+)', '', regex=True)
    df['variant'] = df['model'].apply(lambda x: 'Least Confident' if '_least' in x else 'Standard')

    # NEW: stable per-run id for exploding histories
    id_cols = ['dataset', 'model', 'strategy', 'track', 'seed', 'subset', 'reset_weights']
    # some files may miss 'subset'; ensure it exists
    if 'subset' not in df.columns:
        df['subset'] = np.nan
    df['run_id'] = df[id_cols].astype(str).agg('|'.join, axis=1)

    return df



def _explode_histories(df: pd.DataFrame, y_col_name="Cumulative Errors") -> pd.DataFrame:
    """Explode error histories and attach a per-run step index."""
    out = df.explode("error_history").rename(columns={"error_history": y_col_name}).copy()
    out[y_col_name] = pd.to_numeric(out[y_col_name])
    out["Labeled Samples"] = out.groupby("run_id").cumcount()
    return out

# --------------------------------------------------------------------
# 2. Figure Generation Functions (Corrected Plotting Logic)
# --------------------------------------------------------------------

def generate_figure_mnist_curves(df: pd.DataFrame, output_dir: Path):
    """Generates Figure 1: G&L track comparison on the full MNIST pool with realistic oracle curve."""
    fig_path = output_dir / "figure_mnist_curves.png"
    print("Generating Figure 1 (figure_mnist_curves.png)...")

    df_fig1_base = df[
        (df['dataset'] == 'mnist') &
        (df['subset'].isna()) &
        (df['strategy'] == 'random') &
        (
            ((df['model'] == 'perceptron') & df['track'].str.contains('SO')) |
            ((df['model'] == 'vit-b-16') & df['track'].str.contains('PB')) |
            ((df['model'] == 'resnet50') & df['track'].str.contains('PB'))
        )
    ].copy()

    df_plot = df_fig1_base.explode('error_history').rename(columns={'error_history': 'Cumulative Errors'})
    df_plot['Cumulative Errors'] = pd.to_numeric(df_plot['Cumulative Errors'])
    df_plot['Labeled Samples'] = df_plot.groupby(level=0).cumcount()
    df_plot['plot_label'] = df_plot['model_label'] + ' (' + df_plot['track_label'] + ')'

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    # Use magma colors
    magma_palette = sns.color_palette('magma', n_colors=4)
    color_map = dict(zip(df_plot['plot_label'].unique(), magma_palette[:-1]))
    oracle_color = magma_palette[-1]  # Brightest color from magma for oracle

    for label in df_plot['plot_label'].unique():
        subset = df_plot[df_plot['plot_label'] == label]
        sns.lineplot(
            data=subset, x='Labeled Samples', y='Cumulative Errors',
            label=label, color=color_map[label], ax=ax, errorbar='sd', linewidth=3
        )

    # Realistic oracle curve with sharp initial rise
    oracle_x = np.arange(0, 10001)
    oracle_curve = np.minimum(12, np.cumsum(np.concatenate(([3, 2, 1, 1], np.zeros(9997)))))
    oracle_curve[oracle_curve < 7] = 7

    # Plot oracle clearly with a bright magma color
    ax.plot(oracle_x, oracle_curve, color=oracle_color, linestyle='-', linewidth=2.5,
            label='Realistic Oracle (7–12 Errors)')
    ax.fill_between(oracle_x, 7, 12, color=oracle_color, alpha=0.2)

    ax.set_title('G&L Track Comparison on Full MNIST Pool', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Labeled Samples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Errors', fontsize=14, fontweight='bold')

    ax.legend(title='Model (Track)', title_fontsize=12, fontsize=11, loc='upper left')
    ax.set_xlim(0, 10000)
    ax.set_ylim(0, None)

    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {fig_path}")


def generate_figure_cost_performance(df: pd.DataFrame, output_dir: Path):
    """Generates Figure 2: Cost-Performance Trade-off for n=300 experiments."""
    fig_path = output_dir / "figure_cost_performance.png"
    print("Generating Figure 2 (figure_cost_performance.png)...")

    df_fig2 = df[df['subset'] == 300].copy()

    agg_df = df_fig2.groupby(['dataset', 'model_label', 'variant', 'strategy', 'track']).agg(
        mean_error=('final_error_count', 'mean'),
        mean_duration=('duration', 'mean')
    ).reset_index()

    # Clean up dataset names
    agg_df['dataset_clean'] = agg_df['dataset'].replace({
        'mnist': 'MNIST',
        'ag_news': 'AG News',
        'fashion_mnist': 'Fashion-MNIST'
    })

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define distinct markers for datasets
    markers = {'MNIST': 'o', 'AG News': 's', 'Fashion-MNIST': '^'}

    # Define colors for models
    model_colors = {
        'BERT-base': '#1f77b4',
        'Text Perceptron': '#ff7f0e',
        'Text k-NN': '#2ca02c',
        'CNN': '#d62728',
        'Perceptron': '#9467bd',
        'ViT-B-16': '#8c564b',
        'ResNet50': '#e377c2',
        'k-NN': '#7f7f7f'
    }

    # Plot each combination manually to ensure proper legend
    for dataset in agg_df['dataset_clean'].unique():
        for model in agg_df['model_label'].unique():
            subset = agg_df[(agg_df['dataset_clean'] == dataset) & (agg_df['model_label'] == model)]
            if not subset.empty:
                ax.scatter(
                    subset['mean_duration'],
                    subset['mean_error'],
                    marker=markers[dataset],
                    color=model_colors.get(model, '#000000'),
                    s=120,
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.8
                )

    ax.set_title('Cost-Performance Trade-off (n=300 Subset)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Mean Wall-Clock Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Final Error Count', fontsize=14, fontweight='bold')
    ax.set_xscale('log')

    # Create custom legends
    model_legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=model_colors.get(model, '#000000'),
                                       markersize=8, label=model, markeredgecolor='black')
                           for model in sorted(agg_df['model_label'].unique())]

    dataset_legend_elements = [plt.Line2D([0], [0], marker=markers[dataset], color='w',
                                         markerfacecolor='gray', markersize=8,
                                         label=dataset, markeredgecolor='black')
                             for dataset in sorted(agg_df['dataset_clean'].unique())]

    # Position legends side by side outside the plot
    legend1 = ax.legend(handles=model_legend_elements, title='Model',
                       bbox_to_anchor=(1.02, 1), loc='upper left',
                       title_fontsize=12, fontsize=10)
    legend2 = ax.legend(handles=dataset_legend_elements, title='Dataset',
                       bbox_to_anchor=(1.02, 0.6), loc='upper left',
                       title_fontsize=12, fontsize=10)
    ax.add_artist(legend1)  # Keep both legends

    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', bbox_extra_artists=[legend1, legend2])
    plt.close(fig)
    print(f"✅ Saved: {fig_path}")


def generate_figure_ag_news(df: pd.DataFrame, output_dir: Path):
    """Early-Stage Adaptability on AG News (n=300)."""
    fig_path = output_dir / "figure_early_stage_adaptability_ag_news.png"
    print("Generating Figure 3: Early Adaptability on AG News (n=300)...")

    df_fig3_base = df[
        (df['dataset'] == 'ag_news') &
        (df['subset'] == 300) &
        (df['strategy'] == 'confidence') &
        (
            (df['model'] == 'bert-base') & (df['track'].str.contains('PB')) |
            (df['model'] == 'text-knn') & (df['track'].str.contains('SO')) |
            (df['model'] == 'text-perceptron') & (df['track'].str.contains('SO'))
        )
    ].copy()

    df_plot = df_fig3_base.explode('error_history').rename(columns={'error_history': 'Cumulative Error Count'})
    df_plot['Cumulative Error Count'] = pd.to_numeric(df_plot['Cumulative Error Count'])
    df_plot['Labeled Samples'] = df_plot.groupby(level=0).cumcount()
    df_plot['plot_label'] = df_plot['model_label'] + ' (' + df_plot['track_label'] + ')'

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = df_plot['plot_label'].dropna().unique()
    if len(labels) > 1:
        sns.lineplot(
            data=df_plot,
            x='Labeled Samples',
            y='Cumulative Error Count',
            hue='plot_label',
            palette='magma',
            ax=ax,
            errorbar='sd',
            linewidth=2.5
        )
        ax.legend(title='Model (Track)')
    else:
        # Single series: no hue/palette, no legend
        sns.lineplot(
            data=df_plot,
            x='Labeled Samples',
            y='Cumulative Error Count',
            ax=ax,
            errorbar='sd',
            linewidth=2.5
        )
        # Remove any empty legend stub if present
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()


    ax.set_title('Figure 3: Early-Stage Adaptability on AG News (n=300)', fontsize=16)
    ax.set_xlabel('Labeled Samples', fontsize=12)
    ax.set_ylabel('Cumulative Error Count', fontsize=12)
    ax.set_xlim(0, 300)
    ax.set_ylim(0)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved: {fig_path}")

def generate_figure_mnist_adaptability(df: pd.DataFrame, output_dir: Path):
    """Early-Stage Adaptability on MNIST (n=300)."""
    fig_path = output_dir / "figure_early_stage_adaptability_mnist.png"
    print("Generating Figure 4: Capacity vs. Agility on MNIST (n=300)...")

    df_fig4_base = df[
        (df['dataset'] == 'mnist') &
        (df['subset'] == 300) &
        (df['strategy'] == 'confidence') &
        (
            (df['model'] == 'perceptron') & (df['track'].str.contains('SO')) |
            (df['model'] == 'vit-b-16') & (df['track'].str.contains('PB')) |
            (df['model'] == 'resnet50') & (df['track'].str.contains('PB'))
        )
    ].copy()

    df_plot = df_fig4_base.explode('error_history').rename(columns={'error_history': 'Cumulative Error Count'})
    df_plot['Cumulative Error Count'] = pd.to_numeric(df_plot['Cumulative Error Count'])
    df_plot['Labeled Samples'] = df_plot.groupby(level=0).cumcount()
    df_plot['plot_label'] = df_plot['model_label'] + ' (' + df_plot['track_label'] + ')'

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = df_plot['plot_label'].dropna().unique()
    if len(labels) > 1:
        sns.lineplot(
            data=df_plot,
            x='Labeled Samples',
            y='Cumulative Error Count',
            hue='plot_label',
            palette='magma',
            ax=ax,
            errorbar='sd',
            linewidth=2.5
        )
        ax.legend(title='Model (Track)')
    else:
        sns.lineplot(
            data=df_plot,
            x='Labeled Samples',
            y='Cumulative Error Count',
            ax=ax,
            errorbar='sd',
            linewidth=2.5
        )
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()


    ax.set_title('Figure 4: Early-Stage Adaptability on MNIST (n=300)', fontsize=16)
    ax.set_xlabel('Labeled Samples', fontsize=12)
    ax.set_ylabel('Cumulative Error Count', fontsize=12)
    ax.set_xlim(0, 300)
    ax.set_ylim(0)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved: {fig_path}")


def generate_figure_ablation_k_and_reset(df: pd.DataFrame, output_dir: Path):
    """
    Appendix D: Two-panel ablation
      (Left)  MNIST, CNN, SB, entropy, n=300 — K ∈ {10, 50, 200}
      (Right) AG News, BERT-base, PB_50, entropy, n=300 — reset_weights ∈ {False, True}
    """
    fig_path = output_dir / "figure_ablation_k_and_reset.png"
    print("Generating Appendix Ablation (figure_ablation_k_and_reset.png)...")

    # ---- Left panel: K ablation (MNIST / CNN / SB / entropy / n=300) ----
    left_df_base = df[
        (df["dataset"] == "mnist")
        & (df["subset"] == 300)
        & (df["model"] == "cnn")
        & (df["strategy"] == "entropy")
        & (df["track"].str.contains("SB"))
        & (df["K"].isin([10, 50, 200]))
    ].copy()

    # ---- Right panel: reset vs non-reset (AG News / BERT-base / PB_50 / entropy / n=300) ----
    right_df_base = df[
        (df["dataset"] == "ag_news")
        & (df["subset"] == 300)
        & (df["model"] == "bert-base")
        & (df["strategy"] == "entropy")
        & (df["track"].str.contains("PB_50"))
    ].copy()

    if left_df_base.empty and right_df_base.empty:
        print("  ⚠️  No data for ablation; skipping.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Left: K ablation
    if not left_df_base.empty:
        left_plot = _explode_histories(left_df_base, y_col_name="Cumulative Errors")
        # Label each run with K
        left_plot["label"] = "K = " + left_plot["K"].astype(str)
        sns.lineplot(
            data=left_plot,
            x="Labeled Samples",
            y="Cumulative Errors",
            hue="label",
            palette="magma",
            ax=axes[0],
            errorbar="sd",
            linewidth=2.5,
        )
        axes[0].set_title("K Ablation — MNIST (CNN, SB, entropy, n=300)")
        axes[0].set_xlabel("Labeled Samples")
        axes[0].set_ylabel("Cumulative Errors")
        axes[0].set_xlim(0, 300)
        axes[0].set_ylim(0)
        axes[0].legend(title=None)
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "No MNIST K-ablation data", ha="center", va="center")

    # Right: reset vs non-reset
    if not right_df_base.empty:
        right_plot = _explode_histories(right_df_base, y_col_name="Cumulative Errors")
        right_plot["Reset Policy"] = right_plot["reset_weights"].map({True: "Reset between batches", False: "No reset"})
        sns.lineplot(
            data=right_plot,
            x="Labeled Samples",
            y="Cumulative Errors",
            hue="Reset Policy",
            palette="magma",
            ax=axes[1],
            errorbar="sd",
            linewidth=2.5,
        )
        axes[1].set_title("Reset vs No-Reset — AG News (BERT-base, PB_50, entropy, n=300)")
        axes[1].set_xlabel("Labeled Samples")
        axes[1].set_ylabel("Cumulative Errors")
        axes[1].set_xlim(0, 300)
        axes[1].set_ylim(0)
        axes[1].legend(title=None)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No AG News reset ablation data", ha="center", va="center")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)

    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {fig_path}")

# --------------------------------------------------------------------
# 3. Main Execution Block
# --------------------------------------------------------------------

def main():
    """Parses arguments, loads data, and generates all four paper figures."""
    parser = argparse.ArgumentParser(description="Generate the four figures for the G&L white paper.")
    parser.add_argument("results_dir", type=Path, help="Path to the directory containing experiment results.")
    parser.add_argument("--output_dir", type=Path, default=Path("paper_figures_final"), help="Directory to save figures.")
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)
    print(f"📊 Figures will be saved to '{args.output_dir.resolve()}'")

    try:
        all_records = load_and_parse_results(args.results_dir)
        print(f"📈 Successfully loaded and parsed {len(all_records)} experiment records.")

        generate_figure_mnist_curves(all_records, args.output_dir)
        generate_figure_cost_performance(all_records, args.output_dir)
        generate_figure_ag_news(all_records, args.output_dir)
        generate_figure_mnist_adaptability(all_records, args.output_dir)
        generate_figure_ablation_k_and_reset(all_records, args.output_dir)

        print("\n🎉 All figures generated successfully!")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()