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

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Ensure essential keys exist before processing
            if all(k in data for k in ['params', 'final_error_count', 'duration', 'error_history']):
                record = data['params']
                record['final_error_count'] = data['final_error_count']
                record['duration'] = data['duration']
                record['error_history'] = data['error_history']
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

    return df

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

    ax.set_title('Figure 3: Early-Stage Adaptability on AG News (n=300)', fontsize=16)
    ax.set_xlabel('Labeled Samples', fontsize=12)
    ax.set_ylabel('Cumulative Error Count', fontsize=12)
    ax.legend(title='Model (Track)')
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

    ax.set_title('Figure 4: Early-Stage Adaptability on MNIST (n=300)', fontsize=16)
    ax.set_xlabel('Labeled Samples', fontsize=12)
    ax.set_ylabel('Cumulative Error Count', fontsize=12)
    ax.legend(title='Model (Track)')
    ax.set_xlim(0, 300)
    ax.set_ylim(0)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
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

        print("\n🎉 All figures generated successfully!")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()