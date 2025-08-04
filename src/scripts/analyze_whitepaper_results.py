#!/usr/bin/env python

import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table

# --- Helper Functions (unchanged) ---
def process_file(filepath: Path):
    try:
        with open(filepath, 'r') as f: data = json.load(f)
        params = data.get('params', {}); subset = params.get('subset', 'Full')
        subset_str = str(subset) if subset is not None else 'Full'
        records = []
        for i, error_count in enumerate(data['error_history']):
            records.append({
                'dataset': params.get('dataset'), 'model': params.get('model'),
                'strategy': params.get('strategy'), 'track': params.get('track'),
                'subset': subset_str, 'seed': int(params.get('seed')),
                'step': i + 1, 'error_count': error_count,
                'duration_sec': data.get('duration', float('nan'))
            })
        return records
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"⚠️ Warning: Could not process file {filepath}. Reason: {e}")
        return None

def load_and_filter_data(results_dir, datasets, models, strategies, tracks, subsets):
    all_records = []
    json_files = list(Path(results_dir).rglob('*_results.json'))
    if not json_files: raise FileNotFoundError(f"No '*_results.json' files found in '{results_dir}'.")
    for filepath in json_files:
        records = process_file(filepath)
        if records: all_records.extend(records)
    if not all_records: raise ValueError("No valid result files could be processed.")
    df = pd.DataFrame(all_records)
    initial_rows = len(df)
    if datasets: df = df[df['dataset'].isin(datasets)]
    if models: df = df[df['model'].isin(models)]
    if strategies: df = df[df['strategy'].isin(strategies)]
    if tracks: df = df[df['track'].isin(tracks)]
    if subsets:
        df['subset'] = df['subset'].astype(str)
        df = df[df['subset'].isin(subsets)]
    if df.empty: raise ValueError("No data left after filtering.")
    print(f"🗂️ Loaded {initial_rows} total records, {len(df)} remain after filtering.")
    return df

# --- Main Analysis Functions ---

def generate_summary_table(df):
    """
    Generates and prints a full summary table, with corrected column name case.
    """
    final_step_df = df.loc[df.groupby(['dataset', 'model', 'strategy', 'track', 'subset', 'seed'])['step'].idxmax()]
    summary = final_step_df.groupby(['dataset', 'model', 'strategy', 'track', 'subset']).agg(
        final_error_mean=('error_count', 'mean'),
        final_error_std=('error_count', 'std'),
        duration_mean=('duration_sec', 'mean'),
        duration_std=('duration_sec', 'std'),
        num_seeds=('seed', 'nunique')
    ).reset_index()

    # FIX: Use lowercase column names for sorting
    summary = summary.sort_values(by=["track", "dataset", "model", "strategy"])

    console = Console()
    console.print("\n[bold cyan]📊 Full Results Summary[/bold cyan]")

    table = Table(title="Final Performance and Cost Analysis")
    table.add_column("Track", style="yellow", justify="left")
    table.add_column("Dataset", style="magenta", justify="left")
    table.add_column("Model", style="green", justify="left")
    table.add_column("Strategy", style="blue", justify="left")
    table.add_column("Subset", style="dim")
    table.add_column("Final Error Count", justify="center", style="cyan")
    table.add_column("Wall Time (s)", justify="center", style="red")
    table.add_column("Seeds", justify="center")

    for _, row in summary.iterrows():
        # FIX: Use lowercase column names to access row data
        table.add_row(
            str(row['track']), str(row['dataset']), str(row['model']),
            str(row['strategy']), str(row['subset']),
            f"{row['final_error_mean']:.1f} ± {row['final_error_std']:.1f}",
            f"{row['duration_mean']:.2f} ± {row['duration_std']:.2f}",
            str(row['num_seeds'])
        )

    console.print(table)
    return summary

def generate_latex_table(summary_df, output_dir):
    Path(output_dir).mkdir(exist_ok=True)
    console = Console()
    console.print("\n[bold green]📄 Generating Curated LaTeX Table for Paper...[/bold green]")
    baseline = summary_df[summary_df['strategy'] == 'random'].set_index(['dataset', 'model', 'track', 'subset'])
    active_strategies = summary_df[summary_df['strategy'] != 'random']
    if active_strategies.empty:
        print("⚠️ No active strategies found to compare against baseline. Skipping LaTeX table generation.")
        return
    best_active_idx = active_strategies.loc[active_strategies.groupby(['dataset', 'model', 'track', 'subset'])['final_error_mean'].idxmin()]
    best_active = best_active_idx.set_index(['dataset', 'model', 'track', 'subset'])
    report_df = best_active.join(baseline['final_error_mean'].rename('baseline_error_mean'))
    initial_len = len(report_df)
    report_df.dropna(subset=['baseline_error_mean'], inplace=True)
    if len(report_df) < initial_len:
        console.print(f"[yellow]⚠️ Warning: Dropped {initial_len - len(report_df)} rows from LaTeX table due to missing 'random' baselines.[/yellow]")
    report_df['error_reduction_pct'] = ((report_df['final_error_mean'] - report_df['baseline_error_mean']) / report_df['baseline_error_mean']) * 100
    def format_error_reduction(val):
        if pd.isna(val): return "-"
        if val < -5: return f"\\cellcolor{{green!20}}{val:.1f}"
        elif val > 0: return f"\\cellcolor{{red!20}}{val:.1f}"
        else: return f"{val:.1f}"
    report_df[r'\% Error Reduction'] = report_df['error_reduction_pct'].apply(format_error_reduction)
    report_df['Min Error (Best)'] = report_df.apply(lambda row: f"{row['final_error_mean']:.1f} $\\pm$ {row['final_error_std']:.1f}", axis=1)
    final_latex_df = report_df.reset_index()[[
        'dataset', 'model', 'track', 'strategy', 'Min Error (Best)', 'baseline_error_mean', r'\% Error Reduction'
    ]]
    final_latex_df.rename(columns={
        'dataset': 'Dataset', 'model': 'Model', 'track': 'Track',
        'strategy': 'Best Strategy', 'baseline_error_mean': 'Min Error (Random)'
    }, inplace=True)
    latex_string = final_latex_df.to_latex(
        index=False, escape=False, column_format='llccrrr',
        header=['Dataset', 'Model', 'Track', 'Best Strategy', 'Min Error (Best)', 'Min Error (Random)', r'\% Error $\downarrow$'],
        na_rep="-"
    )
    latex_string = latex_string.replace('\\toprule', '\\begin{table}[ht]\n\\centering\n\\caption{Comparison of the best active learning strategy against a random baseline. Error reduction is colored green for improvement and red for degradation.}\n\\label{tab:main_results}\n\\toprule')
    latex_string = latex_string.replace('\\bottomrule', '\\bottomrule\n\\end{table}')
    latex_path = Path(output_dir) / 'publication_table.tex'
    latex_path.write_text(latex_string)
    console.print(f"✅ LaTeX table saved to [cyan]{latex_path}[/cyan]")
    console.print(latex_string)

def generate_plots(df, summary_df, output_dir):
    Path(output_dir).mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", palette="viridis")
    print("\n🎨 Generating Learning Curve (Error Count vs. Labels) plots...")
    for dataset in df['dataset'].unique():
        if df[df['dataset'] == dataset].empty: continue
        plt.figure(figsize=(12, 7))
        ax = sns.lineplot(data=df[df['dataset'] == dataset], x='step', y='error_count', hue='model', style='strategy', size='subset', errorbar='sd', linewidth=2)
        ax.set_title(f'Learning Curve on {dataset.upper()}', fontsize=16, weight='bold')
        ax.set_xlabel('Number of Labeled Samples', fontsize=12)
        ax.set_ylabel('Error Count (Lower is Better)', fontsize=12)
        plt.legend(title='Model/Strategy/Subset', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_path = Path(output_dir) / f'{dataset}_error_curve.png'
        plt.savefig(plot_path, dpi=300)
        print(f"✅ Saved plot to {plot_path}")
        plt.close()
    print("\n🎨 Generating Cost-Performance Trade-off plot...")
    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(data=summary_df, x='duration_mean', y='final_error_mean', hue='model', style='strategy', size='subset', sizes=(50, 250), alpha=0.8)
    for _, row in summary_df.iterrows():
        plt.errorbar(x=[row['duration_mean']], y=[row['final_error_mean']], xerr=[row['duration_std']], yerr=[row['final_error_std']], fmt='none', ecolor='gray', capsize=3, alpha=0.5)
    ax.set_title('Cost vs. Performance Trade-off', fontsize=16, weight='bold')
    ax.set_xlabel('Mean Wall-Clock Time (seconds)', fontsize=12)
    ax.set_ylabel('Mean Final Error Count (Lower is Better)', fontsize=12)
    plt.legend(title='Model/Strategy/Subset', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plot_path = Path(output_dir) / 'cost_performance_tradeoff.png'
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Saved plot to {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Analyze active learning experiment results from JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("results_dir", type=str, help="Path to the root directory containing JSON result files.")
    parser.add_argument("--output_dir", type=str, default="analysis_output", help="Directory to save generated plots and tables.")
    parser.add_argument("--datasets", nargs='+', default=None, help="Filter by datasets. Default: All.")
    parser.add_argument("--models", nargs='+', default=None, help="Filter by models. Default: All.")
    parser.add_argument("--strategies", nargs='+', default=['entropy', 'least_confidence', 'margin', 'random', 'confidence'], help="Filter by strategies.")
    parser.add_argument("--tracks", nargs='+', default=None, help="Filter by tracks. Default: All.")
    parser.add_argument("--subsets", nargs='+', default=None, help="Filter by subsets. Default: All.")

    args = parser.parse_args()
    console = Console()
    console.print("[bold underline]🔬 Active Learning Analysis Toolkit (v5)[/bold underline]")
    console.print(f"[bold]Active Filters:[/bold]")
    console.print(f"  - [cyan]Datasets[/cyan]:   {args.datasets or 'All'}")
    console.print(f"  - [cyan]Models[/cyan]:     {args.models or 'All'}")
    console.print(f"  - [cyan]Strategies[/cyan]: {args.strategies or 'All'}")
    console.print(f"  - [cyan]Tracks[/cyan]:     {args.tracks or 'All'}")
    console.print(f"  - [cyan]Subsets[/cyan]:    {args.subsets or 'All'}")
    try:
        df = load_and_filter_data(args.results_dir, args.datasets, args.models, args.strategies, args.tracks, args.subsets)
        summary_df = generate_summary_table(df)
        generate_latex_table(summary_df, args.output_dir)
        generate_plots(df, summary_df, args.output_dir)
        console.print("\n[bold green]🎉 Analysis complete![/bold green]")
    except (ValueError, FileNotFoundError) as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    main()