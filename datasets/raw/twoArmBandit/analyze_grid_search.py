#!/usr/bin/env python3
"""
Analyze Grid Search Results

This script analyzes completed and partial grid search runs,
providing a comprehensive report of results and best configurations.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def load_grid_search_summary(grid_dir):
    """Load the grid search summary JSON file"""
    summary_path = Path(grid_dir) / 'grid_search_summary.json'
    
    if not summary_path.exists():
        raise FileNotFoundError(f"Grid search summary not found at {summary_path}")
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    return summary


def extract_run_dataframe(summary):
    """Convert runs to a pandas DataFrame for easy analysis"""
    runs = summary.get('runs', [])
    
    if not runs:
        return pd.DataFrame()
    
    # Extract data for each run
    data = []
    for run in runs:
        row = {
            'run_id': run['run_id'],
            'status': run['status'],
            'execution_time_seconds': run.get('execution_time_seconds', None),
            'completed_at': run.get('completed_at', None),
        }
        
        # Add parameters
        params = run.get('parameters', {})
        for key, value in params.items():
            if key == 'data_path':
                # Extract just the dataset name
                row[key] = Path(value).name
            else:
                row[key] = value
        
        # Add metrics if available
        metrics = run.get('metrics', {})
        for key, value in metrics.items():
            row[key] = value
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def print_summary_header(summary, grid_dir):
    """Print basic summary information"""
    print("\n" + "="*80)
    print("GRID SEARCH ANALYSIS REPORT")
    print("="*80)
    print(f"Grid Directory: {grid_dir}")
    print(f"Started: {summary.get('start_timestamp', 'Unknown')}")
    print(f"Last Updated: {summary.get('last_updated', 'Unknown')}")
    print(f"\nTotal Runs: {summary.get('total_runs', 0)}")
    print(f"Completed: {summary.get('completed_runs', 0)}")
    print(f"Failed: {summary.get('failed_runs', 0)}")
    
    total = summary.get('total_runs', 0)
    completed = summary.get('completed_runs', 0)
    if total > 0:
        progress = (completed / total) * 100
        print(f"Progress: {progress:.1f}%")
        remaining = total - completed - summary.get('failed_runs', 0)
        if remaining > 0:
            print(f"Remaining: {remaining}")
    print("="*80 + "\n")


def print_parameter_space(summary):
    """Print the parameter grid being searched"""
    grid_params = summary.get('grid_parameters', {})
    
    if not grid_params:
        return
    
    print("\nPARAMETER SEARCH SPACE:")
    print("-" * 80)
    for param, values in grid_params.items():
        if param == 'data_path':
            # Show just dataset names
            dataset_names = [Path(p).name for p in values]
            print(f"  {param}: {dataset_names}")
        else:
            print(f"  {param}: {values}")
    print("-" * 80 + "\n")


def print_best_runs(summary, df):
    """Print information about the best runs"""
    print("\nBEST RUNS:")
    print("-" * 80)
    
    # Best Markovian Loss
    if 'best_markovian_run' in summary:
        best_markov = summary['best_markovian_run']
        print(f"\nBest Markovian Loss:")
        print(f"  Run ID: {best_markov['run_id']}")
        print(f"  Loss: {best_markov['loss']:.6f}")
        print(f"  Epoch: {best_markov.get('epoch', 'N/A')}")
        print(f"  Parameters:")
        for key, value in best_markov['parameters'].items():
            if key == 'data_path':
                value = Path(value).name
            print(f"    {key}: {value}")
        print(f"  Output: {best_markov.get('output_dir', 'N/A')}")
    
    # Best Behavior Loss
    if 'best_behavior_run' in summary:
        best_behavior = summary['best_behavior_run']
        print(f"\nBest Behavior Loss:")
        print(f"  Run ID: {best_behavior['run_id']}")
        print(f"  Loss: {best_behavior['loss']:.6f}")
        print(f"  Epoch: {best_behavior.get('epoch', 'N/A')}")
        print(f"  Parameters:")
        for key, value in best_behavior['parameters'].items():
            if key == 'data_path':
                value = Path(value).name
            print(f"    {key}: {value}")
        print(f"  Output: {best_behavior.get('output_dir', 'N/A')}")
    
    # If no best runs in summary, compute from DataFrame
    if 'best_markovian_run' not in summary and 'best_markovian_loss' in df.columns:
        completed_df = df[df['status'] == 'completed'].copy()
        
        if len(completed_df) > 0:
            # Best Markovian
            if 'best_markovian_loss' in completed_df.columns:
                best_markov_idx = completed_df['best_markovian_loss'].idxmin()
                best_markov_row = completed_df.loc[best_markov_idx]
                print(f"\nBest Markovian Loss (computed):")
                print(f"  Run ID: {best_markov_row['run_id']}")
                print(f"  Loss: {best_markov_row['best_markovian_loss']:.6f}")
                print(f"  Epoch: {best_markov_row.get('best_markovian_epoch', 'N/A')}")
            
            # Best Behavior
            if 'best_behavior_loss' in completed_df.columns:
                best_behavior_idx = completed_df['best_behavior_loss'].idxmin()
                best_behavior_row = completed_df.loc[best_behavior_idx]
                print(f"\nBest Behavior Loss (computed):")
                print(f"  Run ID: {best_behavior_row['run_id']}")
                print(f"  Loss: {best_behavior_row['best_behavior_loss']:.6f}")
                print(f"  Epoch: {best_behavior_row.get('best_behavior_epoch', 'N/A')}")
    
    print("-" * 80 + "\n")


def print_failed_runs(df):
    """Print information about failed runs"""
    failed_df = df[df['status'] == 'failed']
    
    if len(failed_df) == 0:
        return
    
    print("\nFAILED RUNS:")
    print("-" * 80)
    for idx, row in failed_df.iterrows():
        print(f"\nRun ID: {row['run_id']}")
        # Print parameter values
        param_cols = [col for col in df.columns if col not in ['run_id', 'status', 
                      'execution_time_seconds', 'completed_at', 'best_markovian_loss',
                      'best_markovian_epoch', 'best_behavior_loss', 'best_behavior_epoch']]
        for col in param_cols:
            if col in row:
                print(f"  {col}: {row[col]}")
    print("-" * 80 + "\n")


def print_statistics(df):
    """Print statistical summary of completed runs"""
    completed_df = df[df['status'] == 'completed'].copy()
    
    if len(completed_df) == 0:
        print("\nNo completed runs to analyze.\n")
        return
    
    print("\nSTATISTICS (Completed Runs):")
    print("-" * 80)
    
    # Execution time statistics
    if 'execution_time_seconds' in completed_df.columns:
        exec_times = completed_df['execution_time_seconds'].dropna()
        if len(exec_times) > 0:
            print(f"\nExecution Time:")
            print(f"  Mean: {exec_times.mean():.1f}s ({exec_times.mean()/60:.1f}m)")
            print(f"  Median: {exec_times.median():.1f}s ({exec_times.median()/60:.1f}m)")
            print(f"  Min: {exec_times.min():.1f}s ({exec_times.min()/60:.1f}m)")
            print(f"  Max: {exec_times.max():.1f}s ({exec_times.max()/60:.1f}m)")
    
    # Loss statistics
    if 'best_markovian_loss' in completed_df.columns:
        markov_losses = completed_df['best_markovian_loss'].dropna()
        if len(markov_losses) > 0:
            print(f"\nMarkovian Loss:")
            print(f"  Mean: {markov_losses.mean():.6f}")
            print(f"  Median: {markov_losses.median():.6f}")
            print(f"  Min: {markov_losses.min():.6f}")
            print(f"  Max: {markov_losses.max():.6f}")
            print(f"  Std: {markov_losses.std():.6f}")
    
    if 'best_behavior_loss' in completed_df.columns:
        behavior_losses = completed_df['best_behavior_loss'].dropna()
        if len(behavior_losses) > 0:
            print(f"\nBehavior Loss:")
            print(f"  Mean: {behavior_losses.mean():.6f}")
            print(f"  Median: {behavior_losses.median():.6f}")
            print(f"  Min: {behavior_losses.min():.6f}")
            print(f"  Max: {behavior_losses.max():.6f}")
            print(f"  Std: {behavior_losses.std():.6f}")
    
    print("-" * 80 + "\n")


def print_detailed_results_table(df):
    """Print a detailed table of all runs"""
    if len(df) == 0:
        return
    
    print("\nDETAILED RESULTS TABLE:")
    print("-" * 80)
    
    # Select key columns for display
    display_cols = ['run_id', 'status']
    
    # Add parameter columns
    param_cols = [col for col in df.columns if col not in ['run_id', 'status', 
                  'execution_time_seconds', 'completed_at', 'best_markovian_loss',
                  'best_markovian_epoch', 'best_behavior_loss', 'best_behavior_epoch']]
    display_cols.extend(param_cols)
    
    # Add metric columns
    if 'best_markovian_loss' in df.columns:
        display_cols.append('best_markovian_loss')
    if 'best_behavior_loss' in df.columns:
        display_cols.append('best_behavior_loss')
    
    # Filter to available columns
    display_cols = [col for col in display_cols if col in df.columns]
    
    # Print table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)
    
    print(df[display_cols].to_string(index=False))
    print("-" * 80 + "\n")


def create_visualizations(df, output_dir):
    """Create visualization plots for the grid search results"""
    completed_df = df[df['status'] == 'completed'].copy()
    
    if len(completed_df) == 0:
        print("No completed runs to visualize.\n")
        return
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / 'analysis_visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    print(f"\nCreating visualizations in {viz_dir}...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Loss comparison plot
    if 'best_markovian_loss' in completed_df.columns and 'best_behavior_loss' in completed_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Markovian loss
        axes[0].bar(completed_df['run_id'], completed_df['best_markovian_loss'])
        axes[0].set_xlabel('Run ID')
        axes[0].set_ylabel('Best Markovian Loss')
        axes[0].set_title('Markovian Loss Across Runs')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Behavior loss
        axes[1].bar(completed_df['run_id'], completed_df['best_behavior_loss'])
        axes[1].set_xlabel('Run ID')
        axes[1].set_ylabel('Best Behavior Loss')
        axes[1].set_title('Behavior Loss Across Runs')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'loss_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Loss comparison plot saved")
    
    # 2. Execution time distribution
    if 'execution_time_seconds' in completed_df.columns:
        exec_times_minutes = completed_df['execution_time_seconds'] / 60
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(exec_times_minutes, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Execution Time (minutes)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Execution Times')
        ax.axvline(exec_times_minutes.mean(), color='red', linestyle='--', 
                   label=f'Mean: {exec_times_minutes.mean():.1f}m')
        ax.axvline(exec_times_minutes.median(), color='green', linestyle='--', 
                   label=f'Median: {exec_times_minutes.median():.1f}m')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'execution_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Execution time distribution plot saved")
    
    # 3. Parameter impact on losses (if we have enough variety)
    param_cols = [col for col in completed_df.columns if col not in 
                  ['run_id', 'status', 'execution_time_seconds', 'completed_at',
                   'best_markovian_loss', 'best_markovian_epoch', 
                   'best_behavior_loss', 'best_behavior_epoch', 'data_path']]
    
    for param in param_cols:
        if completed_df[param].nunique() > 1:  # Only plot if there's variation
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Group by parameter and compute mean losses
                grouped = completed_df.groupby(param).agg({
                    'best_markovian_loss': 'mean',
                    'best_behavior_loss': 'mean'
                }).reset_index()
                
                # Markovian loss
                axes[0].bar(range(len(grouped)), grouped['best_markovian_loss'])
                axes[0].set_xlabel(param)
                axes[0].set_ylabel('Mean Markovian Loss')
                axes[0].set_title(f'Markovian Loss vs {param}')
                axes[0].set_xticks(range(len(grouped)))
                axes[0].set_xticklabels(grouped[param], rotation=45, ha='right')
                
                # Behavior loss
                axes[1].bar(range(len(grouped)), grouped['best_behavior_loss'])
                axes[1].set_xlabel(param)
                axes[1].set_ylabel('Mean Behavior Loss')
                axes[1].set_title(f'Behavior Loss vs {param}')
                axes[1].set_xticks(range(len(grouped)))
                axes[1].set_xticklabels(grouped[param], rotation=45, ha='right')
                
                plt.tight_layout()
                plt.savefig(viz_dir / f'loss_vs_{param}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Loss vs {param} plot saved")
            except Exception as e:
                print(f"  ✗ Could not create plot for {param}: {e}")
    
    print(f"\nVisualizations saved to: {viz_dir}\n")


def export_results_csv(df, output_dir):
    """Export results to CSV file"""
    output_dir = Path(output_dir)
    csv_path = output_dir / 'grid_search_results.csv'
    
    df.to_csv(csv_path, index=False)
    print(f"Results exported to: {csv_path}\n")


def generate_html_report(summary, df, output_dir):
    """Generate an HTML report with all information"""
    output_dir = Path(output_dir)
    html_path = output_dir / 'grid_search_report.html'
    
    completed_df = df[df['status'] == 'completed']
    failed_df = df[df['status'] == 'failed']
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Grid Search Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #555; border-bottom: 2px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
            h3 {{ color: #666; }}
            .summary {{ background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px 20px; }}
            .metric-label {{ font-weight: bold; color: #555; }}
            .metric-value {{ font-size: 1.2em; color: #4CAF50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .best-run {{ background-color: #fff9c4; padding: 15px; margin: 10px 0; border-left: 4px solid #FFC107; }}
            .failed-run {{ background-color: #ffebee; padding: 10px; margin: 5px 0; border-left: 4px solid #f44336; }}
            .progress-bar {{ width: 100%; background-color: #ddd; border-radius: 5px; }}
            .progress-fill {{ height: 30px; background-color: #4CAF50; border-radius: 5px; text-align: center; line-height: 30px; color: white; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Grid Search Analysis Report</h1>
            <p><strong>Grid Directory:</strong> {output_dir}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <div class="metric">
                    <span class="metric-label">Total Runs:</span>
                    <span class="metric-value">{summary.get('total_runs', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Completed:</span>
                    <span class="metric-value">{summary.get('completed_runs', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Failed:</span>
                    <span class="metric-value" style="color: #f44336;">{summary.get('failed_runs', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Progress:</span>
                    <span class="metric-value">{(summary.get('completed_runs', 0) / max(summary.get('total_runs', 1), 1) * 100):.1f}%</span>
                </div>
                
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {(summary.get('completed_runs', 0) / max(summary.get('total_runs', 1), 1) * 100):.1f}%">
                        {summary.get('completed_runs', 0)} / {summary.get('total_runs', 0)}
                    </div>
                </div>
            </div>
    """
    
    # Best runs section
    if 'best_markovian_run' in summary or 'best_behavior_run' in summary:
        html_content += "<h2>Best Runs</h2>"
        
        if 'best_markovian_run' in summary:
            best_markov = summary['best_markovian_run']
            html_content += f"""
            <div class="best-run">
                <h3>Best Markovian Loss</h3>
                <p><strong>Run ID:</strong> {best_markov['run_id']}</p>
                <p><strong>Loss:</strong> {best_markov['loss']:.6f} (Epoch {best_markov.get('epoch', 'N/A')})</p>
                <p><strong>Parameters:</strong></p>
                <ul>
            """
            for key, value in best_markov['parameters'].items():
                if key == 'data_path':
                    value = Path(value).name
                html_content += f"<li><strong>{key}:</strong> {value}</li>"
            html_content += "</ul></div>"
        
        if 'best_behavior_run' in summary:
            best_behavior = summary['best_behavior_run']
            html_content += f"""
            <div class="best-run">
                <h3>Best Behavior Loss</h3>
                <p><strong>Run ID:</strong> {best_behavior['run_id']}</p>
                <p><strong>Loss:</strong> {best_behavior['loss']:.6f} (Epoch {best_behavior.get('epoch', 'N/A')})</p>
                <p><strong>Parameters:</strong></p>
                <ul>
            """
            for key, value in best_behavior['parameters'].items():
                if key == 'data_path':
                    value = Path(value).name
                html_content += f"<li><strong>{key}:</strong> {value}</li>"
            html_content += "</ul></div>"
    
    # Statistics section
    if len(completed_df) > 0:
        html_content += "<h2>Statistics (Completed Runs)</h2><table>"
        
        if 'execution_time_seconds' in completed_df.columns:
            exec_times = completed_df['execution_time_seconds'].dropna()
            if len(exec_times) > 0:
                html_content += f"""
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Mean Execution Time</td><td>{exec_times.mean():.1f}s ({exec_times.mean()/60:.1f}m)</td></tr>
                <tr><td>Median Execution Time</td><td>{exec_times.median():.1f}s ({exec_times.median()/60:.1f}m)</td></tr>
                """
        
        if 'best_markovian_loss' in completed_df.columns:
            markov_losses = completed_df['best_markovian_loss'].dropna()
            if len(markov_losses) > 0:
                html_content += f"""
                <tr><td>Mean Markovian Loss</td><td>{markov_losses.mean():.6f}</td></tr>
                <tr><td>Min Markovian Loss</td><td>{markov_losses.min():.6f}</td></tr>
                """
        
        if 'best_behavior_loss' in completed_df.columns:
            behavior_losses = completed_df['best_behavior_loss'].dropna()
            if len(behavior_losses) > 0:
                html_content += f"""
                <tr><td>Mean Behavior Loss</td><td>{behavior_losses.mean():.6f}</td></tr>
                <tr><td>Min Behavior Loss</td><td>{behavior_losses.min():.6f}</td></tr>
                """
        
        html_content += "</table>"
    
    # All runs table
    if len(df) > 0:
        html_content += "<h2>All Runs</h2><table><tr>"
        
        # Headers
        display_cols = ['run_id', 'status']
        if 'best_markovian_loss' in df.columns:
            display_cols.append('best_markovian_loss')
        if 'best_behavior_loss' in df.columns:
            display_cols.append('best_behavior_loss')
        
        for col in display_cols:
            html_content += f"<th>{col}</th>"
        html_content += "</tr>"
        
        # Rows
        for idx, row in df.iterrows():
            html_content += "<tr>"
            for col in display_cols:
                value = row.get(col, 'N/A')
                if isinstance(value, float):
                    value = f"{value:.6f}"
                html_content += f"<td>{value}</td>"
            html_content += "</tr>"
        
        html_content += "</table>"
    
    # Failed runs section
    if len(failed_df) > 0:
        html_content += "<h2>Failed Runs</h2>"
        for idx, row in failed_df.iterrows():
            html_content += f"""
            <div class="failed-run">
                <strong>Run ID:</strong> {row['run_id']}
            </div>
            """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {html_path}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze grid search results')
    parser.add_argument('grid_dir', type=str, 
                        help='Path to grid search directory')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--no-html', action='store_true',
                        help='Skip HTML report generation')
    parser.add_argument('--csv', action='store_true',
                        help='Export results to CSV')
    
    args = parser.parse_args()
    
    # Load summary
    try:
        summary = load_grid_search_summary(args.grid_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Extract DataFrame
    df = extract_run_dataframe(summary)
    
    # Print reports
    print_summary_header(summary, args.grid_dir)
    print_parameter_space(summary)
    print_best_runs(summary, df)
    print_statistics(df)
    print_failed_runs(df)
    print_detailed_results_table(df)
    
    # Generate visualizations
    if not args.no_viz:
        create_visualizations(df, args.grid_dir)
    
    # Generate HTML report
    if not args.no_html:
        generate_html_report(summary, df, args.grid_dir)
    
    # Export CSV
    if args.csv:
        export_results_csv(df, args.grid_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
