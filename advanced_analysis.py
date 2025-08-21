import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.spatial.distance import cdist

# --- Helper Functions ---

def find_latest_run_dir(log_dir='logs'):
    """Find the latest run directory based on modification time."""
    run_dirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if d.startswith('run_') and os.path.isdir(os.path.join(log_dir, d))]
    if not run_dirs:
        return None
    latest_dir = max(run_dirs, key=os.path.getmtime)
    return latest_dir

def plot_pareto_front(run_dir, save_dir):
    """Plots 3D and 2D projections of the final Pareto front."""
    print("Plotting final Pareto front...")
    front_file = os.path.join(run_dir, 'final_front.npy')
    
    if not os.path.exists(front_file):
        print(f"Could not find 'final_front.npy' in {run_dir}. Skipping Pareto plot.")
        return

    front = np.load(front_file)

    if front.shape[1] < 2:
        print(f"Front has {front.shape[1]} dimensions, cannot plot. Skipping.")
        return
    
    if front.shape[1] == 3:
        # 3D Scatter Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(front[:, 0], front[:, 1], front[:, 2], c='blue', marker='o', s=50, alpha=0.7)
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
        ax.set_title('Final Non-Dominated Front (3D View)')
        plt.savefig(os.path.join(save_dir, 'final_front_3d.png'))
        plt.close()

    # 2D Projections
    num_objs = front.shape[1]
    fig, axes = plt.subplots(1, num_objs, figsize=(6 * num_objs, 5))
    if num_objs == 2: axes = [axes] # make it iterable
    
    pairs = [(i, j) for i in range(num_objs) for j in range(i + 1, num_objs)]
    if len(pairs) > len(axes): # If more pairs than subplots, just use the first few
        pairs = pairs[:len(axes)]

    for i, (ax, pair) in enumerate(zip(axes, pairs)):
        ax.scatter(front[:, pair[0]], front[:, pair[1]], c='blue', marker='o', alpha=0.7)
        ax.set_xlabel(f'Objective {pair[0]+1}')
        ax.set_ylabel(f'Objective {pair[1]+1}')
        ax.set_title(f'Obj {pair[0]+1} vs Obj {pair[1]+1}')
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_front_2d_projections.png'))
    plt.close()
    print("...done.")


def plot_kl_heatmap(run_dir, save_dir):
    """Plots a heatmap of pairwise KL divergences."""
    # This function requires KL data to be saved from the run, which is not implemented yet.
    # Placeholder for future implementation.
    print("KL heatmap plotting is a future feature.")
    pass

def plot_simplex(run_dir, save_dir):
    """Plots sampled preferences on a 2-simplex (for 3 objectives)."""
    # This requires preference data to be saved, which is not implemented yet.
    # Placeholder for future implementation.
    print("Preference simplex plotting is a future feature.")
    pass

def analyze_aggregate_summary(log_dir, save_dir):
    """Reads aggregate_summary.csv and generates comparative plots."""
    print("Analyzing aggregate summary...")
    summary_file = os.path.join(log_dir, 'aggregate_summary.csv')
    if not os.path.exists(summary_file):
        print(f"Could not find 'aggregate_summary.csv' in {log_dir}. Skipping.")
        return

    df = pd.read_csv(summary_file)
    
    # Map boolean 'use_grpo' to a more readable string
    if 'use_grpo' in df.columns:
        df['variant'] = df['use_grpo'].apply(lambda x: 'GRPO' if x else 'PPO')
    else:
        print("WARN: 'use_grpo' column not found. Cannot create comparative plots.")
        return

    # Generate comparative bar plots for key final metrics
    metrics_to_plot = [
        'final_hv_norm', 'final_igd_plus_norm',
        'final_front_size', 'final_kl_median'
    ]
    
    for metric in metrics_to_plot:
        if metric in df.columns:
            plt.figure(figsize=(8, 6))
            sns.barplot(x='variant', y=metric, data=df, capsize=.1)
            plt.title(f'Comparison of Final {metric} (GRPO vs PPO)')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'compare_{metric}_variants.png'))
            plt.close()
    print("...done with comparative plots.")


def analyze_summary(log_dir, save_dir):
    """Reads summary_table.csv and generates aggregate plots and stats."""
    print("Analyzing summary table...")
    summary_file = os.path.join(log_dir, 'summary_table.csv')
    if not os.path.exists(summary_file):
        print(f"Could not find 'summary_table.csv' in {log_dir}. Skipping summary analysis.")
        return

    df = pd.read_csv(summary_file)
    
    # Calculate mean and std dev for key metrics
    stats = df.describe().loc[['mean', 'std']]
    
    # Save stats to a new CSV
    stats_file = os.path.join(save_dir, 'aggregate_summary_stats.csv')
    stats.to_csv(stats_file)
    print(f"Aggregate stats saved to {stats_file}")

    # Generate bar plots for key final metrics
    metrics_to_plot = [
        'final_hv_norm', 'final_igd', 'final_igd_plus', 'final_igd_plus_norm',
        'final_front_size', 'final_kl_median'
    ]
    
    for metric in metrics_to_plot:
        if metric in df.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='seed', y=metric, data=df)
            plt.title(f'Final {metric} per Seed')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'summary_{metric}_by_seed.png'))
            plt.close()
    print("...done.")


def main():
    parser = argparse.ArgumentParser(description="Perform advanced analysis on MORL training logs.")
    parser.add_argument('--run_dir', type=str, help="Path to a specific run directory. If not provided, analyzes the latest run.")
    parser.add_argument('--log_dir', type=str, default='logs', help="Path to the main logging directory.")
    parser.add_argument('--all', action='store_true', help="Run analysis on all run_* directories in the log_dir.")
    
    args = parser.parse_args()

    if args.all:
        run_dirs = [os.path.join(args.log_dir, d) for d in os.listdir(args.log_dir) if d.startswith('run_') and os.path.isdir(os.path.join(args.log_dir, d))]
        print(f"Found {len(run_dirs)} run directories to analyze.")
    elif args.run_dir:
        run_dirs = [args.run_dir]
    else:
        latest_run = find_latest_run_dir(args.log_dir)
        if latest_run:
            run_dirs = [latest_run]
        else:
            run_dirs = []

    if not run_dirs:
        print("No run directories found to analyze.")
        return

    # --- Run analysis on individual directories first ---
    for run_dir in run_dirs:
        print(f"\n--- Analyzing Run: {os.path.basename(run_dir)} ---")
        # Save specific plots into the run's own directory
        specific_save_dir = os.path.join(run_dir, 'analysis')
        os.makedirs(specific_save_dir, exist_ok=True)
        
        plot_pareto_front(run_dir, specific_save_dir)
        # plot_kl_heatmap(run_dir, specific_save_dir) # Future feature
        # plot_simplex(run_dir, specific_save_dir)   # Future feature

    # --- Run aggregate analysis at the top level ---
    general_save_dir = os.path.join(args.log_dir, 'analysis_results')
    os.makedirs(general_save_dir, exist_ok=True)
    
    # This function is now for comparing variants from aggregate_summary.csv
    analyze_aggregate_summary(args.log_dir, general_save_dir)

    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()
