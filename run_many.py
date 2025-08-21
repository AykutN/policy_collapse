import subprocess, os, argparse, json, datetime, itertools

"""Run multiple seeds / variants and aggregate summary tables.
Usage:
python run_many.py --seeds 0 1 2 --episodes 2000 --exact_hv --manifold umap --update_timestep 2000
Generates:
- aggregate_summary.csv (per seed rows)
- aggregate_summary_stats.csv (mean/std across seeds)
"""

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--episodes', type=int, default=5000)
    p.add_argument('--update_timestep', type=int, default=4000)
    p.add_argument('--exact_hv', action='store_true')
    p.add_argument('--manifold', default='none', choices=['none','umap','tsne'])
    p.add_argument('--embed_interval', type=int, default=200)
    p.add_argument('--embed_batch', type=int, default=256)
    p.add_argument('--entropy_coef', type=float, default=0.01)
    p.add_argument('--summary_only', action='store_true', help='Skip training if summaries already exist')
    p.add_argument('--variants', type=str, nargs='+', default=['grpo'], choices=['grpo', 'no-grpo'], help='Specify variants to run')
    return p.parse_args()


def run():
    args = parse()
    summary_rows = []
    # Loop over variants (e.g., 'grpo', 'no-grpo') and seeds
    for variant, seed in itertools.product(args.variants, args.seeds):
        cmd = [
            'python', 'training_morl_example.py',
            '--max_episodes', str(args.episodes),
            '--update_timestep', str(args.update_timestep),
            '--seed', str(seed),
            '--entropy_coef', str(args.entropy_coef),
            '--embed_interval', str(args.embed_interval),
            '--embed_batch', str(args.embed_batch),
            '--manifold', args.manifold,
            '--summary_table'
        ]
        if args.exact_hv: cmd.append('--exact_hv')
        if variant == 'no-grpo':
            cmd.append('--no-grpo')

        print('Running:', ' '.join(cmd))
        
        # --- Find most recent run directory to get its summary ---
        # This is a bit fragile, but works for sequential runs.
        # A better way would be to get the run_id from the training script stdout.
        existing_runs = set(os.listdir('logs'))
        
        subprocess.run(cmd, check=True)
        
        new_runs = set(os.listdir('logs')) - existing_runs
        if not new_runs:
            print(f"[WARN] No new run directory found for variant={variant} seed={seed}. Searching for latest.")
            # Fallback to just getting the latest directory if something went wrong
            all_run_dirs = sorted([os.path.join('logs', d) for d in os.listdir('logs') if d.startswith('run_') and os.path.isdir(os.path.join('logs', d))], key=os.path.getmtime)
            if not all_run_dirs:
                print(f"[ERROR] No run directories found at all. Skipping summary for this run.")
                continue
            latest_run_dir = all_run_dirs[-1]
        else:
            latest_run_dir = os.path.join('logs', new_runs.pop())

        summary_path = os.path.join(latest_run_dir, 'summary.csv')
        if os.path.exists(summary_path):
            import pandas as pd
            try:
                df = pd.read_csv(summary_path)
                if not df.empty:
                    row = df.iloc[0].to_dict()
                    summary_rows.append(row)
                else:
                    print(f"[WARN] Summary file is empty: {summary_path}")
            except pd.errors.EmptyDataError:
                print(f"[WARN] Could not read empty summary file: {summary_path}")
        else:
            print(f"[WARN] Summary file not found: {summary_path}")

    if not summary_rows:
        print("[ERROR] No summary data was collected. Cannot create aggregate files.")
        return

    # --- Aggregation ---
    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    
    # Save raw aggregated data
    agg_path = os.path.join('logs', 'aggregate_summary.csv')
    summary_df.to_csv(agg_path, index=False)
    print(f"Saved aggregate summary to {agg_path}")

    # Calculate and save stats (mean/std) grouped by variant
    stats_df = summary_df.groupby('use_grpo').agg(['mean', 'std']).reset_index()
    
    # Flatten multi-index columns
    stats_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) and col[1] else col[0] for col in stats_df.columns.values]
    
    stats_path = os.path.join('logs', 'aggregate_summary_stats.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved aggregate stats to {stats_path}")


if __name__ == '__main__':
    run()
