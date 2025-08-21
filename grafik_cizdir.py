import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

def find_latest_run_dir(log_dir='logs'):
    """Find the latest run directory based on modification time."""
    run_dirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if d.startswith('run_') and os.path.isdir(os.path.join(log_dir, d))]
    if not run_dirs:
        return None
    latest_dir = max(run_dirs, key=os.path.getmtime)
    return latest_dir

def main():
    parser = argparse.ArgumentParser(description="Generate plots from MORL training logs.")
    parser.add_argument('--run_dir', type=str, help="Path to the specific run directory (e.g., logs/run_...). Overrides automatic detection.")
    parser.add_argument('--kl_clip', type=float, default=5.0, help='Clip threshold for kl_median visualization (e.g., 5.0). Use <=0 to disable.')
    args = parser.parse_args()

    if args.run_dir:
        latest_run_dir = args.run_dir
    else:
        latest_run_dir = find_latest_run_dir()

    if not latest_run_dir or not os.path.isdir(latest_run_dir):
        raise FileNotFoundError(f"Log directory not found: {latest_run_dir}")

    csv_dosya_adi = os.path.join(latest_run_dir, 'metrics.csv')
    if not os.path.isfile(csv_dosya_adi):
        raise FileNotFoundError(f"metrics.csv not found in {latest_run_dir}")

    print(f"Kullanılan CSV: {csv_dosya_adi}")

    # Kayıt klasörü
    kayit_dizini = os.path.join(latest_run_dir, 'grafikler')
    os.makedirs(kayit_dizini, exist_ok=True)

    # CSV oku
    df = pd.read_csv(csv_dosya_adi)
    df.columns = df.columns.str.strip()
    print('Sütunlar:', df.columns.tolist())

    # KL clipping (median & std) for visualization only
    if 'kl_median' in df.columns and args.kl_clip > 0:
        df['kl_median_clipped'] = df['kl_median'].clip(lower=0, upper=args.kl_clip)
    if 'kl_std' in df.columns and args.kl_clip > 0:
        df['kl_std_clipped'] = df['kl_std'].clip(lower=0, upper=args.kl_clip)

    # X ekseni sütunu seç
    x_col = 'update_steps' if 'update_steps' in df.columns else df.columns[0]
    X = df[x_col]

    # Yardımcı plot fonksiyonu
    def safe_plot(y_cols, title, ylabel, fname, styles=None, log_y=False):
        existing = [c for c in y_cols if c in df.columns and df[c].notna().any()]
        if not existing:
            print(f"Skipping plot '{title}' - no data columns found.")
            return
        plt.figure(figsize=(12, 6))
        for i, c in enumerate(existing):
            style = {}
            if styles and i < len(styles):
                style = styles[i]
            plt.plot(X, df[c], label=c, **style)
        plt.xlabel(x_col)
        plt.ylabel(ylabel)
        plt.title(title)
        if log_y:
            plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join(kayit_dizini, fname))
        plt.close()

    # 1) Kayıplar
    safe_plot(['value_loss'], 'Critic Value Loss', 'Loss', 'loss_value.png')

    # 2) PPO / KL
    safe_plot(['kl_div'], 'PPO KL Divergence (Update)', 'KL', 'kl_ppo.png')
    safe_plot(['kl_median', 'kl_std'], 'Policy Diversity (Pairwise KL)', 'KL', 'kl_diversity.png')
    if 'kl_median_clipped' in df.columns:
        safe_plot(['kl_median_clipped'], f'Policy Diversity (KL Median Clipped @ {args.kl_clip})', 'KL (clipped)', 'kl_diversity_clipped.png')

    # 3) Learning Rate
    safe_plot(['lr'], 'Learning Rate', 'LR', 'lr.png')

    # 4) Policy Entropy & Explained Variance
    safe_plot(['policy_entropy'], 'Policy Entropy', 'Entropy', 'entropy.png')
    safe_plot(['explained_variance'], 'Explained Variance', 'Ratio', 'explained_variance.png')

    # 5) Hypervolume (Normalized & Exact)
    safe_plot(['test_hv_normalized', 'test_hv_exact'], 'Test Hypervolume', 'Hypervolume', 'hypervolume_test.png')

    # 6) IGD & IGD+ (Normalized)
    safe_plot(['test_igd', 'test_igd_plus', 'test_igd_plus_norm'], 'IGD Metrics', 'IGD', 'igd_metrics.png', log_y=True)

    # 7) Diversity Metrics
    safe_plot(['front_size'], 'Non-Dominated Front Size', 'Count', 'front_size.png')
    safe_plot(['action_diversity'], 'Action Diversity', 'Mean Action Distance', 'action_diversity.png')

    # 8) Train Rewards
    plt.figure(figsize=(12, 6))
    if 'train_ep_rew_mean' in df.columns:
        plt.plot(X, df['train_ep_rew_mean'], label='Mean Episode Reward')
        if 'train_ep_rew_std' in df.columns:
            plt.fill_between(X,
                             df['train_ep_rew_mean'] - df['train_ep_rew_std'],
                             df['train_ep_rew_mean'] + df['train_ep_rew_std'],
                             alpha=0.2, label='Std Dev')
    plt.xlabel(x_col)
    plt.ylabel('Scalarized Reward')
    plt.title('Training Episode Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(kayit_dizini, 'train_reward.png'))
    plt.close()

    # (Optional) Advantage dispersion if present
    if 'adv_std' in df.columns or 'adv_group_std' in df.columns:
        cols = [c for c in ['adv_std','adv_group_std'] if c in df.columns]
        safe_plot(cols, 'Advantage Std (Global / Group)', 'Std', 'adv_std.png')

    print(f"Tüm grafikler '{kayit_dizini}' dizinine kaydedildi.")

if __name__ == '__main__':
    main()