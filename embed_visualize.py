import numpy as np
import matplotlib.pyplot as plt
import glob, os, argparse

"""Visualize embedding_ep*.npy produced during training.
Plots 2D scatter colored by each preference dimension and saves under the run directory.
Usage:
python embed_visualize.py --run_dir logs/run_20250819_120000
If --run_dir omitted, uses latest.
"""

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', type=str, default=None)
    return p.parse_args()

def latest_run():
    logs = [d for d in os.listdir('logs') if d.startswith('run_')]
    if not logs: raise FileNotFoundError('No run_ directories in logs')
    logs.sort(); return os.path.join('logs', logs[-1])


def main():
    args = parse()
    run_dir = args.run_dir or latest_run()
    emb_files = sorted(glob.glob(os.path.join(run_dir, 'embedding_ep*.npy')))
    if not emb_files:
        print('No embeddings found in', run_dir)
        return
    # Assume parallel saved prefs file names
    for ef in emb_files:
        base = os.path.splitext(os.path.basename(ef))[0]
        ep = base.replace('embedding_ep','')
        pref_file = os.path.join(run_dir, f'embedding_prefs_ep{ep}.npy')
        if not os.path.exists(pref_file):
            print('Missing prefs for', ef); continue
        Z = np.load(ef)
        W = np.load(pref_file)
        out_dir = os.path.join(run_dir, 'embeddings')
        os.makedirs(out_dir, exist_ok=True)
        for k in range(W.shape[1]):
            plt.figure(figsize=(6,5))
            sc = plt.scatter(Z[:,0], Z[:,1], c=W[:,k], cmap='viridis', s=20)
            plt.colorbar(sc, label=f'w[{k}]')
            plt.title(f'Embedding ep{ep} colored by w[{k}]')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'embedding_ep{ep}_w{k}.png'))
            plt.close()
        # combined RGB if 3-dim
        if W.shape[1] == 3:
            plt.figure(figsize=(6,5))
            # Normalize W to [0,1]
            Wn = W / W.max(axis=0, keepdims=True).clip(1e-8)
            plt.scatter(Z[:,0], Z[:,1], c=Wn, s=20)
            plt.title(f'Embedding ep{ep} RGB (prefs)')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'embedding_ep{ep}_rgb.png'))
            plt.close()
        print('Saved embeddings for episode', ep)

if __name__ == '__main__':
    main()
