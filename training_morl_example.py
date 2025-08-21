import time
import uuid
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import torch.nn.functional as F
import csv
import os
import datetime
import random
import json
import argparse
from time import perf_counter
try:
    from pymoo.indicators.hv import HV as ExactHV
except Exception:
    ExactHV = None
from functools import lru_cache
from morl_env_halfcheetah import make_hc_morl

# Helper for immediate flushing (avoid missing UPDATE logs when stdout buffered)
def dprint(enabled, *msg):
    if enabled:
        print(*msg, flush=True)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.prefs = []
        self.logprobs = []
        self.rewards = []          # scalarized
        self.raw_rewards = []       # vector step rewards
        self.is_terminals = []
    
    def clear_memory(self):
        self.actions.clear()
        self.states.clear()
        self.prefs.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.raw_rewards.clear()
        self.is_terminals.clear()

# ---------- FiLM Katmanı ----------
class FiLMLayer(nn.Module):
    def __init__(self, input_dim, cond_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.ReLU(), nn.Linear(128, 2*input_dim)
        )
        self.input_dim = input_dim
    def forward(self, x, cond):
        gb = self.fc(cond)
        g, b = gb[:, :self.input_dim], gb[:, self.input_dim:]
        return g * x + b

# ---------- Actor ----------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, pref_dim, hidden_dim):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh())
        self.film = FiLMLayer(hidden_dim, pref_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim))
        self.action_var = nn.Parameter(torch.full((action_dim,), 0.5))
    def forward(self, s, w):
        h = self.base(s)
        h = self.film(h, w)
        return self.head(h)
    def evaluate(self, s, a, w):
        mean = self.forward(s, w)
        var = self.action_var.expand_as(mean)
        cov = torch.diag_embed(var)
        dist = MultivariateNormal(mean, cov)
        lp = dist.log_prob(a)
        ent = dist.entropy()
        return lp, mean, ent

# ---------- Critic ----------
class Critic(nn.Module):
    def __init__(self, state_dim, pref_dim, hidden_dim):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh())
        self.film = FiLMLayer(hidden_dim, pref_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
    def forward(self, s, w):
        h = self.base(s)
        h = self.film(h, w)
        return self.head(h)

# ---------- Yardımcı: ND güncelleme + IGD ----------
def update_global_nd(global_front: np.ndarray, new_points: np.ndarray):
    if new_points.size == 0:
        return global_front
    if global_front.size == 0:
        combined = new_points
    else:
        combined = np.vstack([global_front, new_points])
    nds = NonDominatedSorting()
    idx = nds.do(combined, only_non_dominated_front=True)
    return combined[idx]

def compute_normalized_igd(global_front: np.ndarray, test_points: np.ndarray, min_ref: np.ndarray, max_ref: np.ndarray):
    if global_front.size == 0 or test_points.size == 0:
        return np.nan, min_ref, max_ref
    min_ref = np.minimum(min_ref, global_front.min(axis=0))
    max_ref = np.maximum(max_ref, global_front.max(axis=0))
    rng = max_ref - min_ref
    rng[rng <= 1e-9] = 1.0
    norm_ref = (global_front - min_ref) / rng
    norm_test = (test_points - min_ref) / rng
    igd_calc = IGD(norm_ref)
    return float(igd_calc(norm_test)), min_ref, max_ref

def compute_normalized_igd_plus(global_front: np.ndarray, test_points: np.ndarray, min_ref: np.ndarray, max_ref: np.ndarray):
    if global_front.size == 0 or test_points.size == 0:
        return np.nan, min_ref, max_ref
    min_ref = np.minimum(min_ref, global_front.min(axis=0))
    max_ref = np.maximum(max_ref, global_front.max(axis=0))
    rng = max_ref - min_ref
    rng[rng <= 1e-9] = 1.0
    norm_ref = (global_front - min_ref) / rng
    norm_test = (test_points - min_ref) / rng
    return igd_plus(norm_ref, norm_test)

def monte_carlo_hv(norm_points: np.ndarray, samples: int = 5000):
    """[0,1]^d uzayında normalize edilmiş (maksimizasyon) ND nokta seti için
    kaba Monte Carlo hypervolume (referans 0 vektörü) tahmini.
    norm_points: shape (n,d) in [0,1]."""
    if norm_points.size == 0:
        return np.nan
    d = norm_points.shape[1]
    # ND filtre
    nd = update_global_nd(np.empty((0,d),dtype=np.float32), norm_points).astype(np.float32)
    U = np.random.rand(samples, d)
    # Dominance: herhangi bir ND noktasının tüm boyutlarda >= U olması
    dominate = (nd[None, ...] >= U[:, None, :]).all(axis=2).any(axis=1)
    return float(dominate.mean())  # [0,1]

# ---------- PPO ----------
class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, pref_dim, hidden_dim, lr, betas, gamma, K_epochs, eps_clip,
                 use_grpo=True, grpo_group_mode='knn', grpo_knn_delta=0.15, ent_coef=0.01,
                 gae_lambda: float = 0.95, target_kl: float = 0.02, lr_min: float = 1e-5, lr_max: float = 3e-3):
        super().__init__()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.use_grpo = use_grpo
        self.grpo_group_mode = grpo_group_mode
        self.grpo_knn_delta = grpo_knn_delta
        self.ent_coef = ent_coef

        self.policy = Actor(state_dim, action_dim, pref_dim, hidden_dim).to(device)
        self.old_policy = Actor(state_dim, action_dim, pref_dim, hidden_dim).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.critic = Critic(state_dim, pref_dim, hidden_dim).to(device)
        self.opt = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ], betas=betas)
        self.mse = nn.MSELoss()

        self.gae_lambda = gae_lambda
        self.target_kl = target_kl
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.base_lr = lr

    def select_action(self, state, pref, memory: Memory):
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        w = torch.as_tensor(pref, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean = self.old_policy(s, w)
            var = self.old_policy.action_var.expand_as(mean)
            dist = MultivariateNormal(mean, torch.diag_embed(var))
            a = dist.sample()
            lp = dist.log_prob(a)
        memory.states.append(s)
        memory.prefs.append(w)
        memory.actions.append(a)
        memory.logprobs.append(lp)
        return a.squeeze(0).cpu().numpy()

    def update(self, memory: Memory):
        # --------- GAE Hazırlık ---------
        states = torch.cat(memory.states, dim=0)
        actions = torch.cat(memory.actions, dim=0)
        prefs = torch.cat(memory.prefs, dim=0)
        old_logp = torch.cat(memory.logprobs, dim=0).detach()
        rewards = torch.as_tensor(memory.rewards, dtype=torch.float32, device=device)
        dones = torch.as_tensor(memory.is_terminals, dtype=torch.float32, device=device)
        with torch.no_grad():
            values = self.critic(states, prefs).squeeze(-1)
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - (dones[t+1] if t < len(rewards)-1 else 0.0)
            next_value = values[t+1] if t < len(rewards)-1 else 0.0
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv_std = float(advantages.std().item())
        group_adv_stds = []

        for _ in range(self.K_epochs):
            logp, _, ent = self.policy.evaluate(states, actions, prefs)
            new_values = self.critic(states, prefs).squeeze(-1)
            ratios = torch.exp(logp - old_logp)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_core = -torch.min(surr1, surr2).mean()
            value_loss = self.mse(new_values, returns)
            entropy_loss = ent.mean()
            if self.use_grpo and prefs.size(0) > 1:
                # KNN Grupları
                if self.grpo_group_mode == 'knn':
                    with torch.no_grad():
                        nprefs = F.normalize(prefs, p=2, dim=1)
                        dmat = 1 - nprefs @ nprefs.t()
                        min_group = min(32, prefs.size(0))
                        groups = []
                        d_np = dmat.cpu().numpy()
                        for i in range(prefs.size(0)):
                            order = np.argsort(d_np[i])
                            close = [j for j in order if d_np[i][j] < self.grpo_knn_delta]
                            if len(close) < min_group:
                                close = order[:min_group].tolist()
                            g_idx = torch.as_tensor(close, dtype=torch.long, device=device)
                            groups.append(g_idx)
                            # group advantage std (normalized advantages subset)
                            group_adv_stds.append(float(advantages[g_idx].std().item()))
                else:
                    all_idx = torch.arange(prefs.size(0), device=device)
                    groups = [all_idx] * prefs.size(0)
                    group_adv_stds.append(float(advantages.std().item()))
                grad_pool = []
                for g_idx in groups:
                    r_g = torch.exp(logp[g_idx] - old_logp[g_idx])
                    s1_g = r_g * advantages[g_idx]
                    s2_g = torch.clamp(r_g, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[g_idx]
                    loss_g = -torch.min(s1_g, s2_g).mean()
                    self.opt.zero_grad(); loss_g.backward(retain_graph=True)
                    grad_vec = torch.cat([p.grad.view(-1) for p in self.policy.parameters() if p.grad is not None])
                    grad_pool.append(grad_vec)
                if grad_pool:
                    avg_grad = torch.mean(torch.stack(grad_pool), dim=0)
                    base_loss = policy_core + 0.5 * value_loss - self.ent_coef * entropy_loss
                    self.opt.zero_grad(); base_loss.backward(retain_graph=True)
                    current = torch.cat([p.grad.view(-1) for p in self.policy.parameters() if p.grad is not None])
                    proj = avg_grad * (torch.dot(current, avg_grad) / (torch.dot(avg_grad, avg_grad) + 1e-8))
                    ptr = 0
                    for p in self.policy.parameters():
                        if p.grad is not None:
                            n = p.numel(); p.grad.data = proj[ptr:ptr+n].view_as(p); ptr += n
                else:
                    loss_full = policy_core + 0.5 * value_loss - self.ent_coef * entropy_loss
                    self.opt.zero_grad(); loss_full.backward()
            else:
                loss_full = policy_core + 0.5 * value_loss - self.ent_coef * entropy_loss
                self.opt.zero_grad(); loss_full.backward()
            self.opt.step()
        self.old_policy.load_state_dict(self.policy.state_dict())
        with torch.no_grad():
            new_logp, _, _ = self.policy.evaluate(states, actions, prefs)
        kl = (old_logp - new_logp).mean().item()
        # Adaptif LR
        cur_lr = self.opt.param_groups[0]['lr']
        if kl > 1.5 * self.target_kl:
            cur_lr = max(self.lr_min, cur_lr / 1.5)
        elif kl < self.target_kl / 1.5:
            cur_lr = min(self.lr_max, cur_lr * 1.1)
        for pg in self.opt.param_groups: pg['lr'] = cur_lr
        explained_var = 1 - torch.var(returns - new_values) / (torch.var(returns) + 1e-8)
        adv_group_std = float(np.mean(group_adv_stds)) if group_adv_stds else float('nan')
        return value_loss.item(), kl, float(explained_var), cur_lr, ent.mean().item(), adv_std, adv_group_std

    def collect_action_mean(self, states, prefs):
        with torch.no_grad():
            return self.policy(states, prefs)

    def get_action_dist(self, states, prefs):
        with torch.no_grad():
            mean = self.policy(states, prefs)
            var = self.policy.action_var.expand_as(mean)
            cov = torch.diag_embed(var)
            return MultivariateNormal(mean, cov)

# ---------- Ana Döngü ----------
def exact_hv(points: np.ndarray, ref=None):
    if ExactHV is None: return np.nan
    if points.size == 0: return np.nan
    d = points.shape[1]
    if ref is None:
        ref = np.zeros(d, dtype=np.float32)
    try:
        hv = ExactHV(ref_point=ref)
        return float(hv(points))
    except Exception:
        return np.nan

def igd_plus(ref: np.ndarray, approx: np.ndarray):
    if ref.size == 0 or approx.size == 0: return np.nan
    # IGD+ : distance only if ref dominates approx in each dim
    dists = []
    for r in ref:
        diff = approx - r
        # positive part
        diff_pos = np.clip(diff, 0, None)
        dists.append(np.linalg.norm(diff_pos, axis=1).min())
    return float(np.mean(dists)) if dists else np.nan

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--log_dir', type=str, default='logs', help='Directory for logs')
    p.add_argument('--max_episodes', type=int, default=50000)
    p.add_argument('--update_timestep', type=int, default=4000)
    p.add_argument('--K_epochs', type=int, default=20)
    p.add_argument('--entropy_coef', type=float, default=0.01)
    p.add_argument('--exact_hv', action='store_true', help='Use exact HV (pymoo) when d<=3')
    p.add_argument('--embed_interval', type=int, default=50, help='Collect embeddings every N episodes')
    p.add_argument('--embed_batch', type=int, default=256, help='Number of (s,w) pairs for manifold')
    p.add_argument('--manifold', choices=['umap','tsne','none'], default='none')
    p.add_argument('--summary_table', action='store_true', help='Write summary_table.csv with final metrics')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-grpo', action='store_true', help='Disable GRPO and use standard PPO update')
    # FAST MODE & TUNABLE EVAL PARAMS
    p.add_argument('--fast', action='store_true', help='Hızlı kip: değerlendirme sıklığını ve test yükünü azaltır')
    p.add_argument('--eval_every', type=int, default=1, help='Her kaç policy update sonrasında test+HV ölç')
    p.add_argument('--igd_every', type=int, default=1, help='Her kaç update sonrasında IGD/IGD+ ölç')
    p.add_argument('--k_test', type=int, default=24, help='Test tercih sayısı')
    p.add_argument('--horizon_test', type=int, default=300, help='Test rollout maksimum adım sayısı')
    p.add_argument('--hv_mc_samples', type=int, default=4000, help='Monte Carlo HV örnek sayısı')
    p.add_argument('--hv_rng_seed', type=int, default=42, help='HV MC için RNG seed')
    p.add_argument('--debug', action='store_true', help='Detaylı debug çıktıları (episode/update log)')
    return p.parse_args()

# ------------- FAST MODE APPLY -------------
def apply_fast_mode(args):
    """Fast mode değerlerini düşürerek hızlandırır."""
    # Sadece varsayılan kaldıysa override et
    if args.eval_every == 1: args.eval_every = 5
    if args.igd_every == 1: args.igd_every = 5
    if args.k_test == 24: args.k_test = 8
    if args.horizon_test == 300: args.horizon_test = 150
    if args.hv_mc_samples == 4000: args.hv_mc_samples = 1000
    print(f"[FAST] eval_every={args.eval_every} igd_every={args.igd_every} k_test={args.k_test} horizon_test={args.horizon_test} hv_samples={args.hv_mc_samples}")
    return args

# ---------- Ana Döngü ----------
def main():
    args = parse_args()
    if args.fast:
        apply_fast_mode(args)
    dprint(args.debug, '[DEBUG] Args yüklendi.')
    env_name = "morl-halfcheetah-v4"  # logical name
    max_episodes = args.max_episodes
    update_timestep = args.update_timestep
    K_epochs = args.K_epochs
    ent_coef = args.entropy_coef
    max_timesteps = 1000
    eps_clip = 0.2
    gamma = 0.99
    lr = 3e-4
    betas = (0.9, 0.999)
    hidden_dim = 256
    use_grpo = not args.no_grpo
    grpo_group_mode = 'knn'
    grpo_knn_delta = 0.15
    random_seed = args.seed
    log_interval = 10
    target_kl = 0.02
    gae_lambda = 0.95

    if random_seed:
        torch.manual_seed(random_seed); np.random.seed(random_seed); random.seed(random_seed)

    # Use custom MORL env wrapper
    if args.debug:
        print('[DEBUG] Env oluşturuluyor...', flush=True)
    env = make_hc_morl(speed_mode="target_speed", target_speed=2.0, friction_rand=False, seed=random_seed, freeze_after_steps=10000)
    test_env = make_hc_morl(speed_mode="target_speed", target_speed=2.0, friction_rand=False, seed=random_seed+123, freeze_after_steps=10000)
    if args.debug:
        print(f'[DEBUG] Env hazır. obs_dim={env.observation_space.shape[0]} act_dim={env.action_space.shape[0]} pref_dim={env.reward_space.shape[0]}', flush=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    pref_dim = env.reward_space.shape[0]

    test_prefs = [np.random.dirichlet(np.ones(pref_dim)).astype(np.float32) for _ in range(args.k_test)]

    memory = Memory()
    ppo = PPO(state_dim, action_dim, pref_dim, hidden_dim, lr, betas, gamma, K_epochs, eps_clip,
              use_grpo, grpo_group_mode, grpo_knn_delta, ent_coef, gae_lambda=gae_lambda, target_kl=target_kl)

    # Logging
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:4]}"
    log_path = os.path.join(args.log_dir, f"run_{run_id}")
    os.makedirs(log_path, exist_ok=True)
    
    # Save metadata
    metadata = {
        "env": env_name,
        "seed": random_seed,
        "use_grpo": use_grpo,
        "args": vars(args)
    }
    with open(os.path.join(log_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # CSV Header and file setup
    csv_path = os.path.join(log_path, "metrics.csv")
    csv_header = [
        "update_steps", "total_timesteps", "time_elapsed",
        "train_ep_rew_mean", "train_ep_rew_std",
        "test_hv_exact", "test_hv_normalized", "test_igd", "test_igd_plus", "test_igd_plus_norm",
        "front_size", "action_diversity", "kl_median", "kl_std",
        "policy_entropy", "explained_variance", "kl_div", "lr", "value_loss",
        "adv_std", "adv_group_std"
    ]
    csvfile = open(csv_path, "w", newline='')
    writer = csv.DictWriter(csvfile, fieldnames=csv_header)
    writer.writeheader()


    global_front = np.empty((0, pref_dim), dtype=np.float32)
    min_ref = np.full(pref_dim, np.inf, dtype=np.float32)
    max_ref = np.full(pref_dim, -np.inf, dtype=np.float32)

    time_step = 0
    steps_since_update = 0

    # EMA stats
    ep_rewards_buffer = []

    # Embedding collection buffers
    embed_states = []
    embed_prefs = []

    update_count = 0
    last_metrics = {
        'igd': np.nan,
        'igd_plus': np.nan,
        'igd_plus_norm': np.nan,
        'hv_train_norm': np.nan,
        'hv_test_norm': np.nan,
        'hv_train_exact': np.nan,
        'hv_test_exact': np.nan
    }
    rng_hv = np.random.default_rng(args.hv_rng_seed)

    start_time = time.time()
    
    for ep in range(1, max_episodes+1):
        if args.debug and ep % 10 == 1:
            print(f'[DEBUG] Episode {ep} başlıyor... time_step={time_step} steps_since_update={steps_since_update} updates={update_count}', flush=True)
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _info_reset = reset_out
        else:
            obs = reset_out
        w = np.random.dirichlet(np.ones(pref_dim)).astype(np.float32)
        ep_vec_sum_raw = np.zeros(pref_dim, dtype=np.float32)
        ep_scalar_reward = 0.0
        
        for t in range(max_timesteps):
            time_step += 1
            steps_since_update += 1
            action = ppo.select_action(obs, w, memory)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, _, terminated, truncated, info = step_out
                done = terminated or truncated
            elif len(step_out) == 4:
                obs, _, done, info = step_out
            else:
                obs, _, done, info = step_out[0], step_out[2], step_out[-1]

            r_vec_norm = info.get('reward_vec', np.zeros(pref_dim))
            r_vec_raw = info.get('reward_vec_raw', r_vec_norm)
            
            scalar_r = float(np.dot(r_vec_norm, w))
            memory.rewards.append(scalar_r)
            memory.raw_rewards.append(r_vec_norm)
            memory.is_terminals.append(done)
            
            ep_scalar_reward += scalar_r
            ep_vec_sum_raw += r_vec_raw

            if len(embed_states) < args.embed_batch and (ep % args.embed_interval == 0):
                embed_states.append(obs if isinstance(obs, np.ndarray) else np.array(obs, dtype=np.float32))
                embed_prefs.append(w)

            need_update = steps_since_update >= update_timestep
            if need_update:
                t0_upd = perf_counter()
                value_loss, kl_div, explained_var, cur_lr, policy_entropy, adv_std, adv_group_std = ppo.update(memory)
                upd_dur = perf_counter() - t0_upd
                
                if args.debug:
                    print(f'[DEBUG] UPDATE triggered at ep={ep} time_step={time_step} batch_steps={len(memory.rewards)} duration={upd_dur:.2f}s', flush=True)
                
                memory.clear_memory()
                steps_since_update = 0
                update_count += 1
                
                do_eval = (update_count % args.eval_every == 0)
                do_igd = (update_count % args.igd_every == 0)
                
                action_diversity = np.nan
                kl_median, kl_std = np.nan, np.nan

                if do_eval:
                    with torch.inference_mode():
                        test_returns = []
                        for tw in test_prefs:
                            reset_t = test_env.reset()
                            if isinstance(reset_t, tuple) and len(reset_t)==2: tobs, _ = reset_t
                            else: tobs = reset_t
                            vec_sum = np.zeros(pref_dim, dtype=np.float32)
                            for _ in range(args.horizon_test):
                                a = ppo.old_policy(torch.as_tensor(tobs, dtype=torch.float32, device=device).unsqueeze(0),
                                                   torch.as_tensor(tw, dtype=torch.float32, device=device).unsqueeze(0))
                                a = a.squeeze(0).cpu().numpy()
                                step_out2 = test_env.step(a)
                                if len(step_out2) == 5:
                                    tobs, _, term2, trunc2, info2 = step_out2
                                    d2 = term2 or trunc2
                                else:
                                    tobs, _, d2, info2 = step_out2[0], step_out2[2], step_out2[-1]
                                
                                tr_vec_norm = info2.get('reward_vec', np.zeros(pref_dim))
                                vec_sum += tr_vec_norm
                                if d2: break
                            test_returns.append(vec_sum)
                        test_returns = np.array(test_returns, dtype=np.float32)

                        # Kaydet: her evaluation bulutunu sakla
                        np.save(os.path.join(log_path, f"test_cloud_u{update_count}.npy"), test_returns)

                        if len(test_prefs) > 1:
                            reset_t = test_env.reset()
                            if isinstance(reset_t, tuple) and len(reset_t)==2: tobs, _ = reset_t
                            else: tobs = reset_t
                            
                            tobs_t = torch.as_tensor(tobs, dtype=torch.float32, device=device).unsqueeze(0)
                            tprefs_t = torch.as_tensor(np.array(test_prefs), dtype=torch.float32, device=device)
                            
                            # Action Diversity
                            action_means = ppo.collect_action_mean(tobs_t.repeat(len(test_prefs), 1), tprefs_t)
                            if action_means.shape[0] > 1:
                                dists = torch.pdist(action_means)
                                action_diversity = dists.mean().item()

                            # Pairwise KL Divergence
                            dists = []
                            for i in range(len(test_prefs)):
                                for j in range(i + 1, len(test_prefs)):
                                    dist1 = ppo.get_action_dist(tobs_t, tprefs_t[i].unsqueeze(0))
                                    dist2 = ppo.get_action_dist(tobs_t, tprefs_t[j].unsqueeze(0))
                                    kl = torch.distributions.kl.kl_divergence(dist1, dist2).item()
                                    dists.append(kl)
                            if dists:
                                kl_median = np.median(dists)
                                kl_std = np.std(dists)

                else:
                    test_returns = np.empty((0,pref_dim), dtype=np.float32)
                
                if do_igd and global_front.size>0 and test_returns.size>0:
                    igd_metric, min_ref, max_ref = compute_normalized_igd(global_front, test_returns, min_ref, max_ref)
                    igd_plus_metric = igd_plus(global_front, test_returns) if global_front.size>0 else np.nan
                    igd_plus_norm_metric = compute_normalized_igd_plus(global_front, test_returns, min_ref, max_ref)
                elif do_igd:
                    igd_metric, igd_plus_metric, igd_plus_norm_metric = np.nan, np.nan, np.nan
                else:
                    igd_metric = last_metrics['igd']
                    igd_plus_metric = last_metrics['igd_plus']
                    igd_plus_norm_metric = last_metrics['igd_plus_norm']

                if do_eval and global_front.size > 0:
                    rng = (max_ref - min_ref); rng[rng <= 1e-9] = 1.0
                    train_norm = np.clip((global_front - min_ref)/rng, 0, 1)
                    hv_train_norm = monte_carlo_hv(train_norm, samples=args.hv_mc_samples)
                else:
                    hv_train_norm = last_metrics['hv_train_norm']

                if do_eval and test_returns.size > 0 and global_front.size > 0:
                    rng = (max_ref - min_ref); rng[rng <= 1e-9] = 1.0
                    test_norm = np.clip((test_returns - min_ref)/rng, 0, 1)
                    hv_test_norm = monte_carlo_hv(test_norm, samples=args.hv_mc_samples)
                else:
                    hv_test_norm = last_metrics['hv_test_norm']

                if do_eval and args.exact_hv and global_front.size>0:
                    hv_train_exact = exact_hv(global_front)
                else:
                    hv_train_exact = last_metrics['hv_train_exact']

                if do_eval and args.exact_hv and test_returns.size>0:
                    hv_test_exact = exact_hv(test_returns)
                else:
                    hv_test_exact = last_metrics['hv_test_exact']

                if do_eval or do_igd:
                    last_metrics.update({
                        'igd': igd_metric, 'igd_plus': igd_plus_metric, 'igd_plus_norm': igd_plus_norm_metric,
                        'hv_train_norm': hv_train_norm, 'hv_test_norm': hv_test_norm,
                        'hv_train_exact': hv_train_exact, 'hv_test_exact': hv_test_exact
                    })

                train_ep_rew_mean = np.mean(ep_rewards_buffer) if ep_rewards_buffer else np.nan
                train_ep_rew_std = np.std(ep_rewards_buffer) if ep_rewards_buffer else np.nan
                ep_rewards_buffer.clear()

                writer.writerow({
                    "update_steps": update_count, "total_timesteps": time_step,
                    "time_elapsed": time.time() - start_time,
                    "train_ep_rew_mean": train_ep_rew_mean, "train_ep_rew_std": train_ep_rew_std,
                    "test_hv_exact": last_metrics['hv_test_exact'], "test_hv_normalized": last_metrics['hv_test_norm'],
                    "test_igd": last_metrics['igd'], "test_igd_plus": last_metrics['igd_plus'], "test_igd_plus_norm": last_metrics['igd_plus_norm'],
                    "front_size": len(global_front), "action_diversity": action_diversity,
                    "kl_median": kl_median, "kl_std": kl_std,
                    "policy_entropy": policy_entropy, "explained_variance": explained_var,
                    "kl_div": kl_div, "lr": cur_lr, "value_loss": value_loss,
                    "adv_std": adv_std, "adv_group_std": adv_group_std
                })
                csvfile.flush()

                if args.debug:
                    print(f"[UPDATE] #{update_count} val={value_loss:.3f} kl={kl_div:.4f} ev={explained_var:.3f} igd={last_metrics['igd']:.4f} hv_test={last_metrics['hv_test_norm']:.3f}", flush=True)
            
            if done:
                break
        
        global_front = update_global_nd(global_front, ep_vec_sum_raw[None, :])
        ep_rewards_buffer.append(ep_scalar_reward)
        
        if ep % log_interval == 0:
            print(f'Ep {ep}\tAvgLen {t+1:.1f}\tFront| {len(global_front)}\tUpdates {update_count}', flush=True)

        if args.debug and update_count == 0 and time_step > 2 * update_timestep:
            print(f"[DEBUG][WARN] Henüz hiç UPDATE tetiklenmedi. time_step={time_step} update_timestep={update_timestep} steps_since_update={steps_since_update}. Parametreleri ve done frekansını kontrol edin.", flush=True)

    # --- Finalization ---
    csvfile.close()
    if args.manifold != 'none' and embed_states:
        embed_path = os.path.join(log_path, 'embeddings')
        os.makedirs(embed_path, exist_ok=True)
        np.save(os.path.join(embed_path, 'states.npy'), np.array(embed_states))
        np.save(os.path.join(embed_path, 'prefs.npy'), np.array(embed_prefs))
        print(f"Saved {len(embed_states)} embeddings to {embed_path}")

    np.save(os.path.join(log_path, 'final_front.npy'), global_front)

    if args.summary_table:
        summary_data = {
            "run_id": run_id, "seed": random_seed, "use_grpo": use_grpo,
            "final_hv_norm": last_metrics['hv_test_norm'], "final_igd": last_metrics['igd'],
            "final_igd_plus": last_metrics['igd_plus'], "final_igd_plus_norm": last_metrics['igd_plus_norm'],
            "final_front_size": len(global_front),
            "final_kl_median": kl_median if 'kl_median' in locals() else np.nan,
            "final_adv_std": adv_std if 'adv_std' in locals() else np.nan,
            "final_adv_group_std": adv_group_std if 'adv_group_std' in locals() else np.nan,
            "total_updates": update_count, "total_time_minutes": (time.time() - start_time) / 60.0
        }
        summary_path = os.path.join(log_path, 'summary.csv')
        with open(summary_path, 'w', newline='') as f:
            s_writer = csv.DictWriter(f, fieldnames=summary_data.keys())
            s_writer.writeheader()
            s_writer.writerow(summary_data)

    # Son test bulutunu (varsa) final olarak da kaydet
    if 'test_returns' in locals() and isinstance(test_returns, np.ndarray) and test_returns.size>0:
        np.save(os.path.join(log_path, 'test_cloud_final.npy'), test_returns)

    if args.debug:
        print('[DEBUG] Eğitim tamamlandı.', flush=True)

if __name__ == '__main__':
    main()
