from __future__ import annotations
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1: x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta**2) * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, M2 / tot, tot

@dataclass
class HCConfig:
    speed_mode: str = "target_speed"  # "raw_speed" | "target_speed"
    target_speed: float = 2.0
    alpha_energy: float = 0.1
    beta_smooth: float = 0.05
    normalize: bool = True
    norm_clip: float = 5.0
    friction_rand: bool = False
    friction_low: float = 0.4
    friction_high: float = 1.5
    seed: Optional[int] = None
    freeze_after_steps: Optional[int] = None  # added: RMS freeze threshold

class HalfCheetahMORL(gym.Wrapper):
    """HalfCheetah-v4 sarmalayıcı: info['reward_vec'] = normalized, info['reward_vec_raw'] = raw"""
    def __init__(self, cfg: HCConfig):
        env = gym.make("HalfCheetah-v4")
        super().__init__(env)
        self.cfg = cfg
        self.dt = float(getattr(self.env.unwrapped, "dt", 0.01))
        self.prev_x = 0.0
        self.prev_action = None
        self.rms = RunningMeanStd((3,)) if cfg.normalize else None
        self._norm_steps = 0  # added counter
        # Reward uzayı (3 amaç)
        self.reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        if cfg.seed is not None:
            try: self.env.reset(seed=cfg.seed)
            except TypeError: pass

    def _maybe_randomize_friction(self):
        if not self.cfg.friction_rand: return
        try:
            model = self.env.unwrapped.model
            fr = model.geom_friction.copy()
            scale = np.random.uniform(self.cfg.friction_low, self.cfg.friction_high)
            fr[:, 0] = fr[:, 0] * scale
            model.geom_friction[:] = fr
        except Exception:
            pass  # bazı mujoco sürümleri farklıdır

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._maybe_randomize_friction()
        try: self.prev_x = float(self.env.unwrapped.data.qpos[0])
        except Exception: self.prev_x = 0.0
        if self.prev_action is None:
            self.prev_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        else:
            self.prev_action.fill(0.0)
        return obs, info

    def step(self, action: np.ndarray):
        obs, _, terminated, truncated, info = self.env.step(action)
        try: x = float(self.env.unwrapped.data.qpos[0])
        except Exception: x = info.get("x_position", 0.0)
        vx = (x - self.prev_x) / self.dt
        self.prev_x = x

        if self.cfg.speed_mode == "raw_speed":
            r_speed = vx
        elif self.cfg.speed_mode == "target_speed":
            r_speed = -abs(vx - self.cfg.target_speed)
        else:
            raise ValueError("speed_mode must be 'raw_speed' or 'target_speed'")

        r_energy = -self.cfg.alpha_energy * float(np.sum(np.square(action)))
        r_smooth = -self.cfg.beta_smooth * float(np.sum(np.square(action - self.prev_action)))
        self.prev_action = action.astype(np.float32, copy=False)

        r_vec_raw = np.array([r_speed, r_energy, r_smooth], dtype=np.float32)
        r_vec = r_vec_raw.copy()
        if self.rms is not None:
            do_update = True
            if self.cfg.freeze_after_steps is not None and self._norm_steps >= self.cfg.freeze_after_steps:
                do_update = False
            if do_update:
                self.rms.update(r_vec)
            self._norm_steps += 1
            std = np.sqrt(np.clip(self.rms.var, 1e-6, None))
            r_vec = (r_vec - self.rms.mean) / std
            r_vec = np.clip(r_vec, -self.cfg.norm_clip, self.cfg.norm_clip)

        info = dict(info)
        info["reward_vec"] = r_vec
        info["reward_vec_raw"] = r_vec_raw
        return obs, 0.0, terminated, truncated, info

def make_hc_morl(**kwargs) -> HalfCheetahMORL:
    return HalfCheetahMORL(HCConfig(**kwargs))

if __name__ == "__main__":
    env = make_hc_morl(speed_mode="target_speed", target_speed=2.0, friction_rand=True, seed=0, freeze_after_steps=10000)
    obs, info = env.reset()
    for _ in range(3):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print("reward_vec:", info["reward_vec"], "raw:", info["reward_vec_raw"])
