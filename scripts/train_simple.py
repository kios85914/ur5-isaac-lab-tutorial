"""Minimal RL Training Loop -- Implement your own PPO! / 最小限のRL訓練ループ -- 自分のPPOを実装しよう！

This script provides a bare-bones training loop using pure PyTorch.
No Gymnasium, no skrl -- just the environment and your algorithm.
このスクリプトは純粋なPyTorchを使った最小限の訓練ループを提供します。
Gymnasium も skrl も使わず、環境とあなたのアルゴリズムだけです。

Usage / 使用方法:
    python scripts/train_simple.py

The script starts with a random policy (reward ~ -1.0).
Implement PPO in the marked sections to improve it!
スクリプトはランダムポリシーで開始します（報酬 ≈ -1.0）。
マークされた部分にPPOを実装して改善しましょう！
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simple RL Training for UR5")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # Run without GUI for faster training / GUIなしで高速訓練

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Imports after simulator launch / シミュレータ起動後のインポート ----

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

from ur5_lab.tasks.ur5_reach_env import UR5ReachEnv, UR5ReachEnvCfg


# =====================================================================
#  1. Environment Setup / 環境セットアップ
# =====================================================================

cfg = UR5ReachEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
env = UR5ReachEnv(cfg)

OBS_DIM = 15    # joint_pos(6) + joint_vel(6) + target_pos(3)
ACT_DIM = 6     # joint position deltas / 関節位置デルタ
DEVICE = env.device

print(f"\n[INFO] Environment ready / 環境準備完了")
print(f"  Num envs / 環境数: {cfg.scene.num_envs}")
print(f"  Obs dim / 観測次元: {OBS_DIM}")
print(f"  Act dim / 行動次元: {ACT_DIM}")


# =====================================================================
#  2. Policy Network / ポリシーネットワーク
# =====================================================================

class Policy(nn.Module):
    """Simple Gaussian policy network. / シンプルなガウスポリシーネットワーク。

    Outputs: mean and log_std for each action dimension.
    出力：各行動次元の平均値と対数標準偏差。
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # Learnable log_std / 学習可能なlog_std

    def forward(self, obs: torch.Tensor):
        """Compute action distribution parameters. / 行動分布パラメータを計算する。"""
        features = self.net(obs)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def sample_action(self, obs: torch.Tensor):
        """Sample action from policy and return (action, log_prob). / ポリシーから行動をサンプリングし (action, log_prob) を返す。"""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.clamp(-1.0, 1.0), log_prob

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions: return (log_prob, entropy). / 行動を評価：(log_prob, エントロピー) を返す。"""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Simple value function network. / シンプルな価値関数ネットワーク。"""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# =====================================================================
#  3. Rollout Buffer / ロールアウトバッファ
# =====================================================================

class RolloutBuffer:
    """Stores trajectory data for one rollout. / 1回のロールアウトの軌跡データを保存する。"""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def compute_returns(self, gamma: float = 0.99, lam: float = 0.95):
        """Compute GAE advantages and returns. / GAEアドバンテージとリターンを計算する。

        Args:
            gamma: Discount factor / 割引率
            lam: GAE lambda / GAE λ
        """
        T = len(self.rewards)
        advantages = torch.zeros(T, self.rewards[0].shape[0], device=self.rewards[0].device)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t].float()) - self.values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - self.dones[t].float()) * last_gae

        returns = advantages + torch.stack(self.values)
        return advantages, returns

    def get_batches(self):
        """Stack all data into tensors. / 全データをテンソルに変換する。"""
        return (
            torch.stack(self.obs),          # (T, N, obs_dim)
            torch.stack(self.actions),      # (T, N, act_dim)
            torch.stack(self.log_probs),    # (T, N)
        )


# =====================================================================
#  4. Training Loop / 訓練ループ
# =====================================================================

# Initialize networks / ネットワークを初期化
policy = Policy(OBS_DIM, ACT_DIM).to(DEVICE)
value_fn = ValueNetwork(OBS_DIM).to(DEVICE)

# =====================================================================
#
#  === YOUR HYPERPARAMETERS HERE / ハイパーパラメータをここに設定 ===
#
# =====================================================================
LEARNING_RATE = 3e-4        # Learning rate / 学習率
GAMMA = 0.99                # Discount factor / 割引率
GAE_LAMBDA = 0.95           # GAE lambda / GAE λ
CLIP_EPS = 0.2              # PPO clip range / PPOクリップ範囲
ENTROPY_COEF = 0.01         # Entropy bonus / エントロピーボーナス
VALUE_COEF = 0.5            # Value loss weight / 価値損失の重み
UPDATE_EPOCHS = 8           # Epochs per update / 更新ごとのエポック数
STEPS_PER_ROLLOUT = 200     # Steps before update / 更新前のステップ数

policy_optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
value_optimizer = optim.Adam(value_fn.parameters(), lr=LEARNING_RATE)

# Logging / ログ
reward_history = deque(maxlen=20)

print(f"\n{'=' * 60}")
print(f"  Starting Training / 訓練開始")
print(f"  Episodes: {args_cli.episodes} | Steps/rollout: {STEPS_PER_ROLLOUT}")
print(f"{'=' * 60}\n")

# Reset environment / 環境をリセット
obs_dict = env.reset()
obs = obs_dict["policy"]  # (N, 15)

for episode in range(args_cli.episodes):
    buffer = RolloutBuffer()

    # ---- Collect rollout / ロールアウト収集 ----
    episode_reward = torch.zeros(cfg.scene.num_envs, device=DEVICE)

    for step_i in range(STEPS_PER_ROLLOUT):
        with torch.no_grad():
            action, log_prob = policy.sample_action(obs)
            value = value_fn(obs)

        # Step environment / 環境をステップ
        obs_dict = env.step(action)
        next_obs = obs_dict["policy"]
        reward = env.reward_buf
        terminated = env.terminated_buf
        truncated = env.truncated_buf
        done = terminated | truncated

        # Store transition / 遷移を保存
        buffer.store(obs, action, log_prob, reward, done, value)
        episode_reward += reward

        obs = next_obs

        # Reset done environments / 完了した環境をリセット
        if done.any():
            reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
            env._reset_idx(reset_ids)

    avg_reward = episode_reward.mean().item() / STEPS_PER_ROLLOUT

    # ================================================================
    #
    #  === YOUR PPO UPDATE HERE / PPO更新をここに実装 ===
    #
    #  You have:
    #    - buffer.obs, buffer.actions, buffer.log_probs  (trajectory data)
    #    - buffer.compute_returns(GAMMA, GAE_LAMBDA)     (compute advantages)
    #    - policy.evaluate(obs, actions)                  (get new log_probs)
    #    - value_fn(obs)                                  (get new values)
    #
    #  Implement the PPO clipped objective:
    #    ratio = exp(new_log_prob - old_log_prob)
    #    clipped = clamp(ratio, 1-eps, 1+eps) * advantage
    #    policy_loss = -min(ratio * advantage, clipped)
    #    value_loss = (returns - values)^2
    #
    # ================================================================

    # --- Reference implementation (uncomment to use) ---
    # --- 参考実装（使用するにはコメントを外す）---

    # advantages, returns = buffer.compute_returns(GAMMA, GAE_LAMBDA)
    # obs_batch, act_batch, old_log_probs = buffer.get_batches()
    # T, N = obs_batch.shape[:2]
    # obs_flat = obs_batch.reshape(T * N, -1)
    # act_flat = act_batch.reshape(T * N, -1)
    # old_lp_flat = old_log_probs.reshape(T * N)
    # adv_flat = advantages.reshape(T * N)
    # ret_flat = returns.reshape(T * N)
    #
    # adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    #
    # for _ in range(UPDATE_EPOCHS):
    #     new_log_probs, entropy = policy.evaluate(obs_flat, act_flat)
    #     values = value_fn(obs_flat)
    #
    #     ratio = torch.exp(new_log_probs - old_lp_flat)
    #     surr1 = ratio * adv_flat
    #     surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_flat
    #     policy_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy.mean()
    #     value_loss = VALUE_COEF * (ret_flat - values).pow(2).mean()
    #
    #     policy_optimizer.zero_grad()
    #     policy_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    #     policy_optimizer.step()
    #
    #     value_optimizer.zero_grad()
    #     value_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(value_fn.parameters(), 1.0)
    #     value_optimizer.step()

    # ---- Logging / ログ ----
    reward_history.append(avg_reward)
    avg_20 = np.mean(reward_history)

    if episode % 10 == 0:
        print(f"  Episode {episode:4d} | Reward (avg): {avg_reward:+.3f} | Reward (20-ep avg): {avg_20:+.3f}")

print(f"\n{'=' * 60}")
print(f"  Training Complete / 訓練完了")
print(f"{'=' * 60}")

# Save model / モデルを保存
os.makedirs("logs", exist_ok=True)
torch.save(policy.state_dict(), "logs/policy.pt")
torch.save(value_fn.state_dict(), "logs/value_fn.pt")
print(f"  Models saved to logs/policy.pt and logs/value_fn.pt")
print(f"  モデルを logs/policy.pt と logs/value_fn.pt に保存しました\n")

env.close()
simulation_app.close()
