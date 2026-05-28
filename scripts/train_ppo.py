"""Train UR5 reach task using PPO / PPO を使用して UR5 到達タスクを訓練する

Usage / 使用方法：
    # Basic training (64 parallel environments) / 基本訓練（64 個の並列環境）
    python scripts/train_ppo.py --task Isaac-UR5-Reach-v0 --num_envs 64

    # Use fewer environments when GPU memory is insufficient / GPU メモリ不足の場合は環境数を減らす
    python scripts/train_ppo.py --task Isaac-UR5-Reach-v0 --num_envs 16

    # Resume training from checkpoint / チェックポイントから訓練を再開
    python scripts/train_ppo.py --task Isaac-UR5-Reach-v0 --checkpoint logs/skrl/.../best_agent.pt

    # Record video / 録画
    python scripts/train_ppo.py --task Isaac-UR5-Reach-v0 --video
"""

from __future__ import annotations

import argparse
import os
import sys
import random
from datetime import datetime

from isaaclab.app import AppLauncher

# ---- CLI arguments / CLIパラメータ ----
parser = argparse.ArgumentParser(description="PPO Training for UR5 Reach Task")
parser.add_argument("--video", action="store_true", help="Record training video")
parser.add_argument("--video_length", type=int, default=200, help="Video length (steps)")
parser.add_argument("--video_interval", type=int, default=2000, help="Video recording interval (steps)")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments")
parser.add_argument("--task", type=str, default="Isaac-UR5-Reach-v0", help="Task ID")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum training iterations")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# Launch simulator / シミュレータを起動
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Imports after simulator launch / シミュレータ起動後の import ----
import gymnasium as gym

import skrl
from packaging import version
from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import ur5_lab  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    """Train skrl PPO agent. / skrl PPO エージェントを訓練する。"""

    # Override settings with CLI arguments / CLI 引数で設定を上書き
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Maximum training steps / 最大訓練ステップ数
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # Random seed / ランダムシード
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]

    # Configure log directory / ログディレクトリを設定
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_torch"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    print(f"[INFO] Training log directory / 訓練ログディレクトリ: {log_dir}")

    # Save configuration files / 設定ファイルを保存
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Load checkpoint (if available) / チェックポイントを読み込む（存在する場合）
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # Create environment / 環境を構築
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Video recording settings / 録画設定
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording training video / 訓練ビデオを録画中...")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap into skrl format / skrl 形式にラップ
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # Create Runner and start training / Runner を作成して訓練を開始
    runner = Runner(env, agent_cfg)

    if resume_path:
        print(f"[INFO] Loading model / モデルを読み込む: {resume_path}")
        runner.agent.load(resume_path)

    # Start training! / 訓練開始！
    runner.run()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
