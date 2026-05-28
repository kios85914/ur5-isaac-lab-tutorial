"""Manually control the UR5 robot -- input joint angles or end-effector position to move the arm / UR5 ロボットを手動制御 -- 関節角度またはエンドエフェクタ位置を入力してアームを動かす

Usage / 使用方法：
    python scripts/manual_joint_control.py --num_envs 1

Commands / 操作コマンド：
    - Enter 6 numbers (space-separated) to set joint target positions / 6 つの数字（スペース区切り）で関節目標位置を設定
    - Enter 'home' to return to home position / 'home' を入力してホームポジションに戻る
    - Enter 'random' to move randomly / 'random' を入力してランダムに動かす
    - Enter 'info' to display current state / 'info' を入力して現在の状態を表示
    - Enter 'quit' to exit / 'quit' を入力して終了

Example / 例：
    > -0.5 -1.0 1.5 -2.0 -1.5 0.0
    > home
    > info
"""

from __future__ import annotations

import argparse
import torch
import threading

from isaaclab.app import AppLauncher

# ---- CLI arguments / CLIパラメータ ----
parser = argparse.ArgumentParser(description="UR5 Manual Joint Control")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments (recommended: 1)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulator (must be done before other imports) / シミュレータを起動（他の import より先に行う必要がある）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Modules that can only be imported after simulator launch / シミュレータ起動後にのみ import できるモジュール ----
import gymnasium as gym
import ur5_lab  # noqa: F401 -- triggers environment registration / 環境登録をトリガー


# Home position (default joint angles) / ホームポジション（デフォルト関節角度）
HOME_JOINTS = [-0.568, -0.658, 1.602, -2.585, -1.606, -1.641]

# Shared state / 共有状態
_command = {"type": "hold", "values": None}
_lock = threading.Lock()
_running = True


def input_thread():
    """Receive user input in a background thread. / バックグラウンドスレッドでユーザー入力を受け取る。"""
    global _running

    print("\n" + "=" * 60)
    print("  UR5 Manual Control Mode / UR5 手動制御モード")
    print("=" * 60)
    print("Commands / コマンド：")
    print("  Enter 6 numbers  → Set joint target angles (radians) / 関節目標角度を設定（ラジアン）")
    print("  ee X Y Z         → Move EE to world position using IK / IKで末端をワールド位置へ移動")
    print("  home             → Return to home position / ホームポジションに戻る")
    print("  random           → Random motion / ランダム動作")
    print("  info             → Display current state / 現在の状態を表示")
    print("  quit             → Exit / 終了")
    print("=" * 60 + "\n")

    while _running:
        try:
            user_input = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            with _lock:
                _command["type"] = "quit"
            break

        if not user_input:
            continue

        with _lock:
            if user_input == "quit" or user_input == "q":
                _command["type"] = "quit"
                break
            elif user_input == "home":
                _command["type"] = "goto"
                _command["values"] = HOME_JOINTS
                print(f"  → Moving to home position / ホームポジションへ移動: {HOME_JOINTS}")
            elif user_input == "random":
                _command["type"] = "random"
                print("  → Executing random motion / ランダム動作を実行")
            elif user_input == "info":
                _command["type"] = "info"
            elif user_input.startswith("ee "):
                try:
                    coords = [float(v) for v in user_input[3:].split()]
                    if len(coords) == 3:
                        _command["type"] = "ik"
                        _command["values"] = coords
                        print(f"  → IK target EE position / IK目標EE位置: ({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f})")
                    else:
                        print("  x 'ee' command needs 3 values (X Y Z) / 'ee' コマンドには3つの値 (X Y Z) が必要です")
                except ValueError:
                    print("  x Cannot parse, usage: ee X Y Z / 解析できません、使用法: ee X Y Z")
            else:
                try:
                    values = [float(v) for v in user_input.split()]
                    if len(values) == 6:
                        _command["type"] = "goto"
                        _command["values"] = values
                        print(f"  → Target joint angles / 目標関節角度: {[f'{v:.3f}' for v in values]}")
                    else:
                        print(f"  x Need 6 numbers, you entered {len(values)} / 6 つの数字が必要です、{len(values)} 個入力されました")
                except ValueError:
                    print("  x Cannot parse. Enter 6 numbers, 'ee X Y Z', or a command / 解析できません。6つの数字、'ee X Y Z'、またはコマンドを入力してください")


def main():
    global _running

    # Create environment / 環境を構築
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg("Isaac-UR5-Reach-v0", num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make("Isaac-UR5-Reach-v0", cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped
    print(f"[INFO] Environment created / 環境構築成功 | Observation space / 観測空間: {env.observation_space} | Action space / 行動空間: {env.action_space}")

    obs, info = env.reset()

    # Start input thread / 入力スレッドを起動
    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    step = 0
    current_target_joints = torch.tensor(HOME_JOINTS, device="cuda:0", dtype=torch.float32)

    try:
        while simulation_app.is_running() and _running:
            action = torch.zeros(env.action_space.shape, device="cuda:0")

            with _lock:
                cmd_type = _command["type"]
                cmd_vals = _command["values"]
                _command["type"] = "hold"
                _command["values"] = None

            if cmd_type == "quit":
                break
            elif cmd_type == "goto":
                # Compute delta from current position to target as action / 現在位置から目標までの差分を動作として計算
                current_joints = unwrapped.get_joint_positions()[0]
                target = torch.tensor(cmd_vals, device="cuda:0", dtype=torch.float32)
                delta = target - current_joints
                # Clamp action magnitude / 動作幅を制限
                action = (delta / unwrapped.cfg.action_scale).unsqueeze(0).clamp(-1.0, 1.0)
                current_target_joints = target
            elif cmd_type == "ik":
                # Compute joint targets via Differential IK / 微分IKで関節目標を計算する
                target_pos = torch.tensor([cmd_vals], device="cuda:0", dtype=torch.float32)
                joint_targets = unwrapped.compute_ik(target_pos)
                current_joints = unwrapped.get_joint_positions()[0]
                delta = joint_targets[0] - current_joints
                action = (delta / unwrapped.cfg.action_scale).unsqueeze(0).clamp(-1.0, 1.0)
                current_target_joints = joint_targets[0]
                ee_pos = unwrapped.get_ee_pos_world()[0]
                print(f"  [IK] Current EE / 現在のEE: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
            elif cmd_type == "random":
                action = torch.rand(1, 6, device="cuda:0") * 2.0 - 1.0
            elif cmd_type == "info":
                joints = unwrapped.get_joint_positions()[0]
                ee_pos = unwrapped.get_ee_pos_world()[0]
                target = unwrapped.get_target_pos_world()[0]
                print(f"\n  Joint angles / 関節角度: [{', '.join(f'{j:.3f}' for j in joints.tolist())}]")
                print(f"  End-effector position / エンドエフェクタ位置: [{', '.join(f'{p:.3f}' for p in ee_pos.tolist())}]")
                print(f"  Target position / 目標位置: [{', '.join(f'{p:.3f}' for p in target.tolist())}]")
                dist = torch.norm(ee_pos - target).item()
                print(f"  Distance to target / 目標までの距離: {dist:.4f} m\n")
            else:
                # Continuously move toward target / 継続的に目標へ向かう
                current_joints = unwrapped.get_joint_positions()[0]
                delta = current_target_joints - current_joints
                if torch.norm(delta) > 0.01:
                    action = (delta / unwrapped.cfg.action_scale).unsqueeze(0).clamp(-1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)

            # Print brief info every 200 steps / 200 ステップごとに簡易情報を表示
            if step % 200 == 0 and step > 0:
                joints = unwrapped.get_joint_positions()[0]
                ee_pos = unwrapped.get_ee_pos_world()[0]
                print(f"  [Step {step}] EE=({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"
                      f" | reward={reward.mean().item():.3f}")

            step += 1

    except KeyboardInterrupt:
        print("\n[INFO] User interrupted / ユーザーによる中断")

    _running = False
    env.close()
    print("[INFO] Environment closed / 環境を閉じました")


if __name__ == "__main__":
    main()
    simulation_app.close()
