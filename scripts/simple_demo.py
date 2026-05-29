"""Simple UR5 Control Demo -- Edit this script to control the robot! / UR5 制御デモ -- このスクリプトを編集してロボットを制御しよう！

This is the simplest way to control the UR5 robot.
No RL, no gym environment -- just direct function calls.
これはUR5ロボットを制御する最もシンプルな方法です。
RL も gym 環境も使わず、関数を直接呼び出すだけです。

Usage / 使用方法:
    python scripts/simple_demo.py
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simple UR5 Control Demo")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Everything below runs after simulator starts / 以下はシミュレータ起動後に実行 ----

import torch
import time

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
import isaaclab.utils.math as math_utils
from isaaclab.sim import SimulationCfg, SimulationContext

from ur5_lab.assets.ur5_cfg import UR5_CFG

# =====================================================================
#  Setup -- Don't modify this section / セットアップ -- ここは変更しないでください
# =====================================================================

# Simulation context / シミュレーションコンテキスト
sim_cfg = SimulationCfg(dt=1.0 / 120.0)
sim = SimulationContext(sim_cfg)
sim.set_camera_view(eye=(3.0, 3.0, 3.0), target=(0.0, 0.0, 1.0))

# Ground + Light / 地面 + ライト
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 1.0)).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))

# Robot / ロボット
robot_cfg = UR5_CFG.replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)

# Start simulation / シミュレーション開始
sim.reset()
robot.update(sim_cfg.dt)

# Joint info / 関節情報
ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
ARM_JOINT_IDS, _ = robot.find_joints(ARM_JOINT_NAMES)

# IK controller / IKコントローラ
ik_controller = DifferentialIKController(
    DifferentialIKControllerCfg(command_type="position", ik_method="pinv"),
    num_envs=1, device=sim.device,
)

# Body index for Jacobian / ヤコビアン用ボディインデックス
ee_body_ids, _ = robot.find_bodies(["ee_link"])
ee_body_idx = ee_body_ids[0]
jacobi_body_idx = ee_body_idx - 1 if robot.is_fixed_base else ee_body_idx

print("\n" + "=" * 50)
print("  UR5 Simple Demo Ready! / UR5 シンプルデモ準備完了！")
print("=" * 50 + "\n")

# =====================================================================
#  Helper Functions / ヘルパー関数
# =====================================================================

def step(n: int = 120):
    """Run n simulation steps (default=120 = 1 second). / n ステップのシミュレーションを実行（デフォルト=120=1秒）。"""
    for _ in range(n):
        sim.step()
        robot.update(sim_cfg.dt)


def get_state() -> dict:
    """Get current robot state. / 現在のロボット状態を取得する。

    Returns dict with: / 以下を含む辞書を返す：
        joint_pos: 6 joint angles (rad) / 6関節角度（ラジアン）
        joint_vel: 6 joint velocities (rad/s) / 6関節速度（ラジアン/秒）
        ee_pos: end-effector position [x, y, z] in world frame (m) / ワールド座標でのEE位置 [x, y, z]（メートル）
    """
    joint_pos = robot.data.joint_pos[0, ARM_JOINT_IDS].tolist()
    joint_vel = robot.data.joint_vel[0, ARM_JOINT_IDS].tolist()

    # EE position from body state / ボディ状態からEE位置を取得
    ee_pos = robot.data.body_pos_w[0, ee_body_idx].tolist()

    return {
        "joint_pos": [round(v, 4) for v in joint_pos],
        "joint_vel": [round(v, 4) for v in joint_vel],
        "ee_pos": [round(v, 4) for v in ee_pos],
    }


def move_joints(target_angles: list[float], duration: float = 1.0):
    """Move robot to target joint angles. / ロボットを目標関節角度に移動する。

    Args:
        target_angles: List of 6 joint angles in radians. / 6つの関節角度のリスト（ラジアン）。
        duration: How long to move (seconds). / 移動時間（秒）。

    Example / 例:
        move_joints([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    """
    assert len(target_angles) == 6, f"Need 6 joint angles, got {len(target_angles)} / 6つの関節角度が必要です"

    target = torch.tensor([target_angles], device=sim.device, dtype=torch.float32)
    n_steps = int(duration * 120)

    robot.set_joint_position_target(target, joint_ids=ARM_JOINT_IDS)
    step(n_steps)

    state = get_state()
    print(f"  Joints / 関節: {state['joint_pos']}")
    print(f"  EE pos / EE位置: {state['ee_pos']}")


def move_to_ee(target_xyz: list[float], duration: float = 1.5):
    """Move end-effector to target position using IK. / IKを使って末端を目標位置に移動する。

    Args:
        target_xyz: Target [x, y, z] in world coordinates (meters). / ワールド座標での目標 [x, y, z]（メートル）。
                    Note: Robot base is at z=1.05m (table height). / 注意：ロボットベースは z=1.05m（テーブル高さ）。
        duration: How long to move (seconds). / 移動時間（秒）。

    Example / 例:
        move_to_ee([0.5, 0.0, 1.3])   # 25cm above table / テーブルから25cm上
        move_to_ee([0.3, 0.3, 1.5])   # left and high / 左上方
    """
    assert len(target_xyz) == 3, f"Need 3 coordinates [x, y, z], got {len(target_xyz)} / 3つの座標が必要です"

    target_pos_w = torch.tensor([target_xyz], device=sim.device, dtype=torch.float32)

    # Get current state / 現在の状態を取得
    robot_pos_w = robot.data.root_state_w[:, :3]
    robot_quat_w = robot.data.root_state_w[:, 3:7]

    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx, :3]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]

    # Convert to base frame / ベースフレームに変換
    target_pos_b, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_quat_w, target_pos_w)
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(robot_pos_w, robot_quat_w, ee_pos_w, ee_quat_w)

    # Compute IK / IKを計算
    ik_controller.set_command(target_pos_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

    jacobian_w = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, ARM_JOINT_IDS]
    base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(robot_quat_w))
    jacobian_b = jacobian_w.clone()
    jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

    joint_pos = robot.data.joint_pos[:, ARM_JOINT_IDS]
    joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos)

    # Apply / 適用
    robot.set_joint_position_target(joint_pos_des, joint_ids=ARM_JOINT_IDS)

    n_steps = int(duration * 120)
    step(n_steps)

    state = get_state()
    print(f"  Target / 目標: {target_xyz}")
    print(f"  EE pos / EE位置: {state['ee_pos']}")
    err = sum((a - b) ** 2 for a, b in zip(target_xyz, state['ee_pos'])) ** 0.5
    print(f"  Error / 誤差: {err:.4f} m")


def home():
    """Return to home position. / ホームポジションに戻る。"""
    print("Moving to home... / ホームに移動中...")
    move_joints([-0.568, -0.658, 1.602, -2.585, -1.606, -1.641])


# =====================================================================
#
#  YOUR CODE HERE -- Edit below! / ここを編集してください！
#
# =====================================================================

# Print initial state / 初期状態を表示
print("Initial state / 初期状態:")
print(get_state())
print()

# --- Example 1: Joint control / 関節制御の例 ---
print("--- Example 1: Joint Control / 関節制御 ---")
home()
print()

# --- Example 2: Move one joint / 1つの関節を動かす ---
print("--- Example 2: Move J1 only / J1のみ動かす ---")
move_joints([-1.5, -0.658, 1.602, -2.585, -1.606, -1.641])
print()

# --- Example 3: IK control / IK制御の例 ---
print("--- Example 3: IK Control / IK制御 ---")
home()
move_to_ee([0.5, 0.0, 1.3])    # 25cm above table / テーブルから25cm上
print()

# --- Example 4: Move to multiple positions / 複数位置に移動 ---
print("--- Example 4: Multiple targets / 複数目標 ---")
targets = [
    [0.5, 0.2, 1.3],   # right / 右
    [0.5, -0.2, 1.3],  # left / 左
    [0.3, 0.0, 1.5],   # high / 上方
    [0.6, 0.0, 1.1],   # low / 下方
]
for i, t in enumerate(targets):
    print(f"  Target {i+1}: {t}")
    move_to_ee(t, duration=1.0)
print()

# --- Your turn! Add your code below / あなたの番！以下にコードを追加 ---
# move_joints([...])
# move_to_ee([...])
# print(get_state())

print("\nDone! Close the window to exit. / 完了！ウィンドウを閉じて終了してください。")

# Keep the window open / ウィンドウを開いたままにする
while simulation_app.is_running():
    sim.step()
    robot.update(sim_cfg.dt)

simulation_app.close()
