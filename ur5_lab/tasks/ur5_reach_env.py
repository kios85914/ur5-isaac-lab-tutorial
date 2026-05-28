"""
UR5 Reach Environment -- Simplified Robot Control Teaching Environment / UR5 リーチ環境 -- 簡略化ロボット制御教育環境

This environment moves the UR5 robot arm's end-effector to a random target position. / この環境はUR5ロボットアームの末端執行器（エンドエフェクタ）をランダムな目標位置に移動させます。
Two usage modes are supported: / 2つの使用モードをサポートします：
  1. RL Training Mode: Standard gym environment for automatic learning with PPO etc. / RLトレーニングモード：PPOなどのアルゴリズムで自動学習するための標準gym環境
  2. Manual Control Mode: Students manually input joint angles to control the robot / 手動制御モード：学生が関節角度を手動入力してロボットを制御する

Observation Space (15 dimensions) / 観測空間（15次元）
-----------------
  joint_pos (6) + joint_vel (6) + target_pos (3)

Action Space (6 dimensions) / 行動空間（6次元）
-----------------
  Position increments (delta) for 6 joints, added to current position after multiplying by action_scale / 6関節の位置増分（デルタ）、action_scaleを掛けて現在位置に加算

Reward Function / 報酬関数
--------
  r = -w_d * dist + w_t * (1 - tanh(dist/s)) + w_a * ||action||^2

  dist = Distance between end-effector and target / 末端執行器と目標の間の距離
"""

from __future__ import annotations

import torch
from typing import Sequence
import gymnasium as gym

# Isaac Lab imports / Isaac Labインポート
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
import isaaclab.utils.math as math_utils

# Local robot configuration / ローカルロボット設定
from ur5_lab.assets.ur5_cfg import UR5_CFG


# =====================================================================
#  Environment Configuration / 環境設定
# =====================================================================

@configclass
class UR5ReachEnvCfg(DirectRLEnvCfg):
    """All tunable parameters for the UR5 Reach environment. / UR5リーチ環境のすべての調整可能パラメータ。

    Modifying parameters here changes the environment behavior without touching environment code. / ここのパラメータを変更するだけで環境の動作を変更でき、環境コードを編集する必要はありません。
    """

    # --- Timing settings / 時間設定 ---
    episode_length_s: float = 5.0       # Seconds per episode / 各エピソードの秒数
    decimation: int = 4                 # Number of physics simulations per control step / 制御ステップごとの物理シミュレーション回数
    action_scale: float = 0.5           # Action scaling factor / 行動スケーリング係数

    # --- Observation and action spaces / 観測空間と行動空間 ---
    #  obs = joint_pos(6) + joint_vel(6) + target_pos(3) = 15
    num_observations: int = 15
    num_actions: int = 6

    observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(15,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))
    state_space = 0

    # --- Scene / シーン ---
    num_envs: int = 16
    env_spacing: float = 3.0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=3.0, replicate_physics=True,
    )

    # --- Simulation / シミュレーション ---
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=4)

    # --- Viewer perspective / 観察視点 ---
    viewer = ViewerCfg(eye=(3.5, 3.5, 3.5), origin_type="world", env_index=0)

    # --- Robot / ロボット ---
    robot_cfg: ArticulationCfg = UR5_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # --- End-effector tracker (FrameTransformer) / 末端追跡器（FrameTransformer）---
    ee_frame_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/ee_link",
                name="end_effector",
                offset=OffsetCfg(pos=[0.12, 0.0, 0.0]),
            ),
        ],
    )

    # --- Target position range (robot coordinate frame) / 目標位置範囲（ロボット座標系）---
    target_pos_range = {
        "x": (0.3, 0.7),    # Forward/backward / 前後
        "y": (-0.4, 0.4),   # Left/right / 左右
        "z": (-0.2, 0.3),   # Up/down (relative to base) / 上下（ベースからの相対値）
    }

    # --- Reward weights / 報酬重み ---
    reward_distance_weight: float = 1.0       # Linear distance penalty / 線形距離ペナルティ
    reward_tanh_weight: float = 2.0           # Tanh reward (high reward when close to target) / tanh報酬（目標に近いほど高い報酬）
    reward_tanh_std: float = 0.1              # Width parameter for tanh / tanhの幅パラメータ
    reward_action_penalty: float = -0.01      # Action magnitude penalty / 行動サイズペナルティ

    # --- Success threshold / 成功閾値 ---
    success_threshold: float = 0.02  # Success if within 2 cm / 2cm以内であれば成功

    # --- Reset settings / リセット設定 ---
    robot_default_joint_pos = [-0.568, -0.658, 1.602, -2.585, -1.606, -1.641]
    reset_joint_noise: float = 0.1   # Random perturbation of joint angles at reset / リセット時の関節角度のランダム擾乱

    # --- Visualization / 可視化 ---
    debug_vis: bool = True


# =====================================================================
#  Environment Implementation / 環境実装
# =====================================================================

class UR5ReachEnv(DirectRLEnv):
    """UR5 reach task environment. / UR5リーチタスク環境。

    The robot must move its end-effector to a random 3D target position. / ロボットは末端執行器をランダムな3D目標位置に移動させなければなりません。
    Episode ends on timeout or successful arrival at the target. / エピソードはタイムアウトまたは目標への到達成功で終了します。
    """

    cfg: UR5ReachEnvCfg

    def __init__(self, cfg: UR5ReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Find indices of the 6 arm joints / 6つのアームジョイントのインデックスを見つける
        self._arm_joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ]
        self._arm_joint_ids, _ = self._robot.find_joints(self._arm_joint_names)

        # Cache joint limits / 関節限界をキャッシュする
        self._joint_lower = self._robot.data.soft_joint_pos_limits[0, self._arm_joint_ids, 0].to(self.device)
        self._joint_upper = self._robot.data.soft_joint_pos_limits[0, self._arm_joint_ids, 1].to(self.device)

        # Joint target position buffer / 関節目標位置バッファ
        self._dof_targets = torch.zeros((self.num_envs, 6), device=self.device)
        # Target EE position (robot coordinate frame) / 目標EE位置（ロボット座標系）
        self._target_pos_b = torch.zeros((self.num_envs, 3), device=self.device)

        # Episode statistics / エピソード統計
        self._episode_sums = {
            "position_error": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
        }

        # Initialize IK controller for manual EE control / 手動EE制御用のIKコントローラを初期化する
        self._ik_controller = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="position", ik_method="pinv"),
            num_envs=self.num_envs,
            device=self.device,
        )
        # Resolve body index for Jacobian lookup / ヤコビアン参照用のボディインデックスを解決する
        ee_body_ids, _ = self._robot.find_bodies(["ee_link"])
        self._ee_body_idx = ee_body_ids[0]
        # For fixed-base robots, Jacobian excludes the base / 固定ベースロボットの場合、ヤコビアンはベースを除外する
        self._jacobi_body_idx = self._ee_body_idx - 1 if self._robot.is_fixed_base else self._ee_body_idx

        # Enable visualization / 可視化を有効にする
        self.set_debug_vis(self.cfg.debug_vis)

        print(f"[UR5ReachEnv] Initialized {self.num_envs} parallel environments / {self.num_envs}個の並列環境を初期化しました")

    # =================================================================
    #  Scene Setup / シーン構築
    # =================================================================

    def _setup_scene(self):
        """Build the scene: robot + end-effector tracker + ground + lighting. / シーンを構築：ロボット + 末端追跡器 + 地面 + ライト。"""
        # Robot / ロボット
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # End-effector tracker / 末端執行器追跡器
        self._ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        # Clone environments / 環境を複製する
        self.scene.clone_environments(copy_from_source=False)

        # Ground / 地面
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        # Lighting / ライト
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 1.0))
        light_cfg.func("/World/Light", light_cfg)

    # =================================================================
    #  Action Processing / 行動処理
    # =================================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        """Store and clip actions. / 行動を保存してクリップする。"""
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        """Convert action increments to joint target positions. / 行動増分を関節目標位置に変換する。

        Steps: / 手順：
          1. Read current joint positions / 現在の関節位置を読み取る
          2. Add (action * scale) as increment / （行動 * スケール）を増分として加算
          3. Clip to joint limits / 関節限界にクリップする
          4. Send to PD controller / PDコントローラーに送信する
        """
        current_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        self._dof_targets = current_pos + self._actions * self.cfg.action_scale
        self._dof_targets = torch.clamp(self._dof_targets, self._joint_lower, self._joint_upper)
        self._robot.set_joint_position_target(self._dof_targets, joint_ids=self._arm_joint_ids)

    # =================================================================
    #  Observations / 観測
    # =================================================================

    def _get_observations(self) -> dict:
        """Compute the observation vector. / 観測ベクトルを計算する。

        Returns {"policy": obs} where obs is a (N, 15) tensor: / {"policy": obs}を返す。obsは(N, 15)テンソル：
          [joint_pos(6), joint_vel(6), target_pos(3)]
        """
        joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._arm_joint_ids]
        target = self._target_pos_b

        obs = torch.cat([joint_pos, joint_vel, target], dim=-1)
        return {"policy": obs}

    # =================================================================
    #  Rewards / 報酬
    # =================================================================

    def _get_rewards(self) -> torch.Tensor:
        """Compute distance-based reward. / 距離ベースの報酬を計算する。

        Reward components: / 報酬の構成：
          1. Linear distance penalty: encourages approaching the target / 線形距離ペナルティ：目標への接近を促す
          2. Tanh reward: high reward when very close / tanh報酬：非常に近い場合に高い報酬
          3. Action penalty: encourages smooth motion / 行動ペナルティ：滑らかな動作を促す
        """
        # End-effector position (world frame) / 末端執行器位置（ワールド座標）
        ee_pos_w = self._ee_frame.data.target_pos_w[..., 0, :]

        # Convert target position to world frame / 目標位置をワールド座標に変換する
        robot_pos_w = self._robot.data.root_state_w[:, :3]
        robot_quat_w = self._robot.data.root_state_w[:, 3:7]
        target_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos_w, robot_quat_w, self._target_pos_b
        )

        # Distance calculation / 距離計算
        dist = torch.norm(ee_pos_w - target_pos_w, p=2, dim=-1)

        # Reward components / 報酬の構成
        r_dist = self.cfg.reward_distance_weight * (-dist)
        r_tanh = self.cfg.reward_tanh_weight * (1.0 - torch.tanh(dist / self.cfg.reward_tanh_std))
        r_action = self.cfg.reward_action_penalty * torch.sum(self._actions ** 2, dim=-1)

        reward = r_dist + r_tanh + r_action

        # Statistics logging / 統計記録
        self._episode_sums["position_error"] += dist
        self._episode_sums["total_reward"] += reward

        return reward

    # =================================================================
    #  Termination Conditions / 終了条件
    # =================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine whether the episode has ended. / エピソードが終了したか判断する。

        terminated: End-effector reached the target (success) / 末端執行器が目標に到達した（成功）
        truncated:  Timeout / タイムアウト
        """
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        ee_pos_w = self._ee_frame.data.target_pos_w[..., 0, :]
        robot_pos_w = self._robot.data.root_state_w[:, :3]
        robot_quat_w = self._robot.data.root_state_w[:, 3:7]
        target_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos_w, robot_quat_w, self._target_pos_b
        )
        dist = torch.norm(ee_pos_w - target_pos_w, p=2, dim=-1)
        terminated = dist < self.cfg.success_threshold

        return terminated, truncated

    # =================================================================
    #  Reset / リセット
    # =================================================================

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset the specified environments. / 指定された環境をリセットする。"""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)
        n = len(env_ids)

        # Log episode statistics / エピソード統計を記録する
        if n > 0:
            extras = {}
            for key, buf in self._episode_sums.items():
                ep_len = self.episode_length_buf[env_ids].clamp(min=1)
                if "error" in key:
                    extras[f"Episode/{key}"] = torch.mean(buf[env_ids] / ep_len).item()
                else:
                    extras[f"Episode/{key}"] = torch.mean(buf[env_ids]).item()
                buf[env_ids] = 0.0
            self.extras["log"] = extras

        # Reset joint positions (with random perturbation) / 関節位置をリセットする（ランダム擾乱付き）
        default = torch.tensor(self.cfg.robot_default_joint_pos, device=self.device, dtype=torch.float32)
        joint_pos = default.unsqueeze(0).expand(n, -1).clone()
        joint_pos += torch.zeros_like(joint_pos).uniform_(
            -self.cfg.reset_joint_noise, self.cfg.reset_joint_noise
        )
        joint_vel = torch.zeros_like(joint_pos)

        self._robot.write_joint_state_to_sim(
            joint_pos, joint_vel, joint_ids=self._arm_joint_ids, env_ids=env_ids
        )

        # Sample new target positions randomly / 新しい目標位置をランダムにサンプリングする
        self._sample_targets(env_ids)

    def _sample_targets(self, env_ids: Sequence[int]):
        """Randomly sample target positions in the robot coordinate frame. / ロボット座標系でランダムに目標位置をサンプリングする。"""
        n = len(env_ids)
        x = torch.zeros(n, device=self.device).uniform_(*self.cfg.target_pos_range["x"])
        y = torch.zeros(n, device=self.device).uniform_(*self.cfg.target_pos_range["y"])
        z = torch.zeros(n, device=self.device).uniform_(*self.cfg.target_pos_range["z"])
        self._target_pos_b[env_ids] = torch.stack([x, y, z], dim=-1)

    # =================================================================
    #  Visualization / 可視化
    # =================================================================

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or toggle target position markers. / 目標位置マーカーを作成または切り替える。"""
        if debug_vis:
            if not hasattr(self, "_target_marker"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.04, 0.04, 0.04)
                marker_cfg.prim_path = "/Visuals/target_marker"
                self._target_marker = VisualizationMarkers(marker_cfg)
            self._target_marker.set_visibility(True)
        else:
            if hasattr(self, "_target_marker"):
                self._target_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update target marker positions. / 目標マーカーの位置を更新する。"""
        robot_pos_w = self._robot.data.root_state_w[:, :3]
        robot_quat_w = self._robot.data.root_state_w[:, 3:7]
        target_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos_w, robot_quat_w, self._target_pos_b
        )
        self._target_marker.visualize(target_pos_w)

    # =================================================================
    #  Convenience Methods (for manual control scripts) / 便利メソッド（手動制御スクリプト用）
    # =================================================================

    def get_ee_pos_world(self) -> torch.Tensor:
        """Get end-effector position in world coordinates (N, 3). / 末端執行器のワールド座標位置を取得する (N, 3)。"""
        return self._ee_frame.data.target_pos_w[..., 0, :].clone()

    def get_target_pos_world(self) -> torch.Tensor:
        """Get target position in world coordinates (N, 3). / 目標のワールド座標位置を取得する (N, 3)。"""
        robot_pos_w = self._robot.data.root_state_w[:, :3]
        robot_quat_w = self._robot.data.root_state_w[:, 3:7]
        pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos_w, robot_quat_w, self._target_pos_b
        )
        return pos_w.clone()

    def get_joint_positions(self) -> torch.Tensor:
        """Get current angles of the 6 arm joints (N, 6). / 6つのアームジョイントの現在角度を取得する (N, 6)。"""
        return self._robot.data.joint_pos[:, self._arm_joint_ids].clone()

    def get_ee_quat_world(self) -> torch.Tensor:
        """Get end-effector orientation (quaternion) in world frame (N, 4). / 末端のワールド座標姿勢（クォータニオン）を取得する (N, 4)。"""
        return self._ee_frame.data.target_quat_w[..., 0, :].clone()

    def compute_ik(self, target_pos_w: torch.Tensor) -> torch.Tensor:
        """Compute joint positions to reach a target EE position using Differential IK.
        微分IKを使用して目標EE位置に到達するための関節位置を計算する。

        Args:
            target_pos_w: Target end-effector position in world frame (N, 3).
                          ワールド座標系での目標エンドエフェクタ位置 (N, 3)。

        Returns:
            Desired joint positions (N, 6). / 目標関節位置 (N, 6)。
        """
        # Get current EE state / 現在のEE状態を取得する
        ee_pos_w = self.get_ee_pos_world()
        ee_quat_w = self.get_ee_quat_world()

        # Transform target to robot base frame / 目標をロボットベースフレームに変換する
        robot_pos_w = self._robot.data.root_state_w[:, :3]
        robot_quat_w = self._robot.data.root_state_w[:, 3:7]
        target_pos_b, _ = math_utils.subtract_frame_transforms(
            robot_pos_w, robot_quat_w, target_pos_w
        )
        # Transform EE to robot base frame / EEをロボットベースフレームに変換する
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            robot_pos_w, robot_quat_w, ee_pos_w, ee_quat_w
        )

        # Set IK target (position-only, keeps current orientation) / IK目標を設定する（位置のみ、現在の姿勢を維持）
        self._ik_controller.set_command(target_pos_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

        # Get Jacobian from PhysX (not available until after first sim step) / PhysXからヤコビアンを取得する（最初のシミュレーションステップ後にのみ利用可能）
        jacobian_w = self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._arm_joint_ids]
        # Transform Jacobian to base frame / ヤコビアンをベースフレームに変換する
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(robot_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

        # Compute desired joint positions / 目標関節位置を計算する
        joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        joint_pos_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos)

        # Clamp to joint limits / 関節限界にクランプする
        joint_pos_des = torch.clamp(joint_pos_des, self._joint_lower, self._joint_upper)

        return joint_pos_des
