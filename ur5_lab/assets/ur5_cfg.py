"""UR5 Robot Configuration File / UR5 ロボット設定ファイル

Defines the ArticulationCfg for UR5 arm + Robotiq 85 gripper. / UR5 マニピュレータ + Robotiq 85 グリッパーの ArticulationCfg を定義する。
Includes: / 内容：
  - USD model path / USD モデルパス
  - Physics properties (rigid body, joints) / 物理特性（剛体、関節）
  - Initial joint angles (home position) / 初期関節角度（ホームポジション）
  - Actuator parameters (stiffness, damping) / アクチュエータパラメータ（剛性、減衰）
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# USD path -- resolved relative to this file, unaffected by working directory / このファイルからの相対パスで解決し、作業ディレクトリに依存しない
_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join(_ASSETS_DIR, "ur5_moveit.usd")

# UR5 + Robotiq-85 configuration / UR5 + Robotiq-85 設定
UR5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
            disable_gravity=True,
        ),
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Robot base height (approx. table height 1.05m) / ロボットベース高さ（約テーブル高さ 1.05m）
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            # Home position -- arm extended forward / ホームポジション -- アームを前方に伸ばした姿勢
            "shoulder_pan_joint": -0.568,
            "shoulder_lift_joint": -0.658,
            "elbow_joint": 1.602,
            "wrist_1_joint": -2.585,
            "wrist_2_joint": -1.606,
            "wrist_3_joint": -1.641,
            # Gripper open / グリッパー開放
            "robotiq_85_left_knuckle_joint": 0.0,
        },
    ),
    actuators={
        # 6-DOF arm / 6自由度アーム
        "arm_actuator": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            velocity_limit_sim=50.0,
            effort_limit_sim=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        # 1-DOF gripper / 1自由度グリッパー
        "gripper_actuator": ImplicitActuatorCfg(
            joint_names_expr=["robotiq_85_left_knuckle_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
)
