# UR5 Robot Control Teaching Project

> **[日本語版 (Japanese)](docs/README_ja.md)** | **[中文版 (Chinese)](docs/README_zh.md)** | **English**

A simplified UR5 robot arm control & reinforcement learning (RL) teaching environment built on NVIDIA Isaac Lab.

---

## Overview

**Goal**: Control the UR5's 6 joints to move the end-effector to randomly generated target positions.

| Item | Description |
|------|-------------|
| Observation Space | 15D: joint angles (6) + joint velocities (6) + target position (3) |
| Action Space | 6D: joint position deltas |
| Reward | Distance-based + tanh bonus + action penalty |
| Success | End-effector within 2cm of target |

---

## Requirements

| Software | Version |
|----------|---------|
| NVIDIA Isaac Sim | 4.5+ |
| Isaac Lab | 2.2+ |
| Python | 3.10+ |
| GPU | RTX 2070+ (RTX 3080+ recommended) |

---

## Installation

```bash
# 1. Activate Isaac Lab Python environment
conda activate isaaclab

# 2. Install this teaching package
cd ur5_teaching
pip install -e .

# 3. Verify installation
python -c "import ur5_lab; print('OK')"
```

---

## Quick Start: Manual Control

```bash
python scripts/manual_joint_control.py --num_envs 1
```

Available commands:

| Command | Description |
|---------|-------------|
| `6 numbers` | Set joint target angles (radians), e.g. `-0.5 -1.0 1.5 -2.0 -1.5 0.0` |
| `ee X Y Z` | Move end-effector to world position via IK, e.g. `ee 0.5 0.0 1.2` |
| `home` | Return to home position |
| `random` | Execute random motion |
| `info` | Show current joint angles, EE position, target position, distance |
| `quit` | Exit |

---

## RL Training

```bash
# Basic training
python scripts/train_ppo.py --task Isaac-UR5-Reach-v0 --num_envs 64

# Low GPU memory
python scripts/train_ppo.py --task Isaac-UR5-Reach-v0 --num_envs 16

# Resume from checkpoint
python scripts/train_ppo.py --task Isaac-UR5-Reach-v0 --checkpoint logs/skrl/.../best_agent.pt
```

Training logs are saved to `logs/skrl/ur5_reach/`.

### PPO Hyperparameters (`agents/ppo_ur5.yaml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| learning_rate | 3e-4 | Learning rate |
| rollouts | 24 | Steps per env before update |
| learning_epochs | 8 | Epochs per update |
| discount_factor | 0.99 | Discount factor |
| timesteps | 150000 | Total training steps |

---

## Project Structure

```
ur5_teaching/
├── README.md                       <- This file
├── docs/
│   ├── README_ja.md                <- Japanese version
│   └── README_zh.md                <- Chinese version
│
├── ur5_lab/                        <- Main package
│   ├── tasks/
│   │   ├── __init__.py             <- gym.register("Isaac-UR5-Reach-v0")
│   │   └── ur5_reach_env.py        <- Core: env definition (obs/action/reward/reset/IK)
│   └── assets/
│       ├── ur5_cfg.py              <- UR5 robot physics config
│       └── ur5_moveit.usd          <- UR5 3D model
│
├── scripts/
│   ├── manual_joint_control.py     <- Manual control (joint + IK)
│   └── train_ppo.py                <- PPO training
│
└── agents/
    └── ppo_ur5.yaml                <- PPO hyperparameters
```

---

## Learning Path

### Step 1: Observe
1. Run `manual_joint_control.py`, observe the robot in Isaac Sim
2. Use `info` to check joint angles and EE position
3. Try different joint angles, observe the robot's motion
4. Use `ee X Y Z` to control via IK, compare with joint control

### Step 2: Understand the Environment
4. Read `ur5_reach_env.py`, understand observation/action/reward definitions
5. Modify reward weights, observe changes in training behavior
6. Modify `target_pos_range` to change task difficulty

### Step 3: RL Training
7. Train with PPO, observe the reward curve
8. Tune hyperparameters (learning_rate, rollouts, etc.)
9. Compare results with different settings

### Step 4: Advanced Challenges
10. Add orientation tracking to the reward
11. Switch to velocity control instead of position control
12. Add obstacles to the scene
