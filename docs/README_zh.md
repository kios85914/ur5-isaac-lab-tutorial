# UR5 機器人控制教學專案

> **[English](../README.md)** | **[日本語版 (Japanese)](README_ja.md)** | **中文**

本專案是一個精簡版的 UR5 機械手臂控制與強化學習 (RL) 教學環境，基於 NVIDIA Isaac Lab。

---

## 專案簡介

**目標**：控制 UR5 的 6 個關節，讓末端執行器移動到隨機生成的目標位置。

| 項目 | 說明 |
|------|------|
| 觀測空間 | 15 維：關節角度(6) + 關節速度(6) + 目標位置(3) |
| 動作空間 | 6 維：關節位置增量 (delta) |
| 獎勵 | 基於末端與目標距離 + tanh 獎勵 + 動作懲罰 |
| 成功條件 | 末端距離目標 < 2cm |

---

## 環境需求

| 軟體 | 版本 |
|------|------|
| NVIDIA Isaac Sim | 4.5+ |
| Isaac Lab | 2.2+ |
| Python | 3.10+ |
| GPU | RTX 2070 以上（建議 RTX 3080+）|

---

## 安裝步驟

```bash
# 1. 啟動 Isaac Lab 的 Python 環境
conda activate isaaclab

# 2. 安裝本教學套件
cd ur5_teaching
pip install -e .

# 3. 確認安裝成功
python -c "import ur5_lab; print('OK')"
```

---

## 快速開始：手動控制

```bash
python scripts/manual_joint_control.py --num_envs 1
```

啟動後可以輸入以下指令：

| 指令 | 說明 |
|------|------|
| `6個數字` | 設定關節目標角度（弧度），例如 `-0.5 -1.0 1.5 -2.0 -1.5 0.0` |
| `ee X Y Z` | 用 IK 移動末端到世界座標位置，例如 `ee 0.5 0.0 1.2` |
| `home` | 回到家位置 |
| `random` | 隨機動作 |
| `info` | 顯示當前關節角度、末端位置、目標位置、距離 |
| `quit` | 離開 |

---

## RL 訓練

```bash
# 基本訓練
python scripts/train_ppo.py --task Isaac-UR5-Reach-v0 --num_envs 64

# GPU 記憶體不足時
python scripts/train_ppo.py --task Isaac-UR5-Reach-v0 --num_envs 16

# 從 checkpoint 繼續
python scripts/train_ppo.py --task Isaac-UR5-Reach-v0 --checkpoint logs/skrl/.../best_agent.pt
```

訓練日誌存在 `logs/skrl/ur5_reach/`。

### PPO 超參數 (`agents/ppo_ur5.yaml`)

| 參數 | 值 | 說明 |
|------|-----|------|
| learning_rate | 3e-4 | 學習率 |
| rollouts | 24 | 每次更新前的步數 |
| learning_epochs | 8 | 每次更新的訓練輪數 |
| discount_factor | 0.99 | 折扣因子 |
| timesteps | 150000 | 總訓練步數 |

---

## 專案結構

```
ur5_teaching/
├── README.md                       ← 英文版（主頁）
├── docs/
│   ├── README_ja.md                ← 日文版
│   └── README_zh.md                ← 中文版（本文件）
│
├── ur5_lab/                        ← 主程式碼
│   ├── tasks/
│   │   ├── __init__.py             ← gym.register("Isaac-UR5-Reach-v0")
│   │   └── ur5_reach_env.py        ← 核心：環境定義（觀測/動作/獎勵/重置/IK）
│   └── assets/
│       ├── ur5_cfg.py              ← UR5 機器人物理參數設定
│       └── ur5_moveit.usd          ← UR5 3D 模型
│
├── scripts/
│   ├── manual_joint_control.py     ← 手動控制（關節 + IK）
│   └── train_ppo.py                ← PPO 訓練腳本
│
└── agents/
    └── ppo_ur5.yaml                ← PPO 超參數設定
```

---

## 學習路線

### 第一步：觀察
1. 執行 `manual_joint_control.py`，在 Isaac Sim 中觀察機器人
2. 用 `info` 指令查看關節角度和末端位置
3. 試著輸入不同的關節角度，觀察機器人的動作
4. 用 `ee X Y Z` 進行 IK 控制，比較與關節控制的差異

### 第二步：理解環境
5. 閱讀 `ur5_reach_env.py`，理解觀測/動作/獎勵的定義
6. 修改獎勵函數的權重，觀察訓練行為的變化
7. 修改 `target_pos_range` 改變任務難度

### 第三步：RL 訓練
8. 用 PPO 開始訓練，觀察獎勵曲線
9. 調整超參數（learning_rate、rollouts 等）
10. 比較不同設定的訓練效果

### 第四步：進階挑戰
11. 加入末端方向 (orientation) 追蹤
12. 改用速度控制而非位置控制
13. 加入桌面障礙物
