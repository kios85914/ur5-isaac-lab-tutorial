# UR5 ロボットアーム制御チュートリアル

> **[English](../README.md)** | **[中文版 (Chinese)](README_zh.md)** | **日本語**

NVIDIA Isaac Lab をベースとした、UR5 ロボットアームの制御と強化学習 (RL) のための簡略化された教育環境です。

---

## 概要

**目標**：UR5 の 6 つの関節を制御し、エンドエフェクタをランダムに生成された目標位置に移動させる。

| 項目 | 説明 |
|------|------|
| 観測空間 | 15 次元：関節角度 (6) + 関節速度 (6) + 目標位置 (3) |
| 行動空間 | 6 次元：関節位置デルタ |
| 報酬 | 距離ベース + tanh ボーナス + 行動ペナルティ |
| 成功条件 | エンドエフェクタが目標から 2cm 以内 |

---

## 環境要件

| ソフトウェア | バージョン |
|-------------|-----------|
| NVIDIA Isaac Sim | 4.5+ |
| Isaac Lab | 2.2+ |
| Python | 3.10+ |
| GPU | RTX 2070 以上（RTX 3080+ 推奨）|

---

## インストール手順

```bash
# 1. Isaac Lab の Python 環境を有効化
conda activate isaaclab

# 2. 教育パッケージをインストール
cd ur5_teaching
pip install -e .

# 3. インストールを確認
python -c "import ur5_lab; print('OK')"
```

---

## クイックスタート：ロボット制御

```bash
python scripts/simple_demo.py
```

`scripts/simple_demo.py` の `YOUR CODE HERE` セクションを編集してロボットを制御：

| 関数 | 説明 | 例 |
|------|------|-----|
| `move_joints([j1,...,j6])` | 目標関節角度に移動 (rad) | `move_joints([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])` |
| `move_to_ee([x, y, z])` | IK でエンドエフェクタをワールド位置へ移動 | `move_to_ee([0.5, 0.0, 1.3])` |
| `get_state()` | 現在の関節角度 + EE 位置を取得 | `print(get_state())` |
| `home()` | ホームポジションに戻る | `home()` |

---

## RL 訓練

```bash
# 基本訓練
python scripts/train_ppo.py --task Isaac-UR5-Reach-v0 --num_envs 64

# GPU メモリ不足の場合
python scripts/train_ppo.py --task Isaac-UR5-Reach-v0 --num_envs 16

# チェックポイントから再開
python scripts/train_ppo.py --task Isaac-UR5-Reach-v0 --checkpoint logs/skrl/.../best_agent.pt
```

訓練ログは `logs/skrl/ur5_reach/` に保存されます。

### PPO ハイパーパラメータ (`agents/ppo_ur5.yaml`)

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| learning_rate | 3e-4 | 学習率 |
| rollouts | 24 | 更新前のステップ数 |
| learning_epochs | 8 | 更新ごとのエポック数 |
| discount_factor | 0.99 | 割引率 |
| timesteps | 150000 | 総訓練ステップ数 |

---

## プロジェクト構成

```
ur5_teaching/
├── README.md                       ← 英語版（メイン）
├── docs/
│   ├── README_ja.md                ← 日本語版（本ファイル）
│   └── README_zh.md                ← 中国語版
│
├── ur5_lab/                        ← メインパッケージ
│   ├── tasks/
│   │   ├── __init__.py             ← gym.register("Isaac-UR5-Reach-v0")
│   │   └── ur5_reach_env.py        ← 核心：環境定義（観測/行動/報酬/リセット/IK）
│   └── assets/
│       ├── ur5_cfg.py              ← UR5 ロボット物理パラメータ設定
│       └── ur5_moveit.usd          ← UR5 3D モデル
│
├── scripts/
│   ├── simple_demo.py              ← 直接制御デモ（関節 + IK）
│   └── train_ppo.py                ← PPO 訓練スクリプト
│
└── agents/
    └── ppo_ur5.yaml                ← PPO ハイパーパラメータ
```

---

## 学習ロードマップ

### ステップ 1：ロボットを制御する
1. `simple_demo.py` を実行し、ロボットがデモ動作を行うのを観察する
2. エディタでスクリプトを開き、`YOUR CODE HERE` セクションを見つける
3. `move_joints()` の引数を変更し、再実行して違いを観察する
4. `move_to_ee()` で異なる XYZ 座標を試す

### ステップ 2：環境の理解
5. `ur5_reach_env.py` を読み、観測/行動/報酬の定義を理解する
6. 報酬関数の重みを変更し、訓練行動の変化を観察する
7. `target_pos_range` を変更してタスクの難易度を調整する

### ステップ 3：RL 訓練
8. PPO で訓練を開始し、報酬カーブを観察する
9. ハイパーパラメータ（learning_rate、rollouts など）を調整する
10. 異なる設定で結果を比較する

### ステップ 4：応用課題
11. エンドエフェクタの姿勢（orientation）追跡を報酬に追加する
12. 位置制御から速度制御に切り替える
13. シーンに障害物を追加する
