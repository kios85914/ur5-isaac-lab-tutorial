"""Environment registration -- registers the UR5 Reach environment in gymnasium. / 環境登録 -- UR5 Reach 環境を gymnasium に登録する。

This file is automatically executed when importing ur5_lab, registering the environment in gymnasium. / ur5_lab をインポートすると自動的に実行され、環境を gymnasium に登録する。
Afterwards the environment can be created with gym.make("Isaac-UR5-Reach-v0"). / その後、gym.make("Isaac-UR5-Reach-v0") で環境を作成できる。
"""

import os
import gymnasium as gym

# Get the path to YAML configuration files under the agents/ directory / agents/ ディレクトリ下の YAML 設定ファイルパスを取得
_AGENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "agents")

gym.register(
    id="Isaac-UR5-Reach-v0",
    entry_point="ur5_lab.tasks.ur5_reach_env:UR5ReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "ur5_lab.tasks.ur5_reach_env:UR5ReachEnvCfg",
        "skrl_cfg_entry_point": os.path.join(_AGENTS_DIR, "ppo_ur5.yaml"),
    },
)
