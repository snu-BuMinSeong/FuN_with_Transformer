from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_simple_yaml


def test_load_simple_yaml_train_config() -> None:
    config = load_simple_yaml(PROJECT_ROOT / "configs" / "train_fun.yaml")

    print("[Config Load]")
    print("seed:", config["seed"])
    print("total_episodes:", config["total_episodes"])
    print("learning_rate:", config["learning_rate"])
    print("goal_update_interval:", config["goal_update_interval"])
    print("hidden_dim:", config["hidden_dim"])

    assert config["seed"] == 123
    assert config["total_episodes"] == 10
    assert config["learning_rate"] == 0.001
    assert config["goal_update_interval"] == 10
    assert config["hidden_dim"] == 64


if __name__ == "__main__":
    test_load_simple_yaml_train_config()
    print("\nAll config tests passed.")
