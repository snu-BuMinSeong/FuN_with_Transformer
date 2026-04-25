from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_simple_yaml


def test_load_simple_yaml_train_config() -> None:
    config = load_simple_yaml(PROJECT_ROOT / "configs" / "train_fun.yaml")

    required_keys = [
        "seed",
        "total_episodes",
        "learning_rate",
        "goal_update_interval",
        "hidden_dim",
    ]

    for key in required_keys:
        assert key in config

    assert isinstance(config["seed"], int)
    assert isinstance(config["total_episodes"], int)
    assert isinstance(config["learning_rate"], (int, float))
    assert isinstance(config["goal_update_interval"], int)
    assert isinstance(config["hidden_dim"], int)

    assert config["total_episodes"] > 0
    assert config["learning_rate"] > 0
    assert config["goal_update_interval"] > 0
    assert config["hidden_dim"] > 0


if __name__ == "__main__":
    test_load_simple_yaml_train_config()
    print("\nAll config tests passed.")
