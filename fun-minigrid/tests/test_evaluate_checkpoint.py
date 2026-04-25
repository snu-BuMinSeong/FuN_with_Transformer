from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_checkpoint import build_model_from_config, evaluate_checkpoint
from src.utils.checkpoint import save_checkpoint


def test_evaluate_checkpoint_loads_saved_model_and_writes_json(tmp_path: Path) -> None:
    config = {
        "env_id": "MiniGrid-DoorKey-5x5-v0",
        "render_mode": None,
        "seed": 1,
        "max_steps": 1,
        "eval_episodes": 1,
        "eval_seed_offset": 1000,
        "goal_update_interval": 10,
        "hidden_dim": 64,
        "goal_size": 16,
        "num_actions": 7,
    }
    config_path = tmp_path / "eval_config.yaml"
    config_path.write_text(
        "\n".join(f"{key}: {'null' if value is None else value}" for key, value in config.items()),
        encoding="utf-8",
    )

    torch.manual_seed(0)
    model = build_model_from_config(config)
    checkpoint_path = tmp_path / "last.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        episode=7,
        config=config,
        best_success_rate=0.5,
    )

    output_path = tmp_path / "checkpoint_eval.json"
    result = evaluate_checkpoint(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        episodes=1,
        seed_offset=10,
        action_mode="argmax",
    )

    assert output_path.exists()
    assert result["checkpoint_episode"] == 7
    assert result["checkpoint_best_success_rate"] == 0.5
    assert result["eval_action_mode"] == "argmax"
    assert result["eval_episodes"] == 1
    assert result["eval_result"]["num_episodes"] == 1
