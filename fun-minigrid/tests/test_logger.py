from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import append_training_log, write_json_summary


def test_append_training_log_writes_expected_fields() -> None:
    log_path = PROJECT_ROOT / "logs" / "test_training_log.csv"
    if log_path.exists():
        log_path.unlink()

    append_training_log(
        str(log_path),
        {
            "episode": 1,
            "total_reward": 0.0,
            "episode_length": 10,
            "success": False,
            "total_loss": -0.01,
            "worker_loss": 0.0,
            "value_loss": 0.25,
            "entropy_bonus": 1.0,
            "manager_loss": 0.0,
            "grad_norm": 0.1,
            "encoder_grad_norm": 0.01,
            "manager_grad_norm": 0.02,
            "worker_grad_norm": 0.03,
            "value_head_grad_norm": 0.04,
            "manager_value_head_grad_norm": 0.05,
            "returns_mean": 0.0,
            "returns_min": 0.0,
            "returns_max": 0.0,
            "advantages_mean": 0.0,
            "advantages_abs_mean": 0.0,
            "values_mean": 0.2,
            "value_min": -0.1,
            "value_max": 0.3,
            "manager_values_mean": 0.1,
            "manager_value_min": -0.2,
            "manager_value_max": 0.2,
            "log_prob_mean": -1.0,
            "log_prob_min": -1.2,
            "log_prob_max": -0.8,
            "entropy_mean": 1.0,
            "entropy_min": 0.9,
            "entropy_max": 1.1,
            "num_steps": 10,
            "action_min": 0,
            "action_max": 6,
            "action_coverage": 0.7,
            "action_histogram": "1,2,3,4,0,0,0",
            "reward_min": 0.0,
            "reward_max": 0.0,
            "nonzero_reward_steps": 0,
            "positive_reward_steps": 0,
            "nonzero_reward_fraction": 0.0,
            "nonzero_return_steps": 0,
            "nonzero_return_fraction": 0.0,
            "has_reward_signal": False,
            "has_return_signal": False,
            "num_goal_updates": 1,
            "final_hidden_norm": 0.5,
            "final_goal_norm": 0.3,
            "final_step_count": 10,
            "reward_moving_avg": 0.0,
            "success_moving_avg": 0.0,
            "loss_moving_avg": -0.01,
        },
    )

    with log_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    print("[Training Logger CSV]")
    print("rows:", len(rows))
    print("columns:", list(rows[0].keys()))

    assert len(rows) == 1
    assert rows[0]["episode"] == "1"
    assert rows[0]["episode_length"] == "10"
    assert rows[0]["action_max"] == "6"
    assert rows[0]["reward_moving_avg"] == "0.0"
    assert rows[0]["encoder_grad_norm"] == "0.01"
    assert rows[0]["entropy_mean"] == "1.0"


def test_write_json_summary_writes_expected_payload() -> None:
    summary_path = PROJECT_ROOT / "logs" / "test_training_summary.json"
    if summary_path.exists():
        summary_path.unlink()

    payload = {
        "seed": 123,
        "num_episodes": 2,
        "final_reward_moving_avg": 0.0,
        "results": [{"episode": 1}, {"episode": 2}],
    }
    write_json_summary(str(summary_path), payload)

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))

    print("\n[Training Logger JSON]")
    print("seed:", loaded["seed"])
    print("num_episodes:", loaded["num_episodes"])
    print("results_len:", len(loaded["results"]))

    assert loaded["seed"] == 123
    assert loaded["num_episodes"] == 2
    assert len(loaded["results"]) == 2


def test_write_json_summary_omits_verbose_episode_fields() -> None:
    summary_path = PROJECT_ROOT / "logs" / "test_compact_training_summary.json"
    if summary_path.exists():
        summary_path.unlink()

    payload = {
        "seed": 123,
        "eval": {
            "episode_seeds": [2001, 2002],
            "episode_results": [
                {"episode": 1, "total_reward": 1.0, "actions": [0, 1, 2]},
                {"episode": 2, "total_reward": 0.0, "actions": [3, 4]},
            ],
        },
    }
    write_json_summary(str(summary_path), payload)

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "episode_seeds" not in loaded["eval"]
    assert loaded["eval"]["episode_results"][0]["total_reward"] == 1.0
    assert "actions" not in loaded["eval"]["episode_results"][0]
    assert "actions" in payload["eval"]["episode_results"][0]


if __name__ == "__main__":
    test_append_training_log_writes_expected_fields()
    test_write_json_summary_writes_expected_payload()
    test_write_json_summary_omits_verbose_episode_fields()
    print("\nAll logger tests passed.")
