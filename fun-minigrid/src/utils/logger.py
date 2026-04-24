from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


LOG_FIELDS = [
    "episode",
    "total_reward",
    "episode_length",
    "success",
    "terminated",
    "truncated",
]

TRAIN_LOG_FIELDS = [
    "episode",
    "total_reward",
    "episode_length",
    "success",
    "total_loss",
    "worker_loss",
    "value_loss",
    "entropy_bonus",
    "manager_loss",
    "grad_norm",
    "encoder_grad_norm",
    "manager_grad_norm",
    "worker_grad_norm",
    "value_head_grad_norm",
    "manager_value_head_grad_norm",
    "returns_mean",
    "returns_min",
    "returns_max",
    "advantages_mean",
    "advantages_abs_mean",
    "values_mean",
    "value_min",
    "value_max",
    "manager_values_mean",
    "manager_value_min",
    "manager_value_max",
    "log_prob_mean",
    "log_prob_min",
    "log_prob_max",
    "entropy_mean",
    "entropy_min",
    "entropy_max",
    "num_steps",
    "action_min",
    "action_max",
    "action_coverage",
    "action_histogram",
    "reward_min",
    "reward_max",
    "nonzero_reward_steps",
    "positive_reward_steps",
    "nonzero_reward_fraction",
    "nonzero_return_steps",
    "nonzero_return_fraction",
    "has_reward_signal",
    "has_return_signal",
    "num_goal_updates",
    "final_hidden_norm",
    "final_goal_norm",
    "final_step_count",
    "reward_moving_avg",
    "success_moving_avg",
    "loss_moving_avg",
]


def ensure_log_dir(log_path: str) -> None:
    """Create the parent directory for a log file if needed."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)


def append_episode_log(log_path: str, episode_result: dict[str, Any]) -> None:
    """Append one episode result to a CSV log file."""
    ensure_log_dir(log_path)

    path = Path(log_path)
    write_header = not path.exists() or path.stat().st_size == 0
    row = {field: episode_result.get(field, "") for field in LOG_FIELDS}

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_training_log(log_path: str, episode_result: dict[str, Any]) -> None:
    """Append one training episode result to a CSV log file."""
    ensure_log_dir(log_path)

    path = Path(log_path)
    write_header = not path.exists() or path.stat().st_size == 0
    row = {field: episode_result.get(field, "") for field in TRAIN_LOG_FIELDS}

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAIN_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_json_summary(path: str, payload: dict[str, Any]) -> None:
    """Write a compact JSON summary file for one training run."""
    ensure_log_dir(path)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
