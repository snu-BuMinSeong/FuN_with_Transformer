from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analyze_manager_goals import _pearson, aggregate_episode_rows


def test_pearson_detects_positive_and_negative_relationships() -> None:
    assert _pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0
    assert _pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == -1.0


def test_aggregate_episode_rows_groups_by_model_and_action_mode() -> None:
    rows = [
        {
            "model_type": "baseline",
            "action_mode": "sample",
            "success": True,
            "return": 1.0,
            "episode_length": 10,
            "num_goal_updates": 2,
            "mean_goal_norm": 1.0,
            "mean_goal_delta_l2": 0.5,
            "mean_goal_cosine_prev": 0.8,
            "mean_policy_entropy": 0.4,
            "mean_top1_prob": 0.7,
            "mean_chosen_action_prob": 0.6,
            "mean_action_kl_on_goal_update": 0.2,
            "top1_action_change_rate_on_goal_update": 0.5,
        },
        {
            "model_type": "baseline",
            "action_mode": "sample",
            "success": False,
            "return": 0.0,
            "episode_length": 20,
            "num_goal_updates": 3,
            "mean_goal_norm": 3.0,
            "mean_goal_delta_l2": 1.5,
            "mean_goal_cosine_prev": 0.6,
            "mean_policy_entropy": 0.8,
            "mean_top1_prob": 0.5,
            "mean_chosen_action_prob": 0.4,
            "mean_action_kl_on_goal_update": 0.4,
            "top1_action_change_rate_on_goal_update": 1.0,
        },
    ]

    aggregate = aggregate_episode_rows(rows)

    assert len(aggregate) == 1
    assert aggregate[0]["model_type"] == "baseline"
    assert aggregate[0]["action_mode"] == "sample"
    assert aggregate[0]["mean_success"] == 0.5
    assert aggregate[0]["mean_episode_length"] == 15.0
    assert aggregate[0]["mean_goal_delta_l2"] == 1.0
