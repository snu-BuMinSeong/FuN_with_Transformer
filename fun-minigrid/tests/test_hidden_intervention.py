from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from diagnose_hidden_intervention import HiddenInterventionPolicy
from src.models.fun import FuNModel


def _policy(intervention: str) -> HiddenInterventionPolicy:
    torch.manual_seed(0)
    model = FuNModel(goal_update_interval=3, hidden_dim=8, goal_size=4, num_actions=7)
    return HiddenInterventionPolicy(
        model=model,
        preprocess_fn=lambda _: np.zeros((3, 7, 7), dtype=np.float32),
        device="cpu",
        action_mode="argmax",
        intervention=intervention,
    )


def test_reset_goal_resets_only_on_goal_update_steps() -> None:
    policy = _policy("reset_goal")
    for _ in range(5):
        policy.act_for_training(None)

    debug = policy.get_intervention_debug_info()
    assert debug["episode_steps"] == 5
    assert debug["episode_goal_updates"] == 2
    assert debug["episode_hidden_resets"] == 2


def test_reset_step_resets_every_step() -> None:
    policy = _policy("reset_step")
    for _ in range(5):
        policy.act_for_training(None)

    debug = policy.get_intervention_debug_info()
    assert debug["episode_steps"] == 5
    assert debug["episode_goal_updates"] == 2
    assert debug["episode_hidden_resets"] == 5


def test_normal_does_not_apply_extra_hidden_resets() -> None:
    policy = _policy("normal")
    for _ in range(5):
        policy.act_for_training(None)

    debug = policy.get_intervention_debug_info()
    assert debug["episode_steps"] == 5
    assert debug["episode_goal_updates"] == 2
    assert debug["episode_hidden_resets"] == 0
