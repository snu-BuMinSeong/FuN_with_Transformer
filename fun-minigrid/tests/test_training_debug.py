from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.make_env import make_env
from src.envs.preprocess import preprocess_obs
from src.models.fun import FuNModel
from src.policies.fun_policy import FuNPolicy
from src.training.trainer import train, train_one_episode


def make_policy(seed: int = 0, action_mode: str = "sample", goal_update_interval: int = 10) -> FuNPolicy:
    torch.manual_seed(seed)
    model = FuNModel(goal_update_interval=goal_update_interval)
    return FuNPolicy(model=model, preprocess_fn=preprocess_obs, action_mode=action_mode)


def test_train_one_episode_debug_fields() -> None:
    env = make_env(seed=42)
    policy = make_policy(seed=0, action_mode="sample", goal_update_interval=3)
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=1e-3)

    result = train_one_episode(
        env=env,
        policy=policy,
        optimizer=optimizer,
        gamma=1.0,
        max_steps=7,
        seed=42,
        entropy_coef=0.01,
        grad_clip_norm=1.0,
    )

    print("[Training Debug One Episode]")
    print("action_min:", result["action_min"])
    print("action_max:", result["action_max"])
    print("reward_min:", result["reward_min"])
    print("reward_max:", result["reward_max"])
    print("values_mean:", result["values_mean"])
    print("goal_shape:", result["goal_shape"])
    print("hidden_state_shape:", result["hidden_state_shape"])
    print("current_goal_shape:", result["current_goal_shape"])
    print("num_goal_updates:", result["num_goal_updates"])
    print("final_step_count:", result["final_step_count"])
    print("final_hidden_norm:", result["final_hidden_norm"])
    print("final_goal_norm:", result["final_goal_norm"])

    assert 0 <= result["action_min"] <= 6
    assert 0 <= result["action_max"] <= 6
    assert result["reward_min"] <= result["reward_max"]
    assert isinstance(result["values_mean"], float)
    assert result["goal_shape"] == (16,)
    assert result["hidden_state_shape"] == (1, 64)
    assert result["current_goal_shape"] == (1, 16)
    assert result["num_goal_updates"] == 3
    assert result["final_step_count"] == 7
    assert result["final_hidden_norm"] >= 0.0
    assert result["final_goal_norm"] >= 0.0

    env.close()


def test_train_ten_episodes_short_run_is_stable() -> None:
    env = make_env(seed=42)
    policy = make_policy(seed=123, action_mode="sample", goal_update_interval=5)
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=1e-3)

    results = train(
        env=env,
        policy=policy,
        optimizer=optimizer,
        num_episodes=10,
        gamma=1.0,
        max_steps=10,
        seed=42,
        entropy_coef=0.01,
        grad_clip_norm=1.0,
    )

    print("\n[Training Debug Ten Episodes]")
    print("num_results:", len(results))
    print("action_mins:", [result["action_min"] for result in results])
    print("action_maxs:", [result["action_max"] for result in results])
    print("goal_updates:", [result["num_goal_updates"] for result in results])
    print("lengths:", [result["episode_length"] for result in results])

    assert len(results) == 10
    assert all(result["episode_length"] == 10 for result in results)
    assert all(0 <= result["action_min"] <= 6 for result in results)
    assert all(0 <= result["action_max"] <= 6 for result in results)
    assert all(result["reward_min"] <= result["reward_max"] for result in results)
    assert all(result["hidden_state_shape"] == (1, 64) for result in results)
    assert all(result["current_goal_shape"] == (1, 16) for result in results)
    assert all(result["num_goal_updates"] == 2 for result in results)
    assert all(result["final_step_count"] == 10 for result in results)

    env.close()


if __name__ == "__main__":
    test_train_one_episode_debug_fields()
    test_train_ten_episodes_short_run_is_stable()
    print("\nAll training debug tests passed.")
