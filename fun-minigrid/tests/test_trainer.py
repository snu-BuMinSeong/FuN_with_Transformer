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


def make_policy(seed: int = 0, action_mode: str = "sample") -> FuNPolicy:
    torch.manual_seed(seed)
    model = FuNModel(goal_update_interval=10)
    return FuNPolicy(model=model, preprocess_fn=preprocess_obs, action_mode=action_mode)


def test_train_one_episode_runs_optimizer_step() -> None:
    env = make_env(seed=42)
    policy = make_policy(seed=0, action_mode="sample")
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=1e-3)
    before_params = [param.detach().clone() for param in policy.model.parameters()]

    result = train_one_episode(
        env=env,
        policy=policy,
        optimizer=optimizer,
        gamma=1.0,
        max_steps=5,
        seed=42,
        entropy_coef=0.01,
        grad_clip_norm=1.0,
    )

    after_params = [param.detach().clone() for param in policy.model.parameters()]
    any_param_changed = any(not torch.equal(before, after) for before, after in zip(before_params, after_params))

    print("[Train One Episode]")
    print("episode_length:", result["episode_length"])
    print("num_steps:", result["num_steps"])
    print("total_loss:", result["total_loss"])
    print("value_loss:", result["value_loss"])
    print("grad_norm:", result["grad_norm"])
    print("params_changed:", any_param_changed)

    assert result["episode_length"] == 5
    assert result["num_steps"] == 5
    assert isinstance(result["total_loss"], float)
    assert isinstance(result["value_loss"], float)
    assert isinstance(result["grad_norm"], float)
    assert result["grad_norm"] >= 0.0
    assert any_param_changed

    env.close()


def test_train_multiple_episodes_returns_summaries() -> None:
    env = make_env(seed=42)
    policy = make_policy(seed=123, action_mode="sample")
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=1e-3)

    results = train(
        env=env,
        policy=policy,
        optimizer=optimizer,
        num_episodes=3,
        gamma=1.0,
        max_steps=4,
        seed=42,
        entropy_coef=0.01,
        grad_clip_norm=1.0,
    )

    print("\n[Train Multiple Episodes]")
    print("num_results:", len(results))
    print("episodes:", [result["episode"] for result in results])
    print("lengths:", [result["episode_length"] for result in results])

    assert len(results) == 3
    assert [result["episode"] for result in results] == [1, 2, 3]
    assert all(result["episode_length"] == 4 for result in results)
    assert all("trajectory" in result for result in results)


def test_train_can_skip_trajectory_storage() -> None:
    env = make_env(seed=42)
    policy = make_policy(seed=456, action_mode="sample")
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=1e-3)

    results = train(
        env=env,
        policy=policy,
        optimizer=optimizer,
        num_episodes=2,
        gamma=1.0,
        max_steps=4,
        seed=42,
        entropy_coef=0.01,
        grad_clip_norm=1.0,
        keep_trajectory=False,
    )

    print("\n[Train Without Trajectory Storage]")
    print("num_results:", len(results))
    print("has_trajectory:", ["trajectory" in result for result in results])

    assert len(results) == 2
    assert all("trajectory" not in result for result in results)

    env.close()


if __name__ == "__main__":
    test_train_one_episode_runs_optimizer_step()
    test_train_multiple_episodes_returns_summaries()
    test_train_can_skip_trajectory_storage()
    print("\nAll trainer tests passed.")
