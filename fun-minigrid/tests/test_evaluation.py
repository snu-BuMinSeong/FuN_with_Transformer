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
from src.training.evaluation import build_eval_comparison, build_eval_comparison_text, evaluate_policy


def make_policy(seed: int = 0, action_mode: str = "argmax") -> FuNPolicy:
    torch.manual_seed(seed)
    model = FuNModel(goal_update_interval=10)
    return FuNPolicy(model=model, preprocess_fn=preprocess_obs, action_mode=action_mode)


def test_evaluate_policy_returns_expected_fields() -> None:
    env = make_env(seed=42)
    policy = make_policy(seed=0, action_mode="argmax")

    result = evaluate_policy(
        env=env,
        policy=policy,
        num_episodes=2,
        max_steps=5,
        seed=42,
    )

    print("[Evaluation]")
    print("num_episodes:", result["num_episodes"])
    print("mean_reward:", result["mean_reward"])
    print("std_reward:", result["std_reward"])
    print("mean_success_rate:", result["mean_success_rate"])
    print("std_success_rate:", result["std_success_rate"])
    print("mean_episode_length:", result["mean_episode_length"])
    print("std_episode_length:", result["std_episode_length"])

    assert result["num_episodes"] == 2
    assert len(result["episode_results"]) == 2
    assert isinstance(result["mean_reward"], float)
    assert isinstance(result["std_reward"], float)
    assert isinstance(result["mean_success_rate"], float)
    assert isinstance(result["std_success_rate"], float)
    assert isinstance(result["mean_episode_length"], float)
    assert isinstance(result["std_episode_length"], float)
    assert len(result["episode_seeds"]) == 2

    env.close()


def test_build_eval_comparison_text_contains_metrics() -> None:
    before = {"mean_reward": 0.0, "mean_success_rate": 0.0, "mean_episode_length": 10.0}
    after = {"mean_reward": 0.1, "mean_success_rate": 0.0, "mean_episode_length": 8.0}
    text = build_eval_comparison_text(before, after)

    print("\n[Evaluation Comparison]")
    print("text:", text)

    assert "mean_reward" in text
    assert "success_rate" in text
    assert "mean_episode_length" in text
    assert "0.000 -> 0.100" in text


def test_build_eval_comparison_returns_deltas() -> None:
    before = {"mean_reward": 0.0, "mean_success_rate": 0.25, "mean_episode_length": 10.0}
    after = {"mean_reward": 1.0, "mean_success_rate": 0.5, "mean_episode_length": 8.0}
    comparison = build_eval_comparison(before, after)

    print("\n[Evaluation Delta]")
    print("comparison:", comparison)

    assert comparison["reward_delta"] == 1.0
    assert comparison["success_rate_delta"] == 0.25
    assert comparison["episode_length_delta"] == -2.0


if __name__ == "__main__":
    test_evaluate_policy_returns_expected_fields()
    test_build_eval_comparison_text_contains_metrics()
    test_build_eval_comparison_returns_deltas()
    print("\nAll evaluation tests passed.")
