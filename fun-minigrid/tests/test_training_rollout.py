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
from src.training.rollout import collect_training_rollout


def make_policy(seed: int = 0, action_mode: str = "sample") -> FuNPolicy:
    torch.manual_seed(seed)
    model = FuNModel(goal_update_interval=10)
    return FuNPolicy(model=model, preprocess_fn=preprocess_obs, action_mode=action_mode)


def test_collect_training_rollout_fields() -> None:
    env = make_env(seed=42)
    policy = make_policy(seed=0, action_mode="sample")

    rollout = collect_training_rollout(env=env, policy=policy, max_steps=5, seed=42)

    print("[Training Rollout]")
    print("episode_length:", rollout["episode_length"])
    print("total_reward:", rollout["total_reward"])
    print("num_observations:", len(rollout["observations"]))
    print("num_actions:", len(rollout["actions"]))
    print("num_log_probs:", len(rollout["log_probs"]))
    print("num_entropies:", len(rollout["entropies"]))
    print("num_values:", len(rollout["values"]))
    print("num_manager_values:", len(rollout["manager_values"]))
    print("first_obs_shape:", rollout["observations"][0].shape)
    print("first_goal_shape:", rollout["goals"][0].shape)
    print("first_value_shape:", rollout["values"][0].shape)
    print("first_manager_value_shape:", rollout["manager_values"][0].shape)
    print("step_indices:", rollout["step_indices"])

    assert rollout["episode_length"] == 5
    assert len(rollout["observations"]) == 5
    assert len(rollout["actions"]) == 5
    assert len(rollout["rewards"]) == 5
    assert len(rollout["dones"]) == 5
    assert len(rollout["terminated"]) == 5
    assert len(rollout["truncated"]) == 5
    assert len(rollout["log_probs"]) == 5
    assert len(rollout["entropies"]) == 5
    assert len(rollout["values"]) == 5
    assert len(rollout["manager_values"]) == 5
    assert len(rollout["goals"]) == 5
    assert len(rollout["goal_updated"]) == 5
    assert len(rollout["step_indices"]) == 5
    assert rollout["observations"][0].shape == (3, 7, 7)
    assert rollout["goals"][0].shape == (16,)
    assert rollout["values"][0].shape == ()
    assert rollout["manager_values"][0].shape == ()
    assert rollout["step_indices"] == [0, 1, 2, 3, 4]
    assert rollout["dones"][-1] is True
    assert rollout["truncated"][-1] is True

    env.close()


def test_collect_training_rollout_is_reproducible_with_seed() -> None:
    env1 = make_env(seed=42)
    env2 = make_env(seed=42)
    policy1 = make_policy(seed=123, action_mode="sample")
    policy2 = make_policy(seed=123, action_mode="sample")

    torch.manual_seed(999)
    rollout1 = collect_training_rollout(env=env1, policy=policy1, max_steps=5, seed=42)
    torch.manual_seed(999)
    rollout2 = collect_training_rollout(env=env2, policy=policy2, max_steps=5, seed=42)

    print("\n[Training Rollout Reproducibility]")
    print("actions 1:", rollout1["actions"])
    print("actions 2:", rollout2["actions"])
    print("rewards 1:", rollout1["rewards"])
    print("rewards 2:", rollout2["rewards"])

    assert rollout1["actions"] == rollout2["actions"]
    assert rollout1["rewards"] == rollout2["rewards"]
    assert rollout1["step_indices"] == rollout2["step_indices"]

    env1.close()
    env2.close()


if __name__ == "__main__":
    test_collect_training_rollout_fields()
    test_collect_training_rollout_is_reproducible_with_seed()
    print("\nAll training rollout tests passed.")
