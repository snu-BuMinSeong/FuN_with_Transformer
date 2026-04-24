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


def make_policy(action_mode: str = "argmax") -> FuNPolicy:
    """Create a deterministic FuNPolicy for lightweight behavior checks."""
    torch.manual_seed(0)
    model = FuNModel(goal_update_interval=10)
    return FuNPolicy(model=model, preprocess_fn=preprocess_obs, action_mode=action_mode)


def test_policy_reset_state() -> None:
    policy = make_policy()
    policy.reset()

    print("[Policy Reset]")
    print("step_count:", policy.step_count)
    print("hidden shape:", policy.hidden_state.shape)
    print("goal shape:", policy.current_goal.shape)

    assert policy.step_count == 0
    assert policy.hidden_state.shape == (1, 64)
    assert policy.current_goal.shape == (1, 16)


def test_policy_act_updates_step_and_returns_valid_action() -> None:
    env = make_env(seed=42)
    obs, _ = env.reset(seed=42)
    policy = make_policy()

    before_step = policy.step_count
    action = policy.act(obs)

    print("\n[Policy Act]")
    print("action:", action)
    print("before_step:", before_step)
    print("after_step:", policy.step_count)

    assert 0 <= action <= 6
    assert policy.step_count == before_step + 1

    env.close()


def test_model_outputs_support_distribution_and_log_prob() -> None:
    env = make_env(seed=42)
    obs, _ = env.reset(seed=42)
    policy = make_policy(action_mode="sample")
    obs_tensor = policy._obs_to_tensor(obs)

    with torch.no_grad():
        out = policy.model(
            obs=obs_tensor,
            hidden_state=policy.hidden_state,
            current_goal=policy.current_goal,
            step_count=policy.step_count,
        )
        sampled_action = out["action_dist"].sample()
        log_prob = out["action_dist"].log_prob(sampled_action)

    print("\n[Policy Model Outputs]")
    print("logits shape     :", out["logits"].shape)
    print("action_probs shape:", out["action_probs"].shape)
    print("sampled_action shape:", sampled_action.shape)
    print("log_prob shape   :", log_prob.shape)
    print("goal_updated     :", out["goal_updated"])

    assert out["logits"].shape == (1, 7)
    assert out["action_probs"].shape == (1, 7)
    assert sampled_action.shape == (1,)
    assert log_prob.shape == (1,)
    assert out["goal_updated"] is True

    env.close()


def test_policy_goal_updates_every_10_steps() -> None:
    env = make_env(seed=42)
    obs, _ = env.reset(seed=42)
    policy = make_policy()

    policy.act(obs)
    goal_after_step_1 = policy.current_goal.clone()
    goal_norm_after_step_1 = policy.get_debug_info()["goal_norm"]

    for _ in range(9):
        policy.act(obs)

    goal_after_step_10 = policy.current_goal.clone()
    goal_norm_after_step_10 = policy.get_debug_info()["goal_norm"]

    policy.act(obs)
    goal_after_step_11 = policy.current_goal.clone()
    goal_norm_after_step_11 = policy.get_debug_info()["goal_norm"]

    print("\n[Policy Goal Update Interval]")
    print("step_count:", policy.step_count)
    print("goal_norm_after_step_1 :", goal_norm_after_step_1)
    print("goal_norm_after_step_10:", goal_norm_after_step_10)
    print("goal_norm_after_step_11:", goal_norm_after_step_11)
    print("goal changed at step 10:", not torch.equal(goal_after_step_1, goal_after_step_11))

    assert policy.step_count == 11
    assert torch.equal(goal_after_step_1, goal_after_step_10)
    assert goal_norm_after_step_1 == goal_norm_after_step_10
    assert not torch.equal(goal_after_step_1, goal_after_step_11)
    assert goal_norm_after_step_1 != goal_norm_after_step_11

    env.close()


def test_policy_act_for_training_returns_rollout_fields() -> None:
    env = make_env(seed=42)
    obs, _ = env.reset(seed=42)
    policy = make_policy(action_mode="sample")

    out = policy.act_for_training(obs)

    print("\n[Policy Training Act]")
    print("action:", out["action"])
    print("log_prob ndim:", out["log_prob"].ndim)
    print("entropy ndim:", out["entropy"].ndim)
    print("goal shape:", out["goal"].shape)
    print("step_index:", out["step_index"])
    print("obs_tensor shape:", out["obs_tensor"].shape)
    print("goal_updated:", out["goal_updated"])

    assert 0 <= out["action"] <= 6
    assert out["log_prob"].ndim == 0
    assert out["entropy"].ndim == 0
    assert out["goal"].shape == (16,)
    assert out["step_index"] == 0
    assert out["obs_tensor"].shape == (3, 7, 7)
    assert out["goal_updated"] is True
    assert policy.step_count == 1

    env.close()


if __name__ == "__main__":
    test_policy_reset_state()
    test_policy_act_updates_step_and_returns_valid_action()
    test_model_outputs_support_distribution_and_log_prob()
    test_policy_goal_updates_every_10_steps()
    test_policy_act_for_training_returns_rollout_fields()

    print("\nAll FuNPolicy tests passed.")
