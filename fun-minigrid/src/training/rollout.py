from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import torch


class StatefulPolicy(Protocol):
    """Policy object that manages its own recurrent state and preprocessing."""

    def reset(self, batch_size: int = 1) -> None:
        """Reset policy state for a new episode."""

    def act(self, obs: Any) -> int:
        """Return an action for a raw environment observation."""


class TrainableStatefulPolicy(StatefulPolicy, Protocol):
    """Stateful policy with rollout outputs needed for training."""

    def act_for_training(self, obs: Any) -> dict[str, Any]:
        """Return action and rollout metadata for one raw observation."""


Policy = Callable[[Any], int] | StatefulPolicy
PreprocessFn = Callable[[Any], Any]


def random_policy(env: Any) -> Policy:
    """Create a policy that samples valid random actions from the environment."""

    def policy(_: Any) -> int:
        return int(env.action_space.sample())

    return policy


def run_episode(
    env: Any,
    policy: Policy,
    preprocess_fn: PreprocessFn | None = None,
    max_steps: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one episode with either a callable or stateful policy."""
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(batch_size=1)

    total_reward = 0.0
    episode_length = 0
    actions: list[int] = []
    terminated = False
    truncated = False
    success_from_info = False

    while not terminated and not truncated:
        if hasattr(policy, "act"):
            action = int(policy.act(obs))
        else:
            policy_obs = preprocess_fn(obs) if preprocess_fn is not None else obs
            action = int(policy(policy_obs))

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        episode_length += 1
        actions.append(action)
        success_from_info = bool(info.get("success", success_from_info))

        if max_steps is not None and episode_length >= max_steps:
            break

    success = success_from_info or (terminated and total_reward > 0.0)

    return {
        "total_reward": total_reward,
        "episode_length": episode_length,
        "terminated": terminated,
        "truncated": truncated,
        "success": success,
        "actions": actions,
        "seed": seed,
    }


def collect_training_rollout(
    env: Any,
    policy: TrainableStatefulPolicy,
    max_steps: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Collect one trajectory with per-step tensors and episode metadata."""
    obs, _ = env.reset(seed=seed)
    policy.reset(batch_size=1)

    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    manager_values: list[torch.Tensor] = []
    goals: list[torch.Tensor] = []
    goal_updated_flags: list[bool] = []
    step_indices: list[int] = []

    total_reward = 0.0
    episode_length = 0
    terminated = False
    truncated = False
    success_from_info = False

    while not terminated and not truncated:
        step_out = policy.act_for_training(obs)
        action = int(step_out["action"])

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        observations.append(np.asarray(step_out["obs_tensor"], dtype=np.float32).copy())
        actions.append(action)
        rewards.append(float(reward))
        dones.append(done)
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        log_probs.append(step_out["log_prob"])
        entropies.append(step_out["entropy"])
        values.append(step_out["value"])
        manager_values.append(step_out["manager_value"])
        goals.append(step_out["goal"])
        goal_updated_flags.append(bool(step_out["goal_updated"]))
        step_indices.append(int(step_out["step_index"]))

        total_reward += float(reward)
        episode_length += 1
        success_from_info = bool(info.get("success", success_from_info))
        obs = next_obs

        if max_steps is not None and episode_length >= max_steps:
            truncated = True
            dones[-1] = True
            truncated_flags[-1] = True
            break

    success = success_from_info or (terminated and total_reward > 0.0)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "terminated": terminated_flags,
        "truncated": truncated_flags,
        "log_probs": log_probs,
        "entropies": entropies,
        "values": values,
        "manager_values": manager_values,
        "goals": goals,
        "goal_updated": goal_updated_flags,
        "step_indices": step_indices,
        "total_reward": total_reward,
        "episode_length": episode_length,
        "success": success,
    }
