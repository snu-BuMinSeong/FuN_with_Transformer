from __future__ import annotations

from typing import Any

import torch


def stack_value_predictions(values: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Return predicted values stacked into shape ``(T,)``."""
    if isinstance(values, list):
        if not values:
            raise ValueError("values must not be empty.")
        stacked = torch.stack(values)
    else:
        stacked = values

    if stacked.ndim != 1:
        raise ValueError(f"Expected values with shape (T,), got {tuple(stacked.shape)}.")

    return stacked


def compute_reward_to_go(
    rewards: list[float] | torch.Tensor,
    gamma: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return per-step discounted reward-to-go with shape ``(T,)``."""
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}.")

    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    if reward_tensor.ndim != 1:
        raise ValueError(f"Expected rewards with shape (T,), got {tuple(reward_tensor.shape)}.")

    returns = torch.zeros_like(reward_tensor)
    running_return = torch.tensor(0.0, dtype=torch.float32, device=reward_tensor.device)

    for step in range(reward_tensor.shape[0] - 1, -1, -1):
        running_return = reward_tensor[step] + gamma * running_return
        returns[step] = running_return

    return returns


def compute_advantages(
    returns: torch.Tensor,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return advantages with shape ``(T,)`` using an optional baseline."""
    if returns.ndim != 1:
        raise ValueError(f"Expected returns with shape (T,), got {tuple(returns.shape)}.")

    if baseline is None:
        return returns.clone()

    if baseline.ndim != 1:
        raise ValueError(f"Expected baseline with shape (T,), got {tuple(baseline.shape)}.")

    if baseline.shape != returns.shape:
        raise ValueError(
            f"Expected baseline shape {tuple(returns.shape)}, got {tuple(baseline.shape)}."
        )

    return returns - baseline


def attach_returns_and_advantages(
    trajectory: dict[str, Any],
    gamma: float = 1.0,
    baseline: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Attach reward-to-go returns and advantages to a rollout trajectory."""
    if "rewards" not in trajectory:
        raise KeyError("trajectory must contain 'rewards'.")

    baseline_tensor = baseline
    if baseline_tensor is None and "values" in trajectory:
        baseline_tensor = stack_value_predictions(trajectory["values"])

    baseline_device = baseline_tensor.device if baseline_tensor is not None else None
    returns = compute_reward_to_go(trajectory["rewards"], gamma=gamma, device=baseline_device)
    advantages = compute_advantages(returns, baseline=baseline_tensor)

    enriched = dict(trajectory)
    enriched["returns"] = returns
    enriched["advantages"] = advantages
    if baseline_tensor is not None:
        enriched["value_predictions"] = baseline_tensor
    enriched["value_targets"] = returns
    return enriched
