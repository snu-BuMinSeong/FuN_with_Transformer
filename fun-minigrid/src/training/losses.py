from __future__ import annotations

from typing import Any

import torch


def stack_log_probs(log_probs: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Return log probabilities stacked into shape ``(T,)``."""
    if isinstance(log_probs, list):
        if not log_probs:
            raise ValueError("log_probs must not be empty.")
        stacked = torch.stack(log_probs)
    else:
        stacked = log_probs

    if stacked.ndim != 1:
        raise ValueError(f"Expected log_probs with shape (T,), got {tuple(stacked.shape)}.")

    return stacked


def stack_entropies(entropies: list[torch.Tensor] | torch.Tensor | None) -> torch.Tensor | None:
    """Return entropies stacked into shape ``(T,)`` when provided."""
    if entropies is None:
        return None

    if isinstance(entropies, list):
        if not entropies:
            raise ValueError("entropies must not be empty when provided.")
        stacked = torch.stack(entropies)
    else:
        stacked = entropies

    if stacked.ndim != 1:
        raise ValueError(f"Expected entropies with shape (T,), got {tuple(stacked.shape)}.")

    return stacked


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


def stack_goal_update_flags(goal_updated: list[bool] | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return goal update flags as a float mask with shape ``(T,)``."""
    if isinstance(goal_updated, list):
        mask = torch.tensor(goal_updated, dtype=torch.float32, device=device)
    else:
        mask = goal_updated.to(device=device, dtype=torch.float32)

    if mask.ndim != 1:
        raise ValueError(f"Expected goal_updated mask with shape (T,), got {tuple(mask.shape)}.")

    return mask


def compute_worker_loss(
    log_probs: list[torch.Tensor] | torch.Tensor,
    advantages: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return the worker policy-gradient loss."""
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}.")

    log_prob_tensor = stack_log_probs(log_probs)
    if advantages.ndim != 1:
        raise ValueError(f"Expected advantages with shape (T,), got {tuple(advantages.shape)}.")
    if log_prob_tensor.shape != advantages.shape:
        raise ValueError(
            f"Expected matching shapes for log_probs and advantages, got "
            f"{tuple(log_prob_tensor.shape)} and {tuple(advantages.shape)}."
        )

    losses = -(log_prob_tensor * advantages.detach())
    return losses.mean() if reduction == "mean" else losses.sum()


def compute_entropy_bonus(
    entropies: list[torch.Tensor] | torch.Tensor | None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return the aggregated entropy bonus term."""
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}.")

    entropy_tensor = stack_entropies(entropies)
    if entropy_tensor is None:
        return torch.tensor(0.0, dtype=torch.float32)

    return entropy_tensor.mean() if reduction == "mean" else entropy_tensor.sum()


def compute_value_loss(
    values: list[torch.Tensor] | torch.Tensor,
    value_targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return mean-squared-error value loss."""
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}.")

    value_tensor = stack_value_predictions(values)
    if value_targets.ndim != 1:
        raise ValueError(f"Expected value_targets with shape (T,), got {tuple(value_targets.shape)}.")
    if value_tensor.shape != value_targets.shape:
        raise ValueError(
            f"Expected matching shapes for values and value_targets, got "
            f"{tuple(value_tensor.shape)} and {tuple(value_targets.shape)}."
        )

    losses = (value_tensor - value_targets) ** 2
    return losses.mean() if reduction == "mean" else losses.sum()


def compute_manager_loss(
    trajectory: dict[str, Any],
    enabled: bool = False,
) -> torch.Tensor:
    """Return a minimal manager loss based on updated goals predicting returns."""
    if not enabled:
        return torch.tensor(0.0, dtype=torch.float32)

    if "manager_values" not in trajectory:
        raise KeyError("trajectory must contain 'manager_values' when manager loss is enabled.")
    if "value_targets" not in trajectory:
        raise KeyError("trajectory must contain 'value_targets' when manager loss is enabled.")
    if "goal_updated" not in trajectory:
        raise KeyError("trajectory must contain 'goal_updated' when manager loss is enabled.")

    manager_values = stack_value_predictions(trajectory["manager_values"])
    value_targets = trajectory["value_targets"]
    if value_targets.ndim != 1:
        raise ValueError(f"Expected value_targets with shape (T,), got {tuple(value_targets.shape)}.")
    if manager_values.shape != value_targets.shape:
        raise ValueError(
            f"Expected matching shapes for manager_values and value_targets, got "
            f"{tuple(manager_values.shape)} and {tuple(value_targets.shape)}."
        )

    update_mask = stack_goal_update_flags(trajectory["goal_updated"], device=manager_values.device)
    if update_mask.shape != manager_values.shape:
        raise ValueError(
            f"Expected goal_updated shape {tuple(manager_values.shape)}, got {tuple(update_mask.shape)}."
        )

    num_updates = update_mask.sum()
    if num_updates.item() == 0:
        return torch.zeros((), dtype=manager_values.dtype, device=manager_values.device)

    per_step_loss = (manager_values - value_targets.detach()) ** 2
    return (per_step_loss * update_mask).sum() / num_updates


def compute_total_loss(
    trajectory: dict[str, Any],
    entropy_coef: float = 0.0,
    value_loss_coef: float = 0.5,
    manager_loss_coef: float = 0.0,
    reduction: str = "mean",
    manager_loss_enabled: bool = False,
) -> dict[str, torch.Tensor]:
    """Return worker, entropy, manager, and total loss terms from one trajectory."""
    if "advantages" not in trajectory:
        raise KeyError("trajectory must contain 'advantages'.")
    if "log_probs" not in trajectory:
        raise KeyError("trajectory must contain 'log_probs'.")
    if "value_targets" not in trajectory:
        raise KeyError("trajectory must contain 'value_targets'.")

    value_source = trajectory.get("value_predictions", trajectory.get("values"))
    if value_source is None:
        raise KeyError("trajectory must contain 'value_predictions' or 'values'.")

    worker_loss = compute_worker_loss(
        log_probs=trajectory["log_probs"],
        advantages=trajectory["advantages"],
        reduction=reduction,
    )
    entropy_bonus = compute_entropy_bonus(
        entropies=trajectory.get("entropies"),
        reduction=reduction,
    ).to(worker_loss.device)
    value_loss = compute_value_loss(
        values=value_source,
        value_targets=trajectory["value_targets"],
        reduction=reduction,
    ).to(worker_loss.device)
    manager_loss = compute_manager_loss(
        trajectory=trajectory,
        enabled=manager_loss_enabled,
    ).to(worker_loss.device)
    total_loss = (
        worker_loss
        + value_loss_coef * value_loss
        - entropy_coef * entropy_bonus
        + manager_loss_coef * manager_loss
    )

    return {
        "worker_loss": worker_loss,
        "value_loss": value_loss,
        "entropy_bonus": entropy_bonus,
        "manager_loss": manager_loss,
        "total_loss": total_loss,
    }
