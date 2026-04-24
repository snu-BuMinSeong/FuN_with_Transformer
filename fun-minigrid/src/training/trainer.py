from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from src.training.losses import compute_total_loss
from src.training.returns import attach_returns_and_advantages
from src.training.rollout import collect_training_rollout


def _check_finite_scalar(name: str, value: torch.Tensor) -> None:
    """Raise an error when a scalar tensor is NaN or inf."""
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor, got shape {tuple(value.shape)}.")
    if not torch.isfinite(value):
        raise ValueError(f"{name} is not finite: {float(value.detach().cpu().item())}.")


def _stack_scalar_tensors(values: list[torch.Tensor]) -> torch.Tensor:
    """Stack scalar tensors into shape ``(T,)``."""
    if not values:
        return torch.empty(0, dtype=torch.float32)
    return torch.stack(values)


def _module_grad_norm(module: torch.nn.Module) -> float:
    """Return the L2 norm of gradients for a module."""
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return total ** 0.5


def _module_has_grad(module: torch.nn.Module) -> bool:
    """Return whether a module has any nonzero gradient."""
    for param in module.parameters():
        if param.grad is None:
            continue
        if torch.count_nonzero(param.grad.detach()).item() > 0:
            return True
    return False


def train_one_episode(
    env: Any,
    policy: Any,
    optimizer: torch.optim.Optimizer,
    gamma: float = 1.0,
    max_steps: int | None = None,
    seed: int | None = None,
    entropy_coef: float = 0.0,
    value_loss_coef: float = 0.5,
    manager_loss_coef: float = 0.0,
    grad_clip_norm: float | None = 1.0,
    keep_trajectory: bool = True,
) -> dict[str, Any]:
    """Run one training episode from rollout collection to optimizer step."""
    model = policy.model
    model.train()

    trajectory = collect_training_rollout(
        env=env,
        policy=policy,
        max_steps=max_steps,
        seed=seed,
    )
    trajectory = attach_returns_and_advantages(trajectory, gamma=gamma)

    optimizer.zero_grad(set_to_none=True)
    loss_dict = compute_total_loss(
        trajectory=trajectory,
        entropy_coef=entropy_coef,
        value_loss_coef=value_loss_coef,
        manager_loss_coef=manager_loss_coef,
        manager_loss_enabled=manager_loss_coef > 0.0,
    )

    _check_finite_scalar("worker_loss", loss_dict["worker_loss"])
    _check_finite_scalar("value_loss", loss_dict["value_loss"])
    _check_finite_scalar("entropy_bonus", loss_dict["entropy_bonus"])
    _check_finite_scalar("manager_loss", loss_dict["manager_loss"])
    _check_finite_scalar("total_loss", loss_dict["total_loss"])

    loss_dict["total_loss"].backward()

    grad_debug = {
        "encoder_grad_norm": _module_grad_norm(model.encoder),
        "manager_grad_norm": _module_grad_norm(model.manager),
        "worker_grad_norm": _module_grad_norm(model.worker),
        "value_head_grad_norm": _module_grad_norm(model.value_head),
        "manager_value_head_grad_norm": _module_grad_norm(model.manager_value_head),
        "encoder_has_grad": _module_has_grad(model.encoder),
        "manager_has_grad": _module_has_grad(model.manager),
        "worker_has_grad": _module_has_grad(model.worker),
        "value_head_has_grad": _module_has_grad(model.value_head),
        "manager_value_head_has_grad": _module_has_grad(model.manager_value_head),
    }

    if grad_clip_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    else:
        grad_norm = torch.tensor(0.0, dtype=torch.float32)

    grad_norm_tensor = torch.as_tensor(grad_norm, dtype=torch.float32)
    if not torch.isfinite(grad_norm_tensor):
        raise ValueError(f"Gradient norm is not finite: {float(grad_norm_tensor.detach().cpu().item())}.")

    optimizer.step()
    model.eval()

    actions = trajectory["actions"]
    rewards = trajectory["rewards"]
    goals = trajectory["goals"]
    goal_updates = trajectory["goal_updated"]
    value_predictions = trajectory["value_predictions"]
    manager_values = trajectory["manager_values"]
    log_probs = _stack_scalar_tensors(trajectory["log_probs"])
    entropies = _stack_scalar_tensors(trajectory["entropies"])
    debug_info = policy.get_debug_info() if hasattr(policy, "get_debug_info") else {}
    action_histogram = [actions.count(action_idx) for action_idx in range(model.num_actions)]
    nonzero_reward_steps = sum(1 for reward in rewards if reward != 0.0)
    positive_reward_steps = sum(1 for reward in rewards if reward > 0.0)
    returns = trajectory["returns"]
    advantages = trajectory["advantages"]

    result = {
        "episode_length": trajectory["episode_length"],
        "total_reward": trajectory["total_reward"],
        "success": trajectory["success"],
        "worker_loss": float(loss_dict["worker_loss"].detach().cpu().item()),
        "value_loss": float(loss_dict["value_loss"].detach().cpu().item()),
        "entropy_bonus": float(loss_dict["entropy_bonus"].detach().cpu().item()),
        "manager_loss": float(loss_dict["manager_loss"].detach().cpu().item()),
        "total_loss": float(loss_dict["total_loss"].detach().cpu().item()),
        "grad_norm": float(grad_norm_tensor.detach().cpu().item()),
        "returns_mean": float(returns.mean().detach().cpu().item()),
        "returns_min": float(returns.min().detach().cpu().item()),
        "returns_max": float(returns.max().detach().cpu().item()),
        "advantages_mean": float(advantages.mean().detach().cpu().item()),
        "advantages_abs_mean": float(advantages.abs().mean().detach().cpu().item()),
        "values_mean": float(value_predictions.mean().detach().cpu().item()),
        "value_min": float(value_predictions.min().detach().cpu().item()),
        "value_max": float(value_predictions.max().detach().cpu().item()),
        "manager_values_mean": float(torch.stack(manager_values).mean().detach().cpu().item()),
        "manager_value_min": float(torch.stack(manager_values).min().detach().cpu().item()),
        "manager_value_max": float(torch.stack(manager_values).max().detach().cpu().item()),
        "log_prob_mean": float(log_probs.mean().detach().cpu().item()),
        "log_prob_min": float(log_probs.min().detach().cpu().item()),
        "log_prob_max": float(log_probs.max().detach().cpu().item()),
        "entropy_mean": float(entropies.mean().detach().cpu().item()),
        "entropy_min": float(entropies.min().detach().cpu().item()),
        "entropy_max": float(entropies.max().detach().cpu().item()),
        "num_steps": len(trajectory["actions"]),
        "action_min": min(actions) if actions else None,
        "action_max": max(actions) if actions else None,
        "action_coverage": float(len(set(actions)) / model.num_actions) if actions else 0.0,
        "action_histogram": ",".join(str(count) for count in action_histogram),
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "nonzero_reward_steps": nonzero_reward_steps,
        "positive_reward_steps": positive_reward_steps,
        "nonzero_reward_fraction": float(nonzero_reward_steps / len(rewards)) if rewards else 0.0,
        "nonzero_return_steps": int(torch.count_nonzero(returns).item()),
        "nonzero_return_fraction": float(torch.count_nonzero(returns).item() / len(rewards)) if rewards else 0.0,
        "has_reward_signal": nonzero_reward_steps > 0,
        "has_return_signal": bool(torch.count_nonzero(returns).item() > 0),
        "goal_shape": tuple(goals[0].shape) if goals else None,
        "hidden_state_shape": tuple(policy.hidden_state.shape) if hasattr(policy, "hidden_state") else None,
        "current_goal_shape": tuple(policy.current_goal.shape) if hasattr(policy, "current_goal") else None,
        "num_goal_updates": int(sum(1 for updated in goal_updates if updated)),
        "final_hidden_norm": float(debug_info.get("hidden_norm", 0.0)),
        "final_goal_norm": float(debug_info.get("goal_norm", 0.0)),
        "final_step_count": int(debug_info.get("step_count", len(actions))),
    }
    result.update(grad_debug)
    if keep_trajectory:
        result["trajectory"] = trajectory
    return result


def train(
    env: Any,
    policy: Any,
    optimizer: torch.optim.Optimizer,
    num_episodes: int,
    gamma: float = 1.0,
    max_steps: int | None = None,
    seed: int | None = None,
    entropy_coef: float = 0.0,
    value_loss_coef: float = 0.5,
    manager_loss_coef: float = 0.0,
    grad_clip_norm: float | None = 1.0,
    keep_trajectory: bool = True,
    episode_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Train for multiple episodes and return per-episode summaries."""
    results: list[dict[str, Any]] = []

    for episode_idx in range(num_episodes):
        episode_seed = None if seed is None else seed + episode_idx
        result = train_one_episode(
            env=env,
            policy=policy,
            optimizer=optimizer,
            gamma=gamma,
            max_steps=max_steps,
            seed=episode_seed,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            manager_loss_coef=manager_loss_coef,
            grad_clip_norm=grad_clip_norm,
            keep_trajectory=keep_trajectory,
        )
        result["episode"] = episode_idx + 1
        if episode_callback is not None:
            episode_callback(result)
        results.append(result)

    return results
