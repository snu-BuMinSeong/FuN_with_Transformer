from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from src.models.fun import FuNModel


class FuNPolicy:
    """Inference-only policy wrapper around ``FuNModel``."""

    def __init__(
        self,
        model: FuNModel,
        preprocess_fn: Callable[[Any], np.ndarray],
        device: torch.device | str | None = None,
        action_mode: str = "sample",
    ) -> None:
        """Initialize model state, preprocessing, and action selection mode."""
        if action_mode not in {"sample", "argmax"}:
            raise ValueError(f"action_mode must be 'sample' or 'argmax', got {action_mode}.")

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocess_fn = preprocess_fn
        self.action_mode = action_mode
        self.hidden_state: torch.Tensor
        self.current_goal: torch.Tensor
        self.step_count = 0
        self.reset()

    def reset(self, batch_size: int = 1) -> None:
        """Reset recurrent policy state for a new episode."""
        if batch_size != 1:
            raise ValueError("FuNPolicy currently supports batch_size=1 only.")

        self.hidden_state = self.model.init_hidden(batch_size=batch_size, device=self.device)
        self.current_goal = self.model.init_goal(batch_size=batch_size, device=self.device)
        self.step_count = 0

    def _obs_to_tensor(self, obs: Any) -> torch.Tensor:
        """Convert a raw environment observation to a batched float tensor."""
        proc_obs = self.preprocess_fn(obs)
        obs_tensor = torch.as_tensor(proc_obs, dtype=torch.float32, device=self.device)

        if obs_tensor.shape != (3, 7, 7):
            raise ValueError(f"Expected preprocessed observation shape (3, 7, 7), got {tuple(obs_tensor.shape)}.")

        return obs_tensor.unsqueeze(0)

    def act_for_training(self, obs: Any) -> dict[str, Any]:
        """Return action selection outputs needed for trajectory collection."""
        obs_tensor = self._obs_to_tensor(obs)
        out = self.model(
            obs=obs_tensor,
            hidden_state=self.hidden_state,
            current_goal=self.current_goal,
            step_count=self.step_count,
        )
        action_dist = out["action_dist"]

        if self.action_mode == "sample":
            action = action_dist.sample()
        else:
            action = torch.argmax(out["logits"], dim=-1)

        log_prob = action_dist.log_prob(action)

        self.hidden_state = out["hidden_state"]
        self.current_goal = out["goal"]

        result = {
            "action": int(action.item()),
            "log_prob": log_prob.squeeze(0),
            "entropy": action_dist.entropy().squeeze(0),
            "value": out["value"].squeeze(0),
            "manager_value": out["manager_value"].squeeze(0),
            "goal": out["goal"].squeeze(0).detach().clone(),
            "goal_updated": bool(out["goal_updated"]),
            "step_index": self.step_count,
            "obs_tensor": obs_tensor.squeeze(0).detach().cpu().numpy().copy(),
        }
        self.step_count += 1
        return result

    def act(self, obs: Any) -> int:
        """Return one action for the current observation and update policy state."""
        with torch.no_grad():
            out = self.act_for_training(obs)
        return int(out["action"])

    def get_debug_info(self) -> dict[str, float | int]:
        """Return lightweight recurrent-state diagnostics."""
        return {
            "step_count": self.step_count,
            "hidden_norm": float(self.hidden_state.norm().item()),
            "goal_norm": float(self.current_goal.norm().item()),
        }
