from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from src.models.encoder import ObservationEncoder
from src.models.manager import AblationManager, Manager
from src.models.worker import Worker


class FuNModel(nn.Module):
    """Minimal FuN-like model with rollout-friendly policy outputs."""

    def __init__(
        self,
        goal_update_interval: int = 10,
        hidden_dim: int = 64,
        goal_size: int = 16,
        num_actions: int = 7,
        manager_type: str = "recurrent",
    ) -> None:
        """Initialize encoder, manager, worker, and goal update interval."""
        super().__init__()
        if goal_update_interval <= 0:
            raise ValueError(f"goal_update_interval must be positive, got {goal_update_interval}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if goal_size <= 0:
            raise ValueError(f"goal_size must be positive, got {goal_size}.")
        if num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {num_actions}.")
        if manager_type not in {"recurrent", "ablation", "feedforward"}:
            raise ValueError(
                "manager_type must be one of 'recurrent', 'ablation', or 'feedforward', "
                f"got {manager_type}."
            )

        self.goal_update_interval = goal_update_interval
        self.hidden_dim = hidden_dim
        self.goal_size = goal_size
        self.num_actions = num_actions
        self.manager_type = manager_type
        self.encoder = ObservationEncoder(embedding_dim=self.hidden_dim)
        manager_cls = Manager if self.manager_type == "recurrent" else AblationManager
        self.manager = manager_cls(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            goal_size=self.goal_size,
        )
        self.worker = Worker(
            state_size=self.hidden_dim,
            goal_size=self.goal_size,
            num_actions=self.num_actions,
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.manager_value_head = nn.Sequential(
            nn.Linear(self.goal_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def init_hidden(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        """Return the initial manager hidden state with shape ``(B, hidden_dim)``."""
        return self.manager.init_hidden(batch_size=batch_size, device=device)

    def init_goal(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        """Return the initial goal tensor with shape ``(B, goal_size)``."""
        return torch.zeros(batch_size, self.goal_size, device=device)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        current_goal: torch.Tensor,
        step_count: int,
    ) -> dict[str, Any]:
        """Encode observation, update goal when scheduled, and return policy outputs."""
        if current_goal.ndim != 2 or current_goal.shape[1] != self.goal_size:
            raise ValueError(
                f"Expected current_goal shape (B, {self.goal_size}), got {tuple(current_goal.shape)}."
            )

        if obs.shape[0] != hidden_state.shape[0] or obs.shape[0] != current_goal.shape[0]:
            raise ValueError(
                "Batch size mismatch: "
                f"obs has batch {obs.shape[0]}, "
                f"hidden_state has batch {hidden_state.shape[0]}, "
                f"current_goal has batch {current_goal.shape[0]}."
            )

        state_emb = self.encoder(obs)
        goal_updated = step_count % self.goal_update_interval == 0

        if goal_updated:
            goal, next_hidden_state = self.manager(state_emb, hidden_state)
        else:
            goal = current_goal
            next_hidden_state = hidden_state

        logits = self.worker(state_emb, goal)
        action_dist = Categorical(logits=logits)
        action_probs = action_dist.probs
        value = self.value_head(state_emb).squeeze(-1)
        manager_value = self.manager_value_head(goal).squeeze(-1)

        return {
            "state_emb": state_emb,
            "goal": goal,
            "hidden_state": next_hidden_state,
            "logits": logits,
            "action_dist": action_dist,
            "action_probs": action_probs,
            "value": value,
            "manager_value": manager_value,
            "goal_updated": goal_updated,
        }
