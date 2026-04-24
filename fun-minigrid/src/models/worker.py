from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class Worker(nn.Module):
    """Minimal Worker that maps state and goal vectors to action logits."""

    def __init__(self, state_size: int = 64, goal_size: int = 16, num_actions: int = 7) -> None:
        """Initialize a small MLP policy head for discrete actions."""
        super().__init__()
        if state_size <= 0:
            raise ValueError(f"state_size must be positive, got {state_size}.")
        if goal_size <= 0:
            raise ValueError(f"goal_size must be positive, got {goal_size}.")
        if num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {num_actions}.")

        self.state_size = state_size
        self.goal_size = goal_size
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(self.state_size + self.goal_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )

    def forward(self, state_emb: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return action logits with shape ``(B, 7)``."""
        if state_emb.ndim != 2 or state_emb.shape[1] != self.state_size:
            raise ValueError(f"Expected state_emb shape (B, 64), got {tuple(state_emb.shape)}.")

        if goal.ndim != 2 or goal.shape[1] != self.goal_size:
            raise ValueError(f"Expected goal shape (B, 16), got {tuple(goal.shape)}.")

        if state_emb.shape[0] != goal.shape[0]:
            raise ValueError(
                "Batch size mismatch: "
                f"state_emb has batch {state_emb.shape[0]}, goal has batch {goal.shape[0]}."
            )

        return self.net(torch.cat([state_emb, goal], dim=1))

    def get_action_distribution(self, state_emb: torch.Tensor, goal: torch.Tensor) -> Categorical:
        """Build a categorical action distribution from the current policy logits."""
        logits = self.forward(state_emb, goal)
        return Categorical(logits=logits)
