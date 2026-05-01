from __future__ import annotations

import torch
from torch import nn


class Manager(nn.Module):
    """Minimal recurrent Manager that maps state embeddings to goal vectors."""

    def __init__(self, input_size: int = 64, hidden_size: int = 64, goal_size: int = 16) -> None:
        """Initialize the GRU memory and goal projection."""
        super().__init__()
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}.")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if goal_size <= 0:
            raise ValueError(f"goal_size must be positive, got {goal_size}.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.goal_size = goal_size

        self.gru = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.goal_head = nn.Linear(self.hidden_size, self.goal_size)

    def init_hidden(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        """Return a zero hidden state with shape ``(B, hidden_size)``."""
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, state_emb: torch.Tensor, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a goal vector and next hidden state."""
        if state_emb.ndim != 2 or state_emb.shape[1] != self.input_size:
            raise ValueError(f"Expected state_emb shape (B, 64), got {tuple(state_emb.shape)}.")

        if hidden_state.ndim != 2 or hidden_state.shape[1] != self.hidden_size:
            raise ValueError(f"Expected hidden_state shape (B, 64), got {tuple(hidden_state.shape)}.")

        if state_emb.shape[0] != hidden_state.shape[0]:
            raise ValueError(
                "Batch size mismatch: "
                f"state_emb has batch {state_emb.shape[0]}, hidden_state has batch {hidden_state.shape[0]}."
            )

        next_hidden_state = self.gru(state_emb, hidden_state)
        goal = self.goal_head(next_hidden_state)
        return goal, next_hidden_state


class AblationManager(nn.Module):
    """Feedforward Manager variant without recurrent memory."""

    def __init__(self, input_size: int = 64, hidden_size: int = 64, goal_size: int = 16) -> None:
        """Initialize a memory-free state-to-goal projection."""
        super().__init__()
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}.")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if goal_size <= 0:
            raise ValueError(f"goal_size must be positive, got {goal_size}.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.goal_size = goal_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.goal_size),
        )

    def init_hidden(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        """Return a zero compatibility hidden state with shape ``(B, hidden_size)``."""
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(
        self,
        state_emb: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a goal vector and an unchanged compatibility hidden state."""
        if state_emb.ndim != 2 or state_emb.shape[1] != self.input_size:
            raise ValueError(f"Expected state_emb shape (B, {self.input_size}), got {tuple(state_emb.shape)}.")

        if hidden_state is None:
            next_hidden_state = self.init_hidden(batch_size=state_emb.shape[0], device=state_emb.device)
        else:
            if hidden_state.ndim != 2 or hidden_state.shape[1] != self.hidden_size:
                raise ValueError(
                    f"Expected hidden_state shape (B, {self.hidden_size}), got {tuple(hidden_state.shape)}."
                )
            if state_emb.shape[0] != hidden_state.shape[0]:
                raise ValueError(
                    "Batch size mismatch: "
                    f"state_emb has batch {state_emb.shape[0]}, hidden_state has batch {hidden_state.shape[0]}."
                )
            next_hidden_state = hidden_state

        goal = self.net(state_emb)
        return goal, next_hidden_state
