from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    episode: int,
    config: dict[str, Any],
    best_success_rate: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint with model, optimizer, and run metadata."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "episode": episode,
        "config": config,
        "best_success_rate": best_success_rate,
        "extra": extra,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: torch.device | str | None = None,
) -> dict[str, Any]:
    """Load a checkpoint into a model and optionally an optimizer."""
    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    return checkpoint
