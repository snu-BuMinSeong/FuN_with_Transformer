from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fun import FuNModel
from src.utils.checkpoint import load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint_restores_model_and_optimizer() -> None:
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "test_checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    torch.manual_seed(0)
    model = FuNModel(goal_update_interval=10, hidden_dim=64, goal_size=16, num_actions=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        episode=100,
        config={"seed": 1, "goal_size": 16},
        best_success_rate=0.25,
        extra={"note": "checkpoint test"},
    )

    assert checkpoint_path.exists()

    restored_model = FuNModel(goal_update_interval=10, hidden_dim=64, goal_size=16, num_actions=7)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.0003)
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        map_location="cpu",
    )

    assert checkpoint["episode"] == 100
    assert checkpoint["config"]["seed"] == 1
    assert checkpoint["best_success_rate"] == 0.25
    assert checkpoint["extra"]["note"] == "checkpoint test"

    for original, restored in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(original, restored)
