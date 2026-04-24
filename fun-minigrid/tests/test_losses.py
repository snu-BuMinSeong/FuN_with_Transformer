from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.losses import (
    compute_entropy_bonus,
    compute_manager_loss,
    compute_total_loss,
    compute_value_loss,
    compute_worker_loss,
)


def test_compute_worker_loss_mean() -> None:
    log_probs = [
        torch.tensor(-0.2, requires_grad=True),
        torch.tensor(-0.4, requires_grad=True),
        torch.tensor(-0.6, requires_grad=True),
    ]
    advantages = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    loss = compute_worker_loss(log_probs=log_probs, advantages=advantages, reduction="mean")

    print("[Worker Loss Mean]")
    print("loss:", float(loss.item()))

    expected = torch.tensor((0.2 + 0.8 + 1.8) / 3.0, dtype=torch.float32)
    assert torch.allclose(loss, expected)


def test_compute_entropy_bonus_mean() -> None:
    entropies = [
        torch.tensor(1.0, requires_grad=True),
        torch.tensor(0.5, requires_grad=True),
        torch.tensor(1.5, requires_grad=True),
    ]
    entropy_bonus = compute_entropy_bonus(entropies=entropies, reduction="mean")

    print("\n[Entropy Bonus Mean]")
    print("entropy_bonus:", float(entropy_bonus.item()))

    assert torch.allclose(entropy_bonus, torch.tensor(1.0))


def test_compute_value_loss_mean() -> None:
    values = [
        torch.tensor(1.0, requires_grad=True),
        torch.tensor(2.0, requires_grad=True),
    ]
    value_targets = torch.tensor([1.5, 1.0], dtype=torch.float32)
    value_loss = compute_value_loss(values=values, value_targets=value_targets, reduction="mean")

    print("\n[Value Loss Mean]")
    print("value_loss:", float(value_loss.item()))

    expected = torch.tensor((((1.0 - 1.5) ** 2) + ((2.0 - 1.0) ** 2)) / 2.0, dtype=torch.float32)
    assert torch.allclose(value_loss, expected)


def test_compute_total_loss_with_entropy_and_placeholder_manager() -> None:
    trajectory = {
        "log_probs": [
            torch.tensor(-0.2, requires_grad=True),
            torch.tensor(-0.4, requires_grad=True),
        ],
        "entropies": [
            torch.tensor(1.0, requires_grad=True),
            torch.tensor(0.5, requires_grad=True),
        ],
        "value_predictions": [
            torch.tensor(0.5, requires_grad=True),
            torch.tensor(1.5, requires_grad=True),
        ],
        "value_targets": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "advantages": torch.tensor([1.0, 2.0], dtype=torch.float32),
    }

    loss_dict = compute_total_loss(
        trajectory=trajectory,
        entropy_coef=0.1,
        value_loss_coef=0.5,
        manager_loss_coef=1.0,
        reduction="mean",
        manager_loss_enabled=False,
    )

    print("\n[Total Loss]")
    print("worker_loss:", float(loss_dict["worker_loss"].item()))
    print("value_loss:", float(loss_dict["value_loss"].item()))
    print("entropy_bonus:", float(loss_dict["entropy_bonus"].item()))
    print("manager_loss:", float(loss_dict["manager_loss"].item()))
    print("total_loss:", float(loss_dict["total_loss"].item()))

    expected_worker = torch.tensor((0.2 + 0.8) / 2.0, dtype=torch.float32)
    expected_value = torch.tensor((((0.5 - 1.0) ** 2) + ((1.5 - 1.0) ** 2)) / 2.0, dtype=torch.float32)
    expected_entropy = torch.tensor((1.0 + 0.5) / 2.0, dtype=torch.float32)
    expected_total = expected_worker + 0.5 * expected_value - 0.1 * expected_entropy

    assert torch.allclose(loss_dict["worker_loss"], expected_worker)
    assert torch.allclose(loss_dict["value_loss"], expected_value)
    assert torch.allclose(loss_dict["entropy_bonus"], expected_entropy)
    assert torch.allclose(loss_dict["manager_loss"], torch.tensor(0.0))
    assert torch.allclose(loss_dict["total_loss"], expected_total)


def test_total_loss_backward_runs() -> None:
    trajectory = {
        "log_probs": [
            torch.tensor(-0.3, requires_grad=True),
            torch.tensor(-0.7, requires_grad=True),
        ],
        "entropies": [
            torch.tensor(0.8, requires_grad=True),
            torch.tensor(0.4, requires_grad=True),
        ],
        "value_predictions": [
            torch.tensor(0.1, requires_grad=True),
            torch.tensor(0.2, requires_grad=True),
        ],
        "value_targets": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "advantages": torch.tensor([2.0, 1.0], dtype=torch.float32),
    }

    loss_dict = compute_total_loss(trajectory=trajectory, entropy_coef=0.05, value_loss_coef=0.5)
    loss_dict["total_loss"].backward()

    print("\n[Total Loss Backward]")
    print("grad log_prob 0:", trajectory["log_probs"][0].grad)
    print("grad log_prob 1:", trajectory["log_probs"][1].grad)
    print("grad value 0    :", trajectory["value_predictions"][0].grad)
    print("grad value 1    :", trajectory["value_predictions"][1].grad)
    print("grad entropy 0 :", trajectory["entropies"][0].grad)
    print("grad entropy 1 :", trajectory["entropies"][1].grad)

    assert trajectory["log_probs"][0].grad is not None
    assert trajectory["log_probs"][1].grad is not None
    assert trajectory["value_predictions"][0].grad is not None
    assert trajectory["value_predictions"][1].grad is not None
    assert trajectory["entropies"][0].grad is not None
    assert trajectory["entropies"][1].grad is not None


def test_compute_manager_loss_with_goal_updates() -> None:
    trajectory = {
        "manager_values": [
            torch.tensor(0.2, requires_grad=True),
            torch.tensor(0.5, requires_grad=True),
            torch.tensor(1.0, requires_grad=True),
        ],
        "value_targets": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        "goal_updated": [True, False, True],
    }
    manager_loss = compute_manager_loss(trajectory, enabled=True)

    print("\n[Manager Loss Active]")
    print("manager_loss:", float(manager_loss.item()))

    expected = torch.tensor((((0.2 - 1.0) ** 2) + ((1.0 - 3.0) ** 2)) / 2.0, dtype=torch.float32)
    assert torch.allclose(manager_loss, expected)


def test_compute_manager_loss_disabled() -> None:
    manager_loss = compute_manager_loss({"dummy": True}, enabled=False)

    print("\n[Manager Loss Disabled]")
    print("manager_loss:", float(manager_loss.item()))

    assert torch.allclose(manager_loss, torch.tensor(0.0))


if __name__ == "__main__":
    test_compute_worker_loss_mean()
    test_compute_entropy_bonus_mean()
    test_compute_value_loss_mean()
    test_compute_total_loss_with_entropy_and_placeholder_manager()
    test_total_loss_backward_runs()
    test_compute_manager_loss_with_goal_updates()
    test_compute_manager_loss_disabled()
    print("\nAll loss tests passed.")
