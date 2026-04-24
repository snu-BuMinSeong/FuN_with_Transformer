from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.returns import (
    attach_returns_and_advantages,
    compute_advantages,
    compute_reward_to_go,
    stack_value_predictions,
)


def test_compute_reward_to_go_without_discount() -> None:
    rewards = [1.0, 2.0, 3.0]
    returns = compute_reward_to_go(rewards, gamma=1.0)

    print("[Reward-to-Go No Discount]")
    print("rewards:", rewards)
    print("returns:", returns.tolist())

    assert returns.shape == (3,)
    assert torch.allclose(returns, torch.tensor([6.0, 5.0, 3.0]))


def test_compute_reward_to_go_with_discount() -> None:
    rewards = [1.0, 2.0, 3.0]
    returns = compute_reward_to_go(rewards, gamma=0.5)

    print("\n[Reward-to-Go Discounted]")
    print("rewards:", rewards)
    print("returns:", returns.tolist())

    expected = torch.tensor([2.75, 3.5, 3.0], dtype=torch.float32)
    assert returns.shape == (3,)
    assert torch.allclose(returns, expected)


def test_compute_advantages_without_baseline() -> None:
    returns = torch.tensor([6.0, 5.0, 3.0], dtype=torch.float32)
    advantages = compute_advantages(returns)

    print("\n[Advantages No Baseline]")
    print("returns:", returns.tolist())
    print("advantages:", advantages.tolist())

    assert advantages.shape == (3,)
    assert torch.allclose(advantages, returns)


def test_compute_advantages_with_baseline() -> None:
    returns = torch.tensor([6.0, 5.0, 3.0], dtype=torch.float32)
    baseline = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32)
    advantages = compute_advantages(returns, baseline=baseline)

    print("\n[Advantages With Baseline]")
    print("returns:", returns.tolist())
    print("baseline:", baseline.tolist())
    print("advantages:", advantages.tolist())

    expected = torch.tensor([5.0, 3.5, 1.0], dtype=torch.float32)
    assert advantages.shape == (3,)
    assert torch.allclose(advantages, expected)


def test_attach_returns_and_advantages_to_trajectory() -> None:
    trajectory = {
        "rewards": [0.0, 1.0, 0.0, 2.0],
        "actions": [1, 2, 3, 4],
        "values": [
            torch.tensor(0.5, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(0.5, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        ],
    }
    enriched = attach_returns_and_advantages(trajectory, gamma=1.0)

    print("\n[Attach Returns And Advantages]")
    print("returns:", enriched["returns"].tolist())
    print("advantages:", enriched["advantages"].tolist())
    print("value_predictions:", enriched["value_predictions"].tolist())

    assert enriched["returns"].shape == (4,)
    assert enriched["advantages"].shape == (4,)
    assert torch.allclose(enriched["returns"], torch.tensor([3.0, 3.0, 2.0, 2.0]))
    assert torch.allclose(enriched["value_predictions"], torch.tensor([0.5, 1.0, 0.5, 0.0]))
    assert torch.allclose(enriched["advantages"], torch.tensor([2.5, 2.0, 1.5, 2.0]))
    assert enriched["actions"] == [1, 2, 3, 4]


def test_stack_value_predictions() -> None:
    values = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    stacked = stack_value_predictions(values)

    print("\n[Stack Value Predictions]")
    print("stacked:", stacked.tolist())

    assert stacked.shape == (3,)
    assert torch.allclose(stacked, torch.tensor([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    test_compute_reward_to_go_without_discount()
    test_compute_reward_to_go_with_discount()
    test_compute_advantages_without_baseline()
    test_compute_advantages_with_baseline()
    test_stack_value_predictions()
    test_attach_returns_and_advantages_to_trajectory()
    print("\nAll return and advantage tests passed.")
