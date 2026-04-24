from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from train import build_progress_line, compute_moving_average


def test_compute_moving_average_uses_trailing_window() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    moving_avg = compute_moving_average(values, window=2)

    print("[Moving Average]")
    print("values:", values)
    print("moving_avg:", moving_avg)

    assert moving_avg == 3.5


def test_compute_moving_average_handles_short_history() -> None:
    moving_avg = compute_moving_average([2.0], window=5)

    print("\n[Moving Average Short History]")
    print("moving_avg:", moving_avg)

    assert moving_avg == 2.0


def test_build_progress_line_contains_moving_averages() -> None:
    line = build_progress_line(
        result={
            "episode": 25,
            "total_reward": 0.0,
            "episode_length": 250,
            "total_loss": -0.01,
            "value_loss": 0.25,
            "manager_loss": 0.05,
            "worker_loss": 0.4,
            "entropy_mean": 1.2,
            "advantages_abs_mean": 0.3,
            "grad_norm": 0.1,
            "encoder_grad_norm": 0.01,
            "manager_grad_norm": 0.02,
            "worker_grad_norm": 0.03,
            "value_head_grad_norm": 0.04,
            "manager_value_head_grad_norm": 0.05,
            "reward_moving_avg": 0.2,
            "success_moving_avg": 0.1,
            "loss_moving_avg": -0.02,
            "nonzero_reward_fraction": 0.0,
            "nonzero_return_fraction": 0.0,
            "action_coverage": 0.5,
            "success": False,
        },
        total_episodes=300,
        best_reward=0.5,
        best_success_rate=0.2,
    )

    print("\n[Progress Line]")
    print("line:", line)

    assert "episode=25/300" in line
    assert "worker_loss=0.400000" in line
    assert "value_loss=0.250000" in line
    assert "manager_loss=0.050000" in line
    assert "entropy=1.200000" in line
    assert "adv_abs=0.300000" in line
    assert "grads[e/m/w/v/mv]=0.0100/0.0200/0.0300/0.0400/0.0500" in line
    assert "reward_signal=0.000" in line
    assert "action_cov=0.500" in line
    assert "reward_ma=0.200" in line
    assert "success_ma=0.100" in line
    assert "loss_ma=-0.020000" in line
    assert "best_reward=0.500" in line
    assert "best_success_ma=0.200" in line


if __name__ == "__main__":
    test_compute_moving_average_uses_trailing_window()
    test_compute_moving_average_handles_short_history()
    test_build_progress_line_contains_moving_averages()
    print("\nAll train utility tests passed.")
