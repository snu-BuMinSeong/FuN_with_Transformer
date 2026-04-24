from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse plotting script arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="logs/week2_train.csv")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--save", type=str, default=None)
    return parser.parse_args()


def to_float(value: str | None, default: float = 0.0) -> float:
    """Convert a CSV field to float with a default fallback."""
    if value is None or value == "":
        return default
    if value.lower() == "true":
        return 1.0
    if value.lower() == "false":
        return 0.0
    return float(value)


def compute_moving_average(values: list[float], window: int) -> list[float]:
    """Return a trailing moving average series."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}.")

    moving_avg: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        moving_avg.append(sum(chunk) / len(chunk))
    return moving_avg


def load_training_log(csv_path: str, window: int) -> dict[str, list[float]]:
    """Load training log CSV and prepare raw and moving-average series."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV log not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"CSV log is empty: {path}")

    episodes = [int(to_float(row.get("episode"), default=float(idx + 1))) for idx, row in enumerate(rows)]
    rewards = [to_float(row.get("total_reward")) for row in rows]
    successes = [to_float(row.get("success")) for row in rows]
    losses = [to_float(row.get("total_loss")) for row in rows]

    reward_ma = [
        to_float(row.get("reward_moving_avg"), default=float("nan"))
        for row in rows
    ]
    success_ma = [
        to_float(row.get("success_moving_avg"), default=float("nan"))
        for row in rows
    ]
    loss_ma = [
        to_float(row.get("loss_moving_avg"), default=float("nan"))
        for row in rows
    ]

    if any(value != value for value in reward_ma):
        reward_ma = compute_moving_average(rewards, window)
    if any(value != value for value in success_ma):
        success_ma = compute_moving_average(successes, window)
    if any(value != value for value in loss_ma):
        loss_ma = compute_moving_average(losses, window)

    return {
        "episodes": episodes,
        "rewards": rewards,
        "successes": successes,
        "losses": losses,
        "reward_ma": reward_ma,
        "success_ma": success_ma,
        "loss_ma": loss_ma,
    }


def main() -> None:
    """Plot reward, success, and loss curves from a training CSV log."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required to run plot_training.py.") from exc

    args = parse_args()
    data = load_training_log(args.csv, window=args.window)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(data["episodes"], data["rewards"], label="reward", alpha=0.5)
    axes[0].plot(data["episodes"], data["reward_ma"], label="reward_ma", linewidth=2)
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Episode vs Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(data["episodes"], data["successes"], label="success", alpha=0.3)
    axes[1].plot(data["episodes"], data["success_ma"], label="success_ma", linewidth=2)
    axes[1].set_ylabel("Success")
    axes[1].set_title("Episode vs Success")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(data["episodes"], data["losses"], label="loss", alpha=0.5)
    axes[2].plot(data["episodes"], data["loss_ma"], label="loss_ma", linewidth=2)
    axes[2].set_ylabel("Loss")
    axes[2].set_xlabel("Episode")
    axes[2].set_title("Episode vs Loss")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved_plot={save_path}")

    plt.show()


if __name__ == "__main__":
    main()
