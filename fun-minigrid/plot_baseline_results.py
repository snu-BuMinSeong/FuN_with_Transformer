from __future__ import annotations

import argparse
import csv
from pathlib import Path


SEEDS = [1, 11, 44]
PLOTS = [
    ("eval_success_rate", "Evaluation Success Rate", "Success Rate", "eval_success_rate.png"),
    ("eval_mean_return", "Evaluation Mean Return", "Mean Return", "eval_mean_return.png"),
    ("eval_mean_episode_length", "Evaluation Episode Length", "Episode Length", "eval_episode_length.png"),
]


def parse_args() -> argparse.Namespace:
    """Parse baseline plotting arguments."""
    parser = argparse.ArgumentParser(description="Plot vanilla FuN baseline evaluation curves.")
    parser.add_argument("--log-root", type=str, default="logs/baseline_fun")
    parser.add_argument("--output-dir", type=str, default="figures/baseline_fun")
    return parser.parse_args()


def _to_float(value: str | None) -> float:
    if value is None or value == "":
        raise ValueError("missing numeric CSV value")
    return float(value)


def load_eval_curve(eval_path: Path) -> dict[str, list[float]]:
    """Load one eval.csv file."""
    with eval_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"eval log is empty: {eval_path}")

    data: dict[str, list[float]] = {"episode": [_to_float(row.get("episode")) for row in rows]}
    for metric, _, _, _ in PLOTS:
        data[metric] = [_to_float(row.get(metric)) for row in rows]
    return data


def main() -> None:
    """Plot seed-level baseline evaluation curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install it with: pip install matplotlib") from exc

    args = parse_args()
    log_root = Path(args.log_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves: dict[int, dict[str, list[float]]] = {}
    for seed in SEEDS:
        eval_path = log_root / f"seed_{seed}" / "eval.csv"
        if not eval_path.exists():
            print(f"skip_missing_seed={seed} path={eval_path}")
            continue
        try:
            curves[seed] = load_eval_curve(eval_path)
        except ValueError as exc:
            print(f"skip_invalid_seed={seed} reason={exc}")

    if not curves:
        print(f"No eval.csv files found under {log_root}. Run training first.")
        return

    for metric, title, ylabel, filename in PLOTS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for seed, data in curves.items():
            ax.plot(data["episode"], data[metric], marker="o", linewidth=1.5, label=f"seed {seed}")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        save_path = output_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved_plot={save_path}")


if __name__ == "__main__":
    main()
