from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any


METRIC_PLOTS = [
    ("sample_success_rate", "Baseline vs Ablation Success Rate", "Success Rate", "baseline_vs_ablation_success_rate.png"),
    ("sample_mean_return", "Baseline vs Ablation Mean Return", "Mean Return", "baseline_vs_ablation_mean_return.png"),
    (
        "sample_mean_episode_length",
        "Baseline vs Ablation Episode Length",
        "Episode Length",
        "baseline_vs_ablation_episode_length.png",
    ),
    (
        "argmax_success_rate",
        "Baseline vs Ablation Argmax Success Rate",
        "Success Rate",
        "baseline_vs_ablation_argmax_success_rate.png",
    ),
]
SEEDWISE_PLOTS = [
    ("sample_success_rate", "Seedwise Success Rate Comparison", "Success Rate", "seedwise_success_rate_comparison.png"),
    ("sample_mean_return", "Seedwise Mean Return Comparison", "Mean Return", "seedwise_mean_return_comparison.png"),
    (
        "sample_mean_episode_length",
        "Seedwise Episode Length Comparison",
        "Episode Length",
        "seedwise_episode_length_comparison.png",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Week 4 baseline vs ablation comparison graphs.")
    parser.add_argument("--csv-path", type=str, default="results/week4_ablation_results.csv")
    parser.add_argument("--output-dir", type=str, default="figures/ablation_fun")
    return parser.parse_args()


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise SystemExit(f"missing_file={csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _value(row: dict[str, str], metric: str) -> float:
    value = _to_float(row.get(metric))
    return 0.0 if value is None else value


def mean_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    means = {row["model"]: row for row in rows if row.get("seed") == "mean"}
    missing = [model for model in ("baseline", "ablation") if model not in means]
    if missing:
        raise SystemExit(f"missing_mean_rows={','.join(missing)}")
    return means


def seed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("seed") != "mean"]


def plot_mean_bar(plt: Any, rows: dict[str, dict[str, str]], metric: str, title: str, ylabel: str, path: Path) -> None:
    models = ["baseline", "ablation"]
    values = [_value(rows[model], metric) for model in models]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(models, values)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot={path}")


def plot_seedwise_grouped(plt: Any, rows: list[dict[str, str]], metric: str, title: str, ylabel: str, path: Path) -> None:
    seeds = sorted({row["seed"] for row in rows}, key=lambda value: int(value))
    by_model_seed = {(row["model"], row["seed"]): row for row in rows}
    x = list(range(len(seeds)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, model in [(-width / 2, "baseline"), (width / 2, "ablation")]:
        values = [_value(by_model_seed.get((model, seed), {}), metric) for seed in seeds]
        ax.bar([idx + offset for idx in x], values, width=width, label=model)
    ax.set_title(title)
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot={path}")


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install it with: pip install matplotlib") from exc

    args = parse_args()
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    means = mean_rows(rows)
    seeds = seed_rows(rows)

    for metric, title, ylabel, filename in METRIC_PLOTS:
        plot_mean_bar(plt, means, metric, title, ylabel, output_dir / filename)

    for metric, title, ylabel, filename in SEEDWISE_PLOTS:
        plot_seedwise_grouped(plt, seeds, metric, title, ylabel, output_dir / filename)


if __name__ == "__main__":
    main()
