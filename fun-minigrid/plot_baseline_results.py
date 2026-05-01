from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


SEEDS = [1, 11, 44]
ARGMAX_CURVE_PLOTS = [
    ("eval_success_rate", "Argmax Evaluation Success Rate", "Success Rate", "argmax_eval_success_rate_curve.png"),
    ("eval_mean_return", "Argmax Evaluation Mean Return", "Mean Return", "eval_mean_return.png"),
    ("eval_mean_episode_length", "Argmax Evaluation Episode Length", "Episode Length", "eval_episode_length.png"),
]


def parse_args() -> argparse.Namespace:
    """Parse baseline plotting arguments."""
    parser = argparse.ArgumentParser(description="Plot vanilla FuN baseline evaluation results.")
    parser.add_argument("--log-root", type=str, default="logs/baseline_fun")
    parser.add_argument("--output-dir", type=str, default="figures/baseline_fun")
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_eval(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    data = _read_json(path)
    eval_result = data.get("eval_result", {})
    return {
        "success_rate": _to_float(eval_result.get("mean_success_rate")) or 0.0,
        "mean_return": _to_float(eval_result.get("mean_reward")) or 0.0,
        "episode_length": _to_float(eval_result.get("mean_episode_length")) or 0.0,
    }


def _eval_csv_final(eval_path: Path) -> dict[str, float] | None:
    if not eval_path.exists():
        return None
    with eval_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None
    final = rows[-1]
    return {
        "success_rate": _to_float(final.get("eval_success_rate")) or 0.0,
        "mean_return": _to_float(final.get("eval_mean_return")) or 0.0,
        "episode_length": _to_float(final.get("eval_mean_episode_length")) or 0.0,
    }


def load_seed_bars(log_root: Path) -> dict[int, dict[str, float]]:
    """Load sample and argmax checkpoint-level metrics for seed bar plots."""
    data: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        seed_dir = log_root / f"seed_{seed}"
        sample = _checkpoint_eval(seed_dir / "checkpoint_eval_best_sample.json")
        if sample is None:
            sample = _eval_csv_final(seed_dir / "eval.csv")
            if sample is None:
                print(f"skip_missing_sample_seed={seed}")
                continue
            print(f"sample_missing_used_eval_fallback_seed={seed}")

        argmax = _checkpoint_eval(seed_dir / "checkpoint_eval_best_argmax.json")
        if argmax is None:
            argmax = _checkpoint_eval(seed_dir / "checkpoint_eval_best.json")
        if argmax is None:
            argmax = _eval_csv_final(seed_dir / "eval.csv")
        if argmax is None:
            argmax = {"success_rate": 0.0, "mean_return": 0.0, "episode_length": 0.0}
            print(f"argmax_missing_seed={seed}")

        data[seed] = {
            "sample_success_rate": sample["success_rate"],
            "sample_mean_return": sample["mean_return"],
            "sample_episode_length": sample["episode_length"],
            "argmax_success_rate": argmax["success_rate"],
        }
    return data


def _plot_bar(plt: Any, labels: list[str], values: list[float], title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot={path}")


def _plot_grouped_success(plt: Any, labels: list[str], sample: list[float], argmax: list[float], path: Path) -> None:
    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar([idx - width / 2 for idx in x], sample, width=width, label="sample", color="#4C78A8")
    ax.bar([idx + width / 2 for idx in x], argmax, width=width, label="argmax", color="#F58518")
    ax.set_title("Argmax vs Sample Success Rate")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Success Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot={path}")


def _to_required_float(value: str | None) -> float:
    if value is None or value == "":
        raise ValueError("missing numeric CSV value")
    return float(value)


def load_eval_curve(eval_path: Path) -> dict[str, list[float]]:
    """Load one argmax eval.csv file."""
    with eval_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"eval log is empty: {eval_path}")

    data: dict[str, list[float]] = {"episode": [_to_required_float(row.get("episode")) for row in rows]}
    for metric, _, _, _ in ARGMAX_CURVE_PLOTS:
        data[metric] = [_to_required_float(row.get(metric)) for row in rows]
    return data


def plot_argmax_curves(plt: Any, log_root: Path, output_dir: Path) -> None:
    """Keep the old eval.csv curves as reference argmax plots."""
    curves: dict[int, dict[str, list[float]]] = {}
    for seed in SEEDS:
        eval_path = log_root / f"seed_{seed}" / "eval.csv"
        if not eval_path.exists():
            print(f"skip_missing_argmax_curve_seed={seed} path={eval_path}")
            continue
        try:
            curves[seed] = load_eval_curve(eval_path)
        except ValueError as exc:
            print(f"skip_invalid_argmax_curve_seed={seed} reason={exc}")

    if not curves:
        return

    for metric, title, ylabel, filename in ARGMAX_CURVE_PLOTS:
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


def main() -> None:
    """Plot seed-level sample checkpoint metrics and reference argmax curves."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install it with: pip install matplotlib") from exc

    args = parse_args()
    log_root = Path(args.log_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bars = load_seed_bars(log_root)
    if not bars:
        print(f"No sample checkpoint eval files found under {log_root}. Run sample evaluation first.")
        return

    labels = [str(seed) for seed in bars]
    sample_success = [bars[seed]["sample_success_rate"] for seed in bars]
    sample_return = [bars[seed]["sample_mean_return"] for seed in bars]
    sample_length = [bars[seed]["sample_episode_length"] for seed in bars]
    argmax_success = [bars[seed]["argmax_success_rate"] for seed in bars]

    _plot_bar(
        plt,
        labels,
        sample_success,
        "Sample Success Rate by Seed",
        "Success Rate",
        output_dir / "sample_success_rate_by_seed.png",
    )
    _plot_bar(
        plt,
        labels,
        sample_return,
        "Sample Mean Return by Seed",
        "Mean Return",
        output_dir / "sample_mean_return_by_seed.png",
    )
    _plot_bar(
        plt,
        labels,
        sample_length,
        "Sample Episode Length by Seed",
        "Episode Length",
        output_dir / "sample_episode_length_by_seed.png",
    )
    _plot_grouped_success(
        plt,
        labels,
        sample_success,
        argmax_success,
        output_dir / "argmax_vs_sample_success_rate.png",
    )
    plot_argmax_curves(plt, log_root, output_dir)


if __name__ == "__main__":
    main()
