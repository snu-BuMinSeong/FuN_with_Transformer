from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SEEDS = [1, 11, 44]
RESULT_COLUMNS = [
    "Seed",
    "Final Success Rate",
    "Final Mean Return",
    "Final Episode Length",
    "Best Success Rate",
    "Best Checkpoint",
]


def parse_args() -> argparse.Namespace:
    """Parse baseline aggregation arguments."""
    parser = argparse.ArgumentParser(description="Aggregate vanilla FuN baseline seed results.")
    parser.add_argument("--log-root", type=str, default="logs/baseline_fun")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: Any) -> float | None:
    if value is None or value == "" or value == "missing":
        return None
    return float(value)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value:.6f}"


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def collect_seed_result(log_root: Path, seed: int) -> dict[str, Any]:
    """Collect the final eval row and checkpoint metadata for one seed."""
    seed_dir = log_root / f"seed_{seed}"
    eval_path = seed_dir / "eval.csv"
    summary_path = seed_dir / "summary.json"
    best_checkpoint = Path("checkpoints") / "baseline_fun" / f"seed_{seed}" / "best.pt"

    result: dict[str, Any] = {
        "Seed": str(seed),
        "Final Success Rate": None,
        "Final Mean Return": None,
        "Final Episode Length": None,
        "Best Success Rate": None,
        "Best Checkpoint": best_checkpoint.as_posix(),
        "missing": [],
    }

    eval_rows: list[dict[str, str]] = []
    if eval_path.exists():
        eval_rows = _read_csv_rows(eval_path)
    else:
        result["missing"].append(str(eval_path))

    if eval_rows:
        final_row = eval_rows[-1]
        result["Final Success Rate"] = _to_float(final_row.get("eval_success_rate"))
        result["Final Mean Return"] = _to_float(final_row.get("eval_mean_return"))
        result["Final Episode Length"] = _to_float(final_row.get("eval_mean_episode_length"))
        result["Best Success Rate"] = max(
            value
            for value in (_to_float(row.get("eval_success_rate")) for row in eval_rows)
            if value is not None
        )
    elif eval_path.exists():
        result["missing"].append(f"{eval_path} is empty")

    if summary_path.exists():
        summary = _read_json(summary_path)
        summary_best = _to_float(summary.get("best_success_rate"))
        if summary_best is not None:
            result["Best Success Rate"] = summary_best
        summary_checkpoint = summary.get("best_checkpoint_path")
        if summary_checkpoint:
            result["Best Checkpoint"] = str(summary_checkpoint).replace("\\", "/")
    else:
        result["missing"].append(str(summary_path))

    return result


def build_rows(seed_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build formatted seed rows plus a mean row."""
    rows: list[dict[str, str]] = []
    for result in seed_results:
        missing = result["missing"]
        rows.append(
            {
                "Seed": result["Seed"],
                "Final Success Rate": _format_metric(result["Final Success Rate"]),
                "Final Mean Return": _format_metric(result["Final Mean Return"]),
                "Final Episode Length": _format_metric(result["Final Episode Length"]),
                "Best Success Rate": _format_metric(result["Best Success Rate"]),
                "Best Checkpoint": "missing" if missing else result["Best Checkpoint"],
            }
        )

    rows.append(
        {
            "Seed": "Mean",
            "Final Success Rate": _format_metric(_mean([row["Final Success Rate"] for row in seed_results])),
            "Final Mean Return": _format_metric(_mean([row["Final Mean Return"] for row in seed_results])),
            "Final Episode Length": _format_metric(_mean([row["Final Episode Length"] for row in seed_results])),
            "Best Success Rate": _format_metric(_mean([row["Best Success Rate"] for row in seed_results])),
            "Best Checkpoint": "-",
        }
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Seed | Final Success Rate | Final Mean Return | Final Episode Length | Best Success Rate | Best Checkpoint |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["Seed"],
                    row["Final Success Rate"],
                    row["Final Mean Return"],
                    row["Final Episode Length"],
                    row["Best Success Rate"],
                    row["Best Checkpoint"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_summary(path: Path, table_text: str) -> None:
    summary = f"""# Week 3 Baseline Summary

## 실험 목적

Vanilla FuN을 memory ablation 실험의 비교 기준으로 고정한다.

## 실험 환경

- MiniGrid-DoorKey-5x5-v0
- total episodes: 1000
- eval interval: 50
- eval episodes: 20
- evaluation action mode: argmax

## 모델 구조

- Observation Encoder
- GRU-based Manager
- Worker
- Value head
- Manager value head

## 사용 Config

- `configs/train_fun_baseline_seed1.yaml`
- `configs/train_fun_baseline_seed11.yaml`
- `configs/train_fun_baseline_seed44.yaml`

## Seed별 결과

{table_text}

## 그래프

- `figures/baseline_fun/eval_success_rate.png`
- `figures/baseline_fun/eval_mean_return.png`
- `figures/baseline_fun/eval_episode_length.png`

## 관찰

- 학습 안정성:
- sparse reward 한계:
- seed별 편차:
- 4주차 memory ablation 비교 기준:
"""
    path.write_text(summary, encoding="utf-8")


def main() -> None:
    """Aggregate baseline eval logs into CSV, Markdown, and summary files."""
    args = parse_args()
    log_root = Path(args.log_root)
    results_dir = Path(args.results_dir)
    seed_results = [collect_seed_result(log_root, seed) for seed in SEEDS]

    if all(result["missing"] for result in seed_results):
        print(f"No baseline results found under {log_root}. Run training first.")
        return

    rows = build_rows(seed_results)
    table_text = markdown_table(rows)

    csv_path = results_dir / "week3_baseline_results.csv"
    md_path = results_dir / "week3_baseline_results.md"
    summary_path = results_dir / "week3_baseline_summary.md"

    write_csv(csv_path, rows)
    md_path.write_text(table_text + "\n", encoding="utf-8")
    write_summary(summary_path, table_text)

    print(table_text)
    print(f"saved_csv={csv_path}")
    print(f"saved_markdown={md_path}")
    print(f"saved_summary={summary_path}")

    for result in seed_results:
        if result["missing"]:
            print(f"seed_{result['Seed']}_missing={', '.join(result['missing'])}")


if __name__ == "__main__":
    main()
