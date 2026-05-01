from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SEEDS = [1, 11, 44]
RESULT_COLUMNS = [
    "Seed",
    "Sample Success Rate",
    "Sample Mean Return",
    "Sample Episode Length",
    "Argmax Success Rate",
    "Argmax Mean Return",
    "Argmax Episode Length",
    "Best Checkpoint",
    "Note",
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


def _compact_checkpoint_eval(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    data = _read_json(path)
    eval_result = data.get("eval_result", {})
    return {
        "success_rate": _to_float(eval_result.get("mean_success_rate")),
        "mean_return": _to_float(eval_result.get("mean_reward")),
        "episode_length": _to_float(eval_result.get("mean_episode_length")),
    }


def _fallback_eval_csv(eval_path: Path) -> dict[str, float] | None:
    if not eval_path.exists():
        return None
    rows = _read_csv_rows(eval_path)
    if not rows:
        return None
    final = rows[-1]
    return {
        "success_rate": _to_float(final.get("eval_success_rate")),
        "mean_return": _to_float(final.get("eval_mean_return")),
        "episode_length": _to_float(final.get("eval_mean_episode_length")),
    }


def _summary_checkpoint(summary_path: Path, default: Path) -> str:
    if not summary_path.exists():
        return default.as_posix()
    summary = _read_json(summary_path)
    checkpoint = summary.get("best_checkpoint_path")
    if not checkpoint:
        return default.as_posix()
    return str(checkpoint).replace("\\", "/")


def _train_row_warning(seed_dir: Path) -> str | None:
    train_path = seed_dir / "train.csv"
    summary_path = seed_dir / "summary.json"
    if not train_path.exists() or not summary_path.exists():
        return None

    rows = _read_csv_rows(train_path)
    summary = _read_json(summary_path)
    final_eval = summary.get("final_eval", {})
    expected = final_eval.get("episode") if isinstance(final_eval, dict) else None
    if isinstance(expected, (int, float)) and len(rows) != int(expected):
        return f"warning: train_rows={len(rows)} summary_final_episode={int(expected)}"
    return None


def collect_seed_result(log_root: Path, seed: int) -> dict[str, Any]:
    """Collect the official sample checkpoint result and reference argmax metric."""
    seed_dir = log_root / f"seed_{seed}"
    eval_path = seed_dir / "eval.csv"
    summary_path = seed_dir / "summary.json"
    sample_path = seed_dir / "checkpoint_eval_best_sample.json"
    argmax_path = seed_dir / "checkpoint_eval_best_argmax.json"
    legacy_argmax_path = seed_dir / "checkpoint_eval_best.json"
    best_checkpoint = Path("checkpoints") / "baseline_fun" / f"seed_{seed}" / "best.pt"

    notes: list[str] = []
    sample = _compact_checkpoint_eval(sample_path)
    if sample is None:
        sample = _fallback_eval_csv(eval_path)
        if sample is None:
            sample = {"success_rate": None, "mean_return": None, "episode_length": None}
            notes.append(f"missing sample eval and eval fallback: {sample_path}")
        else:
            notes.append("sample eval missing; used eval.csv fallback")

    argmax = _compact_checkpoint_eval(argmax_path)
    if argmax is None:
        argmax = _compact_checkpoint_eval(legacy_argmax_path)
    if argmax is None:
        fallback = _fallback_eval_csv(eval_path)
        argmax = fallback
        if fallback is None:
            argmax = {"success_rate": None, "mean_return": None, "episode_length": None}
            notes.append("argmax eval missing")

    row_warning = _train_row_warning(seed_dir)
    if row_warning:
        notes.append(row_warning)

    return {
        "Seed": str(seed),
        "Sample Success Rate": sample["success_rate"],
        "Sample Mean Return": sample["mean_return"],
        "Sample Episode Length": sample["episode_length"],
        "Argmax Success Rate": argmax["success_rate"],
        "Argmax Mean Return": argmax["mean_return"],
        "Argmax Episode Length": argmax["episode_length"],
        "Best Checkpoint": _summary_checkpoint(summary_path, best_checkpoint),
        "Note": "; ".join(notes),
    }


def build_rows(seed_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build formatted seed rows plus a mean row."""
    rows: list[dict[str, str]] = []
    for result in seed_results:
        rows.append(
            {
                "Seed": result["Seed"],
                "Sample Success Rate": _format_metric(result["Sample Success Rate"]),
                "Sample Mean Return": _format_metric(result["Sample Mean Return"]),
                "Sample Episode Length": _format_metric(result["Sample Episode Length"]),
                "Argmax Success Rate": _format_metric(result["Argmax Success Rate"]),
                "Argmax Mean Return": _format_metric(result["Argmax Mean Return"]),
                "Argmax Episode Length": _format_metric(result["Argmax Episode Length"]),
                "Best Checkpoint": result["Best Checkpoint"],
                "Note": result["Note"],
            }
        )

    rows.append(
        {
            "Seed": "Mean",
            "Sample Success Rate": _format_metric(_mean([row["Sample Success Rate"] for row in seed_results])),
            "Sample Mean Return": _format_metric(_mean([row["Sample Mean Return"] for row in seed_results])),
            "Sample Episode Length": _format_metric(_mean([row["Sample Episode Length"] for row in seed_results])),
            "Argmax Success Rate": _format_metric(_mean([row["Argmax Success Rate"] for row in seed_results])),
            "Argmax Mean Return": _format_metric(_mean([row["Argmax Mean Return"] for row in seed_results])),
            "Argmax Episode Length": _format_metric(_mean([row["Argmax Episode Length"] for row in seed_results])),
            "Best Checkpoint": "-",
            "Note": "sample mean over available seed checkpoint evaluations",
        }
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]], include_note: bool = False) -> str:
    headers = [
        "Seed",
        "Sample Success Rate",
        "Sample Mean Return",
        "Sample Episode Length",
        "Argmax Success Rate",
        "Argmax Mean Return",
        "Argmax Episode Length",
        "Best Checkpoint",
    ]
    if include_note:
        headers.append("Note")

    lines = [
        "| " + " | ".join(headers) + " |",
        "|---:|---:|---:|---:|---:|---:|---" + ("|---|" if include_note else "|"),
    ]
    for row in rows:
        values = [row[header] for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(path: Path, table_text: str) -> None:
    summary = f"""# Week 3 Baseline Summary

## 실험 목적

Vanilla FuN을 memory ablation 실험의 비교 기준으로 고정한다.

## 중요한 수정 사항

기존 argmax evaluation은 모든 seed에서 success가 0에 가깝게 나왔지만, sample evaluation에서는 학습 성능이 확인되었다.
따라서 본 프로젝트에서는 stochastic policy 특성을 고려하여 sample evaluation을 공식 baseline 평가 기준으로 사용하고, argmax evaluation은 참고 지표로 둔다.

## 실험 환경

- MiniGrid-DoorKey-5x5-v0
- total episodes: 1000
- eval interval: 50
- eval episodes: 20
- official evaluation action mode: sample
- reference evaluation action mode: argmax

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

## 해석

- 학습은 완전히 실패한 것이 아니다.
- train success와 sample checkpoint evaluation success가 확인되므로 policy distribution은 task-relevant behavior를 일부 학습했다.
- argmax는 특정 action 반복 또는 행동 다양성 제거로 인해 DoorKey task에서 실패하는 것으로 해석된다.
- 따라서 4주차 memory ablation 비교에서도 sample evaluation을 primary metric으로 사용해야 한다.
- seed별 편차가 존재하므로 평균뿐 아니라 seed별 결과를 함께 보고해야 한다.

## 그래프

- `figures/baseline_fun/sample_success_rate_by_seed.png`
- `figures/baseline_fun/sample_mean_return_by_seed.png`
- `figures/baseline_fun/sample_episode_length_by_seed.png`
- `figures/baseline_fun/argmax_vs_sample_success_rate.png`
- `figures/baseline_fun/argmax_eval_success_rate_curve.png`

## 한계

- seed별 편차가 크다.
- DoorKey sparse reward 특성상 안정적인 학습은 아직 어렵다.
- argmax deterministic policy 성능은 낮거나 0이다.
- sample evaluation은 stochastic하므로 평가 episode 수에 따른 분산이 있다.

## 4주차 비교 기준

primary:

- sample success rate
- sample mean return
- sample mean episode length

secondary:

- argmax success rate
- train stability
- seed variance
"""
    path.write_text(summary, encoding="utf-8")


def main() -> None:
    """Aggregate baseline eval logs into CSV, Markdown, and summary files."""
    args = parse_args()
    log_root = Path(args.log_root)
    results_dir = Path(args.results_dir)
    seed_results = [collect_seed_result(log_root, seed) for seed in SEEDS]

    rows = build_rows(seed_results)
    table_text = markdown_table(rows)

    csv_path = results_dir / "week3_baseline_results.csv"
    md_path = results_dir / "week3_baseline_results.md"
    summary_path = results_dir / "week3_baseline_summary.md"

    write_csv(csv_path, rows)
    md_path.write_text(markdown_table(rows, include_note=True) + "\n", encoding="utf-8")
    write_summary(summary_path, table_text)

    print(table_text)
    print(f"saved_csv={csv_path}")
    print(f"saved_markdown={md_path}")
    print(f"saved_summary={summary_path}")

    for result in seed_results:
        if result["Note"]:
            print(f"seed_{result['Seed']}_note={result['Note']}")


if __name__ == "__main__":
    main()
