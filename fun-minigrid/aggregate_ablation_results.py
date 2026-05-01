from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SEEDS = [1, 11, 44]
MODELS = {
    "baseline": "logs/baseline_fun",
    "ablation": "logs/ablation_fun",
}
CSV_COLUMNS = [
    "model",
    "seed",
    "sample_success_rate",
    "sample_mean_return",
    "sample_mean_episode_length",
    "sample_std_return",
    "sample_std_success_rate",
    "sample_std_episode_length",
    "argmax_success_rate",
    "argmax_mean_return",
    "argmax_mean_episode_length",
    "checkpoint_episode",
    "checkpoint_best_success_rate",
    "final_episode",
    "train_best_success_rate",
    "final_success_moving_avg",
    "final_reward_moving_avg",
    "episodes_with_reward_signal",
    "episodes_with_return_signal",
    "mean_entropy",
    "mean_action_coverage",
    "mean_encoder_grad_norm",
    "mean_manager_grad_norm",
    "mean_worker_grad_norm",
    "mean_value_head_grad_norm",
    "mean_manager_value_head_grad_norm",
]
MEAN_METRICS = [
    "sample_success_rate",
    "sample_mean_return",
    "sample_mean_episode_length",
    "argmax_success_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Week 4 FuN memory ablation results.")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def _read_json(path: Path, missing: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        missing.append(str(path).replace("\\", "/"))
        print(f"missing_file={path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        normalized_path = str(path).replace("\\", "/")
        missing.append(f"{normalized_path} (invalid JSON: {exc})")
        print(f"invalid_json={path} reason={exc}")
        return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _mean(values: list[Any]) -> float | None:
    numbers = [_to_float(value) for value in values]
    present = [value for value in numbers if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _eval_metrics(data: dict[str, Any] | None) -> dict[str, float | None]:
    eval_result = data.get("eval_result", {}) if isinstance(data, dict) else {}
    if not isinstance(eval_result, dict):
        eval_result = {}
    return {
        "mean_success_rate": _to_float(eval_result.get("mean_success_rate")),
        "mean_reward": _to_float(eval_result.get("mean_reward")),
        "mean_episode_length": _to_float(eval_result.get("mean_episode_length")),
        "std_reward": _to_float(eval_result.get("std_reward")),
        "std_success_rate": _to_float(eval_result.get("std_success_rate")),
        "std_episode_length": _to_float(eval_result.get("std_episode_length")),
    }


def collect_row(model: str, log_root: Path, seed: int, missing: list[str]) -> dict[str, Any]:
    seed_dir = log_root / f"seed_{seed}"
    sample = _read_json(seed_dir / "checkpoint_eval_best_sample.json", missing)
    argmax = _read_json(seed_dir / "checkpoint_eval_best_argmax.json", missing)
    summary = _read_json(seed_dir / "summary.json", missing)

    sample_metrics = _eval_metrics(sample)
    argmax_metrics = _eval_metrics(argmax)
    summary_data = summary if isinstance(summary, dict) else {}

    return {
        "model": model,
        "seed": seed,
        "sample_success_rate": sample_metrics["mean_success_rate"],
        "sample_mean_return": sample_metrics["mean_reward"],
        "sample_mean_episode_length": sample_metrics["mean_episode_length"],
        "sample_std_return": sample_metrics["std_reward"],
        "sample_std_success_rate": sample_metrics["std_success_rate"],
        "sample_std_episode_length": sample_metrics["std_episode_length"],
        "argmax_success_rate": argmax_metrics["mean_success_rate"],
        "argmax_mean_return": argmax_metrics["mean_reward"],
        "argmax_mean_episode_length": argmax_metrics["mean_episode_length"],
        "checkpoint_episode": sample.get("checkpoint_episode") if isinstance(sample, dict) else None,
        "checkpoint_best_success_rate": sample.get("checkpoint_best_success_rate") if isinstance(sample, dict) else None,
        "final_episode": summary_data.get("final_episode"),
        "train_best_success_rate": summary_data.get("best_success_rate"),
        "final_success_moving_avg": summary_data.get("final_success_moving_avg"),
        "final_reward_moving_avg": summary_data.get("final_reward_moving_avg"),
        "episodes_with_reward_signal": summary_data.get("episodes_with_reward_signal"),
        "episodes_with_return_signal": summary_data.get("episodes_with_return_signal"),
        "mean_entropy": summary_data.get("mean_entropy"),
        "mean_action_coverage": summary_data.get("mean_action_coverage"),
        "mean_encoder_grad_norm": summary_data.get("mean_encoder_grad_norm"),
        "mean_manager_grad_norm": summary_data.get("mean_manager_grad_norm"),
        "mean_worker_grad_norm": summary_data.get("mean_worker_grad_norm"),
        "mean_value_head_grad_norm": summary_data.get("mean_value_head_grad_norm"),
        "mean_manager_value_head_grad_norm": summary_data.get("mean_manager_value_head_grad_norm"),
    }


def mean_row(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        column: (model if column == "model" else "mean" if column == "seed" else _mean([row[column] for row in rows]))
        for column in CSV_COLUMNS
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format(row.get(column)) for column in CSV_COLUMNS})


def markdown_table(headers: list[str], align: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(align) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def model_means(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    means: dict[str, dict[str, float | None]] = {}
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model and row["seed"] != "mean"]
        means[model] = {
            "mean_sample_success_rate": _mean([row["sample_success_rate"] for row in model_rows]),
            "mean_sample_return": _mean([row["sample_mean_return"] for row in model_rows]),
            "mean_sample_episode_length": _mean([row["sample_mean_episode_length"] for row in model_rows]),
            "mean_argmax_success_rate": _mean([row["argmax_success_rate"] for row in model_rows]),
        }
    return means


def difference_summary(means: dict[str, dict[str, float | None]]) -> dict[str, float | None]:
    pairs = [
        ("sample_success_rate", "mean_sample_success_rate"),
        ("sample_return", "mean_sample_return"),
        ("sample_episode_length", "mean_sample_episode_length"),
        ("argmax_success_rate", "mean_argmax_success_rate"),
    ]
    diff: dict[str, float | None] = {}
    for output_key, mean_key in pairs:
        baseline = means["baseline"][mean_key]
        ablation = means["ablation"][mean_key]
        diff[output_key] = None if baseline is None or ablation is None else ablation - baseline
    return diff


def write_results_markdown(path: Path, rows: list[dict[str, Any]], means: dict[str, dict[str, float | None]]) -> None:
    seed_rows = [row for row in rows if row["seed"] != "mean"]
    table1 = markdown_table(
        [
            "Model",
            "Seed",
            "Sample Success Rate",
            "Sample Mean Return",
            "Sample Episode Length",
            "Argmax Success Rate",
        ],
        ["---", "---:", "---:", "---:", "---:", "---:"],
        [
            [
                str(row["model"]),
                str(row["seed"]),
                _format(row["sample_success_rate"]),
                _format(row["sample_mean_return"]),
                _format(row["sample_mean_episode_length"]),
                _format(row["argmax_success_rate"]),
            ]
            for row in seed_rows
        ],
    )
    table2 = markdown_table(
        [
            "Model",
            "Mean Sample Success Rate",
            "Mean Sample Return",
            "Mean Episode Length",
            "Mean Argmax Success Rate",
        ],
        ["---", "---:", "---:", "---:", "---:"],
        [
            [
                model,
                _format(values["mean_sample_success_rate"]),
                _format(values["mean_sample_return"]),
                _format(values["mean_sample_episode_length"]),
                _format(values["mean_argmax_success_rate"]),
            ]
            for model, values in means.items()
        ],
    )
    diff = difference_summary(means)
    table3 = markdown_table(
        ["Metric", "Ablation - Baseline"],
        ["---", "---:"],
        [
            ["Success rate difference", _format(diff["sample_success_rate"])],
            ["Mean return difference", _format(diff["sample_return"])],
            ["Episode length difference", _format(diff["sample_episode_length"])],
            ["Argmax success difference", _format(diff["argmax_success_rate"])],
        ],
    )
    path.write_text(
        "# Week 4 Baseline vs Ablation Results\n\n"
        "## Table 1. Best checkpoint sample evaluation\n\n"
        f"{table1}\n\n"
        "## Table 2. Model mean comparison\n\n"
        f"{table2}\n\n"
        "## Table 3. Difference\n\n"
        "Ablation - baseline.\n\n"
        f"{table3}\n",
        encoding="utf-8",
    )


def write_summary_markdown(
    path: Path,
    means: dict[str, dict[str, float | None]],
    diff: dict[str, float | None],
    missing: list[str],
) -> None:
    missing_note = "No missing files were detected."
    if missing:
        missing_lines = "\n".join(f"- {item}" for item in missing)
        missing_note = f"The following expected files were missing or invalid:\n\n{missing_lines}"

    text = f"""# Week 4 Memory Ablation Summary

## 1. Experiment Goal

- Vanilla FuN의 GRU Manager memory를 제거했을 때 성능이 어떻게 변하는지 확인한다.

## 2. Compared Models

- Vanilla FuN: GRUCell 기반 recurrent Manager
- Ablation FuN: feedforward AblationManager, current state embedding만 사용

## 3. Experimental Setup

- Environment: MiniGrid-DoorKey-5x5-v0
- Seeds: 1, 11, 44
- total_episodes: 1000
- eval_episodes: 100 for checkpoint evaluation
- Primary metric: sample success rate, sample mean return, sample episode length
- Secondary metric: argmax success rate

## 4. Main Results

- Baseline mean sample success rate: {_format(means["baseline"]["mean_sample_success_rate"])}
- Ablation mean sample success rate: {_format(means["ablation"]["mean_sample_success_rate"])}
- Difference: {_format(diff["sample_success_rate"])}
- Baseline mean return: {_format(means["baseline"]["mean_sample_return"])}
- Ablation mean return: {_format(means["ablation"]["mean_sample_return"])}
- Difference: {_format(diff["sample_return"])}
- Baseline mean episode length: {_format(means["baseline"]["mean_sample_episode_length"])}
- Ablation mean episode length: {_format(means["ablation"]["mean_sample_episode_length"])}
- Difference: {_format(diff["sample_episode_length"])}
- Baseline mean argmax success rate: {_format(means["baseline"]["mean_argmax_success_rate"])}
- Ablation mean argmax success rate: {_format(means["ablation"]["mean_argmax_success_rate"])}
- Difference: {_format(diff["argmax_success_rate"])}

## 5. Interpretation

- AblationManager가 vanilla FuN보다 높거나 비슷한 성능을 보이면, MiniGrid DoorKey-5x5에서는 recurrent memory가 필수적이지 않았다고 해석할 수 있다.
- Feedforward Manager가 더 단순해서 짧은 학습 조건에서 더 안정적으로 최적화되었을 가능성이 있다.
- Sample 평가에서는 성공하지만 argmax 평가는 둘 다 실패하면, 현재 정책은 deterministic path policy라기보다 stochastic sampling에 의존한다고 해석할 수 있다.

## 6. Limitations

- Seed가 3개라 통계적으로 강한 결론은 아니다.
- DoorKey-5x5는 작은 환경이라 recurrent memory의 장점이 잘 드러나지 않을 수 있다.
- Argmax evaluation이 모두 실패하므로 정책 안정성에는 한계가 있다.
- FuN 논문식 Manager objective를 완전히 재현한 것은 아니다.

## 7. Next Step

- 결과 그래프 생성
- 필요하면 더 어려운 DoorKey 환경 또는 longer training에서 재검증
- 이후 TransformerManager는 선택적 확장으로 진행

## Missing File Note

{missing_note}
"""
    path.write_text(text, encoding="utf-8")


def write_json_summary(path: Path, means: dict[str, dict[str, float | None]], diff: dict[str, float | None]) -> None:
    data = {
        "baseline": means["baseline"],
        "ablation": means["ablation"],
        "difference_ablation_minus_baseline": diff,
        "seeds": SEEDS,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    missing: list[str] = []

    seed_rows: list[dict[str, Any]] = []
    for model, log_root in MODELS.items():
        root = Path(log_root)
        model_rows = [collect_row(model, root, seed, missing) for seed in SEEDS]
        seed_rows.extend(model_rows)

    all_rows: list[dict[str, Any]] = []
    for model in MODELS:
        current = [row for row in seed_rows if row["model"] == model]
        all_rows.extend(current)
        all_rows.append(mean_row(model, current))

    means = model_means(seed_rows)
    diff = difference_summary(means)

    write_csv(results_dir / "week4_ablation_results.csv", all_rows)
    write_results_markdown(results_dir / "week4_ablation_results.md", seed_rows, means)
    write_summary_markdown(results_dir / "week4_ablation_summary.md", means, diff, missing)
    write_json_summary(results_dir / "week4_baseline_vs_ablation_summary.json", means, diff)

    print(f"saved_csv={results_dir / 'week4_ablation_results.csv'}")
    print(f"saved_markdown={results_dir / 'week4_ablation_results.md'}")
    print(f"saved_summary={results_dir / 'week4_ablation_summary.md'}")
    print(f"saved_json={results_dir / 'week4_baseline_vs_ablation_summary.json'}")
    print(
        "baseline_mean_sample_success_rate="
        f"{_format(means['baseline']['mean_sample_success_rate'])}"
    )
    print(
        "ablation_mean_sample_success_rate="
        f"{_format(means['ablation']['mean_sample_success_rate'])}"
    )


if __name__ == "__main__":
    main()
