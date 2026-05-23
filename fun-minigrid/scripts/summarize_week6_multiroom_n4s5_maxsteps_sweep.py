from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
LOG_ROOT = ROOT / "logs" / "multiroom_n4s5_maxsteps_sweep"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "multiroom_n4s5_maxsteps_sweep"

RUNS = [
    ("Vanilla FuN", "baseline", 1, 250, "baseline_ms250_seed_1"),
    ("Vanilla FuN", "baseline", 1, 500, "baseline_ms500_seed_1"),
    ("Vanilla FuN", "baseline", 1, 750, "baseline_ms750_seed_1"),
    ("AblationManager", "ablation", 1, 250, "ablation_ms250_seed_1"),
    ("AblationManager", "ablation", 1, 500, "ablation_ms500_seed_1"),
    ("AblationManager", "ablation", 1, 750, "ablation_ms750_seed_1"),
    ("AblationManager", "ablation", 1, 1000, "ablation_ms1000_seed_1"),
]

TABLE_COLUMNS = [
    "Model",
    "Seed",
    "Max Steps",
    "Train Success Count",
    "Train Success Rate",
    "Final100 Train Success",
    "Final Sample Success",
    "Final Argmax Success",
    "Final Sample Return",
    "Final Argmax Return",
    "Final Sample Length",
    "Final Argmax Length",
    "Reward Signal Episodes",
    "Best Checkpoint",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def as_float(value: str | int | float | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def as_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def fmt(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def has_nonfinite(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        for value in row.values():
            lower = str(value).lower()
            if lower in {"nan", "inf", "+inf", "-inf", "infinity", "-infinity"}:
                return True
    return False


def summarize_run(model: str, key: str, seed: int, max_steps: int, dirname: str) -> dict[str, Any]:
    log_dir = LOG_ROOT / dirname
    checkpoint_dir = CHECKPOINT_ROOT / dirname
    train_rows = read_csv_rows(log_dir / "train.csv")
    eval_rows = read_csv_rows(log_dir / "eval.csv")
    summary = read_json(log_dir / "summary.json")

    success_count = sum(1 for row in train_rows if as_bool(row.get("success")))
    reward_signal_episodes = sum(1 for row in train_rows if as_bool(row.get("has_reward_signal")))
    train_success_rate = success_count / len(train_rows) if train_rows else 0.0
    final100 = train_rows[-100:]
    final100_success = (
        sum(1 for row in final100 if as_bool(row.get("success"))) / len(final100)
        if final100
        else 0.0
    )
    final_eval = eval_rows[-1] if eval_rows else {}
    lengths = [as_float(row.get("episode_length")) for row in train_rows]
    final100_lengths = [as_float(row.get("episode_length")) for row in final100]

    best_checkpoint = checkpoint_dir / "best.pt"
    last_checkpoint = checkpoint_dir / "last.pt"
    row_count = len(train_rows)
    final_episode = int(summary.get("final_episode", 0) or 0)

    return {
        "Model": model,
        "run_key": key,
        "Seed": seed,
        "Max Steps": max_steps,
        "Train Success Count": success_count,
        "Train Success Rate": train_success_rate,
        "Final100 Train Success": final100_success,
        "Final Sample Success": as_float(final_eval.get("eval_sample_success_rate")),
        "Final Argmax Success": as_float(final_eval.get("eval_argmax_success_rate")),
        "Final Sample Return": as_float(final_eval.get("eval_sample_mean_return")),
        "Final Argmax Return": as_float(final_eval.get("eval_argmax_mean_return")),
        "Final Sample Length": as_float(final_eval.get("eval_sample_mean_episode_length")),
        "Final Argmax Length": as_float(final_eval.get("eval_argmax_mean_episode_length")),
        "Reward Signal Episodes": reward_signal_episodes,
        "Best Checkpoint": str(best_checkpoint.relative_to(ROOT)) if best_checkpoint.exists() else "",
        "Last Checkpoint Exists": last_checkpoint.exists(),
        "Best Checkpoint Exists": best_checkpoint.exists(),
        "train_rows": row_count,
        "eval_rows": len(eval_rows),
        "summary_final_episode": final_episode,
        "complete": row_count == 5000 and final_episode == 5000,
        "nonfinite": has_nonfinite(train_rows) or has_nonfinite(eval_rows),
        "mean_episode_length": mean(lengths) if lengths else 0.0,
        "final100_episode_length": mean(final100_lengths) if final100_lengths else 0.0,
        "time_limit_bound_final100": bool(final100_lengths)
        and all(abs(length - max_steps) < 1e-9 for length in final100_lengths),
        "log_dir": str(log_dir.relative_to(ROOT)),
        "checkpoint_dir": str(checkpoint_dir.relative_to(ROOT)),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt(row.get(column)) for column in TABLE_COLUMNS})


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(TABLE_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in TABLE_COLUMNS) + " |")
    return "\n".join(lines)


def first_reward_condition(rows: list[dict[str, Any]]) -> str:
    rewarded = [row for row in rows if int(row["Reward Signal Episodes"]) > 0]
    if not rewarded:
        return "None"
    rewarded.sort(key=lambda row: (int(row["Max Steps"]), str(row["Model"])))
    first = rewarded[0]
    return f"{first['Model']} max_steps={first['Max Steps']}"


def best_candidate(rows: list[dict[str, Any]]) -> str:
    argmax_positive = [row for row in rows if float(row["Final Argmax Success"]) > 0.0]
    if argmax_positive:
        argmax_positive.sort(
            key=lambda row: (float(row["Final Argmax Success"]), float(row["Final Sample Success"])),
            reverse=True,
        )
        row = argmax_positive[0]
        return f"{row['Model']} max_steps={row['Max Steps']} (argmax eval success > 0)"

    sample_positive = [row for row in rows if float(row["Final Sample Success"]) > 0.0]
    if sample_positive:
        sample_positive.sort(key=lambda row: float(row["Final Sample Success"]), reverse=True)
        row = sample_positive[0]
        return f"{row['Model']} max_steps={row['Max Steps']} (sample eval success > 0)"

    train_positive = [
        row
        for row in rows
        if int(row["Train Success Count"]) > 0 or int(row["Reward Signal Episodes"]) > 0
    ]
    if train_positive:
        train_positive.sort(
            key=lambda row: (int(row["Reward Signal Episodes"]), int(row["Train Success Count"])),
            reverse=True,
        )
        row = train_positive[0]
        return f"{row['Model']} max_steps={row['Max Steps']} (train reward signal only)"

    return "No N4-S5 from-scratch candidate under current settings; lower to N3 or change exploration/training stability."


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    incomplete = [row for row in rows if not row["complete"]]
    lines = [
        "# Week 6 MultiRoom-N4-S5 Max Steps Sweep Results",
        "",
        markdown_table(rows),
        "",
        "## Run Integrity",
    ]
    for row in rows:
        lines.append(
            "- "
            f"{row['Model']} max_steps={row['Max Steps']}: "
            f"train_rows={row['train_rows']}, eval_rows={row['eval_rows']}, "
            f"summary_final_episode={row['summary_final_episode']}, "
            f"best.pt={row['Best Checkpoint Exists']}, last.pt={row['Last Checkpoint Exists']}, "
            f"nonfinite={row['nonfinite']}, "
            f"final100_time_limit_bound={row['time_limit_bound_final100']}"
        )

    lines.extend(
        [
            "",
            "## Analysis",
            "",
            f"1. First reward signal condition: {first_reward_condition(rows)}.",
            "2. Vanilla vs Ablation first reward signal: "
            + (
                "No reward signal in either model."
                if first_reward_condition(rows) == "None"
                else first_reward_condition(rows)
            ),
            "3. Eval sample success > 0: "
            + (", ".join(
                f"{row['Model']} max_steps={row['Max Steps']}"
                for row in rows
                if float(row["Final Sample Success"]) > 0.0
            ) or "None"),
            "4. Eval argmax success > 0: "
            + (", ".join(
                f"{row['Model']} max_steps={row['Max Steps']}"
                for row in rows
                if float(row["Final Argmax Success"]) > 0.0
            ) or "None"),
            "5. Episode length tied to max_steps: "
            + (", ".join(
                f"{row['Model']} max_steps={row['Max Steps']}={row['time_limit_bound_final100']}"
                for row in rows
            )),
            "6. Continue N4-S5 or lower to N3: "
            + (
                "Wait for all runs to complete before deciding."
                if incomplete
                else (
                    "Continue N4-S5 only for the positive eval/reward-signal candidate."
                    if any(float(row["Final Sample Success"]) > 0.0 or int(row["Reward Signal Episodes"]) > 0 for row in rows)
                    else "Lower to N3 under the current from-scratch settings."
                )
            ),
            f"7. Next max_steps candidate: {best_candidate(rows)}",
        ]
    )
    if incomplete:
        lines.extend(
            [
                "",
                "## Incomplete Runs",
                "",
                "The table above is partial because at least one run has not reached 5000 episodes.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    complete = all(row["complete"] for row in rows)
    lines = [
        "# Week 6 MultiRoom-N4-S5 Max Steps Sweep Summary",
        "",
        f"- Complete: {complete}",
        f"- First reward signal: {first_reward_condition(rows)}",
        f"- Recommended next candidate: {best_candidate(rows)}",
        f"- Runs with sample eval success > 0: {sum(1 for row in rows if float(row['Final Sample Success']) > 0.0)}",
        f"- Runs with argmax eval success > 0: {sum(1 for row in rows if float(row['Final Argmax Success']) > 0.0)}",
        f"- Runs with nonfinite values: {sum(1 for row in rows if row['nonfinite'])}",
        "",
        markdown_table(rows),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ms1000_followup(rows: list[dict[str, Any]]) -> None:
    ablation_rows = [
        row
        for row in rows
        if row["run_key"] == "ablation" and int(row["Max Steps"]) in {750, 1000}
    ]
    ablation_rows.sort(key=lambda row: int(row["Max Steps"]))
    ms750 = next((row for row in ablation_rows if int(row["Max Steps"]) == 750), None)
    ms1000 = next((row for row in ablation_rows if int(row["Max Steps"]) == 1000), None)

    if ms750 is None or ms1000 is None:
        train_signal_answer = "Cannot compare until both max_steps=750 and max_steps=1000 rows exist."
        train_delta_text = ""
    else:
        train_delta = int(ms1000["Train Success Count"]) - int(ms750["Train Success Count"])
        train_signal_answer = "Yes" if train_delta > 0 else "No"
        train_delta_text = (
            f"ms750={ms750['Train Success Count']}, "
            f"ms1000={ms1000['Train Success Count']}, delta={train_delta}"
        )

    sample_positive = bool(ms1000 and float(ms1000["Final Sample Success"]) > 0.0)
    argmax_positive = bool(ms1000 and float(ms1000["Final Argmax Success"]) > 0.0)
    length_bound = bool(ms1000 and ms1000["time_limit_bound_final100"])

    if sample_positive:
        recommendation = (
            "Continue N4-S5. Next run should extend AblationManager max_steps=1000 "
            "to seeds 11 and 44."
        )
    elif ms750 is not None and ms1000 is not None and int(ms1000["Train Success Count"]) > int(ms750["Train Success Count"]):
        recommendation = (
            "N4-S5 still has only train reward signal. Consider one max_steps=1250 probe "
            "only if compute budget is acceptable; otherwise lower difficulty to N3-S5 or N4-S4."
        )
    else:
        recommendation = (
            "Do not keep expanding N4-S5 under the current settings. Lower difficulty to N3-S5 "
            "or N4-S4 before spending more compute."
        )

    result_lines = [
        "# Week 6 MultiRoom-N4-S5 AblationManager ms1000 Follow-up",
        "",
        markdown_table(ablation_rows),
        "",
        "## Integrity",
    ]
    for row in ablation_rows:
        result_lines.append(
            "- "
            f"max_steps={row['Max Steps']}: train_rows={row['train_rows']}, "
            f"eval_rows={row['eval_rows']}, summary_final_episode={row['summary_final_episode']}, "
            f"best.pt={row['Best Checkpoint Exists']}, last.pt={row['Last Checkpoint Exists']}, "
            f"nonfinite={row['nonfinite']}, final100_time_limit_bound={row['time_limit_bound_final100']}"
        )
    result_lines.extend(
        [
            "",
            "## Answers",
            "",
            f"A. Train reward signal increased from 750 to 1000: {train_signal_answer}. {train_delta_text}",
            f"B. Sample eval success became > 0: {'Yes' if sample_positive else 'No'}.",
            f"C. Argmax eval success became > 0: {'Yes' if argmax_positive else 'No'}.",
            f"D. Episode length remains tied to max_steps: {'Yes' if length_bound else 'No'}.",
            f"E. Continue N4-S5: {'Yes' if sample_positive else 'Not as the main path'}.",
            "F. Lower difficulty: "
            + ("No, extend seeds first." if sample_positive else "Yes, prefer N3-S5 or N4-S4."),
            "",
            "## Recommendation",
            "",
            recommendation,
        ]
    )
    (RESULTS_DIR / "week6_multiroom_n4s5_ms1000_ablation_result.md").write_text(
        "\n".join(result_lines) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# Week 6 MultiRoom-N4-S5 AblationManager ms1000 Summary",
        "",
        f"- Complete: {bool(ms1000 and ms1000['complete'])}",
        f"- Train reward signal 750 -> 1000: {train_signal_answer}. {train_delta_text}",
        f"- ms1000 sample eval success > 0: {sample_positive}",
        f"- ms1000 argmax eval success > 0: {argmax_positive}",
        f"- ms1000 final100 episode length tied to max_steps: {length_bound}",
        f"- Recommendation: {recommendation}",
        "",
        markdown_table(ablation_rows),
    ]
    (RESULTS_DIR / "week6_multiroom_n4s5_ms1000_ablation_summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(*run) for run in RUNS]
    write_csv(rows, RESULTS_DIR / "week6_multiroom_n4s5_maxsteps_sweep_results.csv")
    write_markdown(rows, RESULTS_DIR / "week6_multiroom_n4s5_maxsteps_sweep_results.md")
    write_summary(rows, RESULTS_DIR / "week6_multiroom_n4s5_maxsteps_sweep_summary.md")
    write_ms1000_followup(rows)


if __name__ == "__main__":
    main()
