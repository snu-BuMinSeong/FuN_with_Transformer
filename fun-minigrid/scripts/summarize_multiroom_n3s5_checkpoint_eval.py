from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


RAW_DIR = Path("logs/multiroom_n3s5_seed1/checkpoint_eval_100")
RESULTS_DIR = Path("results")
SUMMARY_CSV = RESULTS_DIR / "multiroom_n3_s5_checkpoint_eval_summary.csv"
SUMMARY_MD = RESULTS_DIR / "multiroom_n3_s5_checkpoint_eval_summary.md"
ANALYSIS_MD = RESULTS_DIR / "multiroom_n3_s5_checkpoint_eval_analysis.md"

CHECKPOINT_ORDER = {
    "best": 0,
    "best_argmax": 1,
    "best_sample": 2,
    "episode_5000": 3,
    "last": 4,
}
MODEL_ORDER = {"baseline": 0, "ablation": 1}
POLICY_ORDER = {"sample": 0, "argmax": 1}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(RAW_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        result = payload["eval_result"]
        checkpoint_name = path.stem.split("_", 1)[1].rsplit("_", 1)[0]
        model = path.stem.split("_", 1)[0]
        eval_policy = payload["eval_action_mode"]
        rows.append(
            {
                "model": model,
                "checkpoint": checkpoint_name,
                "eval_policy": eval_policy,
                "success_rate": float(result["mean_success_rate"]),
                "std_success_rate": float(result["std_success_rate"]),
                "avg_return": float(result["mean_reward"]),
                "std_return": float(result["std_reward"]),
                "avg_episode_length": float(result["mean_episode_length"]),
                "std_episode_length": float(result["std_episode_length"]),
                "min_return": float(result["min_reward"]),
                "max_return": float(result["max_reward"]),
                "min_episode_length": float(result["min_episode_length"]),
                "max_episode_length": float(result["max_episode_length"]),
                "eval_episodes": int(payload["eval_episodes"]),
                "eval_seed_offset": int(payload["eval_seed_offset"]),
                "checkpoint_episode": payload.get("checkpoint_episode"),
                "checkpoint_best_success_rate": payload.get("checkpoint_best_success_rate"),
                "raw_eval_json": str(path),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            MODEL_ORDER.get(row["model"], 99),
            CHECKPOINT_ORDER.get(row["checkpoint"], 99),
            POLICY_ORDER.get(row["eval_policy"], 99),
        ),
    )


def _write_csv(rows: list[dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "checkpoint",
        "eval_policy",
        "success_rate",
        "std_success_rate",
        "avg_return",
        "std_return",
        "avg_episode_length",
        "std_episode_length",
        "min_return",
        "max_return",
        "min_episode_length",
        "max_episode_length",
        "eval_episodes",
        "eval_seed_offset",
        "checkpoint_episode",
        "checkpoint_best_success_rate",
        "raw_eval_json",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| model | checkpoint | eval_policy | success_rate | avg_return | avg_episode_length | std_return | std_episode_length |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["checkpoint"],
                    row["eval_policy"],
                    _fmt(row["success_rate"]),
                    _fmt(row["avg_return"]),
                    _fmt(row["avg_episode_length"]),
                    _fmt(row["std_return"]),
                    _fmt(row["std_episode_length"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _best_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["model"], row["eval_policy"])
        current = best.get(key)
        score = (row["success_rate"], row["avg_return"], -row["avg_episode_length"])
        current_score = None
        if current is not None:
            current_score = (
                current["success_rate"],
                current["avg_return"],
                -current["avg_episode_length"],
            )
        if current is None or score > current_score:
            best[key] = row
    return best


def _policy_rows(rows: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["eval_policy"] == policy]


def _write_summary_md(rows: list[dict[str, Any]]) -> None:
    content = [
        "# MultiRoom-N3-S5 Checkpoint Evaluation Summary",
        "",
        "Evaluation conditions:",
        "",
        "- Environment: `MiniGrid-MultiRoom-N3-S5-v0`",
        "- Seed: training config seed `1`",
        "- Evaluation episodes: `100`",
        "- Evaluation seed offset: `9000`",
        "- Max steps: config `max_steps=1000`",
        "- Policies: `sample` and `argmax`",
        "",
        _table(rows),
        "",
        f"Raw JSON files are under `{RAW_DIR}`.",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(content), encoding="utf-8")


def _write_analysis_md(rows: list[dict[str, Any]]) -> None:
    best = _best_rows(rows)
    sample_rows = _policy_rows(rows, "sample")
    argmax_rows = _policy_rows(rows, "argmax")

    baseline_argmax_best = best.get(("baseline", "argmax"))
    ablation_argmax_best = best.get(("ablation", "argmax"))
    baseline_sample_best = best.get(("baseline", "sample"))
    ablation_sample_best = best.get(("ablation", "sample"))

    def best_sentence(row: dict[str, Any] | None) -> str:
        if row is None:
            return "No result."
        return (
            f"`{row['checkpoint']}` with success {_fmt(row['success_rate'])}, "
            f"return {_fmt(row['avg_return'])}, length {_fmt(row['avg_episode_length'])}"
        )

    def row_for(model: str, checkpoint: str, policy: str) -> dict[str, Any] | None:
        for row in rows:
            if row["model"] == model and row["checkpoint"] == checkpoint and row["eval_policy"] == policy:
                return row
        return None

    lines = [
        "# MultiRoom-N3-S5 Checkpoint Evaluation Analysis",
        "",
        "## Evaluation Setup",
        "",
        "All checkpoints were evaluated with the same config-defined environment and max-step limit.",
        "The evaluation used 100 episodes per checkpoint/policy pair with seed offset 9000.",
        "Both stochastic `sample` action selection and deterministic `argmax` action selection were evaluated.",
        "",
        "## Overall Comparison",
        "",
        f"- Best baseline sample result: {best_sentence(baseline_sample_best)}.",
        f"- Best ablation sample result: {best_sentence(ablation_sample_best)}.",
        f"- Best baseline argmax result: {best_sentence(baseline_argmax_best)}.",
        f"- Best ablation argmax result: {best_sentence(ablation_argmax_best)}.",
        "",
        "## Full Results",
        "",
        _table(rows),
        "",
        "## Checkpoint Selection Notes",
        "",
    ]

    for model in ["baseline", "ablation"]:
        for policy in ["sample", "argmax"]:
            selected = best.get((model, policy))
            lines.append(f"- `{model}` under `{policy}` evaluation: {best_sentence(selected)}.")

    lines.extend(["", "## best.pt / best_argmax.pt / best_sample.pt Consistency", ""])
    for model in ["baseline", "ablation"]:
        sample_best = row_for(model, "best_sample", "sample")
        argmax_best = row_for(model, "best_argmax", "argmax")
        generic_sample = row_for(model, "best", "sample")
        generic_argmax = row_for(model, "best", "argmax")
        lines.append(
            f"- `{model}`: `best_sample.pt` sample={_fmt(sample_best['success_rate']) if sample_best else 'NA'} "
            f"/ return={_fmt(sample_best['avg_return']) if sample_best else 'NA'}; "
            f"`best_argmax.pt` argmax={_fmt(argmax_best['success_rate']) if argmax_best else 'NA'} "
            f"/ return={_fmt(argmax_best['avg_return']) if argmax_best else 'NA'}; "
            f"`best.pt` sample={_fmt(generic_sample['success_rate']) if generic_sample else 'NA'}, "
            f"argmax={_fmt(generic_argmax['success_rate']) if generic_argmax else 'NA'}."
        )

    lines.extend(["", "## last.pt vs best checkpoints", ""])
    for model in ["baseline", "ablation"]:
        for policy in ["sample", "argmax"]:
            last = row_for(model, "last", policy)
            selected = best.get((model, policy))
            if last is None or selected is None:
                continue
            lines.append(
                f"- `{model}` `{policy}`: `last.pt` success {_fmt(last['success_rate'])}, "
                f"return {_fmt(last['avg_return'])}, length {_fmt(last['avg_episode_length'])}; "
                f"best observed checkpoint `{selected['checkpoint']}` success {_fmt(selected['success_rate'])}, "
                f"return {_fmt(selected['avg_return'])}, length {_fmt(selected['avg_episode_length'])}."
            )

    lines.extend(["", "## Recommendation", ""])
    if baseline_argmax_best and ablation_argmax_best:
        winner = "baseline"
        if (
            ablation_argmax_best["success_rate"],
            ablation_argmax_best["avg_return"],
            -ablation_argmax_best["avg_episode_length"],
        ) > (
            baseline_argmax_best["success_rate"],
            baseline_argmax_best["avg_return"],
            -baseline_argmax_best["avg_episode_length"],
        ):
            winner = "ablation"
        lines.append(
            f"For a deterministic headline metric, use `best_argmax.pt` evaluated with `argmax` for each model, "
            f"and report the full checkpoint table as a robustness check. Under the best observed argmax comparison, "
            f"`{winner}` is ahead by the success/return/length tie-break rule."
        )
    lines.append(
        "If the paper needs a single pre-registered checkpoint rule independent of evaluation outcomes, "
        "prefer reporting `best_argmax.pt` for argmax evaluation and `best_sample.pt` for sample evaluation. "
        "If the goal is final-training-state performance, report `episode_5000.pt` or `last.pt` separately rather than mixing it with best checkpoint selection."
    )

    lines.extend(["", f"Summary CSV: `{SUMMARY_CSV}`", f"Raw eval logs: `{RAW_DIR}`", ""])
    ANALYSIS_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = _load_rows()
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 eval JSON files, found {len(rows)} in {RAW_DIR}.")
    _write_csv(rows)
    _write_summary_md(rows)
    _write_analysis_md(rows)
    print(f"wrote {SUMMARY_CSV}")
    print(f"wrote {SUMMARY_MD}")
    print(f"wrote {ANALYSIS_MD}")


if __name__ == "__main__":
    main()
