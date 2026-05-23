from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from evaluate_checkpoint import build_model_from_config
from src.envs.make_env import make_env
from src.envs.preprocess import preprocess_obs
from src.policies.fun_policy import FuNPolicy
from src.training.rollout import run_episode
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_simple_yaml
from src.utils.seed import set_seed


INTERVENTIONS = ("normal", "reset_goal", "reset_step")
ACTION_MODES = ("sample", "argmax")
DEFAULT_BATCH_SEEDS = (1, 11, 44)
DEFAULT_RESULTS_DIR = Path("results/multiroom_n4s4_hidden_intervention")


class HiddenInterventionPolicy(FuNPolicy):
    """Evaluation policy with diagnostic hidden-state interventions."""

    def __init__(
        self,
        *args: Any,
        intervention: str = "normal",
        **kwargs: Any,
    ) -> None:
        if intervention not in INTERVENTIONS:
            raise ValueError(f"intervention must be one of {INTERVENTIONS}, got {intervention}.")
        self.intervention = intervention
        self.total_hidden_resets = 0
        self.total_goal_updates = 0
        super().__init__(*args, **kwargs)

    def reset(self, batch_size: int = 1) -> None:
        super().reset(batch_size=batch_size)
        self.episode_hidden_resets = 0
        self.episode_goal_updates = 0
        self.episode_steps = 0

    def _zero_hidden(self) -> None:
        self.hidden_state = torch.zeros_like(self.hidden_state)
        self.episode_hidden_resets += 1
        self.total_hidden_resets += 1

    def act_for_training(self, obs: Any) -> dict[str, Any]:
        goal_update_due = self.step_count % self.model.goal_update_interval == 0
        if self.intervention == "reset_step":
            self._zero_hidden()
        elif self.intervention == "reset_goal" and goal_update_due:
            self._zero_hidden()

        out = super().act_for_training(obs)
        if bool(out["goal_updated"]):
            self.episode_goal_updates += 1
            self.total_goal_updates += 1
        self.episode_steps += 1
        return out

    def get_intervention_debug_info(self) -> dict[str, int | str]:
        return {
            "intervention": self.intervention,
            "episode_hidden_resets": self.episode_hidden_resets,
            "episode_goal_updates": self.episode_goal_updates,
            "episode_steps": self.episode_steps,
            "total_hidden_resets": self.total_hidden_resets,
            "total_goal_updates": self.total_goal_updates,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose recurrent hidden-state use in a saved FuN baseline.")
    parser.add_argument("--config", type=str, help="Path to a baseline training config YAML.")
    parser.add_argument("--checkpoint", type=str, help="Path to the final checkpoint .pt file.")
    parser.add_argument("--seed", type=int, choices=list(DEFAULT_BATCH_SEEDS), help="Training seed.")
    parser.add_argument("--action-mode", choices=ACTION_MODES, help="Evaluation action mode.")
    parser.add_argument("--intervention", choices=INTERVENTIONS, help="Hidden-state intervention.")
    parser.add_argument("--episodes", type=int, default=100, help="Evaluation episodes per condition.")
    parser.add_argument("--output", type=str, help="Single-run output JSON path.")
    parser.add_argument(
        "--eval-seed-offset",
        type=int,
        default=None,
        help="Override config eval_seed_offset. Defaults to the config value.",
    )
    parser.add_argument(
        "--batch-n4s4-baseline",
        action="store_true",
        help="Run seed 1/11/44 x sample/argmax x normal/reset_goal/reset_step.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Batch output directory.",
    )
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _load_summary(seed: int) -> dict[str, Any]:
    summary_path = Path(f"logs/multiroom_n4s4_seed{seed}/baseline/summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing baseline summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _resolve_batch_paths(seed: int) -> tuple[Path, Path]:
    summary = _load_summary(seed)
    config_path = Path(str(summary["config_path"]))
    checkpoint_path = Path(str(summary["final_episode_checkpoint_path"]))
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config recorded in summary: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing final checkpoint recorded in summary: {checkpoint_path}")
    return config_path, checkpoint_path


def _validate_baseline_config(config: dict[str, Any], seed: int) -> None:
    if str(config.get("env_id")) != "MiniGrid-MultiRoom-N4-S4-v0":
        raise ValueError(f"Expected MiniGrid-MultiRoom-N4-S4-v0, got {config.get('env_id')}.")
    if str(config.get("manager_type", "recurrent")) != "recurrent":
        raise ValueError(f"Expected recurrent baseline manager, got {config.get('manager_type')}.")
    if int(config.get("seed")) != seed:
        raise ValueError(f"Config seed mismatch: expected {seed}, got {config.get('seed')}.")


def evaluate_hidden_intervention(
    config_path: str | Path,
    checkpoint_path: str | Path,
    seed: int,
    action_mode: str,
    intervention: str,
    episodes: int,
    output_path: str | Path | None = None,
    eval_seed_offset: int | None = None,
) -> dict[str, Any]:
    cfg = load_simple_yaml(config_path)
    _validate_baseline_config(cfg, seed)
    eval_seed_offset = int(eval_seed_offset if eval_seed_offset is not None else cfg.get("eval_seed_offset", 1000))
    eval_seed_start = seed + eval_seed_offset
    max_steps = int(cfg.get("max_steps", 100)) if cfg.get("max_steps") is not None else None

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg, device=device)
    checkpoint = load_checkpoint(checkpoint_path, model=model, optimizer=None, map_location=device)
    policy = HiddenInterventionPolicy(
        model=model,
        preprocess_fn=preprocess_obs,
        device=device,
        action_mode=action_mode,
        intervention=intervention,
    )

    env = make_env(
        env_id=str(cfg.get("env_id")),
        render_mode=cfg.get("render_mode"),
        seed=seed,
        max_episode_steps=max_steps,
    )
    episode_results: list[dict[str, Any]] = []
    try:
        for episode_idx in range(episodes):
            episode_seed = eval_seed_start + episode_idx
            result = run_episode(env=env, policy=policy, max_steps=max_steps, seed=episode_seed)
            result["episode"] = episode_idx + 1
            result["intervention_debug"] = policy.get_intervention_debug_info()
            episode_results.append(result)
    finally:
        env.close()

    returns = [float(result["total_reward"]) for result in episode_results]
    lengths = [float(result["episode_length"]) for result in episode_results]
    successes = [1.0 if bool(result["success"]) else 0.0 for result in episode_results]
    success_lengths = [
        float(result["episode_length"])
        for result in episode_results
        if bool(result["success"])
    ]
    reset_counts = [
        int(result["intervention_debug"]["episode_hidden_resets"])
        for result in episode_results
    ]
    goal_update_counts = [
        int(result["intervention_debug"]["episode_goal_updates"])
        for result in episode_results
    ]

    summary = {
        "model": "baseline",
        "seed": seed,
        "action_mode": action_mode,
        "intervention": intervention,
        "eval_episodes": episodes,
        "eval_seed_offset": eval_seed_offset,
        "eval_seed_start": eval_seed_start,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_episode": checkpoint.get("episode"),
        "config_path": str(config_path),
        "success_rate": _mean(successes),
        "mean_return": _mean(returns),
        "std_return": _std(returns),
        "mean_episode_length": _mean(lengths),
        "std_episode_length": _std(lengths),
        "mean_success_episode_length": _mean(success_lengths) if success_lengths else None,
        "hidden_reset_count_total": sum(reset_counts),
        "goal_update_count_total": sum(goal_update_counts),
        "mean_hidden_resets_per_episode": _mean([float(value) for value in reset_counts]),
        "mean_goal_updates_per_episode": _mean([float(value) for value in goal_update_counts]),
    }
    result = {
        **summary,
        "successes": [bool(result["success"]) for result in episode_results],
        "returns": returns,
        "episode_lengths": lengths,
        "episode_seeds": [result["seed"] for result in episode_results],
        "episode_results": episode_results,
    }
    jsonable_result = _jsonable(result)
    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(jsonable_result, indent=2, ensure_ascii=False), encoding="utf-8")
    return jsonable_result


def _raw_filename(seed: int, action_mode: str, intervention: str) -> str:
    return f"seed{seed}_{action_mode}_{intervention}.json"


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "model",
        "seed",
        "action_mode",
        "intervention",
        "eval_episodes",
        "success_rate",
        "mean_return",
        "std_return",
        "mean_episode_length",
        "std_episode_length",
        "mean_success_episode_length",
        "hidden_reset_count_total",
        "goal_update_count_total",
        "mean_hidden_resets_per_episode",
        "mean_goal_updates_per_episode",
        "checkpoint_path",
        "checkpoint_episode",
        "config_path",
        "eval_seed_offset",
        "eval_seed_start",
    ]
    return {field: result.get(field) for field in fields}


def _write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_rows(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    return grouped


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_drop_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition = {
        (int(row["seed"]), str(row["action_mode"]), str(row["intervention"])): row
        for row in rows
    }
    drop_rows: list[dict[str, Any]] = []
    for seed in DEFAULT_BATCH_SEEDS:
        for action_mode in ACTION_MODES:
            normal = by_condition[(seed, action_mode, "normal")]
            for intervention in ("reset_goal", "reset_step"):
                target = by_condition[(seed, action_mode, intervention)]
                drop_rows.append(
                    {
                        "seed": seed,
                        "action_mode": action_mode,
                        "intervention": intervention,
                        "success_drop": float(normal["success_rate"]) - float(target["success_rate"]),
                        "return_drop": float(normal["mean_return"]) - float(target["mean_return"]),
                        "ep_len_change": float(target["mean_episode_length"]) - float(normal["mean_episode_length"]),
                    }
                )
    return drop_rows


def _write_summary_md(rows: list[dict[str, Any]], output_path: Path) -> None:
    lines: list[str] = ["# MultiRoom N4-S4 Hidden Intervention Summary", ""]
    lines.extend(
        [
            "## Seed-level results",
            "",
            "| seed | action_mode | intervention | success | return | ep_len | success_ep_len |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (int(item["seed"]), str(item["action_mode"]), str(item["intervention"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    str(row["action_mode"]),
                    str(row["intervention"]),
                    _fmt(float(row["success_rate"])),
                    _fmt(float(row["mean_return"])),
                    _fmt(float(row["mean_episode_length"])),
                    _fmt(row["mean_success_episode_length"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Three-seed means by action mode",
            "",
            "| action_mode | intervention | mean_success | mean_return | mean_ep_len | mean_success_ep_len |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for key, group in sorted(_group_rows(rows, "action_mode", "intervention").items()):
        action_mode, intervention = key
        success_lengths = [
            float(row["mean_success_episode_length"])
            for row in group
            if row["mean_success_episode_length"] is not None and row["mean_success_episode_length"] != ""
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(action_mode),
                    str(intervention),
                    _fmt(_mean([float(row["success_rate"]) for row in group])),
                    _fmt(_mean([float(row["mean_return"]) for row in group])),
                    _fmt(_mean([float(row["mean_episode_length"]) for row in group])),
                    _fmt(_mean(success_lengths) if success_lengths else None),
                ]
            )
            + " |"
        )

    drop_rows = build_drop_rows(rows)
    lines.extend(
        [
            "",
            "## Intervention drops from normal",
            "",
            "| action_mode | intervention | success_drop | return_drop | ep_len_change |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for key, group in sorted(_group_rows(drop_rows, "action_mode", "intervention").items()):
        action_mode, intervention = key
        lines.append(
            "| "
            + " | ".join(
                [
                    str(action_mode),
                    str(intervention),
                    _fmt(_mean([float(row["success_drop"]) for row in group])),
                    _fmt(_mean([float(row["return_drop"]) for row in group])),
                    _fmt(_mean([float(row["ep_len_change"]) for row in group])),
                ]
            )
            + " |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _interpret(rows: list[dict[str, Any]]) -> str:
    drop_rows = build_drop_rows(rows)
    by_intervention = _group_rows(drop_rows, "intervention")
    reset_goal_success_drop = _mean([float(row["success_drop"]) for row in by_intervention[("reset_goal",)]])
    reset_step_success_drop = _mean([float(row["success_drop"]) for row in by_intervention[("reset_step",)]])

    if abs(reset_goal_success_drop) <= 0.02 and abs(reset_step_success_drop) <= 0.02:
        return "The final Baseline policy does not appear to rely strongly on recurrent hidden state."
    if reset_step_success_drop > reset_goal_success_drop and reset_step_success_drop > 0.02:
        return "Hidden state may contribute to goal generation and final action stability."
    return "Hidden intervention effects are limited or mixed across action modes and seeds."


def _write_analysis_md(rows: list[dict[str, Any]], output_path: Path) -> None:
    drop_rows = build_drop_rows(rows)
    lines = [
        "# Hidden Intervention Analysis",
        "",
        "This diagnostic evaluates only final Baseline recurrent-manager checkpoints; no retraining was run.",
        "",
        "## Normal Check",
        "",
    ]
    for key, group in sorted(_group_rows(rows, "action_mode", "intervention").items()):
        action_mode, intervention = key
        if intervention != "normal":
            continue
        lines.append(
            f"- {action_mode}: 3-seed mean success={_mean([float(row['success_rate']) for row in group]):.4f}, "
            f"mean_return={_mean([float(row['mean_return']) for row in group]):.4f}, "
            f"mean_ep_len={_mean([float(row['mean_episode_length']) for row in group]):.4f}"
        )
    lines.extend(["", "## Drops From Normal", ""])
    for key, group in sorted(_group_rows(drop_rows, "action_mode", "intervention").items()):
        action_mode, intervention = key
        lines.append(
            f"- {action_mode} {intervention}: success_drop={_mean([float(row['success_drop']) for row in group]):.4f}, "
            f"return_drop={_mean([float(row['return_drop']) for row in group]):.4f}, "
            f"ep_len_change={_mean([float(row['ep_len_change']) for row in group]):.4f}"
        )
    lines.extend(["", "## First-pass Interpretation", "", _interpret(rows), ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_batch(results_dir: Path, episodes: int, eval_seed_offset: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in DEFAULT_BATCH_SEEDS:
        config_path, checkpoint_path = _resolve_batch_paths(seed)
        for action_mode in ACTION_MODES:
            for intervention in INTERVENTIONS:
                output_path = results_dir / _raw_filename(seed, action_mode, intervention)
                result = evaluate_hidden_intervention(
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    seed=seed,
                    action_mode=action_mode,
                    intervention=intervention,
                    episodes=episodes,
                    output_path=output_path,
                    eval_seed_offset=eval_seed_offset,
                )
                row = _summary_row(result)
                rows.append(row)
                print(
                    f"seed={seed} action_mode={action_mode} intervention={intervention} "
                    f"success={row['success_rate']:.3f} return={row['mean_return']:.3f} "
                    f"ep_len={row['mean_episode_length']:.3f} output={output_path}"
                )

    _write_summary_csv(rows, results_dir / "hidden_intervention_summary.csv")
    _write_summary_md(rows, results_dir / "hidden_intervention_summary.md")
    _write_analysis_md(rows, results_dir / "hidden_intervention_analysis.md")
    return rows


def main() -> None:
    args = parse_args()
    if args.batch_n4s4_baseline:
        run_batch(Path(args.results_dir), episodes=args.episodes, eval_seed_offset=args.eval_seed_offset)
        return

    required = {
        "--config": args.config,
        "--checkpoint": args.checkpoint,
        "--seed": args.seed,
        "--action-mode": args.action_mode,
        "--intervention": args.intervention,
        "--output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"Missing required arguments for single run: {', '.join(missing)}")

    result = evaluate_hidden_intervention(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        action_mode=args.action_mode,
        intervention=args.intervention,
        episodes=args.episodes,
        output_path=args.output,
        eval_seed_offset=args.eval_seed_offset,
    )
    print(
        f"seed={result['seed']} action_mode={result['action_mode']} intervention={result['intervention']} "
        f"success={result['success_rate']:.3f} return={result['mean_return']:.3f} "
        f"ep_len={result['mean_episode_length']:.3f} output={args.output}"
    )


if __name__ == "__main__":
    main()
