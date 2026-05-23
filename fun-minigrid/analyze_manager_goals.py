from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from evaluate_checkpoint import build_model_from_config
from src.envs.make_env import make_env
from src.envs.preprocess import preprocess_obs
from src.models.fun import FuNModel
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_simple_yaml
from src.utils.seed import set_seed


MODELS = ("baseline", "ablation")
ACTION_MODES = ("sample", "argmax")
SEEDS = (1, 11, 44)
RESULTS_DIR = Path("results/multiroom_n4s4_manager_goal_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log and analyze Manager goals for final N4-S4 checkpoints.")
    parser.add_argument("--config", type=str, help="Path to config YAML for a single run.")
    parser.add_argument("--checkpoint", type=str, help="Path to final checkpoint for a single run.")
    parser.add_argument("--model-type", choices=MODELS, help="baseline or ablation.")
    parser.add_argument("--seed", type=int, choices=list(SEEDS), help="Training seed.")
    parser.add_argument("--action-mode", choices=ACTION_MODES, help="Evaluation action mode.")
    parser.add_argument("--episodes", type=int, default=100, help="Evaluation episodes per condition.")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR), help="Output directory.")
    parser.add_argument("--eval-seed-offset", type=int, default=None, help="Override config eval_seed_offset.")
    parser.add_argument("--batch-n4s4", action="store_true", help="Run all model/seed/action-mode combinations.")
    return parser.parse_args()


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _std(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return None
    if len(clean) == 1:
        return 0.0
    mean = sum(clean) / len(clean)
    return float((sum((value - mean) ** 2 for value in clean) / len(clean)) ** 0.5)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys, strict=False)
        if x is not None and y is not None and not math.isnan(float(x)) and not math.isnan(float(y))
    ]
    if len(pairs) < 2:
        return None
    x_values = [pair[0] for pair in pairs]
    y_values = [pair[1] for pair in pairs]
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denom_x = sum((x - mean_x) ** 2 for x in x_values)
    denom_y = sum((y - mean_y) ** 2 for y in y_values)
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return float(numerator / ((denom_x * denom_y) ** 0.5))


def _safe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _goal_to_text(goal: torch.Tensor) -> str:
    values = goal.detach().cpu().flatten().tolist()
    return json.dumps([round(float(value), 6) for value in values], separators=(",", ":"))


def _kl_prev_current(prev_probs: torch.Tensor, current_probs: torch.Tensor) -> float:
    eps = 1e-12
    prev = prev_probs.clamp_min(eps)
    current = current_probs.clamp_min(eps)
    return float((prev * (prev.log() - current.log())).sum().item())


class GoalLoggingPolicy:
    def __init__(
        self,
        model: FuNModel,
        device: torch.device | str,
        action_mode: str,
    ) -> None:
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode}.")
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.action_mode = action_mode
        self.reset()

    def reset(self) -> None:
        self.hidden_state = self.model.init_hidden(batch_size=1, device=self.device)
        self.current_goal = self.model.init_goal(batch_size=1, device=self.device)
        self.step_count = 0
        self.goal_age = 0
        self.prev_goal: torch.Tensor | None = None
        self.prev_probs: torch.Tensor | None = None
        self.prev_top1_action: int | None = None

    def act_and_log(self, obs: Any) -> tuple[int, dict[str, Any]]:
        obs_array = preprocess_obs(obs)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model(
                obs=obs_tensor,
                hidden_state=self.hidden_state,
                current_goal=self.current_goal,
                step_count=self.step_count,
            )
            probs = out["action_probs"].squeeze(0).detach()
            dist = out["action_dist"]
            if self.action_mode == "sample":
                action_tensor = dist.sample()
            else:
                action_tensor = torch.argmax(out["logits"], dim=-1)
            action = int(action_tensor.item())
            entropy = float(dist.entropy().squeeze(0).item())

        goal = out["goal"].squeeze(0).detach()
        goal_updated = bool(out["goal_updated"])
        if goal_updated:
            goal_age = 0
        else:
            goal_age = self.goal_age + 1

        if self.prev_goal is None:
            goal_delta_l2 = None
            goal_cosine_prev = None
        else:
            goal_delta_l2 = float(torch.linalg.vector_norm(goal - self.prev_goal).item())
            denom = float(torch.linalg.vector_norm(goal).item() * torch.linalg.vector_norm(self.prev_goal).item())
            goal_cosine_prev = None if denom <= 0.0 else float(torch.dot(goal, self.prev_goal).item() / denom)

        top1_prob, top1_action_tensor = torch.max(probs, dim=0)
        top1_action = int(top1_action_tensor.item())
        chosen_action_prob = float(probs[action].item())
        if self.prev_probs is not None and goal_updated:
            action_kl = _kl_prev_current(self.prev_probs, probs)
            top1_changed = bool(self.prev_top1_action != top1_action)
        else:
            action_kl = None
            top1_changed = None

        trace = {
            "step": self.step_count,
            "action": action,
            "goal_updated": goal_updated,
            "goal_age": goal_age,
            "goal_norm": float(torch.linalg.vector_norm(goal).item()),
            "goal_delta_l2": goal_delta_l2,
            "goal_cosine_prev": goal_cosine_prev,
            "policy_entropy": entropy,
            "top1_prob": float(top1_prob.item()),
            "chosen_action_prob": chosen_action_prob,
            "top1_action": top1_action,
            "action_changed_from_prev_step": None if self.prev_top1_action is None else bool(self.prev_top1_action != top1_action),
            "hidden_norm": float(torch.linalg.vector_norm(out["hidden_state"].squeeze(0).detach()).item()),
            "action_prob_kl_from_prev": action_kl,
            "top1_action_changed_on_goal_update": top1_changed,
            "goal_vector": _goal_to_text(goal),
            "action_probs": json.dumps([round(float(value), 8) for value in probs.cpu().tolist()], separators=(",", ":")),
        }

        self.hidden_state = out["hidden_state"].detach()
        self.current_goal = out["goal"].detach()
        self.step_count += 1
        self.goal_age = goal_age
        self.prev_goal = goal
        self.prev_probs = probs
        self.prev_top1_action = top1_action
        return action, trace


RAW_FIELDS = [
    "model_type",
    "seed",
    "action_mode",
    "episode_id",
    "episode_seed",
    "step",
    "reward",
    "done",
    "success",
    "action",
    "goal_updated",
    "goal_age",
    "goal_norm",
    "goal_delta_l2",
    "goal_cosine_prev",
    "policy_entropy",
    "top1_prob",
    "chosen_action_prob",
    "top1_action",
    "action_changed_from_prev_step",
    "hidden_norm",
    "action_prob_kl_from_prev",
    "top1_action_changed_on_goal_update",
    "goal_vector",
    "action_probs",
]


EPISODE_FIELDS = [
    "model_type",
    "seed",
    "action_mode",
    "episode_id",
    "episode_seed",
    "success",
    "return",
    "episode_length",
    "num_goal_updates",
    "mean_goal_norm",
    "std_goal_norm",
    "mean_goal_delta_l2",
    "std_goal_delta_l2",
    "mean_goal_cosine_prev",
    "std_goal_cosine_prev",
    "mean_policy_entropy",
    "mean_top1_prob",
    "mean_chosen_action_prob",
    "mean_action_kl_on_goal_update",
    "top1_action_change_rate_on_goal_update",
]


EVENT_FIELDS = [
    "model_type",
    "seed",
    "action_mode",
    "episode_id",
    "step",
    "success",
    "episode_length",
    "goal_delta_l2",
    "goal_cosine_prev",
    "action_prob_kl_from_prev",
    "top1_action_changed_on_goal_update",
    "entropy_after_update",
    "top1_prob_after_update",
]


AGG_FIELDS = [
    "model_type",
    "action_mode",
    "mean_success",
    "mean_return",
    "mean_episode_length",
    "mean_num_goal_updates",
    "mean_goal_norm",
    "mean_goal_delta_l2",
    "mean_goal_cosine_prev",
    "mean_policy_entropy",
    "mean_top1_prob",
    "mean_chosen_action_prob",
    "mean_action_kl_on_goal_update",
    "mean_top1_action_change_rate_on_goal_update",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_config(config: dict[str, Any], model_type: str, seed: int) -> None:
    expected_manager = "recurrent" if model_type == "baseline" else "ablation"
    if str(config.get("env_id")) != "MiniGrid-MultiRoom-N4-S4-v0":
        raise ValueError(f"Expected MiniGrid-MultiRoom-N4-S4-v0, got {config.get('env_id')}.")
    if str(config.get("manager_type")) != expected_manager:
        raise ValueError(f"Expected manager_type={expected_manager}, got {config.get('manager_type')}.")
    if int(config.get("seed")) != seed:
        raise ValueError(f"Expected seed={seed}, got {config.get('seed')}.")


def _summary_path(model_type: str, seed: int) -> Path:
    return Path(f"logs/multiroom_n4s4_seed{seed}/{model_type}/summary.json")


def resolve_paths_from_summary(model_type: str, seed: int) -> tuple[Path, Path]:
    summary_path = _summary_path(model_type, seed)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config_path = Path(str(summary["config_path"]))
    checkpoint_path = Path(str(summary["final_episode_checkpoint_path"]))
    if not config_path.exists():
        raise FileNotFoundError(f"Summary config path does not exist: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Summary final checkpoint path does not exist: {checkpoint_path}")
    return config_path, checkpoint_path


def run_goal_trace(
    config_path: str | Path,
    checkpoint_path: str | Path,
    model_type: str,
    seed: int,
    action_mode: str,
    episodes: int,
    output_dir: str | Path,
    eval_seed_offset: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir = Path(output_dir)
    config = load_simple_yaml(config_path)
    _validate_config(config, model_type, seed)
    eval_seed_offset = int(eval_seed_offset if eval_seed_offset is not None else config.get("eval_seed_offset", 1000))
    eval_seed_start = seed + eval_seed_offset
    max_steps = int(config.get("max_steps", 100)) if config.get("max_steps") is not None else None

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config, device=device)
    checkpoint = load_checkpoint(checkpoint_path, model=model, optimizer=None, map_location=device)
    if int(checkpoint.get("episode")) != int(config.get("total_episodes")):
        raise ValueError(f"Checkpoint episode {checkpoint.get('episode')} is not final episode {config.get('total_episodes')}.")

    policy = GoalLoggingPolicy(model=model, device=device, action_mode=action_mode)
    env = make_env(
        env_id=str(config.get("env_id")),
        render_mode=config.get("render_mode"),
        seed=seed,
        max_episode_steps=max_steps,
    )

    raw_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    try:
        for episode_idx in range(episodes):
            episode_id = episode_idx + 1
            episode_seed = eval_seed_start + episode_idx
            obs, _ = env.reset(seed=episode_seed)
            policy.reset()
            total_reward = 0.0
            terminated = False
            truncated = False
            success_from_info = False
            episode_trace: list[dict[str, Any]] = []

            while not terminated and not truncated:
                action, trace = policy.act_and_log(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                success_from_info = bool(info.get("success", success_from_info))
                done = bool(terminated or truncated)
                if max_steps is not None and int(trace["step"]) + 1 >= max_steps and not done:
                    truncated = True
                    done = True

                row = {
                    "model_type": model_type,
                    "seed": seed,
                    "action_mode": action_mode,
                    "episode_id": episode_id,
                    "episode_seed": episode_seed,
                    "reward": float(reward),
                    "done": done,
                    "success": False,
                    **trace,
                }
                episode_trace.append(row)
                obs = next_obs
                if done:
                    break

            success = bool(success_from_info or (terminated and total_reward > 0.0))
            episode_length = len(episode_trace)
            for row in episode_trace:
                row["success"] = success
                raw_rows.append(row)

            goal_norms = [float(row["goal_norm"]) for row in episode_trace]
            goal_deltas = [_safe_float(row["goal_delta_l2"]) for row in episode_trace if row["goal_delta_l2"] not in ("", None)]
            goal_cosines = [_safe_float(row["goal_cosine_prev"]) for row in episode_trace if row["goal_cosine_prev"] not in ("", None)]
            entropies = [float(row["policy_entropy"]) for row in episode_trace]
            top1_probs = [float(row["top1_prob"]) for row in episode_trace]
            chosen_probs = [float(row["chosen_action_prob"]) for row in episode_trace]
            update_rows = [row for row in episode_trace if bool(row["goal_updated"])]
            update_kl = [
                _safe_float(row["action_prob_kl_from_prev"])
                for row in update_rows
                if row["action_prob_kl_from_prev"] not in ("", None)
            ]
            update_top1_changes = [
                1.0 if bool(row["top1_action_changed_on_goal_update"]) else 0.0
                for row in update_rows
                if row["top1_action_changed_on_goal_update"] not in ("", None)
            ]
            episode_rows.append(
                {
                    "model_type": model_type,
                    "seed": seed,
                    "action_mode": action_mode,
                    "episode_id": episode_id,
                    "episode_seed": episode_seed,
                    "success": success,
                    "return": total_reward,
                    "episode_length": episode_length,
                    "num_goal_updates": len(update_rows),
                    "mean_goal_norm": _mean(goal_norms),
                    "std_goal_norm": _std(goal_norms),
                    "mean_goal_delta_l2": _mean([value for value in goal_deltas if value is not None]),
                    "std_goal_delta_l2": _std([value for value in goal_deltas if value is not None]),
                    "mean_goal_cosine_prev": _mean([value for value in goal_cosines if value is not None]),
                    "std_goal_cosine_prev": _std([value for value in goal_cosines if value is not None]),
                    "mean_policy_entropy": _mean(entropies),
                    "mean_top1_prob": _mean(top1_probs),
                    "mean_chosen_action_prob": _mean(chosen_probs),
                    "mean_action_kl_on_goal_update": _mean([value for value in update_kl if value is not None]),
                    "top1_action_change_rate_on_goal_update": _mean(update_top1_changes),
                }
            )
            for row in update_rows:
                if row["step"] == 0:
                    continue
                event_rows.append(
                    {
                        "model_type": model_type,
                        "seed": seed,
                        "action_mode": action_mode,
                        "episode_id": episode_id,
                        "step": row["step"],
                        "success": success,
                        "episode_length": episode_length,
                        "goal_delta_l2": row["goal_delta_l2"],
                        "goal_cosine_prev": row["goal_cosine_prev"],
                        "action_prob_kl_from_prev": row["action_prob_kl_from_prev"],
                        "top1_action_changed_on_goal_update": row["top1_action_changed_on_goal_update"],
                        "entropy_after_update": row["policy_entropy"],
                        "top1_prob_after_update": row["top1_prob"],
                    }
                )
    finally:
        env.close()

    raw_path = output_dir / "raw" / f"{model_type}_seed{seed}_{action_mode}_goal_trace.csv"
    _write_csv(raw_path, raw_rows, RAW_FIELDS)
    print(
        f"{model_type} seed={seed} action_mode={action_mode} episodes={len(episode_rows)} "
        f"success={_mean([1.0 if row['success'] else 0.0 for row in episode_rows]):.3f} raw={raw_path}"
    )
    return raw_rows, episode_rows, event_rows


def aggregate_episode_rows(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate_rows: list[dict[str, Any]] = []
    for model_type in MODELS:
        for action_mode in ACTION_MODES:
            group = [
                row
                for row in episode_rows
                if row["model_type"] == model_type and row["action_mode"] == action_mode
            ]
            if not group:
                continue
            aggregate_rows.append(
                {
                    "model_type": model_type,
                    "action_mode": action_mode,
                    "mean_success": _mean([1.0 if row["success"] else 0.0 for row in group]),
                    "mean_return": _mean([float(row["return"]) for row in group]),
                    "mean_episode_length": _mean([float(row["episode_length"]) for row in group]),
                    "mean_num_goal_updates": _mean([float(row["num_goal_updates"]) for row in group]),
                    "mean_goal_norm": _mean([float(row["mean_goal_norm"]) for row in group]),
                    "mean_goal_delta_l2": _mean([float(row["mean_goal_delta_l2"]) for row in group]),
                    "mean_goal_cosine_prev": _mean([float(row["mean_goal_cosine_prev"]) for row in group]),
                    "mean_policy_entropy": _mean([float(row["mean_policy_entropy"]) for row in group]),
                    "mean_top1_prob": _mean([float(row["mean_top1_prob"]) for row in group]),
                    "mean_chosen_action_prob": _mean([float(row["mean_chosen_action_prob"]) for row in group]),
                    "mean_action_kl_on_goal_update": _mean(
                        [
                            float(row["mean_action_kl_on_goal_update"])
                            for row in group
                            if row["mean_action_kl_on_goal_update"] is not None
                        ]
                    ),
                    "mean_top1_action_change_rate_on_goal_update": _mean(
                        [
                            float(row["top1_action_change_rate_on_goal_update"])
                            for row in group
                            if row["top1_action_change_rate_on_goal_update"] is not None
                        ]
                    ),
                }
            )
    return aggregate_rows


def _event_aggregate_rows(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_type in MODELS:
        for action_mode in ACTION_MODES:
            group = [
                row
                for row in event_rows
                if row["model_type"] == model_type and row["action_mode"] == action_mode
            ]
            if not group:
                continue
            rows.append(
                {
                    "model_type": model_type,
                    "action_mode": action_mode,
                    "mean_goal_delta_on_update": _mean([float(row["goal_delta_l2"]) for row in group]),
                    "mean_action_kl_on_update": _mean(
                        [
                            float(row["action_prob_kl_from_prev"])
                            for row in group
                            if row["action_prob_kl_from_prev"] not in ("", None)
                        ]
                    ),
                    "top1_change_rate": _mean(
                        [1.0 if bool(row["top1_action_changed_on_goal_update"]) else 0.0 for row in group]
                    ),
                }
            )
    return rows


def _correlation_rows(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action_mode in ACTION_MODES:
        for model_type in MODELS:
            group = [
                row
                for row in episode_rows
                if row["model_type"] == model_type and row["action_mode"] == action_mode
            ]
            rows.append(
                {
                    "action_mode": action_mode,
                    "model_type": model_type,
                    "corr_ep_len_goal_delta": _pearson(
                        [float(row["episode_length"]) for row in group],
                        [float(row["mean_goal_delta_l2"]) for row in group],
                    ),
                    "corr_ep_len_goal_cosine": _pearson(
                        [float(row["episode_length"]) for row in group],
                        [float(row["mean_goal_cosine_prev"]) for row in group],
                    ),
                }
            )
    return rows


def make_figures(output_dir: Path, aggregate_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]], event_rows: list[dict[str, Any]]) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="manager-goal-mpl-"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping figures because matplotlib is unavailable: {exc}")
        return

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    def bar_metric(metric: str, filename: str, ylabel: str) -> None:
        labels = []
        values = []
        for row in aggregate_rows:
            labels.append(f"{row['model_type']}\n{row['action_mode']}")
            values.append(float(row[metric]))
        plt.figure(figsize=(7, 4))
        plt.bar(labels, values, color=["#4c78a8", "#4c78a8", "#f58518", "#f58518"])
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=160)
        plt.close()

    bar_metric("mean_goal_delta_l2", "mean_goal_delta_by_model_and_mode.png", "Mean goal delta L2")
    bar_metric("mean_goal_cosine_prev", "mean_goal_cosine_by_model_and_mode.png", "Mean goal cosine")
    bar_metric("mean_action_kl_on_goal_update", "goal_update_action_kl_by_model_and_mode.png", "KL(prev || current)")

    for action_mode in ACTION_MODES:
        plt.figure(figsize=(6, 4))
        for model_type, color in (("baseline", "#4c78a8"), ("ablation", "#f58518")):
            group = [
                row
                for row in episode_rows
                if row["model_type"] == model_type and row["action_mode"] == action_mode
            ]
            plt.scatter(
                [float(row["mean_goal_delta_l2"]) for row in group],
                [float(row["episode_length"]) for row in group],
                s=14,
                alpha=0.6,
                label=model_type,
                color=color,
            )
        plt.xlabel("Episode mean goal delta L2")
        plt.ylabel("Episode length")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / f"episode_length_vs_goal_delta_{action_mode}.png", dpi=160)
        plt.close()


def write_markdown(output_dir: Path, aggregate_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]], event_rows: list[dict[str, Any]]) -> None:
    event_aggs = _event_aggregate_rows(event_rows)
    corr_rows = _correlation_rows(episode_rows)
    by_agg = {(row["model_type"], row["action_mode"]): row for row in aggregate_rows}

    lines: list[str] = [
        "# Manager Goal Results",
        "",
        "## 3-seed aggregate table",
        "",
        "| model | action_mode | success | return | ep_len | goal_delta | goal_cosine | entropy | top1_prob | update_action_kl |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model_type"],
                    row["action_mode"],
                    _fmt(row["mean_success"]),
                    _fmt(row["mean_return"]),
                    _fmt(row["mean_episode_length"]),
                    _fmt(row["mean_goal_delta_l2"]),
                    _fmt(row["mean_goal_cosine_prev"]),
                    _fmt(row["mean_policy_entropy"]),
                    _fmt(row["mean_top1_prob"]),
                    _fmt(row["mean_action_kl_on_goal_update"]),
                ]
            )
            + " |"
        )

    metrics = [
        ("mean_success", "success"),
        ("mean_return", "return"),
        ("mean_episode_length", "ep_len"),
        ("mean_goal_delta_l2", "goal_delta"),
        ("mean_goal_cosine_prev", "goal_cosine"),
        ("mean_policy_entropy", "entropy"),
        ("mean_top1_prob", "top1_prob"),
        ("mean_action_kl_on_goal_update", "update_action_kl"),
    ]
    lines.extend(
        [
            "",
            "## Baseline vs Ablation differences",
            "",
            "| action_mode | metric | baseline | ablation | difference |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for action_mode in ACTION_MODES:
        if ("baseline", action_mode) not in by_agg or ("ablation", action_mode) not in by_agg:
            continue
        for metric_key, metric_name in metrics:
            baseline = by_agg[("baseline", action_mode)][metric_key]
            ablation = by_agg[("ablation", action_mode)][metric_key]
            lines.append(
                f"| {action_mode} | {metric_name} | {_fmt(baseline)} | {_fmt(ablation)} | {_fmt(float(ablation) - float(baseline))} |"
            )

    lines.extend(
        [
            "",
            "## Episode length correlation table",
            "",
            "| action_mode | model | corr(ep_len, goal_delta) | corr(ep_len, goal_cosine) |",
            "|---|---|---:|---:|",
        ]
    )
    for row in corr_rows:
        lines.append(
            f"| {row['action_mode']} | {row['model_type']} | {_fmt(row['corr_ep_len_goal_delta'])} | {_fmt(row['corr_ep_len_goal_cosine'])} |"
        )

    lines.extend(
        [
            "",
            "## Goal update event summary table",
            "",
            "| model | action_mode | mean_goal_delta_on_update | mean_action_kl_on_update | top1_change_rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in event_aggs:
        lines.append(
            f"| {row['model_type']} | {row['action_mode']} | {_fmt(row['mean_goal_delta_on_update'])} | {_fmt(row['mean_action_kl_on_update'])} | {_fmt(row['top1_change_rate'])} |"
        )
    (output_dir / "manager_goal_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    analysis: list[str] = [
        "# Manager Goal Analysis",
        "",
        "Q1. Is Ablation more goal-stable than Baseline?",
        "",
    ]
    for action_mode in ACTION_MODES:
        if ("baseline", action_mode) not in by_agg or ("ablation", action_mode) not in by_agg:
            continue
        b = by_agg[("baseline", action_mode)]
        a = by_agg[("ablation", action_mode)]
        analysis.append(
            f"- {action_mode}: Ablation-Baseline goal_delta={float(a['mean_goal_delta_l2']) - float(b['mean_goal_delta_l2']):.4f}, "
            f"goal_cosine={float(a['mean_goal_cosine_prev']) - float(b['mean_goal_cosine_prev']):.4f}."
        )
    analysis.extend(["", "Q2. Does a goal update change the Worker action distribution?", ""])
    for row in event_aggs:
        analysis.append(
            f"- {row['model_type']} {row['action_mode']}: update KL={row['mean_action_kl_on_update']:.4f}, "
            f"top1 change rate={row['top1_change_rate']:.4f}."
        )
    analysis.extend(["", "Q3. Which model connects goal updates to stronger policy changes?", ""])
    for action_mode in ACTION_MODES:
        baseline_matches = [
            row for row in event_aggs if row["model_type"] == "baseline" and row["action_mode"] == action_mode
        ]
        ablation_matches = [
            row for row in event_aggs if row["model_type"] == "ablation" and row["action_mode"] == action_mode
        ]
        if not baseline_matches or not ablation_matches:
            continue
        b = baseline_matches[0]
        a = ablation_matches[0]
        stronger = "ablation" if float(a["mean_action_kl_on_update"]) > float(b["mean_action_kl_on_update"]) else "baseline"
        analysis.append(f"- {action_mode}: {stronger} has the larger mean update KL.")
    analysis.extend(["", "Q4. Is goal instability linked to longer episodes?", ""])
    for row in corr_rows:
        analysis.append(
            f"- {row['model_type']} {row['action_mode']}: corr_len_delta={_fmt(row['corr_ep_len_goal_delta'])}, "
            f"corr_len_cosine={_fmt(row['corr_ep_len_goal_cosine'])}."
        )
    analysis.extend(
        [
            "",
            "Q5. What does this suggest about faster Ablation learning?",
            "",
            "These final-checkpoint diagnostics can suggest whether Ablation learned a simpler and more stable goal interface, but they do not by themselves prove the learning-time mechanism because intermediate checkpoints are unavailable.",
        ]
    )
    (output_dir / "manager_goal_analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")


def run_batch(output_dir: Path, episodes: int, eval_seed_offset: int | None = None) -> None:
    all_episode_rows: list[dict[str, Any]] = []
    all_event_rows: list[dict[str, Any]] = []
    for model_type in MODELS:
        for seed in SEEDS:
            config_path, checkpoint_path = resolve_paths_from_summary(model_type, seed)
            for action_mode in ACTION_MODES:
                _, episode_rows, event_rows = run_goal_trace(
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    model_type=model_type,
                    seed=seed,
                    action_mode=action_mode,
                    episodes=episodes,
                    output_dir=output_dir,
                    eval_seed_offset=eval_seed_offset,
                )
                all_episode_rows.extend(episode_rows)
                all_event_rows.extend(event_rows)

    summaries_dir = output_dir / "summaries"
    aggregate_rows = aggregate_episode_rows(all_episode_rows)
    _write_csv(summaries_dir / "goal_episode_summary.csv", all_episode_rows, EPISODE_FIELDS)
    _write_csv(summaries_dir / "goal_update_event_summary.csv", all_event_rows, EVENT_FIELDS)
    _write_csv(summaries_dir / "goal_aggregate_summary.csv", aggregate_rows, AGG_FIELDS)
    make_figures(output_dir, aggregate_rows, all_episode_rows, all_event_rows)
    write_markdown(output_dir, aggregate_rows, all_episode_rows, all_event_rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.batch_n4s4:
        run_batch(output_dir=output_dir, episodes=args.episodes, eval_seed_offset=args.eval_seed_offset)
        return

    required = {
        "--config": args.config,
        "--checkpoint": args.checkpoint,
        "--model-type": args.model_type,
        "--seed": args.seed,
        "--action-mode": args.action_mode,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"Missing required arguments for single run: {', '.join(missing)}")
    _, episode_rows, event_rows = run_goal_trace(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        seed=args.seed,
        action_mode=args.action_mode,
        episodes=args.episodes,
        output_dir=output_dir,
        eval_seed_offset=args.eval_seed_offset,
    )
    summaries_dir = output_dir / "summaries"
    aggregate_rows = aggregate_episode_rows(episode_rows)
    _write_csv(summaries_dir / "goal_episode_summary.csv", episode_rows, EPISODE_FIELDS)
    _write_csv(summaries_dir / "goal_update_event_summary.csv", event_rows, EVENT_FIELDS)
    _write_csv(summaries_dir / "goal_aggregate_summary.csv", aggregate_rows, AGG_FIELDS)
    make_figures(output_dir, aggregate_rows, episode_rows, event_rows)
    write_markdown(output_dir, aggregate_rows, episode_rows, event_rows)


if __name__ == "__main__":
    main()
