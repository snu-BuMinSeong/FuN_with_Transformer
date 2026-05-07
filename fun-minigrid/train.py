from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.envs.make_env import make_env
from src.envs.preprocess import preprocess_obs
from src.models.fun import FuNModel
from src.policies.fun_policy import FuNPolicy
from src.training.evaluation import build_eval_comparison, build_eval_comparison_text, evaluate_policy
from src.training.trainer import train
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import load_simple_yaml
from src.utils.logger import append_eval_log, append_training_log, write_json_summary
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse lightweight training script arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_fun.yaml")
    return parser.parse_args()


def compute_moving_average(values: list[float], window: int) -> float:
    """Return the trailing moving average for the given window size."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}.")
    if not values:
        return 0.0

    tail = values[-window:]
    return float(sum(tail) / len(tail))


def build_progress_line(
    result: dict[str, Any],
    total_episodes: int,
    best_reward: float,
    best_success_rate: float,
) -> str:
    """Return a compact training progress summary line."""
    return (
        f"episode={result['episode']}/{total_episodes} "
        f"reward={result['total_reward']:.3f} "
        f"length={result['episode_length']} "
        f"worker_loss={result['worker_loss']:.6f} "
        f"loss={result['total_loss']:.6f} "
        f"value_loss={result['value_loss']:.6f} "
        f"manager_loss={result['manager_loss']:.6f} "
        f"entropy={result['entropy_mean']:.6f} "
        f"adv_abs={result['advantages_abs_mean']:.6f} "
        f"grad_norm={result['grad_norm']:.6f} "
        f"grads[e/m/w/v/mv]="
        f"{result['encoder_grad_norm']:.4f}/"
        f"{result['manager_grad_norm']:.4f}/"
        f"{result['worker_grad_norm']:.4f}/"
        f"{result['value_head_grad_norm']:.4f}/"
        f"{result['manager_value_head_grad_norm']:.4f} "
        f"reward_ma={result['reward_moving_avg']:.3f} "
        f"success_ma={result['success_moving_avg']:.3f} "
        f"loss_ma={result['loss_moving_avg']:.6f} "
        f"reward_signal={result['nonzero_reward_fraction']:.3f} "
        f"return_signal={result['nonzero_return_fraction']:.3f} "
        f"action_cov={result['action_coverage']:.3f} "
        f"best_reward={best_reward:.3f} "
        f"best_success_ma={best_success_rate:.3f} "
        f"success={result['success']}"
    )


def normalize_eval_result(episode: int, eval_result: dict[str, Any]) -> dict[str, Any]:
    """Map evaluation.py metrics to the baseline eval CSV schema."""
    return {
        "episode": episode,
        "eval_success_rate": float(eval_result["mean_success_rate"]),
        "eval_mean_return": float(eval_result["mean_reward"]),
        "eval_std_return": float(eval_result["std_reward"]),
        "eval_mean_episode_length": float(eval_result["mean_episode_length"]),
        "eval_std_episode_length": float(eval_result["std_episode_length"]),
        "eval_episode_seeds": ",".join(str(seed) for seed in eval_result["episode_seeds"]),
    }


def normalize_mode_eval_result(
    episode: int,
    mode: str,
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    """Map evaluation.py metrics to mode-specific eval CSV columns."""
    return {
        "episode": episode,
        f"eval_{mode}_success_rate": float(eval_result["mean_success_rate"]),
        f"eval_{mode}_mean_return": float(eval_result["mean_reward"]),
        f"eval_{mode}_std_return": float(eval_result["std_reward"]),
        f"eval_{mode}_mean_episode_length": float(eval_result["mean_episode_length"]),
        f"eval_{mode}_std_episode_length": float(eval_result["std_episode_length"]),
        f"eval_{mode}_episode_seeds": ",".join(str(seed) for seed in eval_result["episode_seeds"]),
    }


def get_eval_action_modes(config: dict[str, Any]) -> list[str]:
    """Return configured eval action modes with backward-compatible fallback."""
    raw_modes = config.get("eval_action_modes")
    if raw_modes is None:
        raw_modes = [str(config.get("eval_action_mode", "argmax"))]
    elif isinstance(raw_modes, str):
        raw_modes = [raw_modes]
    elif isinstance(raw_modes, list):
        raw_modes = [str(mode) for mode in raw_modes]
    else:
        raise ValueError(f"eval_action_modes must be a list or string, got {type(raw_modes).__name__}.")

    modes: list[str] = []
    for mode in raw_modes:
        if mode not in {"sample", "argmax"}:
            raise ValueError(f"eval action mode must be 'sample' or 'argmax', got {mode}.")
        if mode not in modes:
            modes.append(mode)

    if not modes:
        raise ValueError("eval_action_modes must contain at least one mode.")
    return modes


def get_primary_eval_action_mode(config: dict[str, Any], eval_action_modes: list[str]) -> str:
    """Return the mode used for legacy eval fields and best.pt selection."""
    primary_mode = str(config.get("eval_action_mode", eval_action_modes[0]))
    if primary_mode not in eval_action_modes:
        primary_mode = eval_action_modes[0]
    return primary_mode


def main() -> None:
    """Train vanilla FuN from config and save per-episode logs."""
    args = parse_args()
    cfg = load_simple_yaml(args.config)

    seed = int(cfg.get("seed", 123))
    moving_avg_window = int(cfg.get("moving_avg_window", 5))
    log_interval = int(cfg.get("log_interval", 25))
    log_dir = Path(str(cfg.get("log_dir", "logs")))
    checkpoint_dir = Path(str(cfg.get("checkpoint_dir", "checkpoints")))
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(cfg.get("log_path", log_dir / "train.csv"))
    eval_log_path = str(cfg.get("eval_log_path", log_dir / "eval.csv"))
    summary_path = str(cfg.get("summary_path", log_dir / "summary.json"))
    last_checkpoint_path = checkpoint_dir / "last.pt"
    best_checkpoint_path = checkpoint_dir / "best.pt"
    best_sample_checkpoint_path = checkpoint_dir / "best_sample.pt"
    best_argmax_checkpoint_path = checkpoint_dir / "best_argmax.pt"
    total_episodes = int(cfg.get("total_episodes", 10))
    eval_interval = int(cfg.get("eval_interval", total_episodes))
    eval_episodes = int(cfg.get("eval_episodes", 1))
    eval_seed_offset = int(cfg.get("eval_seed_offset", 1000))
    max_steps = int(cfg.get("max_steps", 100)) if cfg.get("max_steps") is not None else None
    eval_action_modes = get_eval_action_modes(cfg)
    primary_eval_action_mode = get_primary_eval_action_mode(cfg, eval_action_modes)
    resume_from_checkpoint = cfg.get("resume_from_checkpoint")
    resume_optimizer_state = bool(cfg.get("resume_optimizer_state", False))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(
        env_id=str(cfg.get("env_id", "MiniGrid-DoorKey-5x5-v0")),
        render_mode=cfg.get("render_mode"),
        seed=seed,
        max_episode_steps=max_steps,
    )
    model = FuNModel(
        goal_update_interval=int(cfg.get("goal_update_interval", 10)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        goal_size=int(cfg.get("goal_size", cfg.get("goal_dim", 16))),
        num_actions=int(cfg.get("num_actions", 7)),
        manager_type=str(cfg.get("manager_type", "recurrent")),
    ).to(device)
    policy = FuNPolicy(
        model=model,
        preprocess_fn=preprocess_obs,
        device=device,
        action_mode=str(cfg.get("action_mode", "sample")),
    )
    eval_policies = {
        mode: FuNPolicy(
            model=model,
            preprocess_fn=preprocess_obs,
            device=device,
            action_mode=mode,
        )
        for mode in eval_action_modes
    }
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
    )

    resume_info: dict[str, Any] = {
        "enabled": resume_from_checkpoint is not None,
        "checkpoint_path": str(resume_from_checkpoint) if resume_from_checkpoint is not None else None,
        "checkpoint_episode": None,
        "checkpoint_best_success_rate": None,
        "optimizer_state_requested": resume_optimizer_state,
        "optimizer_state_loaded": False,
    }
    if resume_from_checkpoint is not None:
        checkpoint = load_checkpoint(
            resume_from_checkpoint,
            model=model,
            optimizer=optimizer if resume_optimizer_state else None,
            map_location=device,
        )
        resume_info["checkpoint_episode"] = checkpoint.get("episode")
        resume_info["checkpoint_best_success_rate"] = checkpoint.get("best_success_rate")
        resume_info["optimizer_state_loaded"] = bool(
            resume_optimizer_state and checkpoint.get("optimizer_state_dict") is not None
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(cfg.get("learning_rate", 1e-3))

    print(f"device={device}")
    print(
        "train_config="
        f"episodes={cfg.get('total_episodes')} "
        f"lr={cfg.get('learning_rate')} "
        f"gamma={cfg.get('gamma')} "
        f"value_loss_coef={cfg.get('value_loss_coef', 0.5)} "
        f"manager_loss_coef={cfg.get('manager_loss_coef', 0.0)} "
        f"goal_update_interval={cfg.get('goal_update_interval')} "
        f"hidden_dim={cfg.get('hidden_dim')} "
        f"moving_avg_window={moving_avg_window} "
        f"eval_interval={eval_interval} "
        f"eval_episodes={eval_episodes} "
        f"eval_action_modes={','.join(eval_action_modes)} "
        f"primary_eval_action_mode={primary_eval_action_mode} "
        f"eval_seed_offset={eval_seed_offset} "
        f"resume_from_checkpoint={resume_from_checkpoint} "
        f"resume_optimizer_state_loaded={resume_info['optimizer_state_loaded']} "
        f"log_interval={log_interval}"
    )

    try:
        pre_evals = {
            mode: evaluate_policy(
                env=env,
                policy=eval_policy,
                num_episodes=eval_episodes,
                max_steps=max_steps,
                seed=seed + eval_seed_offset,
            )
            for mode, eval_policy in eval_policies.items()
        }
        pre_eval = pre_evals[primary_eval_action_mode]
        print(
            "pre_eval="
            f"mode={primary_eval_action_mode} "
            f"mean_reward={pre_eval['mean_reward']:.3f} "
            f"std_reward={pre_eval['std_reward']:.3f} "
            f"success_rate={pre_eval['mean_success_rate']:.3f} "
            f"std_success_rate={pre_eval['std_success_rate']:.3f} "
            f"mean_length={pre_eval['mean_episode_length']:.3f} "
            f"std_length={pre_eval['std_episode_length']:.3f}"
        )

        reward_history: list[float] = []
        success_history: list[float] = []
        loss_history: list[float] = []
        results: list[dict[str, Any]] = []
        eval_results: list[dict[str, Any]] = []
        best_success_rate = float("-inf")
        best_success_by_mode = {mode: float("-inf") for mode in eval_action_modes}
        best_checkpoint_path_by_mode = {
            "sample": best_sample_checkpoint_path,
            "argmax": best_argmax_checkpoint_path,
        }

        def on_episode_end(result: dict[str, Any]) -> None:
            nonlocal best_success_rate
            reward_history.append(float(result["total_reward"]))
            success_history.append(1.0 if bool(result["success"]) else 0.0)
            loss_history.append(float(result["total_loss"]))
            result["reward_moving_avg"] = compute_moving_average(reward_history, moving_avg_window)
            result["success_moving_avg"] = compute_moving_average(success_history, moving_avg_window)
            result["loss_moving_avg"] = compute_moving_average(loss_history, moving_avg_window)
            append_training_log(log_path, result)
            results.append(dict(result))

            should_print = (
                result["episode"] == 1
                or result["episode"] % log_interval == 0
                or result["episode"] == total_episodes
            )
            if should_print:
                print(
                    build_progress_line(
                        result=result,
                        total_episodes=total_episodes,
                        best_reward=max(reward_history),
                        best_success_rate=max(
                            compute_moving_average(success_history[:idx], moving_avg_window)
                            for idx in range(1, len(success_history) + 1)
                        ),
                    )
                )

            episode = int(result["episode"])
            if eval_interval > 0 and episode % eval_interval == 0:
                interval_evals = {
                    mode: evaluate_policy(
                        env=env,
                        policy=eval_policy,
                        num_episodes=eval_episodes,
                        max_steps=max_steps,
                        seed=seed + eval_seed_offset + episode,
                    )
                    for mode, eval_policy in eval_policies.items()
                }
                primary_eval = interval_evals[primary_eval_action_mode]
                eval_row = normalize_eval_result(episode, primary_eval)
                for mode, mode_eval in interval_evals.items():
                    eval_row.update(normalize_mode_eval_result(episode, mode, mode_eval))
                append_eval_log(eval_log_path, eval_row)
                eval_results.append({**eval_row, "raw_eval": primary_eval, "raw_evals": interval_evals})

                current_success_rate = float(eval_row["eval_success_rate"])
                if current_success_rate >= best_success_rate:
                    best_success_rate = current_success_rate
                    save_checkpoint(
                        best_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        episode=episode,
                        config=cfg,
                        best_success_rate=best_success_rate,
                        extra={"eval": eval_row},
                    )

                for mode, mode_eval in interval_evals.items():
                    mode_success_rate = float(mode_eval["mean_success_rate"])
                    if mode_success_rate >= best_success_by_mode[mode]:
                        best_success_by_mode[mode] = mode_success_rate
                        mode_checkpoint_path = best_checkpoint_path_by_mode[mode]
                        save_checkpoint(
                            mode_checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            episode=episode,
                            config=cfg,
                            best_success_rate=mode_success_rate,
                            extra={"eval": eval_row, "eval_action_mode": mode},
                        )
                save_checkpoint(
                    last_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    episode=episode,
                    config=cfg,
                    best_success_rate=best_success_rate,
                    extra={"eval": eval_row},
                )

        train(
            env=env,
            policy=policy,
            optimizer=optimizer,
            num_episodes=total_episodes,
            gamma=float(cfg.get("gamma", 1.0)),
            max_steps=max_steps,
            seed=seed,
            entropy_coef=float(cfg.get("entropy_coef", 0.0)),
            value_loss_coef=float(cfg.get("value_loss_coef", 0.5)),
            manager_loss_coef=float(cfg.get("manager_loss_coef", 0.0)),
            grad_clip_norm=float(cfg.get("grad_clip_norm", 1.0))
            if cfg.get("grad_clip_norm") is not None
            else None,
            keep_trajectory=False,
            episode_callback=on_episode_end,
        )

        post_evals = {
            mode: evaluate_policy(
                env=env,
                policy=eval_policy,
                num_episodes=eval_episodes,
                max_steps=max_steps,
                seed=seed + 2 * eval_seed_offset,
            )
            for mode, eval_policy in eval_policies.items()
        }
        post_eval = post_evals[primary_eval_action_mode]
        final_episode_checkpoint_path = checkpoint_dir / f"episode_{total_episodes}.pt"
        save_checkpoint(
            final_episode_checkpoint_path,
            model=model,
            optimizer=optimizer,
            episode=total_episodes,
            config=cfg,
            best_success_rate=best_success_rate if best_success_rate != float("-inf") else None,
            extra={"final_eval": normalize_eval_result(total_episodes, post_eval)},
        )
        eval_comparison = build_eval_comparison(pre_eval, post_eval)
        comparison_text = build_eval_comparison_text(pre_eval, post_eval)
        print(
            "post_eval="
            f"mode={primary_eval_action_mode} "
            f"mean_reward={post_eval['mean_reward']:.3f} "
            f"std_reward={post_eval['std_reward']:.3f} "
            f"success_rate={post_eval['mean_success_rate']:.3f} "
            f"std_success_rate={post_eval['std_success_rate']:.3f} "
            f"mean_length={post_eval['mean_episode_length']:.3f} "
            f"std_length={post_eval['std_episode_length']:.3f}"
        )
        print(f"comparison={comparison_text}")

        summary: dict[str, Any] = {
            "config": cfg,
            "config_path": args.config,
            "seed": seed,
            "num_episodes": len(results),
            "final_episode": len(results),
            "moving_avg_window": moving_avg_window,
            "eval_episodes": eval_episodes,
            "eval_seed_offset": eval_seed_offset,
            "eval_action_modes": eval_action_modes,
            "primary_eval_action_mode": primary_eval_action_mode,
            "resume": resume_info,
            "best_success_rate": best_success_rate if best_success_rate != float("-inf") else None,
            "best_checkpoint_path": str(best_checkpoint_path),
            "best_sample_success_rate": best_success_by_mode.get("sample")
            if best_success_by_mode.get("sample") != float("-inf")
            else None,
            "best_sample_checkpoint_path": str(best_sample_checkpoint_path)
            if "sample" in eval_action_modes
            else None,
            "best_argmax_success_rate": best_success_by_mode.get("argmax")
            if best_success_by_mode.get("argmax") != float("-inf")
            else None,
            "best_argmax_checkpoint_path": str(best_argmax_checkpoint_path)
            if "argmax" in eval_action_modes
            else None,
            "last_checkpoint_path": str(last_checkpoint_path),
            "final_episode_checkpoint_path": str(final_episode_checkpoint_path),
            "train_log_path": log_path,
            "eval_log_path": eval_log_path,
            "final_eval": normalize_eval_result(total_episodes, post_eval),
            "final_evals": {
                mode: normalize_eval_result(total_episodes, mode_eval)
                for mode, mode_eval in post_evals.items()
            },
            "final_reward_moving_avg": reward_history and compute_moving_average(reward_history, moving_avg_window),
            "final_success_moving_avg": success_history
            and compute_moving_average(success_history, moving_avg_window),
            "final_loss_moving_avg": loss_history and compute_moving_average(loss_history, moving_avg_window),
            "best_reward": max(reward_history) if reward_history else 0.0,
            "best_success": max(success_history) if success_history else 0.0,
            "best_reward_moving_avg": max(
                compute_moving_average(reward_history[:idx], moving_avg_window)
                for idx in range(1, len(reward_history) + 1)
            )
            if reward_history
            else 0.0,
            "best_success_moving_avg": max(
                compute_moving_average(success_history[:idx], moving_avg_window)
                for idx in range(1, len(success_history) + 1)
            )
            if success_history
            else 0.0,
            "episodes_with_reward_signal": sum(1 for result in results if result["has_reward_signal"]),
            "episodes_with_return_signal": sum(1 for result in results if result["has_return_signal"]),
            "mean_worker_loss": sum(result["worker_loss"] for result in results) / len(results) if results else 0.0,
            "mean_value_loss": sum(result["value_loss"] for result in results) / len(results) if results else 0.0,
            "mean_manager_loss": sum(result["manager_loss"] for result in results) / len(results) if results else 0.0,
            "mean_entropy": sum(result["entropy_mean"] for result in results) / len(results) if results else 0.0,
            "mean_action_coverage": sum(result["action_coverage"] for result in results) / len(results)
            if results
            else 0.0,
            "mean_encoder_grad_norm": sum(result["encoder_grad_norm"] for result in results) / len(results)
            if results
            else 0.0,
            "mean_manager_grad_norm": sum(result["manager_grad_norm"] for result in results) / len(results)
            if results
            else 0.0,
            "mean_worker_grad_norm": sum(result["worker_grad_norm"] for result in results) / len(results)
            if results
            else 0.0,
            "mean_value_head_grad_norm": sum(result["value_head_grad_norm"] for result in results) / len(results)
            if results
            else 0.0,
            "mean_manager_value_head_grad_norm": sum(
                result["manager_value_head_grad_norm"] for result in results
            )
            / len(results)
            if results
            else 0.0,
            "pre_eval": pre_eval,
            "pre_evals": pre_evals,
            "post_eval": post_eval,
            "post_evals": post_evals,
            "interval_evals": eval_results,
            "eval_comparison": eval_comparison,
            "comparison_text": comparison_text,
            "results": [
                {
                    key: value
                    for key, value in result.items()
                    if key != "trajectory"
                }
                for result in results
            ],
        }
        write_json_summary(summary_path, summary)
    finally:
        env.close()


if __name__ == "__main__":
    main()
