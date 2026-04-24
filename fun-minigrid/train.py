from __future__ import annotations

import argparse
from typing import Any

import torch

from src.envs.make_env import make_env
from src.envs.preprocess import preprocess_obs
from src.models.fun import FuNModel
from src.policies.fun_policy import FuNPolicy
from src.training.evaluation import build_eval_comparison, build_eval_comparison_text, evaluate_policy
from src.training.trainer import train
from src.utils.config import load_simple_yaml
from src.utils.logger import append_training_log, write_json_summary
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


def main() -> None:
    """Train vanilla FuN from config and save per-episode logs."""
    args = parse_args()
    cfg = load_simple_yaml(args.config)

    seed = int(cfg.get("seed", 123))
    moving_avg_window = int(cfg.get("moving_avg_window", 5))
    log_interval = int(cfg.get("log_interval", 25))
    log_path = str(cfg.get("log_path", "logs/week2_train.csv"))
    summary_path = str(cfg.get("summary_path", "logs/week2_train_summary.json"))
    eval_episodes = int(cfg.get("eval_episodes", 1))
    eval_seed_offset = int(cfg.get("eval_seed_offset", 1000))
    max_steps = int(cfg.get("max_steps", 100)) if cfg.get("max_steps") is not None else None
    total_episodes = int(cfg.get("total_episodes", 10))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(
        env_id=str(cfg.get("env_id", "MiniGrid-DoorKey-5x5-v0")),
        render_mode=cfg.get("render_mode"),
        seed=seed,
    )
    model = FuNModel(
        goal_update_interval=int(cfg.get("goal_update_interval", 10)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        goal_size=int(cfg.get("goal_size", 16)),
        num_actions=int(cfg.get("num_actions", 7)),
    ).to(device)
    policy = FuNPolicy(
        model=model,
        preprocess_fn=preprocess_obs,
        device=device,
        action_mode=str(cfg.get("action_mode", "sample")),
    )
    eval_policy = FuNPolicy(
        model=model,
        preprocess_fn=preprocess_obs,
        device=device,
        action_mode=str(cfg.get("eval_action_mode", "argmax")),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
    )

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
        f"eval_episodes={eval_episodes} "
        f"eval_seed_offset={eval_seed_offset} "
        f"log_interval={log_interval}"
    )

    try:
        pre_eval = evaluate_policy(
            env=env,
            policy=eval_policy,
            num_episodes=eval_episodes,
            max_steps=max_steps,
            seed=seed + eval_seed_offset,
        )
        print(
            "pre_eval="
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

        def on_episode_end(result: dict[str, Any]) -> None:
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

        post_eval = evaluate_policy(
            env=env,
            policy=eval_policy,
            num_episodes=eval_episodes,
            max_steps=max_steps,
            seed=seed + 2 * eval_seed_offset,
        )
        eval_comparison = build_eval_comparison(pre_eval, post_eval)
        comparison_text = build_eval_comparison_text(pre_eval, post_eval)
        print(
            "post_eval="
            f"mean_reward={post_eval['mean_reward']:.3f} "
            f"std_reward={post_eval['std_reward']:.3f} "
            f"success_rate={post_eval['mean_success_rate']:.3f} "
            f"std_success_rate={post_eval['std_success_rate']:.3f} "
            f"mean_length={post_eval['mean_episode_length']:.3f} "
            f"std_length={post_eval['std_episode_length']:.3f}"
        )
        print(f"comparison={comparison_text}")

        summary: dict[str, Any] = {
            "config_path": args.config,
            "seed": seed,
            "num_episodes": len(results),
            "moving_avg_window": moving_avg_window,
            "eval_episodes": eval_episodes,
            "eval_seed_offset": eval_seed_offset,
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
            "post_eval": post_eval,
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
