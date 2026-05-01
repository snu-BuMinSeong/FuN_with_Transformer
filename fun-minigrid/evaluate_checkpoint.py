from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.envs.make_env import make_env
from src.envs.preprocess import preprocess_obs
from src.models.fun import FuNModel
from src.policies.fun_policy import FuNPolicy
from src.training.evaluation import evaluate_policy
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_simple_yaml
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse checkpoint evaluation arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a saved vanilla FuN checkpoint.")
    parser.add_argument("--config", required=True, type=str, help="Path to the training config YAML.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the checkpoint .pt file.")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path.")
    parser.add_argument("--episodes", type=int, default=None, help="Override config eval_episodes.")
    parser.add_argument("--seed-offset", type=int, default=None, help="Override config eval_seed_offset.")
    parser.add_argument("--action-mode", type=str, default="argmax", choices=["argmax", "sample"])
    return parser.parse_args()


def build_model_from_config(config: dict[str, Any], device: torch.device | str | None = None) -> FuNModel:
    """Create a FuN model from a flat training config."""
    model = FuNModel(
        goal_update_interval=int(config.get("goal_update_interval", 10)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        goal_size=int(config.get("goal_size", config.get("goal_dim", 16))),
        num_actions=int(config.get("num_actions", 7)),
        manager_type=str(config.get("manager_type", "recurrent")),
    )
    if device is not None:
        model = model.to(device)
    return model


def build_policy_from_config(
    config: dict[str, Any],
    model: FuNModel,
    device: torch.device | str,
    action_mode: str = "argmax",
) -> FuNPolicy:
    """Create an evaluation policy from a model and config."""
    return FuNPolicy(
        model=model,
        preprocess_fn=preprocess_obs,
        device=device,
        action_mode=action_mode,
    )


def _to_jsonable(value: Any) -> Any:
    """Convert common numeric containers to JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    episodes: int | None = None,
    seed_offset: int | None = None,
    action_mode: str = "argmax",
) -> dict[str, Any]:
    """Load a checkpoint and evaluate it with the config-defined environment."""
    cfg = load_simple_yaml(config_path)
    seed = int(cfg.get("seed", 123))
    eval_episodes = int(episodes if episodes is not None else cfg.get("eval_episodes", 1))
    eval_seed_offset = int(seed_offset if seed_offset is not None else cfg.get("eval_seed_offset", 1000))
    max_steps = int(cfg.get("max_steps", 100)) if cfg.get("max_steps") is not None else None

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg, device=device)
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        map_location=device,
    )
    policy = build_policy_from_config(
        config=cfg,
        model=model,
        device=device,
        action_mode=action_mode,
    )

    env = make_env(
        env_id=str(cfg.get("env_id", "MiniGrid-DoorKey-5x5-v0")),
        render_mode=cfg.get("render_mode"),
        seed=seed,
    )
    try:
        eval_result = evaluate_policy(
            env=env,
            policy=policy,
            num_episodes=eval_episodes,
            max_steps=max_steps,
            seed=seed + eval_seed_offset,
        )
    finally:
        env.close()

    result = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_episode": checkpoint.get("episode"),
        "checkpoint_best_success_rate": checkpoint.get("best_success_rate"),
        "eval_action_mode": action_mode,
        "eval_episodes": eval_episodes,
        "eval_seed_offset": eval_seed_offset,
        "eval_result": eval_result,
    }
    jsonable_result = _to_jsonable(result)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(jsonable_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return jsonable_result


def print_evaluation_summary(result: dict[str, Any]) -> None:
    """Print a compact human-readable evaluation summary."""
    eval_result = result["eval_result"]
    print("[Checkpoint Evaluation]")
    print(f"config: {result['config_path']}")
    print(f"checkpoint: {result['checkpoint_path']}")
    print(f"checkpoint_episode: {result['checkpoint_episode']}")
    print(f"checkpoint_best_success_rate: {result['checkpoint_best_success_rate']}")
    print(f"action_mode: {result['eval_action_mode']}")
    print(f"eval_episodes: {result['eval_episodes']}")
    print(f"eval_seed_offset: {result['eval_seed_offset']}")
    print(
        "metrics: "
        f"mean_reward={eval_result['mean_reward']:.3f}, "
        f"std_reward={eval_result['std_reward']:.3f}, "
        f"success_rate={eval_result['mean_success_rate']:.3f}, "
        f"std_success_rate={eval_result['std_success_rate']:.3f}, "
        f"mean_episode_length={eval_result['mean_episode_length']:.3f}, "
        f"std_episode_length={eval_result['std_episode_length']:.3f}"
    )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    result = evaluate_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        episodes=args.episodes,
        seed_offset=args.seed_offset,
        action_mode=args.action_mode,
    )
    print_evaluation_summary(result)
    if args.output is not None:
        print(f"output: {args.output}")


if __name__ == "__main__":
    main()
