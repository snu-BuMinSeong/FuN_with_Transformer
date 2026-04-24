from __future__ import annotations

import statistics
from typing import Any

from src.training.rollout import run_episode


def _safe_mean(values: list[float]) -> float:
    """Return the mean of a non-empty list."""
    return float(sum(values) / len(values))


def _safe_std(values: list[float]) -> float:
    """Return the population standard deviation, or 0 for one value."""
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def evaluate_policy(
    env: Any,
    policy: Any,
    num_episodes: int,
    max_steps: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run evaluation episodes and return aggregate metrics."""
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}.")

    episode_results: list[dict[str, Any]] = []
    rewards: list[float] = []
    successes: list[float] = []
    lengths: list[float] = []
    episode_seeds: list[int | None] = []

    for episode_idx in range(num_episodes):
        episode_seed = None if seed is None else seed + episode_idx
        result = run_episode(
            env=env,
            policy=policy,
            max_steps=max_steps,
            seed=episode_seed,
        )
        result["episode"] = episode_idx + 1
        episode_results.append(result)
        rewards.append(float(result["total_reward"]))
        successes.append(1.0 if bool(result["success"]) else 0.0)
        lengths.append(float(result["episode_length"]))
        episode_seeds.append(episode_seed)

    return {
        "num_episodes": num_episodes,
        "mean_reward": _safe_mean(rewards),
        "std_reward": _safe_std(rewards),
        "mean_success_rate": _safe_mean(successes),
        "std_success_rate": _safe_std(successes),
        "mean_episode_length": _safe_mean(lengths),
        "std_episode_length": _safe_std(lengths),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "min_episode_length": min(lengths),
        "max_episode_length": max(lengths),
        "episode_seeds": episode_seeds,
        "episode_results": episode_results,
    }


def build_eval_comparison(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    """Return delta metrics for pre/post evaluation."""
    return {
        "reward_delta": float(after["mean_reward"]) - float(before["mean_reward"]),
        "success_rate_delta": float(after["mean_success_rate"]) - float(before["mean_success_rate"]),
        "episode_length_delta": float(after["mean_episode_length"]) - float(before["mean_episode_length"]),
    }


def build_eval_comparison_text(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Build a short comparison sentence for pre/post evaluation."""
    comparison = build_eval_comparison(before, after)
    reward_delta = comparison["reward_delta"]
    success_delta = comparison["success_rate_delta"]
    length_delta = comparison["episode_length_delta"]

    if reward_delta > 0.0 or success_delta > 0.0:
        trend = "학습 후 성능이 학습 전보다 일부 개선되었다."
    elif reward_delta == 0.0 and success_delta == 0.0 and length_delta == 0.0:
        trend = "학습 전후 평균 reward, success rate, episode length 차이는 없었다."
    else:
        trend = "학습 후 성능이 학습 전보다 좋아졌다고 보기 어렵다."

    return (
        f"{trend} "
        f"mean_reward: {before['mean_reward']:.3f} -> {after['mean_reward']:.3f}, "
        f"success_rate: {before['mean_success_rate']:.3f} -> {after['mean_success_rate']:.3f}, "
        f"mean_episode_length: {before['mean_episode_length']:.3f} -> {after['mean_episode_length']:.3f}."
    )
