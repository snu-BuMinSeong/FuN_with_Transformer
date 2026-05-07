from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
from gymnasium import Env
from minigrid.wrappers import ImgObsWrapper


def make_env(
    env_id: str = "MiniGrid-DoorKey-5x5-v0",
    render_mode: str | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
) -> Env:
    """Create a MiniGrid environment with image observations."""
    kwargs = {"render_mode": render_mode}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make(env_id, **kwargs)
    if max_episode_steps is not None and hasattr(env.unwrapped, "max_steps"):
        env.unwrapped.max_steps = max_episode_steps
    env = ImgObsWrapper(env)

    if seed is not None:
        env.reset(seed=seed)

    return env
