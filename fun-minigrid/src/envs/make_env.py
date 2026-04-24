from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
from gymnasium import Env
from minigrid.wrappers import ImgObsWrapper


def make_env(
    env_id: str = "MiniGrid-DoorKey-5x5-v0",
    render_mode: str | None = None,
    seed: int | None = None,
) -> Env:
    """Create a MiniGrid environment with image observations."""
    env = gym.make(env_id, render_mode=render_mode)
    env = ImgObsWrapper(env)

    if seed is not None:
        env.reset(seed=seed)

    return env
