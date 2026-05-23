from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
from gymnasium import Env
from gymnasium.envs.registration import register, registry
from minigrid.wrappers import ImgObsWrapper

CUSTOM_MULTIROOM_ENVS = {
    "MiniGrid-MultiRoom-N3-S5-v0": {"minNumRooms": 3, "maxNumRooms": 3, "maxRoomSize": 5},
    "MiniGrid-MultiRoom-N4-S4-v0": {"minNumRooms": 4, "maxNumRooms": 4, "maxRoomSize": 4},
}


def register_custom_envs() -> None:
    """Register project-specific MiniGrid variants missing from upstream."""
    for env_id, kwargs in CUSTOM_MULTIROOM_ENVS.items():
        if env_id in registry:
            continue
        register(
            id=env_id,
            entry_point="minigrid.envs:MultiRoomEnv",
            kwargs=kwargs,
        )


def make_env(
    env_id: str = "MiniGrid-DoorKey-5x5-v0",
    render_mode: str | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
) -> Env:
    """Create a MiniGrid environment with image observations."""
    register_custom_envs()

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
