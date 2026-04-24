from __future__ import annotations

from typing import Any


def compute_success(
    terminated: bool,
    truncated: bool,
    total_reward: float,
    info: dict[str, Any] | None = None,
) -> bool:
    """Return whether an episode should count as successful.

    If the environment provides an explicit ``success`` flag in ``info``, use it.
    Otherwise, count only positively rewarded terminated episodes as success.
    Truncated episodes are not considered successful by default.
    """
    if info is not None and "success" in info:
        return bool(info["success"])

    return bool(terminated and not truncated and total_reward > 0.0)
