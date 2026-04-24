from __future__ import annotations

from typing import Any

import numpy as np


def preprocess_obs(obs: Any) -> np.ndarray:
    """Convert an ImgObsWrapper observation to a CHW float32 array.

    MiniGrid's ImgObsWrapper returns image observations in HWC format.
    This function keeps the values unnormalized for week 1 and only changes
    dtype and layout so later code can consume observations consistently.
    """
    obs_array = np.asarray(obs, dtype=np.float32)

    if obs_array.ndim == 3:
        obs_array = np.transpose(obs_array, (2, 0, 1))

    return obs_array
