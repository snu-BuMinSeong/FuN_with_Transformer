from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.make_env import make_env
from src.envs.preprocess import preprocess_obs
from src.models.fun import FuNModel


def test_fun_with_real_env_obs() -> None:
    env = make_env()
    obs, info = env.reset()

    proc = preprocess_obs(obs)
    obs_tensor = torch.tensor(proc, dtype=torch.float32).unsqueeze(0)

    model = FuNModel(goal_update_interval=10)
    hidden = model.init_hidden(batch_size=1)
    goal = model.init_goal(batch_size=1)

    out = model(obs_tensor, hidden, goal, step_count=0)

    print("[Real Env Obs Test]")
    print("obs_tensor shape  :", obs_tensor.shape)
    print("state_emb shape   :", out["state_emb"].shape)
    print("goal shape        :", out["goal"].shape)
    print("hidden_state shape:", out["hidden_state"].shape)
    print("logits shape      :", out["logits"].shape)

    assert out["state_emb"].shape == (1, 64)
    assert out["goal"].shape == (1, 16)
    assert out["hidden_state"].shape == (1, 64)
    assert out["logits"].shape == (1, 7)

    env.close()


if __name__ == "__main__":
    test_fun_with_real_env_obs()
    print("Real observation test passed.")
