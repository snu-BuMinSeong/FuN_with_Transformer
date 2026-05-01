from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.encoder import ObservationEncoder
from src.models.fun import FuNModel
from src.models.manager import AblationManager, Manager
from src.models.worker import Worker


def test_encoder_shape() -> None:
    batch_size = 2
    obs = torch.randn(batch_size, 3, 7, 7, dtype=torch.float32)

    model = ObservationEncoder()
    out = model(obs)

    print("[Encoder]")
    print("input shape :", obs.shape)
    print("output shape:", out.shape)

    assert out.shape == (batch_size, 64), f"Expected (2, 64), got {out.shape}"


def test_manager_shape() -> None:
    batch_size = 2
    state_emb = torch.randn(batch_size, 64, dtype=torch.float32)

    model = Manager()
    hidden = model.init_hidden(batch_size)

    goal, next_hidden = model(state_emb, hidden)

    print("\n[Manager]")
    print("state_emb shape   :", state_emb.shape)
    print("hidden shape      :", hidden.shape)
    print("goal shape        :", goal.shape)
    print("next_hidden shape :", next_hidden.shape)

    assert hidden.shape == (batch_size, 64), f"Expected hidden (2, 64), got {hidden.shape}"
    assert goal.shape == (batch_size, 16), f"Expected goal (2, 16), got {goal.shape}"
    assert next_hidden.shape == (batch_size, 64), f"Expected next_hidden (2, 64), got {next_hidden.shape}"


def test_ablation_manager_shape() -> None:
    batch_size = 2
    state_emb = torch.randn(batch_size, 64, dtype=torch.float32)

    model = AblationManager()
    hidden = model.init_hidden(batch_size)

    goal, next_hidden = model(state_emb, hidden)

    print("\n[AblationManager]")
    print("state_emb shape   :", state_emb.shape)
    print("hidden shape      :", hidden.shape)
    print("goal shape        :", goal.shape)
    print("next_hidden shape :", next_hidden.shape)

    assert hidden.shape == (batch_size, 64), f"Expected hidden (2, 64), got {hidden.shape}"
    assert goal.shape == (batch_size, 16), f"Expected goal (2, 16), got {goal.shape}"
    assert next_hidden.shape == (batch_size, 64), f"Expected next_hidden (2, 64), got {next_hidden.shape}"


def test_ablation_manager_goal_ignores_hidden_state() -> None:
    batch_size = 2
    state_emb = torch.randn(batch_size, 64, dtype=torch.float32)
    hidden_a = torch.randn(batch_size, 64, dtype=torch.float32)
    hidden_b = torch.randn(batch_size, 64, dtype=torch.float32)

    model = AblationManager()
    goal_a, next_hidden_a = model(state_emb, hidden_a)
    goal_b, next_hidden_b = model(state_emb, hidden_b)

    print("\n[AblationManager Hidden Independence]")
    print("goal_a shape:", goal_a.shape)
    print("goal_b shape:", goal_b.shape)
    print("goals equal :", torch.allclose(goal_a, goal_b))

    assert torch.allclose(goal_a, goal_b)
    assert torch.equal(next_hidden_a, hidden_a)
    assert torch.equal(next_hidden_b, hidden_b)


def test_worker_shape() -> None:
    batch_size = 2
    state_emb = torch.randn(batch_size, 64, dtype=torch.float32)
    goal = torch.randn(batch_size, 16, dtype=torch.float32)

    model = Worker()
    logits = model(state_emb, goal)
    action_dist = model.get_action_distribution(state_emb, goal)

    print("\n[Worker]")
    print("state_emb shape :", state_emb.shape)
    print("goal shape      :", goal.shape)
    print("logits shape    :", logits.shape)
    print("probs shape     :", action_dist.probs.shape)

    assert logits.shape == (batch_size, 7), f"Expected logits (2, 7), got {logits.shape}"
    assert action_dist.probs.shape == (batch_size, 7), (
        f"Expected probs (2, 7), got {action_dist.probs.shape}"
    )


def test_fun_shape() -> None:
    batch_size = 2
    obs = torch.randn(batch_size, 3, 7, 7, dtype=torch.float32)

    model = FuNModel(goal_update_interval=10)
    hidden = model.init_hidden(batch_size)
    goal = model.init_goal(batch_size)

    out = model(obs, hidden, goal, step_count=0)

    print("\n[FuNModel]")
    print("obs shape         :", obs.shape)
    print("state_emb shape   :", out["state_emb"].shape)
    print("goal shape        :", out["goal"].shape)
    print("hidden_state shape:", out["hidden_state"].shape)
    print("logits shape      :", out["logits"].shape)
    print("value shape       :", out["value"].shape)
    print("manager_value shape:", out["manager_value"].shape)
    print("action_probs shape:", out["action_probs"].shape)
    print("goal_updated      :", out["goal_updated"])

    assert out["state_emb"].shape == (batch_size, 64), (
        f'Expected state_emb (2, 64), got {out["state_emb"].shape}'
    )
    assert out["goal"].shape == (batch_size, 16), f'Expected goal (2, 16), got {out["goal"].shape}'
    assert out["hidden_state"].shape == (batch_size, 64), (
        f'Expected hidden_state (2, 64), got {out["hidden_state"].shape}'
    )
    assert out["logits"].shape == (batch_size, 7), f'Expected logits (2, 7), got {out["logits"].shape}'
    assert out["value"].shape == (batch_size,), f'Expected value (2,), got {out["value"].shape}'
    assert out["manager_value"].shape == (batch_size,), (
        f'Expected manager_value (2,), got {out["manager_value"].shape}'
    )
    assert out["action_probs"].shape == (batch_size, 7), (
        f'Expected action_probs (2, 7), got {out["action_probs"].shape}'
    )
    assert out["goal_updated"] is True


def test_fun_ablation_shape_and_schema() -> None:
    batch_size = 2
    obs = torch.randn(batch_size, 3, 7, 7, dtype=torch.float32)
    expected_keys = {
        "state_emb",
        "goal",
        "hidden_state",
        "logits",
        "action_dist",
        "action_probs",
        "value",
        "manager_value",
        "goal_updated",
    }

    recurrent_model = FuNModel(goal_update_interval=10)
    ablation_model = FuNModel(goal_update_interval=10, manager_type="ablation")
    hidden = ablation_model.init_hidden(batch_size)
    goal = ablation_model.init_goal(batch_size)

    recurrent_out = recurrent_model(
        obs,
        recurrent_model.init_hidden(batch_size),
        recurrent_model.init_goal(batch_size),
        step_count=0,
    )
    out = ablation_model(obs, hidden, goal, step_count=0)

    print("\n[FuNModel Ablation]")
    print("keys              :", sorted(out.keys()))
    print("goal shape        :", out["goal"].shape)
    print("hidden_state shape:", out["hidden_state"].shape)
    print("logits shape      :", out["logits"].shape)

    assert set(out.keys()) == expected_keys
    assert set(out.keys()) == set(recurrent_out.keys())
    assert out["logits"].shape == (batch_size, 7)
    assert out["goal"].shape == (batch_size, 16)
    assert out["hidden_state"].shape == (batch_size, 64)
    assert out["goal_updated"] is True


def test_fun_goal_update_behavior() -> None:
    batch_size = 2
    obs = torch.randn(batch_size, 3, 7, 7, dtype=torch.float32)

    model = FuNModel(goal_update_interval=10)
    hidden = model.init_hidden(batch_size)
    goal = model.init_goal(batch_size)

    out_step_0 = model(obs, hidden, goal, step_count=0)
    out_step_1 = model(obs, out_step_0["hidden_state"], out_step_0["goal"], step_count=1)

    print("\n[FuNModel Goal Update Behavior]")
    print("step 0 goal shape :", out_step_0["goal"].shape)
    print("step 1 goal shape :", out_step_1["goal"].shape)
    print("step 0 updated    :", out_step_0["goal_updated"])
    print("step 1 updated    :", out_step_1["goal_updated"])

    assert out_step_0["goal"].shape == (batch_size, 16)
    assert out_step_1["goal"].shape == (batch_size, 16)
    assert out_step_0["hidden_state"].shape == (batch_size, 64)
    assert out_step_1["hidden_state"].shape == (batch_size, 64)
    assert out_step_0["goal_updated"] is True
    assert out_step_1["goal_updated"] is False


def test_fun_ablation_goal_update_behavior() -> None:
    batch_size = 2
    obs = torch.randn(batch_size, 3, 7, 7, dtype=torch.float32)

    model = FuNModel(goal_update_interval=10, manager_type="ablation")
    hidden = model.init_hidden(batch_size)
    goal = model.init_goal(batch_size)

    out_step_0 = model(obs, hidden, goal, step_count=0)
    out_step_1 = model(obs, out_step_0["hidden_state"], out_step_0["goal"], step_count=1)

    print("\n[FuNModel Ablation Goal Update Behavior]")
    print("step 0 updated:", out_step_0["goal_updated"])
    print("step 1 updated:", out_step_1["goal_updated"])

    assert out_step_0["goal_updated"] is True
    assert out_step_1["goal_updated"] is False
    assert torch.equal(out_step_1["goal"], out_step_0["goal"])
    assert torch.equal(out_step_1["hidden_state"], out_step_0["hidden_state"])


if __name__ == "__main__":
    test_encoder_shape()
    test_manager_shape()
    test_ablation_manager_shape()
    test_ablation_manager_goal_ignores_hidden_state()
    test_worker_shape()
    test_fun_shape()
    test_fun_ablation_shape_and_schema()
    test_fun_goal_update_behavior()
    test_fun_ablation_goal_update_behavior()

    print("\nAll shape tests passed.")
