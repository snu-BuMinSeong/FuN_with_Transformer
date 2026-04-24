from __future__ import annotations

import argparse

import torch

from src.envs.make_env import make_env
from src.envs.preprocess import preprocess_obs
from src.models.fun import FuNModel
from src.policies.fun_policy import FuNPolicy
from src.training.rollout import run_episode
from src.utils.logger import append_episode_log
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse lightweight script arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Run a few FuNPolicy MiniGrid episodes and save CSV logs."""
    args = parse_args()
    seed = 123
    num_episodes = 3
    log_path = f"logs/week1_fun_policy_run_{args.run_id}.csv"

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(seed=seed)
    model = FuNModel(goal_update_interval=10).to(device)
    policy = FuNPolicy(
        model=model,
        preprocess_fn=preprocess_obs,
        device=device,
        action_mode="sample",
    )

    print(f"device={device}")

    try:
        for episode in range(1, num_episodes + 1):
            result = run_episode(
                env=env,
                policy=policy,
                seed=seed + episode,
            )
            result["episode"] = episode

            print(
                f"run={args.run_id} "
                f"episode={episode} "
                f"reward={result['total_reward']:.3f} "
                f"length={result['episode_length']} "
                f"success={result['success']}"
            )
            if hasattr(policy, "get_debug_info"):
                print(f"debug={policy.get_debug_info()}")
            append_episode_log(log_path, result)
    finally:
        env.close()


if __name__ == "__main__":
    main()
