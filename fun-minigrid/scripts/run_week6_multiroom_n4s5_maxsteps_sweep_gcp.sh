#!/usr/bin/env bash
set -euo pipefail

cd /home/fumin0193/FuN_with_Transformer/fun-minigrid

PYTHON=/home/fumin0193/.venv/bin/python

start_run() {
  local session_name="$1"
  local config_path="$2"
  local log_dir="$3"
  local checkpoint_dir="$4"

  mkdir -p "$log_dir" "$checkpoint_dir"
  tmux new-session -d -s "$session_name" \
    "$PYTHON train.py --config $config_path > $log_dir/run_stdout.log 2>&1"
}

start_run \
  mr_n4s5_base_ms250_s1 \
  configs/train_fun_baseline_multiroom_n4s5_ms250_seed1.yaml \
  logs/multiroom_n4s5_maxsteps_sweep/baseline_ms250_seed_1 \
  checkpoints/multiroom_n4s5_maxsteps_sweep/baseline_ms250_seed_1

start_run \
  mr_n4s5_base_ms500_s1 \
  configs/train_fun_baseline_multiroom_n4s5_ms500_seed1.yaml \
  logs/multiroom_n4s5_maxsteps_sweep/baseline_ms500_seed_1 \
  checkpoints/multiroom_n4s5_maxsteps_sweep/baseline_ms500_seed_1

start_run \
  mr_n4s5_base_ms750_s1 \
  configs/train_fun_baseline_multiroom_n4s5_ms750_seed1.yaml \
  logs/multiroom_n4s5_maxsteps_sweep/baseline_ms750_seed_1 \
  checkpoints/multiroom_n4s5_maxsteps_sweep/baseline_ms750_seed_1

start_run \
  mr_n4s5_ablate_ms250_s1 \
  configs/train_fun_ablation_multiroom_n4s5_ms250_seed1.yaml \
  logs/multiroom_n4s5_maxsteps_sweep/ablation_ms250_seed_1 \
  checkpoints/multiroom_n4s5_maxsteps_sweep/ablation_ms250_seed_1

start_run \
  mr_n4s5_ablate_ms500_s1 \
  configs/train_fun_ablation_multiroom_n4s5_ms500_seed1.yaml \
  logs/multiroom_n4s5_maxsteps_sweep/ablation_ms500_seed_1 \
  checkpoints/multiroom_n4s5_maxsteps_sweep/ablation_ms500_seed_1

start_run \
  mr_n4s5_ablate_ms750_s1 \
  configs/train_fun_ablation_multiroom_n4s5_ms750_seed1.yaml \
  logs/multiroom_n4s5_maxsteps_sweep/ablation_ms750_seed_1 \
  checkpoints/multiroom_n4s5_maxsteps_sweep/ablation_ms750_seed_1

tmux ls
