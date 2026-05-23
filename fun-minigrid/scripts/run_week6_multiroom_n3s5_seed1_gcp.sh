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
    "$PYTHON -u train.py --config $config_path > $log_dir/run_stdout.log 2>&1"
}

start_run \
  mr_n3s5_baseline_s1 \
  configs/train_fun_baseline_multiroom_n3s5_ms1000_seed1.yaml \
  logs/multiroom_n3s5_seed1/baseline \
  checkpoints/multiroom_n3s5_seed1/baseline

start_run \
  mr_n3s5_ablation_s1 \
  configs/train_fun_ablation_multiroom_n3s5_ms1000_seed1.yaml \
  logs/multiroom_n3s5_seed1/ablation \
  checkpoints/multiroom_n3s5_seed1/ablation

tmux ls
