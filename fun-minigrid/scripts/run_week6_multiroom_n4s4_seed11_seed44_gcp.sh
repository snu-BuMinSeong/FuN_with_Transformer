#!/usr/bin/env bash
set -euo pipefail

cd /home/fumin0193/FuN_with_Transformer/fun-minigrid

PYTHON="${PYTHON:-/home/fumin0193/.venv/bin/python}"

start_run() {
  local session_name="$1"
  local config_path="$2"
  local log_dir="$3"
  local checkpoint_dir="$4"

  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "SKIP existing tmux session ${session_name}"
    return 0
  fi

  mkdir -p "$log_dir" "$checkpoint_dir"
  tmux new-session -d -s "$session_name" \
    "$PYTHON -u train.py --config $config_path > $log_dir/run_stdout.log 2>&1"
  echo "STARTED ${session_name} config=${config_path}"
}

start_run \
  mr_n4s4_baseline_s11 \
  configs/train_fun_baseline_multiroom_n4s4_ms1000_seed11.yaml \
  logs/multiroom_n4s4_seed11/baseline \
  checkpoints/multiroom_n4s4_seed11/baseline

start_run \
  mr_n4s4_baseline_s44 \
  configs/train_fun_baseline_multiroom_n4s4_ms1000_seed44.yaml \
  logs/multiroom_n4s4_seed44/baseline \
  checkpoints/multiroom_n4s4_seed44/baseline

start_run \
  mr_n4s4_ablation_s11 \
  configs/train_fun_ablation_multiroom_n4s4_ms1000_seed11.yaml \
  logs/multiroom_n4s4_seed11/ablation \
  checkpoints/multiroom_n4s4_seed11/ablation

start_run \
  mr_n4s4_ablation_s44 \
  configs/train_fun_ablation_multiroom_n4s4_ms1000_seed44.yaml \
  logs/multiroom_n4s4_seed44/ablation \
  checkpoints/multiroom_n4s4_seed44/ablation

tmux ls
