#!/usr/bin/env bash
set -euo pipefail

cd /home/fumin0193/FuN_with_Transformer/fun-minigrid

SESSION=multiroom_n4s5_ablation_ms1000_seed1
PYTHON=/home/fumin0193/.venv/bin/python
CONFIG=configs/train_fun_ablation_multiroom_n4s5_ms1000_seed1.yaml
LOG_DIR=logs/multiroom_n4s5_maxsteps_sweep/ablation_ms1000_seed_1
CHECKPOINT_DIR=checkpoints/multiroom_n4s5_maxsteps_sweep/ablation_ms1000_seed_1

while tmux has-session -t "$SESSION" 2>/dev/null; do
  date
  if [ -f "$LOG_DIR/train.csv" ]; then
    tail -n +2 "$LOG_DIR/train.csv" | wc -l
  else
    echo 0
  fi
  sleep 300
done

"$PYTHON" evaluate_checkpoint.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT_DIR/best.pt" \
  --action-mode sample \
  --output "$LOG_DIR/checkpoint_eval_best_sample.json"

"$PYTHON" evaluate_checkpoint.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT_DIR/best.pt" \
  --action-mode argmax \
  --output "$LOG_DIR/checkpoint_eval_best_argmax.json"

"$PYTHON" scripts/summarize_week6_multiroom_n4s5_maxsteps_sweep.py

echo "postprocess_done"
date
