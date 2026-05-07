#!/usr/bin/env bash
set -euo pipefail

cd /home/fumin0193/FuN_with_Transformer/fun-minigrid

SESSIONS=(
  mr_n4s5_base_ms250_s1
  mr_n4s5_base_ms500_s1
  mr_n4s5_base_ms750_s1
  mr_n4s5_ablate_ms250_s1
  mr_n4s5_ablate_ms500_s1
  mr_n4s5_ablate_ms750_s1
)

while true; do
  running=0
  for session in "${SESSIONS[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      running=$((running + 1))
    fi
  done

  date
  echo "running_sessions=$running"
  for train_log in logs/multiroom_n4s5_maxsteps_sweep/*_seed_1/train.csv; do
    [ -f "$train_log" ] || continue
    rows=$(tail -n +2 "$train_log" | wc -l)
    echo "$train_log rows=$rows"
  done

  if [ "$running" -eq 0 ]; then
    break
  fi
  sleep 300
done

/home/fumin0193/.venv/bin/python scripts/summarize_week6_multiroom_n4s5_maxsteps_sweep.py
echo "summary_written=results/week6_multiroom_n4s5_maxsteps_sweep_summary.md"
