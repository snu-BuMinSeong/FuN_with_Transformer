#!/usr/bin/env bash
set -euo pipefail

cd /home/fumin0193/FuN_with_Transformer/fun-minigrid

PYTHON_BIN="${PYTHON_BIN:-/home/fumin0193/.venv/bin/python}"
RUN_ROOT="logs/gcp_run_logs/week6_multiroom_n4s5_seed1_pilot"
mkdir -p "${RUN_ROOT}"

echo "START week6_multiroom_n4s5_seed1_pilot $(date --iso-8601=seconds)"
echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" --version
nvidia-smi || true

run_train() {
  local name="$1"
  local config="$2"
  local log_path="${RUN_ROOT}/${name}.log"

  echo "RUN ${name} config=${config} start=$(date --iso-8601=seconds)"
  "${PYTHON_BIN}" train.py --config "${config}" 2>&1 | tee "${log_path}"
  echo "DONE ${name} end=$(date --iso-8601=seconds)"
}

has_reward_signal() {
  local summary_path="$1"
  "${PYTHON_BIN}" - "${summary_path}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

episodes_with_reward = int(data.get("episodes_with_reward_signal") or 0)
best_reward = float(data.get("best_reward") or 0.0)
best_success = float(data.get("best_success") or 0.0)

print(
    f"summary={path} "
    f"episodes_with_reward_signal={episodes_with_reward} "
    f"best_reward={best_reward:.6f} "
    f"best_success={best_success:.6f}"
)
raise SystemExit(0 if episodes_with_reward > 0 or best_reward > 0.0 or best_success > 0.0 else 1)
PY
}

run_train "baseline_stage1_5000" "configs/train_fun_baseline_multiroom_n4s5_seed1.yaml"
run_train "ablation_stage1_5000" "configs/train_fun_ablation_multiroom_n4s5_seed1.yaml"

baseline_summary="logs/two_stage_multiroom_n4s5/baseline/seed_1/stage1_5000/summary.json"
ablation_summary="logs/two_stage_multiroom_n4s5/ablation/seed_1/stage1_5000/summary.json"

if has_reward_signal "${baseline_summary}"; then
  run_train "baseline_stage2_argmax_ft_1000" "configs/finetune_fun_baseline_multiroom_n4s5_seed1.yaml"
else
  echo "SKIP baseline_stage2_argmax_ft_1000 no baseline reward signal"
fi

if has_reward_signal "${ablation_summary}"; then
  run_train "ablation_stage2_argmax_ft_1000" "configs/finetune_fun_ablation_multiroom_n4s5_seed1.yaml"
else
  echo "SKIP ablation_stage2_argmax_ft_1000 no ablation reward signal"
fi

if [[ -d "checkpoints/two_stage_multiroom_n4s5/baseline/seed_1/stage2_argmax_ft_1000" || -d "checkpoints/two_stage_multiroom_n4s5/ablation/seed_1/stage2_argmax_ft_1000" ]]; then
  echo "RUN final_eval_100 start=$(date --iso-8601=seconds)"
  scripts/evaluate_week6_multiroom_n4s5_seed1_gcp.sh 2>&1 | tee "${RUN_ROOT}/final_eval_100.log"
  echo "DONE final_eval_100 end=$(date --iso-8601=seconds)"
else
  echo "SKIP final_eval_100 no Stage 2 checkpoints"
fi

echo "ALL_DONE week6_multiroom_n4s5_seed1_pilot $(date --iso-8601=seconds)"
