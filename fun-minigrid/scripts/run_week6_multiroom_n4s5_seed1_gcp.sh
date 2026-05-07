#!/usr/bin/env bash
set -euo pipefail

cd /home/fumin0193/FuN_with_Transformer/fun-minigrid

PYTHON_BIN="${PYTHON_BIN:-/home/fumin0193/.venv/bin/python}"
RUN_ROOT="logs/gcp_run_logs/week6_multiroom_n4s5_seed1"
mkdir -p "${RUN_ROOT}"

echo "START week6_multiroom_n4s5_seed1 $(date --iso-8601=seconds)"
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

run_train "baseline_stage1_5000" "configs/train_fun_baseline_multiroom_n4s5_seed1.yaml"
run_train "ablation_stage1_5000" "configs/train_fun_ablation_multiroom_n4s5_seed1.yaml"

if [[ -f "checkpoints/two_stage_multiroom_n4s5/baseline/seed_1/stage1_5000/best_sample.pt" ]]; then
  run_train "baseline_stage2_argmax_ft_1000" "configs/finetune_fun_baseline_multiroom_n4s5_seed1.yaml"
else
  echo "SKIP baseline_stage2_argmax_ft_1000 missing baseline best_sample.pt"
fi

if [[ -f "checkpoints/two_stage_multiroom_n4s5/ablation/seed_1/stage1_5000/best_sample.pt" ]]; then
  run_train "ablation_stage2_argmax_ft_1000" "configs/finetune_fun_ablation_multiroom_n4s5_seed1.yaml"
else
  echo "SKIP ablation_stage2_argmax_ft_1000 missing ablation best_sample.pt"
fi

echo "ALL_DONE week6_multiroom_n4s5_seed1 $(date --iso-8601=seconds)"
