#!/usr/bin/env bash
set -euo pipefail

cd /home/fumin0193/FuN_with_Transformer/fun-minigrid

PYTHON_BIN="${PYTHON_BIN:-/home/fumin0193/.venv/bin/python}"
OUT_ROOT="logs/two_stage_multiroom_n4s5/final_eval_100"
mkdir -p "${OUT_ROOT}"

echo "START week6_multiroom_n4s5_seed1_final_eval $(date --iso-8601=seconds)"

eval_one() {
  local model="$1"
  local config="$2"
  local checkpoint_name="$3"
  local checkpoint_path="$4"
  local mode="$5"
  local output="${OUT_ROOT}/${model}_${checkpoint_name}_${mode}.json"

  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "SKIP missing checkpoint ${checkpoint_path}"
    return
  fi

  echo "EVAL model=${model} checkpoint=${checkpoint_name} mode=${mode}"
  "${PYTHON_BIN}" evaluate_checkpoint.py \
    --config "${config}" \
    --checkpoint "${checkpoint_path}" \
    --episodes 100 \
    --seed-offset 9000 \
    --action-mode "${mode}" \
    --output "${output}"
}

for model in baseline ablation; do
  config="configs/finetune_fun_${model}_multiroom_n4s5_seed1.yaml"
  stage2_dir="checkpoints/two_stage_multiroom_n4s5/${model}/seed_1/stage2_argmax_ft_1000"

  for checkpoint_name in best_sample best_argmax last; do
    checkpoint_path="${stage2_dir}/${checkpoint_name}.pt"
    for mode in sample argmax; do
      eval_one "${model}" "${config}" "${checkpoint_name}" "${checkpoint_path}" "${mode}"
    done
  done
done

echo "ALL_DONE week6_multiroom_n4s5_seed1_final_eval $(date --iso-8601=seconds)"
