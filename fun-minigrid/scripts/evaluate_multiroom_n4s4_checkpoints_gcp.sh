#!/usr/bin/env bash
set -euo pipefail

cd /home/fumin0193/FuN_with_Transformer/fun-minigrid

PYTHON_BIN="${PYTHON_BIN:-/home/fumin0193/.venv/bin/python}"
OUT_ROOT="logs/multiroom_n4s4_seed1/checkpoint_eval_100"
mkdir -p "${OUT_ROOT}"

export CUDA_VISIBLE_DEVICES=""

echo "START multiroom_n4s4_checkpoint_eval $(date --iso-8601=seconds)"
echo "python=${PYTHON_BIN}"
echo "out_root=${OUT_ROOT}"
echo "episodes=100 seed_offset=9000 max_parallel=${MAX_PARALLEL:-4}"

eval_one() {
  local model="$1"
  local config="$2"
  local checkpoint_name="$3"
  local checkpoint_path="$4"
  local mode="$5"
  local output="${OUT_ROOT}/${model}_${checkpoint_name}_${mode}.json"
  local stdout_log="${OUT_ROOT}/${model}_${checkpoint_name}_${mode}.log"

  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "SKIP missing checkpoint ${checkpoint_path}"
    return 1
  fi

  echo "EVAL model=${model} checkpoint=${checkpoint_name} mode=${mode} start=$(date --iso-8601=seconds)"
  "${PYTHON_BIN}" evaluate_checkpoint.py \
    --config "${config}" \
    --checkpoint "${checkpoint_path}" \
    --episodes 100 \
    --seed-offset 9000 \
    --action-mode "${mode}" \
    --output "${output}" \
    > "${stdout_log}" 2>&1
  echo "DONE model=${model} checkpoint=${checkpoint_name} mode=${mode} end=$(date --iso-8601=seconds)"
}

wait_for_slot() {
  local max_parallel="${MAX_PARALLEL:-4}"
  while [[ "$(jobs -pr | wc -l)" -ge "${max_parallel}" ]]; do
    sleep 2
  done
}

for model in baseline ablation; do
  config="configs/train_fun_${model}_multiroom_n4s4_ms1000_seed1.yaml"
  checkpoint_dir="checkpoints/multiroom_n4s4_seed1/${model}"

  for checkpoint_name in best best_argmax best_sample episode_5000 last; do
    checkpoint_path="${checkpoint_dir}/${checkpoint_name}.pt"
    for mode in sample argmax; do
      wait_for_slot
      eval_one "${model}" "${config}" "${checkpoint_name}" "${checkpoint_path}" "${mode}" &
    done
  done
done

wait

echo "ALL_DONE multiroom_n4s4_checkpoint_eval $(date --iso-8601=seconds)"
