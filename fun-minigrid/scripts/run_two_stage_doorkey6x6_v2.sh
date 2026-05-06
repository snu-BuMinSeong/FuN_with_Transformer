#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fumin0193/FuN_with_Transformer/two_stage_doorkey6x6_workspace}"
PY="${PY:-/home/fumin0193/.venv/bin/python}"
cd "$ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_LOG_DIR="logs/gcp_run_logs/two_stage_doorkey6x6_v2"
CONFIG_DIR="configs/two_stage_doorkey6x6_v2"
mkdir -p "$RUN_LOG_DIR" "$CONFIG_DIR"

for src in configs/*stage*.yaml; do
  dst="$CONFIG_DIR/$(basename "$src")"
  sed 's#two_stage_doorkey6x6/#two_stage_doorkey6x6_v2/#g' "$src" > "$dst"
done

"$PY" -m py_compile train.py evaluate_checkpoint.py src/utils/config.py

run_batch() {
  local stage="$1"
  shift
  local -a names=("$@")
  local -a pids=()

  for name in "${names[@]}"; do
    local cfg="$CONFIG_DIR/${name}.yaml"
    local out="$RUN_LOG_DIR/${name}.out"
    echo "START $stage $name $(date -Is)" | tee -a "$RUN_LOG_DIR/orchestrator.log"
    "$PY" train.py --config "$cfg" > "$out" 2>&1 &
    pids+=("$!")

    if [ "${#pids[@]}" -ge 2 ]; then
      for pid in "${pids[@]}"; do
        wait "$pid"
      done
      pids=()
      echo "BATCH_DONE $stage $(date -Is)" | tee -a "$RUN_LOG_DIR/orchestrator.log"
    fi
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "STAGE_DONE $stage $(date -Is)" | tee -a "$RUN_LOG_DIR/orchestrator.log"
}

stage1_names=(
  train_fun_baseline_doorkey6x6_seed1_stage1_5000
  train_fun_baseline_doorkey6x6_seed11_stage1_5000
  train_fun_baseline_doorkey6x6_seed44_stage1_5000
  train_fun_ablation_doorkey6x6_seed1_stage1_5000
  train_fun_ablation_doorkey6x6_seed11_stage1_5000
  train_fun_ablation_doorkey6x6_seed44_stage1_5000
)

stage2_names=(
  train_fun_baseline_doorkey6x6_seed1_stage2_argmax_ft_1000
  train_fun_baseline_doorkey6x6_seed11_stage2_argmax_ft_1000
  train_fun_baseline_doorkey6x6_seed44_stage2_argmax_ft_1000
  train_fun_ablation_doorkey6x6_seed1_stage2_argmax_ft_1000
  train_fun_ablation_doorkey6x6_seed11_stage2_argmax_ft_1000
  train_fun_ablation_doorkey6x6_seed44_stage2_argmax_ft_1000
)

run_batch stage1 "${stage1_names[@]}"
run_batch stage2 "${stage2_names[@]}"

eval_one() {
  local model="$1"
  local seed="$2"
  local cfg="$3"
  local ckpt_name="$4"
  local mode="$5"
  local stage_dir="stage2_argmax_ft_1000"
  local ckpt="checkpoints/two_stage_doorkey6x6_v2/${model}/seed_${seed}/${stage_dir}/${ckpt_name}.pt"
  local out="logs/two_stage_doorkey6x6_v2/${model}/seed_${seed}/${stage_dir}/eval_${ckpt_name}_${mode}_100ep.json"
  echo "EVAL ${model} seed=${seed} ${ckpt_name} ${mode} $(date -Is)" | tee -a "$RUN_LOG_DIR/orchestrator.log"
  "$PY" evaluate_checkpoint.py --config "$cfg" --checkpoint "$ckpt" --output "$out" --episodes 100 --action-mode "$mode"
}

for model in baseline ablation; do
  for seed in 1 11 44; do
    cfg="$CONFIG_DIR/train_fun_${model}_doorkey6x6_seed${seed}_stage2_argmax_ft_1000.yaml"
    for ckpt_name in best_sample best_argmax last; do
      for mode in sample argmax; do
        eval_one "$model" "$seed" "$cfg" "$ckpt_name" "$mode"
      done
    done
  done
done

echo "ALL_DONE $(date -Is)" | tee -a "$RUN_LOG_DIR/orchestrator.log"
