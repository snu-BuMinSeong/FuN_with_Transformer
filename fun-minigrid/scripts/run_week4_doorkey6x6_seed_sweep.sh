#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/fumin0193/FuN_with_Transformer/fun-minigrid"
PY="/home/fumin0193/.venv/bin/python"
cd "$ROOT"

timestamp="$(date +%Y%m%d_%H%M%S)"

"$PY" -m py_compile train.py evaluate_checkpoint.py

"$PY" - <<'PY'
from pathlib import Path
from src.utils.config import load_simple_yaml

configs = [
    ("baseline", 1, "configs/train_fun_baseline_doorkey6x6_seed1.yaml", "recurrent"),
    ("baseline", 11, "configs/train_fun_baseline_doorkey6x6_seed11.yaml", "recurrent"),
    ("baseline", 44, "configs/train_fun_baseline_doorkey6x6_seed44.yaml", "recurrent"),
    ("ablation", 1, "configs/train_fun_ablation_doorkey6x6_seed1.yaml", "ablation"),
    ("ablation", 11, "configs/train_fun_ablation_doorkey6x6_seed11.yaml", "ablation"),
    ("ablation", 44, "configs/train_fun_ablation_doorkey6x6_seed44.yaml", "ablation"),
]

common = {
    "env_id": "MiniGrid-DoorKey-6x6-v0",
    "total_episodes": 1000,
    "eval_interval": 50,
    "eval_episodes": 20,
    "action_mode": "sample",
    "eval_action_mode": "sample",
}

for family, seed, path, expected_manager in configs:
    cfg = load_simple_yaml(path)
    for key, expected in common.items():
        actual = cfg.get(key)
        if actual != expected:
            raise SystemExit(f"{path}: expected {key}={expected!r}, got {actual!r}")
    if int(cfg.get("seed")) != seed:
        raise SystemExit(f"{path}: expected seed={seed}, got {cfg.get('seed')!r}")
    manager = str(cfg.get("manager_type", "recurrent"))
    if manager != expected_manager:
        raise SystemExit(f"{path}: expected manager_type={expected_manager!r}, got {manager!r}")
    expected_log = f"logs/{family}_fun_doorkey6x6/seed_{seed}"
    expected_ckpt = f"checkpoints/{family}_fun_doorkey6x6/seed_{seed}"
    if str(cfg.get("log_dir")) != expected_log:
        raise SystemExit(f"{path}: expected log_dir={expected_log!r}, got {cfg.get('log_dir')!r}")
    if str(cfg.get("checkpoint_dir")) != expected_ckpt:
        raise SystemExit(f"{path}: expected checkpoint_dir={expected_ckpt!r}, got {cfg.get('checkpoint_dir')!r}")

for smoke in [
    "logs/baseline_fun_doorkey6x6/smoke",
    "logs/ablation_fun_doorkey6x6/smoke",
    "checkpoints/baseline_fun_doorkey6x6/smoke",
    "checkpoints/ablation_fun_doorkey6x6/smoke",
]:
    print(f"smoke_path {'exists' if Path(smoke).exists() else 'missing'}: {smoke}")

print("config validation passed")
PY

for seed in 1 11 44; do
  for family in baseline ablation; do
    for base in logs checkpoints; do
      target="$base/${family}_fun_doorkey6x6/seed_${seed}"
      if [[ -e "$target" ]]; then
        backup="$base/${family}_fun_doorkey6x6/seed_${seed}_old_${timestamp}"
        mv "$target" "$backup"
        echo "moved existing $target -> $backup"
      fi
    done
  done
done

run_train() {
  local family="$1"
  local seed="$2"
  local config="configs/train_fun_${family}_doorkey6x6_seed${seed}.yaml"
  local log_dir="logs/${family}_fun_doorkey6x6/seed_${seed}"
  mkdir -p "$log_dir"
  echo "===== train ${family} seed ${seed} ====="
  "$PY" train.py --config "$config" 2>&1 | tee "$log_dir/run_stdout.log"
}

run_eval() {
  local family="$1"
  local seed="$2"
  local config="configs/train_fun_${family}_doorkey6x6_seed${seed}.yaml"
  local ckpt="checkpoints/${family}_fun_doorkey6x6/seed_${seed}/best.pt"
  local log_dir="logs/${family}_fun_doorkey6x6/seed_${seed}"
  for mode in sample argmax; do
    echo "===== eval ${family} seed ${seed} best ${mode} ====="
    "$PY" evaluate_checkpoint.py \
      --config "$config" \
      --checkpoint "$ckpt" \
      --episodes 20 \
      --action-mode "$mode" \
      --output "$log_dir/best_eval_${mode}.json" \
      2>&1 | tee "$log_dir/best_eval_${mode}.log"
  done
}

for seed in 1 11 44; do
  run_train baseline "$seed"
  run_train ablation "$seed"
done

for seed in 1 11 44; do
  run_eval baseline "$seed"
  run_eval ablation "$seed"
done

"$PY" - <<'PY'
import csv
import json
from pathlib import Path

rows = []
for family in ["baseline", "ablation"]:
    for seed in [1, 11, 44]:
        log_dir = Path(f"logs/{family}_fun_doorkey6x6/seed_{seed}")
        eval_csv = log_dir / "eval.csv"
        train_csv = log_dir / "train.csv"
        summary_json = log_dir / "summary.json"
        last_eval = {}
        if eval_csv.exists():
            with eval_csv.open(newline="", encoding="utf-8") as f:
                eval_rows = list(csv.DictReader(f))
            if eval_rows:
                last_eval = eval_rows[-1]
        train_tail = {}
        if train_csv.exists():
            with train_csv.open(newline="", encoding="utf-8") as f:
                train_rows = list(csv.DictReader(f))
            if train_rows:
                train_tail = train_rows[-1]
        summary = json.loads(summary_json.read_text(encoding="utf-8")) if summary_json.exists() else {}
        evals = {}
        for mode in ["sample", "argmax"]:
            path = log_dir / f"best_eval_{mode}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                result = data["eval_result"]
                evals[mode] = {
                    "success_rate": result["mean_success_rate"],
                    "mean_reward": result["mean_reward"],
                    "mean_episode_length": result["mean_episode_length"],
                    "checkpoint_episode": data.get("checkpoint_episode"),
                }
        rows.append({
            "family": family,
            "seed": seed,
            "best_success_rate": summary.get("best_success_rate"),
            "best_reward": summary.get("best_reward"),
            "last_train_success_ma": train_tail.get("success_moving_avg"),
            "last_train_reward_ma": train_tail.get("reward_moving_avg"),
            "last_eval_success_rate": last_eval.get("eval_success_rate"),
            "last_eval_mean_return": last_eval.get("eval_mean_return"),
            "best_eval_sample": evals.get("sample", {}),
            "best_eval_argmax": evals.get("argmax", {}),
        })

out_dir = Path("results")
out_dir.mkdir(exist_ok=True)
json_path = out_dir / "week4_doorkey6x6_seed_sweep_summary.json"
md_path = out_dir / "week4_doorkey6x6_seed_sweep_summary.md"
json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

lines = [
    "# Week 4 DoorKey-6x6 Seed Sweep Summary",
    "",
    "Scope: vanilla FuN baseline and AblationManager on MiniGrid-DoorKey-6x6-v0, seeds 1/11/44, 1000 episodes, sample training/eval action mode.",
    "",
    "| family | seed | train best_success | final eval_success | best sample_success | best argmax_success | best sample_return | best argmax_return |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    sample = row["best_eval_sample"]
    argmax = row["best_eval_argmax"]
    def fmt(value):
        if value in (None, ""):
            return ""
        return f"{float(value):.3f}"
    lines.append(
        f"| {row['family']} | {row['seed']} | "
        f"{fmt(row['best_success_rate'])} | {fmt(row['last_eval_success_rate'])} | "
        f"{fmt(sample.get('success_rate'))} | {fmt(argmax.get('success_rate'))} | "
        f"{fmt(sample.get('mean_reward'))} | {fmt(argmax.get('mean_reward'))} |"
    )
lines.extend([
    "",
    "Generated files:",
    "- logs/*_fun_doorkey6x6/seed_{1,11,44}/train.csv",
    "- logs/*_fun_doorkey6x6/seed_{1,11,44}/eval.csv",
    "- logs/*_fun_doorkey6x6/seed_{1,11,44}/best_eval_sample.json",
    "- logs/*_fun_doorkey6x6/seed_{1,11,44}/best_eval_argmax.json",
    "- checkpoints/*_fun_doorkey6x6/seed_{1,11,44}/best.pt",
    "- checkpoints/*_fun_doorkey6x6/seed_{1,11,44}/last.pt",
])
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {json_path}")
print(f"wrote {md_path}")
PY

echo "all week4 doorkey6x6 seed sweep tasks completed"
