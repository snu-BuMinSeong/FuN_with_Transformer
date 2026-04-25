#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python train.py --config configs/train_fun_baseline_seed1.yaml
python train.py --config configs/train_fun_baseline_seed11.yaml
python train.py --config configs/train_fun_baseline_seed44.yaml
