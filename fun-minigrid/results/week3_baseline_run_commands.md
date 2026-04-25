# Week 3 Baseline Run Commands

Run commands from the `fun-minigrid` directory unless noted otherwise.

Important: this VM previously had disk pressure. On GCP, always run `df -h` before `pip install` or long training.

## GCP Basic Checks

```bash
df -h
nvidia-smi
python3 --version
```

## Repository Setup

Fresh clone:

```bash
cd /home/fumin0193
git clone <REPO_URL>
cd fun-minigrid
```

If the repository is already cloned:

```bash
cd /home/fumin0193/fun-minigrid
git pull
```

## Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check PyTorch and CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Smoke Test

```bash
python train.py --config configs/train_fun_baseline_seed1.yaml
```

## tmux Training

Seed 1:

```bash
tmux new -s fun_seed1
source .venv/bin/activate
python train.py --config configs/train_fun_baseline_seed1.yaml
```

Seed 11:

```bash
tmux new -s fun_seed11
source .venv/bin/activate
python train.py --config configs/train_fun_baseline_seed11.yaml
```

Seed 44:

```bash
tmux new -s fun_seed44
source .venv/bin/activate
python train.py --config configs/train_fun_baseline_seed44.yaml
```

Detach from tmux with `Ctrl+B`, then `D`.

Reattach:

```bash
tmux attach -t fun_seed1
tmux attach -t fun_seed11
tmux attach -t fun_seed44
```

## Progress Logs

```bash
tail -f logs/baseline_fun/seed_1/train.csv
tail -f logs/baseline_fun/seed_11/train.csv
tail -f logs/baseline_fun/seed_44/train.csv
```

## Scripted Runs

```bash
bash scripts/run_baseline_seeds.sh
```

For one seed:

```bash
bash scripts/run_seed1.sh
bash scripts/run_seed11.sh
bash scripts/run_seed44.sh
```

## Result Checks

```bash
ls logs/baseline_fun/seed_1
ls checkpoints/baseline_fun/seed_1
```

For all seeds:

```bash
ls logs/baseline_fun/seed_1
ls logs/baseline_fun/seed_11
ls logs/baseline_fun/seed_44
ls checkpoints/baseline_fun/seed_1
ls checkpoints/baseline_fun/seed_11
ls checkpoints/baseline_fun/seed_44
```

Expected output paths:

| Seed | Train Log | Eval Log | Summary | Checkpoints |
|---:|---|---|---|---|
| 1 | `logs/baseline_fun/seed_1/train.csv` | `logs/baseline_fun/seed_1/eval.csv` | `logs/baseline_fun/seed_1/summary.json` | `checkpoints/baseline_fun/seed_1/` |
| 11 | `logs/baseline_fun/seed_11/train.csv` | `logs/baseline_fun/seed_11/eval.csv` | `logs/baseline_fun/seed_11/summary.json` | `checkpoints/baseline_fun/seed_11/` |
| 44 | `logs/baseline_fun/seed_44/train.csv` | `logs/baseline_fun/seed_44/eval.csv` | `logs/baseline_fun/seed_44/summary.json` | `checkpoints/baseline_fun/seed_44/` |

## Checkpoint Evaluation

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed1.yaml \
  --checkpoint checkpoints/baseline_fun/seed_1/best.pt \
  --output logs/baseline_fun/seed_1/checkpoint_eval_best.json
```

Also evaluate `last.pt`:

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed1.yaml \
  --checkpoint checkpoints/baseline_fun/seed_1/last.pt \
  --output logs/baseline_fun/seed_1/checkpoint_eval_last.json
```

Seed 11:

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed11.yaml \
  --checkpoint checkpoints/baseline_fun/seed_11/best.pt \
  --output logs/baseline_fun/seed_11/checkpoint_eval_best.json
```

Seed 44:

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed44.yaml \
  --checkpoint checkpoints/baseline_fun/seed_44/best.pt \
  --output logs/baseline_fun/seed_44/checkpoint_eval_best.json
```

## Result Collection

```bash
python aggregate_baseline_results.py
python plot_baseline_results.py
```
