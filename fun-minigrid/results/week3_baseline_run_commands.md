# Week 3 Baseline Run Commands

Run from the `fun-minigrid` directory.

## Individual Seeds

```bash
python train.py --config configs/train_fun_baseline_seed1.yaml
python train.py --config configs/train_fun_baseline_seed11.yaml
python train.py --config configs/train_fun_baseline_seed44.yaml
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

## tmux Example

```bash
tmux new -s fun_seed1
source .venv/bin/activate
python train.py --config configs/train_fun_baseline_seed1.yaml
```

Detach from tmux with `Ctrl+B`, then `D`.

Reattach later:

```bash
tmux attach -t fun_seed1
```

## Output Paths

| Seed | Train Log | Eval Log | Summary | Checkpoints |
|---:|---|---|---|---|
| 1 | `logs/baseline_fun/seed_1/train.csv` | `logs/baseline_fun/seed_1/eval.csv` | `logs/baseline_fun/seed_1/summary.json` | `checkpoints/baseline_fun/seed_1/` |
| 11 | `logs/baseline_fun/seed_11/train.csv` | `logs/baseline_fun/seed_11/eval.csv` | `logs/baseline_fun/seed_11/summary.json` | `checkpoints/baseline_fun/seed_11/` |
| 44 | `logs/baseline_fun/seed_44/train.csv` | `logs/baseline_fun/seed_44/eval.csv` | `logs/baseline_fun/seed_44/summary.json` | `checkpoints/baseline_fun/seed_44/` |
