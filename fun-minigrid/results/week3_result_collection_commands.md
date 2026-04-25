# Week 3 Result Collection Commands

Run from the `fun-minigrid` directory after seed training has produced `eval.csv`, `summary.json`, and checkpoints.

## Aggregate Seed Results

```bash
python aggregate_baseline_results.py
```

Outputs:

- `results/week3_baseline_results.csv`
- `results/week3_baseline_results.md`
- `results/week3_baseline_summary.md`

## Plot Baseline Curves

```bash
python plot_baseline_results.py
```

Outputs:

- `figures/baseline_fun/eval_success_rate.png`
- `figures/baseline_fun/eval_mean_return.png`
- `figures/baseline_fun/eval_episode_length.png`

If matplotlib is missing:

```bash
pip install matplotlib
```

## Checkpoint Evaluation

Seed 1:

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed1.yaml \
  --checkpoint checkpoints/baseline_fun/seed_1/best.pt \
  --output logs/baseline_fun/seed_1/checkpoint_eval_best.json
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
