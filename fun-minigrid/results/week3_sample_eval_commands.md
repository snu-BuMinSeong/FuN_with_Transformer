# Week 3 Sample Evaluation Commands

The official Week 3 baseline metric is checkpoint evaluation with `--action-mode sample`.
Argmax evaluation is kept as a secondary/reference metric.

## Seed 1

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed1.yaml \
  --checkpoint checkpoints/baseline_fun/seed_1/best.pt \
  --action-mode sample \
  --episodes 100 \
  --output logs/baseline_fun/seed_1/checkpoint_eval_best_sample.json
```

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed1.yaml \
  --checkpoint checkpoints/baseline_fun/seed_1/best.pt \
  --action-mode argmax \
  --episodes 100 \
  --output logs/baseline_fun/seed_1/checkpoint_eval_best_argmax.json
```

## Seed 11

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed11.yaml \
  --checkpoint checkpoints/baseline_fun/seed_11/best.pt \
  --action-mode sample \
  --episodes 100 \
  --output logs/baseline_fun/seed_11/checkpoint_eval_best_sample.json
```

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed11.yaml \
  --checkpoint checkpoints/baseline_fun/seed_11/best.pt \
  --action-mode argmax \
  --episodes 100 \
  --output logs/baseline_fun/seed_11/checkpoint_eval_best_argmax.json
```

## Seed 44

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed44.yaml \
  --checkpoint checkpoints/baseline_fun/seed_44/best.pt \
  --action-mode sample \
  --episodes 100 \
  --output logs/baseline_fun/seed_44/checkpoint_eval_best_sample.json
```

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed44.yaml \
  --checkpoint checkpoints/baseline_fun/seed_44/best.pt \
  --action-mode argmax \
  --episodes 100 \
  --output logs/baseline_fun/seed_44/checkpoint_eval_best_argmax.json
```

## Seed 1 Mixed Log Cleanup

Seed 1 currently has `train.csv` rows that do not match `summary.json`'s final episode count.
Do this cleanup only when you are ready to rerun seed 1 from scratch:

```bash
mv logs/baseline_fun/seed_1 logs/baseline_fun/seed_1_old_mixed
mv checkpoints/baseline_fun/seed_1 checkpoints/baseline_fun/seed_1_old_mixed
python train.py --config configs/train_fun_baseline_seed1.yaml
```

Then rerun both sample and argmax checkpoint evaluations for seed 1.
