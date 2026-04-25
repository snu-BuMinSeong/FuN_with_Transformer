# Week 3 Checkpoint Evaluation Commands

Run from the `fun-minigrid` directory after a checkpoint exists.

## Seed 1

Evaluate `last.pt` and save the result JSON:

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed1.yaml \
  --checkpoint checkpoints/baseline_fun/seed_1/last.pt \
  --output logs/baseline_fun/seed_1/checkpoint_eval_last.json
```

Evaluate `best.pt` and save the result JSON:

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed1.yaml \
  --checkpoint checkpoints/baseline_fun/seed_1/best.pt \
  --output logs/baseline_fun/seed_1/checkpoint_eval_best.json
```

Override the number of evaluation episodes:

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed1.yaml \
  --checkpoint checkpoints/baseline_fun/seed_1/best.pt \
  --episodes 50 \
  --output logs/baseline_fun/seed_1/checkpoint_eval_best_50ep.json
```

## Seed 11

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed11.yaml \
  --checkpoint checkpoints/baseline_fun/seed_11/last.pt \
  --output logs/baseline_fun/seed_11/checkpoint_eval_last.json

python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed11.yaml \
  --checkpoint checkpoints/baseline_fun/seed_11/best.pt \
  --output logs/baseline_fun/seed_11/checkpoint_eval_best.json
```

## Seed 44

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed44.yaml \
  --checkpoint checkpoints/baseline_fun/seed_44/last.pt \
  --output logs/baseline_fun/seed_44/checkpoint_eval_last.json

python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed44.yaml \
  --checkpoint checkpoints/baseline_fun/seed_44/best.pt \
  --output logs/baseline_fun/seed_44/checkpoint_eval_best.json
```

## GCP

```bash
cd /home/fumin0193/fun-minigrid
source .venv/bin/activate

python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed1.yaml \
  --checkpoint checkpoints/baseline_fun/seed_1/best.pt \
  --output logs/baseline_fun/seed_1/checkpoint_eval_best.json
```

Evaluation JSON files should be saved next to the seed's logs:

- `logs/baseline_fun/seed_1/checkpoint_eval_last.json`
- `logs/baseline_fun/seed_1/checkpoint_eval_best.json`
- `logs/baseline_fun/seed_11/checkpoint_eval_last.json`
- `logs/baseline_fun/seed_11/checkpoint_eval_best.json`
- `logs/baseline_fun/seed_44/checkpoint_eval_last.json`
- `logs/baseline_fun/seed_44/checkpoint_eval_best.json`
