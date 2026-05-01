# Week 3 Diagnosis Commands

```bash
python -m py_compile diagnose_baseline.py
```

```bash
python diagnose_baseline.py
```

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
  --config configs/train_fun_baseline_seed11.yaml \
  --checkpoint checkpoints/baseline_fun/seed_11/best.pt \
  --action-mode sample \
  --episodes 100 \
  --output logs/baseline_fun/seed_11/checkpoint_eval_best_sample.json
```

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_baseline_seed44.yaml \
  --checkpoint checkpoints/baseline_fun/seed_44/best.pt \
  --action-mode sample \
  --episodes 100 \
  --output logs/baseline_fun/seed_44/checkpoint_eval_best_sample.json
```

```bash
python train.py --config configs/train_fun_empty5x5_sanity_seed1.yaml
```

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_empty5x5_sanity_seed1.yaml \
  --checkpoint checkpoints/sanity_empty5x5/seed_1/best.pt \
  --action-mode argmax \
  --episodes 100 \
  --output logs/sanity_empty5x5/seed_1/checkpoint_eval_best_argmax.json
```

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_empty5x5_sanity_seed1.yaml \
  --checkpoint checkpoints/sanity_empty5x5/seed_1/best.pt \
  --action-mode sample \
  --episodes 100 \
  --output logs/sanity_empty5x5/seed_1/checkpoint_eval_best_sample.json
```
