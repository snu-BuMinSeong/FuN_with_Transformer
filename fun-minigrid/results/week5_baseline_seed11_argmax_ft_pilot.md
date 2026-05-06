# Week 5 Baseline Seed 11 Argmax Fine-Tuning Pilot

## 1. Purpose

AblationManager low-entropy FT1000 improved DoorKey-6x6 deterministic argmax performance while preserving sample success. This pilot checks whether the same fine-tuning recipe also helps Vanilla FuN Baseline seed 11.

Seed 11 was selected because it was the strongest Baseline 3000 run under sample evaluation. Baseline seed 1 and seed 44 were not fine-tuned in this step.

No model architecture, AblationManager code, reward shaping, PPO, HER, or TransformerManager changes were made. Existing 3000 and AblationManager FT1000 paths were not used as output paths.

## 2. GCP Execution

Executed on GCP in a separate workspace:

- `/home/fumin0193/FuN_with_Transformer/baseline_argmax_ft_pilot_workspace/fun-minigrid`

Command:

```bash
python train.py --config configs/train_fun_baseline_doorkey6x6_seed11_argmax_ft_pilot.yaml
```

stdout/stderr log:

- `logs/gcp_run_logs/baseline_argmax_ft_pilot_seed11.out`

Pre-run verification:

- `python -m py_compile train.py evaluate_checkpoint.py`: passed locally and on GCP.
- `python -m pytest -q`: 48 passed locally before the GCP run.

Post-copy verification:

- `train.csv` rows: 1000
- `eval.csv` rows: 20
- `summary.json` `final_episode`: 1000
- no `NaN` or `inf` strings were found in local `train.csv` or `eval.csv`
- `best_sample.pt`, `best_argmax.pt`, `last.pt`, and `episode_1000.pt` are present

## 3. Baseline 3000 Pre-State

| Model | Seed | 3000 Sample Success 100ep | 3000 Argmax Success 100ep | 3000 Sample Return | 3000 Argmax Return |
|---|---:|---:|---:|---:|---:|
| Vanilla FuN | 11 | 1.000 | 0.340 | 0.907 | 0.329 |

## 4. AblationManager FT1000 Reference

| Model | Seed | FT1000 Sample Success 100ep | FT1000 Argmax Success 100ep | FT1000 Sample Return | FT1000 Argmax Return |
|---|---:|---:|---:|---:|---:|
| AblationManager | 11 | 1.000 | 0.530 | 0.945 | 0.512 |

## 5. Fine-Tuning Settings

Config:

- `configs/train_fun_baseline_doorkey6x6_seed11_argmax_ft_pilot.yaml`

Settings:

- resume checkpoint: `checkpoints/baseline_fun_doorkey6x6_3000/seed_11/best.pt`
- resume checkpoint episode: 3000
- manager type: `recurrent`
- total episodes: 1000
- learning rate: 0.0001
- entropy coef: 0.003
- value loss coef: 0.5
- manager loss coef: 0.1
- grad clip norm: 1.0
- eval interval: 50
- eval episodes: 50
- eval action modes: sample, argmax
- log dir: `logs/baseline_fun_doorkey6x6_baseline_argmax_ft_pilot/seed_11`
- checkpoint dir: `checkpoints/baseline_fun_doorkey6x6_baseline_argmax_ft_pilot/seed_11`
- optimizer state resumed: no

`manager_type: recurrent` is the code-compatible Baseline setting. The requested `manager_type: baseline` is not a valid `FuNModel` manager type, while omitting the field also defaults to recurrent.

Optimizer state was intentionally not resumed because the fine-tuning learning rate and entropy coefficient differ from the original 3000-episode run. The run loaded model weights and used a fresh Adam optimizer.

Mean training entropy during pilot: 0.931.

## 6. Fine-Tuning Eval Curve

Training-time eval used 50 episodes per interval:

| Episode | Sample Success | Argmax Success | Sample Return | Argmax Return | Sample Length | Argmax Length |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1.000 | 0.440 | 0.912 | 0.423 | 35.000 | 146.880 |
| 100 | 1.000 | 0.360 | 0.918 | 0.345 | 32.940 | 166.040 |
| 150 | 0.980 | 0.020 | 0.895 | 0.019 | 39.000 | 245.380 |
| 200 | 1.000 | 0.260 | 0.925 | 0.245 | 29.820 | 191.000 |
| 250 | 1.000 | 0.380 | 0.936 | 0.366 | 25.480 | 160.540 |
| 300 | 1.000 | 0.560 | 0.942 | 0.537 | 23.300 | 119.100 |
| 350 | 1.000 | 0.340 | 0.936 | 0.323 | 25.760 | 171.880 |
| 400 | 1.000 | 0.600 | 0.939 | 0.579 | 24.420 | 108.340 |
| 450 | 1.000 | 0.640 | 0.939 | 0.619 | 24.280 | 98.260 |
| 500 | 1.000 | 0.540 | 0.942 | 0.519 | 23.180 | 123.500 |
| 550 | 1.000 | 0.400 | 0.948 | 0.385 | 20.940 | 155.840 |
| 600 | 1.000 | 0.640 | 0.947 | 0.608 | 21.320 | 102.860 |
| 650 | 1.000 | 0.620 | 0.940 | 0.599 | 24.100 | 103.580 |
| 700 | 1.000 | 0.620 | 0.945 | 0.597 | 21.860 | 104.140 |
| 750 | 1.000 | 0.700 | 0.943 | 0.675 | 22.740 | 85.080 |
| 800 | 1.000 | 0.380 | 0.950 | 0.366 | 20.020 | 160.480 |
| 850 | 1.000 | 0.580 | 0.951 | 0.558 | 19.640 | 113.760 |
| 900 | 1.000 | 0.560 | 0.944 | 0.540 | 22.360 | 118.160 |
| 950 | 1.000 | 0.600 | 0.950 | 0.580 | 19.900 | 108.120 |
| 1000 | 0.980 | 0.420 | 0.934 | 0.406 | 23.400 | 150.640 |

Training-time best checkpoints:

- `best_sample.pt`: episode 950, sample eval success 1.000
- `best_argmax.pt`: episode 750, argmax eval success 0.700

## 7. Post Fine-Tuning 100ep Evaluation

| Checkpoint | Sample Success 100ep | Argmax Success 100ep | Sample Return | Argmax Return | Sample Length | Argmax Length |
|---|---:|---:|---:|---:|---:|---:|
| best_sample.pt | 1.000 | 0.560 | 0.947 | 0.541 | 21.140 | 117.780 |
| best_argmax.pt | 1.000 | 0.580 | 0.947 | 0.559 | 21.090 | 113.260 |
| last.pt | 1.000 | 0.570 | 0.949 | 0.550 | 20.250 | 115.340 |
| episode_1000.pt | 1.000 | 0.570 | 0.949 | 0.550 | 20.250 | 115.340 |

## 8. Before/After Comparison

Best result is selected by highest 100ep argmax success.

| Model | Seed | Before Argmax | After Best Argmax | Delta Argmax | Before Sample | After Sample | Sample Preserved |
|---|---:|---:|---:|---:|---:|---:|---|
| Vanilla FuN | 11 | 0.340 | 0.580 | +0.240 | 1.000 | 1.000 | Yes |

Argmax episode length improved from 169.440 before fine-tuning to 113.260 for `best_argmax.pt`.

## 9. AblationManager Comparison

| Model | Seed | Fine-tuning Type | Sample Success | Argmax Success | Argmax Length |
|---|---:|---|---:|---:|---:|
| Vanilla FuN | 11 | baseline FT pilot | 1.000 | 0.580 | 113.260 |
| AblationManager | 11 | ablation FT1000 | 1.000 | 0.530 | 124.800 |

For seed 11 only, Vanilla FuN after fine-tuning slightly outperformed AblationManager FT1000 under 100ep argmax evaluation.

## 10. Judgment

Success criteria:

- Baseline seed 11 argmax improves over 0.340: passed. It reached 0.580.
- Argmax success reaches at least 0.500: passed.
- Sample success remains at least 0.900: passed. It stayed at 1.000.
- Argmax episode length decreases below 169.440: passed. It reached 113.260.

Interpretation:

The pilot is a strong success. Low-entropy fine-tuning is effective for Vanilla FuN when the starting checkpoint already has high sample success. This does not imply seed 1 and seed 44 should immediately be fine-tuned, because their baseline sample performance was weak and may require additional task learning before deterministic fine-tuning.

## 11. Next Step Recommendation

Recommendation: A with the seed caveat.

- Baseline seed 11 improved strongly and sample success was preserved.
- Report that Baseline can also benefit from low-entropy fine-tuning on a seed where the stochastic policy has already solved the task.
- Do not run Baseline seed 1/44 fine-tuning yet. Their sample success was not reliably solved, so they need separate judgment about additional training or longer 5000-episode runs before deterministic fine-tuning.
- AblationManager remains the stronger overall result because it solved sample evaluation across all seeds before fine-tuning and improved argmax consistently across all seeds.

## 12. Report Interpretation Draft

1. Vanilla FuN seed 11 shows that low-entropy fine-tuning is not specific to AblationManager; it can also improve deterministic execution when the baseline policy has already learned the task under sample evaluation.

2. Baseline seed 11 argmax success increased from 0.340 to 0.580 while sample success remained at 1.000, indicating that the fine-tuning primarily improved policy determinism rather than task acquisition.

3. The Baseline seed 11 pilot slightly exceeded the AblationManager seed 11 FT1000 argmax result, but this comparison is seed-limited and does not overturn the cross-seed AblationManager advantage.

4. Because Baseline seed 1 and seed 44 had weak sample performance before fine-tuning, applying the same deterministic fine-tuning to those seeds is premature.

5. Overall, the results support a two-stage interpretation: first learn a successful stochastic DoorKey policy, then apply low-entropy fine-tuning to improve argmax deployment behavior.
