# Week 5 Low-Entropy Fine-Tuning Smoke V2: AblationManager All Seeds

## 1. Purpose

AblationManager DoorKey-6x6 3000-episode checkpoints had high sample success but weak argmax success. This smoke test runs low-entropy fine-tuning for seeds 1, 11, and 44 with:

- `entropy_coef: 0.003`
- `total_episodes: 500`
- `eval_episodes: 50`
- dual sample/argmax eval
- separate `best_sample.pt` and `best_argmax.pt`

No model architecture, reward shaping, PPO, HER, or TransformerManager changes were made. Existing 3000-run paths were not used as output paths.

## 2. GCP Execution

Executed on GCP in a separate workspace:

- `/home/fumin0193/FuN_with_Transformer/argmax_ft_smoke_v2_workspace/fun-minigrid`

The original GCP project directory did not contain the `_argmax_ft_smoke_v2` output paths before this run. Existing local 3000 checkpoints were copied into the separate GCP workspace as resume sources.

Parallel execution:

- Seed 1 was launched first. The initial shell grouping command only started seed 1 because the seed 11/44 redirections were evaluated from the wrong working directory.
- After detecting this, seed 11 and seed 44 were launched concurrently as background jobs.
- Seed outputs did not mix; stdout/stderr were written to separate files.

Seed commands:

```bash
python train.py --config configs/smoke_train_fun_ablation_doorkey6x6_seed1_argmax_ft_v2.yaml
python train.py --config configs/smoke_train_fun_ablation_doorkey6x6_seed11_argmax_ft_v2.yaml
python train.py --config configs/smoke_train_fun_ablation_doorkey6x6_seed44_argmax_ft_v2.yaml
```

Seed stdout/stderr logs:

- `logs/gcp_run_logs/argmax_ft_smoke_v2_seed1.out`
- `logs/gcp_run_logs/argmax_ft_smoke_v2_seed11.out`
- `logs/gcp_run_logs/argmax_ft_smoke_v2_seed44.out`

Verification:

- `python -m py_compile train.py evaluate_checkpoint.py`: passed
- full test suite after results were copied back: 48 passed

## 3. Before Fine-Tuning

Existing 3000 checkpoint performance:

| Seed | Before Sample Success 100ep | Before Argmax Success 100ep | Before Sample Return | Before Argmax Return |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 0.450 | 0.912 | 0.437 |
| 11 | 0.990 | 0.040 | 0.880 | 0.038 |
| 44 | 1.000 | 0.180 | 0.894 | 0.173 |

Before mean argmax success: 0.223.

## 4. Fine-Tuning Settings

Common settings:

- total episodes: 500
- learning rate: 0.0001
- entropy coef: 0.003
- value loss coef: 0.5
- manager loss coef: 0.1
- grad clip norm: 1.0
- eval interval: 50
- eval episodes: 50
- eval action modes: sample, argmax
- optimizer state resumed: no

Optimizer state was intentionally not resumed because fine-tuning changed `learning_rate` and `entropy_coef`; each run loaded model weights and used a fresh Adam optimizer.

Seed-specific paths:

| Seed | Resume Checkpoint | Log Dir | Checkpoint Dir |
|---:|---|---|---|
| 1 | `checkpoints/ablation_fun_doorkey6x6_3000/seed_1/best.pt` | `logs/ablation_fun_doorkey6x6_argmax_ft_smoke_v2/seed_1` | `checkpoints/ablation_fun_doorkey6x6_argmax_ft_smoke_v2/seed_1` |
| 11 | `checkpoints/ablation_fun_doorkey6x6_3000/seed_11/best.pt` | `logs/ablation_fun_doorkey6x6_argmax_ft_smoke_v2/seed_11` | `checkpoints/ablation_fun_doorkey6x6_argmax_ft_smoke_v2/seed_11` |
| 44 | `checkpoints/ablation_fun_doorkey6x6_3000/seed_44/best.pt` | `logs/ablation_fun_doorkey6x6_argmax_ft_smoke_v2/seed_44` | `checkpoints/ablation_fun_doorkey6x6_argmax_ft_smoke_v2/seed_44` |

Output checks:

| Seed | Train Rows | Eval Rows | Final Episode | NaN/Inf in CSV | Required Checkpoints |
|---:|---:|---:|---:|---|---|
| 1 | 500 | 10 | 500 | No | present |
| 11 | 500 | 10 | 500 | No | present |
| 44 | 500 | 10 | 500 | No | present |

Mean training entropy:

| Seed | Mean Entropy |
|---:|---:|
| 1 | 1.049 |
| 11 | 1.235 |
| 44 | 1.168 |

## 5. Fine-Tuning Eval Curve

Training-time eval used 50 episodes per interval:

| Seed | Episode | Sample Success | Argmax Success | Sample Return | Argmax Return | Sample Length | Argmax Length |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 1.000 | 0.400 | 0.935 | 0.390 | 26.160 | 154.080 |
| 1 | 100 | 1.000 | 0.440 | 0.938 | 0.427 | 24.640 | 145.020 |
| 1 | 150 | 1.000 | 0.320 | 0.932 | 0.310 | 27.080 | 174.040 |
| 1 | 200 | 1.000 | 0.440 | 0.939 | 0.425 | 24.560 | 145.920 |
| 1 | 250 | 1.000 | 0.420 | 0.945 | 0.408 | 21.980 | 149.940 |
| 1 | 300 | 1.000 | 0.420 | 0.943 | 0.406 | 22.620 | 150.760 |
| 1 | 350 | 1.000 | 0.340 | 0.943 | 0.328 | 22.780 | 169.900 |
| 1 | 400 | 1.000 | 0.520 | 0.941 | 0.504 | 23.440 | 126.320 |
| 1 | 450 | 1.000 | 0.680 | 0.945 | 0.659 | 21.980 | 88.440 |
| 1 | 500 | 1.000 | 0.680 | 0.949 | 0.658 | 20.220 | 88.700 |
| 11 | 50 | 1.000 | 0.200 | 0.885 | 0.193 | 46.080 | 202.760 |
| 11 | 100 | 1.000 | 0.040 | 0.890 | 0.038 | 43.860 | 240.640 |
| 11 | 150 | 1.000 | 0.080 | 0.897 | 0.077 | 41.320 | 231.160 |
| 11 | 200 | 1.000 | 0.020 | 0.908 | 0.019 | 36.640 | 245.240 |
| 11 | 250 | 1.000 | 0.280 | 0.914 | 0.268 | 34.280 | 184.700 |
| 11 | 300 | 1.000 | 0.280 | 0.919 | 0.271 | 32.520 | 183.560 |
| 11 | 350 | 1.000 | 0.240 | 0.926 | 0.232 | 29.640 | 193.240 |
| 11 | 400 | 1.000 | 0.240 | 0.925 | 0.233 | 29.860 | 192.660 |
| 11 | 450 | 1.000 | 0.440 | 0.929 | 0.425 | 28.380 | 146.200 |
| 11 | 500 | 1.000 | 0.340 | 0.927 | 0.328 | 29.160 | 169.700 |
| 44 | 50 | 1.000 | 0.280 | 0.900 | 0.271 | 39.960 | 183.560 |
| 44 | 100 | 1.000 | 0.200 | 0.920 | 0.194 | 32.140 | 202.520 |
| 44 | 150 | 1.000 | 0.200 | 0.919 | 0.194 | 32.420 | 202.500 |
| 44 | 200 | 1.000 | 0.180 | 0.916 | 0.174 | 33.500 | 207.400 |
| 44 | 250 | 1.000 | 0.300 | 0.942 | 0.291 | 23.300 | 178.560 |
| 44 | 300 | 1.000 | 0.220 | 0.929 | 0.214 | 28.420 | 197.240 |
| 44 | 350 | 1.000 | 0.420 | 0.931 | 0.406 | 27.740 | 150.640 |
| 44 | 400 | 1.000 | 0.360 | 0.936 | 0.350 | 25.560 | 164.120 |
| 44 | 450 | 1.000 | 0.360 | 0.935 | 0.350 | 25.960 | 164.180 |
| 44 | 500 | 1.000 | 0.340 | 0.935 | 0.330 | 25.940 | 169.160 |

Training-time best argmax eval:

| Seed | Best Argmax Eval Episode | Best Argmax Eval Success |
|---:|---:|---:|
| 1 | 500 | 0.680 |
| 11 | 450 | 0.440 |
| 44 | 350 | 0.420 |

## 6. Post Fine-Tuning 100ep Evaluation

| Seed | Checkpoint | Sample Success 100ep | Argmax Success 100ep | Sample Return | Argmax Return | Sample Length | Argmax Length |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | best_sample.pt | 1.000 | 0.540 | 0.944 | 0.524 | 22.220 | 121.530 |
| 1 | best_argmax.pt | 1.000 | 0.540 | 0.944 | 0.524 | 22.220 | 121.530 |
| 1 | last.pt | 1.000 | 0.540 | 0.944 | 0.524 | 22.220 | 121.530 |
| 11 | best_sample.pt | 1.000 | 0.370 | 0.930 | 0.358 | 27.870 | 162.180 |
| 11 | best_argmax.pt | 1.000 | 0.360 | 0.931 | 0.348 | 27.550 | 164.810 |
| 11 | last.pt | 1.000 | 0.370 | 0.930 | 0.358 | 27.870 | 162.180 |
| 44 | best_sample.pt | 1.000 | 0.440 | 0.945 | 0.426 | 22.180 | 145.430 |
| 44 | best_argmax.pt | 1.000 | 0.450 | 0.930 | 0.436 | 28.020 | 142.930 |
| 44 | last.pt | 1.000 | 0.440 | 0.945 | 0.426 | 22.180 | 145.430 |

## 7. Before/After Comparison

Best after result is selected by highest 100ep argmax success among `best_sample.pt`, `best_argmax.pt`, and `last.pt`.

| Seed | Before Argmax Success | Best After Argmax Success | Delta Argmax | Before Sample Success | After Sample Success | Sample Preserved |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.450 | 0.540 | +0.090 | 1.000 | 1.000 | Yes |
| 11 | 0.040 | 0.370 | +0.330 | 0.990 | 1.000 | Yes |
| 44 | 0.180 | 0.450 | +0.270 | 1.000 | 1.000 | Yes |

Mean argmax success:

- before: 0.223
- after: 0.453
- delta: +0.230

## 8. Judgment

Success criteria:

- Seed 11 argmax improves from 0.040 to at least 0.200: passed. It reached 0.370 on 100ep evaluation.
- Mean argmax improves over the previous mean: passed. Mean argmax improved from 0.223 to 0.453.
- Sample success remains at least 0.800 for each seed: passed. All post-FT sample success values are 1.000.
- `best_argmax.pt` is clearly better than old `best.pt`: partially passed.

Details:

- Seed 1 improved from 0.450 to 0.540, and all three final checkpoints tied because the best argmax checkpoint was also the final checkpoint.
- Seed 11 improved strongly from 0.040 to 0.370. In 100ep re-evaluation, `best_sample.pt` and `last.pt` slightly beat `best_argmax.pt`, although `best_argmax.pt` was correctly selected from the 50ep interval eval at episode 450.
- Seed 44 improved from 0.180 to 0.450. `best_argmax.pt` was the best 100ep argmax checkpoint and also had the shortest argmax episode length.

There is no sign of policy collapse. Sample performance stayed at 1.000 across all seeds and training-time sample eval was also consistently 1.000.

Seed variance remains, but the direction is positive for every seed. The biggest recovery is seed 11, which was the main target.

## 9. Recommendation

Recommendation: A. expand AblationManager low-entropy fine-tuning to the full follow-up experiment.

Suggested full run:

- resume from AblationManager DoorKey-6x6 3000 checkpoints
- `entropy_coef: 0.003`
- `learning_rate: 0.0001`
- `total_episodes: 1000`
- `eval_interval: 50`
- `eval_episodes: 50`
- keep dual eval enabled
- save `best_sample.pt`, `best_argmax.pt`, and `last.pt`
- keep output paths separate from the existing 3000 runs

Rationale:

- All three seeds improved in 100ep argmax evaluation.
- Seed 11 crossed the requested success threshold.
- Mean argmax roughly doubled while sample success was preserved.
- 500 episodes already improved deterministic success; 1000 episodes may stabilize the late-stage argmax policy and reduce interval-eval noise.

Additional recommendation:

- Implement temperature evaluation before or alongside the 1000-episode full run if time permits. The policies still show a gap between sample and argmax, so temperature sweeps can identify whether near-greedy decoding is more reliable than exact argmax.
