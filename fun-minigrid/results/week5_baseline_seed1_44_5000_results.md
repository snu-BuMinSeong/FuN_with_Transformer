# Week 5 Baseline Seed 1/44 Additional Training to 5000

## 1. Purpose

Baseline seed 1 and seed 44 had low sample success at the 3000-episode checkpoint. Because low-entropy fine-tuning is intended to improve determinism after the policy has learned the task, this step first extends Baseline seed 1/44 training before considering low-entropy fine-tuning.

This run resumes from the existing 3000 checkpoints and trains 2000 additional episodes, for a cumulative budget of 5000 episodes.

No model architecture, AblationManager code, reward shaping, PPO, HER, or TransformerManager changes were made. Existing 3000 paths and Baseline seed 11 fine-tuning paths were not used as output paths.

## 2. GCP Execution

Executed on GCP in a separate workspace:

- `/home/fumin0193/FuN_with_Transformer/baseline_5000_workspace/fun-minigrid`

Parallel execution:

- seed 1 and seed 44 were launched concurrently with background `nohup` jobs.
- the shell waited for both PIDs to finish.
- stdout/stderr logs were saved separately.

Commands:

```bash
python train.py --config configs/train_fun_baseline_doorkey6x6_seed1_5000.yaml
python train.py --config configs/train_fun_baseline_doorkey6x6_seed44_5000.yaml
```

stdout/stderr logs:

- `logs/gcp_run_logs/baseline_5000_seed1.out`
- `logs/gcp_run_logs/baseline_5000_seed44.out`

Verification:

- `python -m py_compile train.py evaluate_checkpoint.py`: passed locally and on GCP.
- `python -m pytest -q`: 48 passed locally before the GCP run.

## 3. 3000 Pre-State

| Seed | 3000 Sample Success 100ep | 3000 Argmax Success 100ep | 3000 Sample Return | 3000 Argmax Return |
|---:|---:|---:|---:|---:|
| 1 | 0.080 | 0.000 | 0.048 | 0.000 |
| 44 | 0.190 | 0.000 | 0.123 | 0.000 |

## 4. Additional Training Settings

Common settings:

- resume checkpoint: existing Baseline DoorKey-6x6 3000 `best.pt`
- additional episodes: 2000
- cumulative episodes: 5000
- learning rate: 0.0003
- entropy coef: 0.01
- value loss coef: 0.5
- manager loss coef: 0.1
- grad clip norm: 1.0
- eval interval: 50
- eval episodes: 50
- eval action modes: sample, argmax
- optimizer state resumed: no

The training script counts the resumed run from episode 1 to 2000. The source checkpoint episode plus additional episodes gives the cumulative episode count used below. For seed 1, the source `best.pt` is checkpoint episode 2500, but this experiment is interpreted against the 3000-run stage and labeled as the 5000 cumulative follow-up requested here.

Seed-specific paths:

| Seed | Resume Checkpoint | Log Dir | Checkpoint Dir |
|---:|---|---|---|
| 1 | `checkpoints/baseline_fun_doorkey6x6_3000/seed_1/best.pt` | `logs/baseline_fun_doorkey6x6_5000/seed_1` | `checkpoints/baseline_fun_doorkey6x6_5000/seed_1` |
| 44 | `checkpoints/baseline_fun_doorkey6x6_3000/seed_44/best.pt` | `logs/baseline_fun_doorkey6x6_5000/seed_44` | `checkpoints/baseline_fun_doorkey6x6_5000/seed_44` |

Output checks:

| Seed | Train Rows | Eval Rows | Run Final Episode | Cumulative Episode | NaN/Inf in CSV | Required Checkpoints |
|---:|---:|---:|---:|---:|---|---|
| 1 | 2000 | 40 | 2000 | 5000 | No | present |
| 44 | 2000 | 40 | 2000 | 5000 | No | present |

Both `episode_2000.pt` and an `episode_5000.pt` copy were created in the new `_5000` checkpoint directory for each seed. The checkpoint payload still reports run episode 2000.

## 5. Additional Training Eval Curve

Training-time eval used 50 episodes per interval:

| Seed | Run Episode | Cumulative Episode | Sample Success | Argmax Success | Sample Return | Argmax Return | Sample Length | Argmax Length |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 3050 | 0.080 | 0.000 | 0.051 | 0.000 | 241.540 | 250.000 |
| 1 | 100 | 3100 | 0.080 | 0.000 | 0.045 | 0.000 | 244.200 | 250.000 |
| 1 | 150 | 3150 | 0.040 | 0.000 | 0.024 | 0.000 | 246.540 | 250.000 |
| 1 | 200 | 3200 | 0.040 | 0.000 | 0.018 | 0.000 | 248.860 | 250.000 |
| 1 | 250 | 3250 | 0.100 | 0.000 | 0.050 | 0.000 | 245.140 | 250.000 |
| 1 | 300 | 3300 | 0.180 | 0.000 | 0.117 | 0.000 | 230.100 | 250.000 |
| 1 | 350 | 3350 | 0.100 | 0.000 | 0.054 | 0.000 | 243.220 | 250.000 |
| 1 | 400 | 3400 | 0.160 | 0.000 | 0.095 | 0.000 | 235.960 | 250.000 |
| 1 | 450 | 3450 | 0.140 | 0.000 | 0.087 | 0.000 | 236.320 | 250.000 |
| 1 | 500 | 3500 | 0.160 | 0.000 | 0.103 | 0.000 | 232.920 | 250.000 |
| 1 | 550 | 3550 | 0.100 | 0.000 | 0.061 | 0.000 | 240.620 | 250.000 |
| 1 | 600 | 3600 | 0.080 | 0.000 | 0.054 | 0.000 | 240.520 | 250.000 |
| 1 | 650 | 3650 | 0.160 | 0.000 | 0.101 | 0.000 | 233.800 | 250.000 |
| 1 | 700 | 3700 | 0.160 | 0.000 | 0.113 | 0.000 | 228.720 | 250.000 |
| 1 | 750 | 3750 | 0.120 | 0.000 | 0.063 | 0.000 | 242.920 | 250.000 |
| 1 | 800 | 3800 | 0.220 | 0.000 | 0.132 | 0.000 | 230.220 | 250.000 |
| 1 | 850 | 3850 | 0.160 | 0.000 | 0.098 | 0.000 | 235.000 | 250.000 |
| 1 | 900 | 3900 | 0.220 | 0.000 | 0.138 | 0.000 | 227.860 | 250.000 |
| 1 | 950 | 3950 | 0.280 | 0.000 | 0.188 | 0.000 | 216.740 | 250.000 |
| 1 | 1000 | 4000 | 0.360 | 0.000 | 0.216 | 0.000 | 217.720 | 250.000 |
| 1 | 1050 | 4050 | 0.380 | 0.000 | 0.244 | 0.000 | 209.340 | 250.000 |
| 1 | 1100 | 4100 | 0.440 | 0.000 | 0.307 | 0.000 | 193.260 | 250.000 |
| 1 | 1150 | 4150 | 0.740 | 0.000 | 0.503 | 0.000 | 159.800 | 250.000 |
| 1 | 1200 | 4200 | 0.600 | 0.000 | 0.432 | 0.000 | 167.160 | 250.000 |
| 1 | 1250 | 4250 | 0.860 | 0.040 | 0.621 | 0.036 | 130.580 | 241.100 |
| 1 | 1300 | 4300 | 0.980 | 0.000 | 0.757 | 0.000 | 94.060 | 250.000 |
| 1 | 1350 | 4350 | 0.940 | 0.000 | 0.729 | 0.000 | 99.460 | 250.000 |
| 1 | 1400 | 4400 | 0.980 | 0.220 | 0.809 | 0.210 | 73.260 | 197.260 |
| 1 | 1450 | 4450 | 1.000 | 0.080 | 0.835 | 0.077 | 66.000 | 230.820 |
| 1 | 1500 | 4500 | 1.000 | 0.180 | 0.861 | 0.174 | 55.680 | 207.940 |
| 1 | 1550 | 4550 | 1.000 | 0.260 | 0.900 | 0.252 | 40.100 | 188.940 |
| 1 | 1600 | 4600 | 1.000 | 0.320 | 0.916 | 0.307 | 33.560 | 176.900 |
| 1 | 1650 | 4650 | 1.000 | 0.220 | 0.893 | 0.213 | 42.860 | 196.820 |
| 1 | 1700 | 4700 | 1.000 | 0.480 | 0.923 | 0.464 | 30.620 | 133.900 |
| 1 | 1750 | 4750 | 1.000 | 0.620 | 0.922 | 0.601 | 31.260 | 102.860 |
| 1 | 1800 | 4800 | 1.000 | 0.580 | 0.921 | 0.560 | 31.480 | 112.100 |
| 1 | 1850 | 4850 | 1.000 | 0.420 | 0.924 | 0.407 | 30.600 | 150.620 |
| 1 | 1900 | 4900 | 1.000 | 0.600 | 0.915 | 0.577 | 34.180 | 108.560 |
| 1 | 1950 | 4950 | 1.000 | 0.620 | 0.907 | 0.600 | 37.260 | 103.060 |
| 1 | 2000 | 5000 | 1.000 | 0.560 | 0.910 | 0.542 | 36.100 | 119.640 |
| 44 | 50 | 3050 | 0.220 | 0.000 | 0.130 | 0.000 | 230.900 | 250.000 |
| 44 | 100 | 3100 | 0.280 | 0.000 | 0.186 | 0.000 | 217.540 | 250.000 |
| 44 | 150 | 3150 | 0.300 | 0.000 | 0.189 | 0.000 | 219.540 | 250.000 |
| 44 | 200 | 3200 | 0.340 | 0.000 | 0.203 | 0.000 | 219.780 | 250.000 |
| 44 | 250 | 3250 | 0.260 | 0.000 | 0.162 | 0.000 | 224.380 | 250.000 |
| 44 | 300 | 3300 | 0.540 | 0.000 | 0.362 | 0.000 | 186.400 | 250.000 |
| 44 | 350 | 3350 | 0.440 | 0.000 | 0.287 | 0.000 | 201.140 | 250.000 |
| 44 | 400 | 3400 | 0.480 | 0.000 | 0.334 | 0.000 | 188.540 | 250.000 |
| 44 | 450 | 3450 | 0.540 | 0.000 | 0.346 | 0.000 | 192.760 | 250.000 |
| 44 | 500 | 3500 | 0.700 | 0.000 | 0.532 | 0.000 | 142.020 | 250.000 |
| 44 | 550 | 3550 | 0.780 | 0.080 | 0.557 | 0.074 | 144.040 | 231.500 |
| 44 | 600 | 3600 | 0.780 | 0.100 | 0.548 | 0.093 | 147.980 | 226.640 |
| 44 | 650 | 3650 | 0.580 | 0.040 | 0.381 | 0.039 | 184.560 | 240.250 |
| 44 | 700 | 3700 | 0.980 | 0.000 | 0.749 | 0.000 | 97.420 | 250.000 |
| 44 | 750 | 3750 | 0.980 | 0.080 | 0.827 | 0.078 | 66.220 | 230.610 |
| 44 | 800 | 3800 | 1.000 | 0.140 | 0.884 | 0.135 | 46.600 | 216.140 |
| 44 | 850 | 3850 | 1.000 | 0.060 | 0.896 | 0.058 | 41.460 | 235.450 |
| 44 | 900 | 3900 | 1.000 | 0.320 | 0.918 | 0.308 | 32.920 | 176.260 |
| 44 | 950 | 3950 | 1.000 | 0.240 | 0.896 | 0.233 | 41.560 | 191.870 |
| 44 | 1000 | 4000 | 1.000 | 0.300 | 0.908 | 0.289 | 36.980 | 180.860 |
| 44 | 1050 | 4050 | 0.980 | 0.380 | 0.897 | 0.350 | 38.300 | 157.250 |
| 44 | 1100 | 4100 | 1.000 | 0.640 | 0.915 | 0.612 | 33.840 | 97.800 |
| 44 | 1150 | 4150 | 1.000 | 0.460 | 0.910 | 0.439 | 36.160 | 137.760 |
| 44 | 1200 | 4200 | 1.000 | 0.440 | 0.923 | 0.420 | 30.800 | 145.990 |
| 44 | 1250 | 4250 | 1.000 | 0.420 | 0.923 | 0.406 | 30.840 | 153.110 |
| 44 | 1300 | 4300 | 1.000 | 0.420 | 0.920 | 0.395 | 31.940 | 158.100 |
| 44 | 1350 | 4350 | 1.000 | 0.560 | 0.919 | 0.533 | 32.400 | 119.760 |
| 44 | 1400 | 4400 | 1.000 | 0.760 | 0.919 | 0.697 | 32.280 | 69.300 |
| 44 | 1450 | 4450 | 1.000 | 0.740 | 0.925 | 0.700 | 29.940 | 67.460 |
| 44 | 1500 | 4500 | 1.000 | 0.680 | 0.925 | 0.643 | 30.080 | 84.800 |
| 44 | 1550 | 4550 | 1.000 | 0.520 | 0.936 | 0.502 | 25.720 | 122.280 |
| 44 | 1600 | 4600 | 1.000 | 0.840 | 0.929 | 0.777 | 28.280 | 51.700 |
| 44 | 1650 | 4650 | 1.000 | 0.880 | 0.934 | 0.828 | 26.580 | 38.980 |
| 44 | 1700 | 4700 | 1.000 | 0.740 | 0.943 | 0.705 | 22.740 | 63.240 |
| 44 | 1750 | 4750 | 1.000 | 0.940 | 0.932 | 0.906 | 27.100 | 24.360 |
| 44 | 1800 | 4800 | 1.000 | 0.660 | 0.930 | 0.634 | 27.900 | 86.260 |
| 44 | 1850 | 4850 | 1.000 | 0.840 | 0.935 | 0.791 | 26.060 | 49.040 |
| 44 | 1900 | 4900 | 1.000 | 0.800 | 0.935 | 0.767 | 25.840 | 59.800 |
| 44 | 1950 | 4950 | 1.000 | 0.840 | 0.936 | 0.798 | 25.540 | 47.940 |
| 44 | 2000 | 5000 | 1.000 | 0.940 | 0.942 | 0.904 | 23.280 | 24.680 |

## 6. 5000 Checkpoint 100ep Evaluation

| Seed | Checkpoint | Sample Success 100ep | Argmax Success 100ep | Sample Return | Argmax Return | Sample Length | Argmax Length |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | best_sample.pt | 1.000 | 0.610 | 0.910 | 0.589 | 35.880 | 106.090 |
| 1 | best_argmax.pt | 1.000 | 0.660 | 0.917 | 0.631 | 33.080 | 96.730 |
| 1 | last.pt | 1.000 | 0.610 | 0.910 | 0.589 | 35.880 | 106.090 |
| 1 | episode_2000.pt | 1.000 | 0.610 | 0.910 | 0.589 | 35.880 | 106.090 |
| 1 | episode_5000.pt | 1.000 | 0.610 | 0.910 | 0.589 | 35.880 | 106.090 |
| 44 | best_sample.pt | 1.000 | 0.920 | 0.942 | 0.866 | 23.390 | 41.730 |
| 44 | best_argmax.pt | 1.000 | 0.920 | 0.942 | 0.866 | 23.390 | 41.730 |
| 44 | last.pt | 1.000 | 0.920 | 0.942 | 0.866 | 23.390 | 41.730 |
| 44 | episode_2000.pt | 1.000 | 0.920 | 0.942 | 0.866 | 23.390 | 41.730 |
| 44 | episode_5000.pt | 1.000 | 0.920 | 0.942 | 0.866 | 23.390 | 41.730 |

## 7. Before/After Comparison

Best sample checkpoint is used for sample readiness. Best argmax is the highest 100ep argmax result among evaluated checkpoints.

| Seed | 3000 Sample | 5000 Best Sample | Delta Sample | 3000 Argmax | 5000 Best Argmax | Delta Argmax | Sample Ready for FT |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.080 | 1.000 | +0.920 | 0.000 | 0.660 | +0.660 | Yes |
| 44 | 0.190 | 1.000 | +0.810 | 0.000 | 0.920 | +0.920 | Yes |

Both seeds now exceed the 0.800 sample-readiness threshold.

## 8. Reference Comparison

| Model | Seed | Stage | Sample Success | Argmax Success |
|---|---:|---|---:|---:|
| Vanilla FuN | 11 | 3000 | 1.000 | 0.340 |
| Vanilla FuN | 11 | low-entropy FT pilot | 1.000 | 0.580 |
| AblationManager | mean | FT1000 | 1.000 | 0.577 |
| Vanilla FuN | 1 | 5000 additional training | 1.000 | 0.660 |
| Vanilla FuN | 44 | 5000 additional training | 1.000 | 0.920 |

## 9. Judgment

Both Baseline seed 1 and seed 44 reached sample success 1.000 after additional training. That means both are now valid candidates for low-entropy fine-tuning if further deterministic optimization is desired.

However, argmax also improved strongly during the additional training itself:

- seed 1: 0.000 -> 0.660
- seed 44: 0.000 -> 0.920

This reduces the urgency of immediate low-entropy fine-tuning for these seeds. The previous weakness was largely insufficient task learning at 3000 episodes, not only stochastic policy execution.

Seed 44 is especially strong after 5000 cumulative episodes and already exceeds the AblationManager FT1000 mean argmax success. Seed 1 is also above the 0.500 threshold, but lower than seed 44.

Baseline seed sensitivity remains real: seed 11 solved the task by 3000 episodes, while seed 1/44 needed longer training to recover.

## 10. Next Step Recommendation

Recommendation: A, with a practical caveat.

- Both seed 1 and seed 44 are sample-ready for low-entropy fine-tuning.
- If the goal is to maximize Baseline deterministic performance, run low-entropy fine-tuning on both seeds using `entropy_coef: 0.003`, `learning_rate: 0.0001`, and additional 1000 episodes.
- If the goal is to finish the report efficiently, the 5000 additional-training result is already strong enough to show that Baseline can recover with longer training, but it needs more training budget than AblationManager.

Recommended report framing:

- Use Baseline seed 11 FT pilot as evidence that low-entropy FT can help when the task is already learned.
- Use Baseline seed 1/44 5000 results as evidence that their 3000 failure was mostly sample-policy learning deficiency.
- Keep AblationManager as the more sample-efficient and cross-seed stable method at 3000 episodes.

## 11. Report Interpretation Draft

1. Vanilla FuN seed 1 and seed 44 did not fail permanently; extending training from the 3000-stage checkpoints allowed both seeds to reach sample success 1.000.

2. The large improvement in sample success shows that the earlier Baseline seed 1/44 weakness was primarily insufficient task learning within 3000 episodes, rather than only a deterministic decoding problem.

3. Argmax success also improved during longer training, reaching 0.660 for seed 1 and 0.920 for seed 44, so additional training alone can produce more deterministic Baseline behavior in some seeds.

4. Compared with AblationManager, Vanilla FuN can eventually reach strong DoorKey-6x6 performance, but it required more training budget and showed stronger seed sensitivity.

5. These results support the conclusion that AblationManager is more stable at the 3000-episode budget, while Baseline can recover under extended training and may further benefit from low-entropy fine-tuning once sample success is high.
