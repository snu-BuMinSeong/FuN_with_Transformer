# Week 5 AblationManager Argmax Fine-Tuning 1000 Results

## 1. Purpose

AblationManager DoorKey-6x6 reached high sample success after 3000 episodes, but argmax success remained weak. The 500-episode smoke v2 showed that low-entropy fine-tuning with `entropy_coef: 0.003` improved argmax success while preserving sample success.

This run is the full AblationManager argmax-improvement fine-tuning experiment for seeds 1, 11, and 44:

- resume from existing 3000-episode AblationManager checkpoints
- `learning_rate: 0.0001`
- `entropy_coef: 0.003`
- `total_episodes: 1000`
- dual sample/argmax evaluation
- separate `best_sample.pt`, `best_argmax.pt`, `last.pt`, and `episode_1000.pt`

No model architecture, reward shaping, PPO, HER, or TransformerManager changes were made. Existing 3000 and smoke v2 output paths were not used as FT1000 output paths.

## 2. GCP Execution

Executed on GCP in a separate workspace:

- `/home/fumin0193/FuN_with_Transformer/argmax_ft_1000_workspace/fun-minigrid`

Parallel execution:

- Seeds 1, 11, and 44 were launched concurrently with background `nohup` jobs.
- The shell waited for all three PIDs to finish.
- Seed stdout/stderr logs were written to separate files, so outputs did not mix.

Seed commands:

```bash
python train.py --config configs/train_fun_ablation_doorkey6x6_seed1_argmax_ft_1000.yaml
python train.py --config configs/train_fun_ablation_doorkey6x6_seed11_argmax_ft_1000.yaml
python train.py --config configs/train_fun_ablation_doorkey6x6_seed44_argmax_ft_1000.yaml
```

Seed stdout/stderr logs:

- `logs/gcp_run_logs/argmax_ft_1000_seed1.out`
- `logs/gcp_run_logs/argmax_ft_1000_seed11.out`
- `logs/gcp_run_logs/argmax_ft_1000_seed44.out`

Pre-run verification:

- `python -m py_compile train.py evaluate_checkpoint.py`: passed locally and on GCP.
- `python -m pytest -q`: 48 passed locally before the GCP run.

Post-copy verification:

- each seed has 1000 `train.csv` rows
- each seed has 20 `eval.csv` rows
- each seed summary has `final_episode: 1000`
- no `NaN` or `inf` strings were found in local `train.csv` or `eval.csv`
- required checkpoints are present for all seeds

## 3. 3000 Checkpoint Baseline

| Seed | 3000 Sample Success 100ep | 3000 Argmax Success 100ep | 3000 Sample Return | 3000 Argmax Return |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 0.450 | 0.912 | 0.437 |
| 11 | 0.990 | 0.040 | 0.880 | 0.038 |
| 44 | 1.000 | 0.180 | 0.894 | 0.173 |

## 4. Smoke V2 Reference

| Seed | Smoke V2 Best Argmax Success | Smoke V2 Sample Success |
|---:|---:|---:|
| 1 | 0.540 | 1.000 |
| 11 | 0.370 | 1.000 |
| 44 | 0.450 | 1.000 |

## 5. Fine-Tuning Settings

Common settings:

- total episodes: 1000
- learning rate: 0.0001
- entropy coef: 0.003
- value loss coef: 0.5
- manager loss coef: 0.1
- grad clip norm: 1.0
- eval interval: 50
- eval episodes: 50
- eval action modes: sample, argmax
- optimizer state resumed: no

Optimizer state was intentionally not resumed because the fine-tuning learning rate and entropy coefficient differ from the original 3000-episode run. Each run loaded model weights and used a fresh Adam optimizer.

Seed-specific paths:

| Seed | Resume Checkpoint | Log Dir | Checkpoint Dir |
|---:|---|---|---|
| 1 | `checkpoints/ablation_fun_doorkey6x6_3000/seed_1/best.pt` | `logs/ablation_fun_doorkey6x6_argmax_ft_1000/seed_1` | `checkpoints/ablation_fun_doorkey6x6_argmax_ft_1000/seed_1` |
| 11 | `checkpoints/ablation_fun_doorkey6x6_3000/seed_11/best.pt` | `logs/ablation_fun_doorkey6x6_argmax_ft_1000/seed_11` | `checkpoints/ablation_fun_doorkey6x6_argmax_ft_1000/seed_11` |
| 44 | `checkpoints/ablation_fun_doorkey6x6_3000/seed_44/best.pt` | `logs/ablation_fun_doorkey6x6_argmax_ft_1000/seed_44` | `checkpoints/ablation_fun_doorkey6x6_argmax_ft_1000/seed_44` |

Mean training entropy:

| Seed | Mean Entropy |
|---:|---:|
| 1 | 0.921 |
| 11 | 1.090 |
| 44 | 1.033 |

## 6. Fine-Tuning Eval Curve

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
| 1 | 550 | 1.000 | 0.400 | 0.953 | 0.387 | 18.640 | 155.120 |
| 1 | 600 | 1.000 | 0.540 | 0.953 | 0.525 | 18.700 | 121.020 |
| 1 | 650 | 1.000 | 0.520 | 0.947 | 0.505 | 21.140 | 126.200 |
| 1 | 700 | 1.000 | 0.620 | 0.947 | 0.600 | 21.260 | 102.840 |
| 1 | 750 | 1.000 | 0.540 | 0.949 | 0.523 | 20.500 | 121.800 |
| 1 | 800 | 1.000 | 0.500 | 0.948 | 0.484 | 20.640 | 131.400 |
| 1 | 850 | 1.000 | 0.540 | 0.947 | 0.524 | 21.380 | 121.440 |
| 1 | 900 | 1.000 | 0.620 | 0.951 | 0.601 | 19.460 | 102.580 |
| 1 | 950 | 1.000 | 0.620 | 0.954 | 0.602 | 18.320 | 102.340 |
| 1 | 1000 | 1.000 | 0.680 | 0.955 | 0.658 | 18.080 | 88.860 |
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
| 11 | 550 | 1.000 | 0.220 | 0.934 | 0.212 | 26.440 | 198.140 |
| 11 | 600 | 1.000 | 0.400 | 0.941 | 0.388 | 23.440 | 154.960 |
| 11 | 650 | 1.000 | 0.300 | 0.930 | 0.290 | 28.060 | 179.000 |
| 11 | 700 | 1.000 | 0.340 | 0.936 | 0.326 | 25.720 | 170.520 |
| 11 | 750 | 1.000 | 0.480 | 0.937 | 0.463 | 25.340 | 136.980 |
| 11 | 800 | 1.000 | 0.260 | 0.935 | 0.252 | 26.140 | 188.400 |
| 11 | 850 | 1.000 | 0.280 | 0.935 | 0.271 | 26.160 | 183.640 |
| 11 | 900 | 1.000 | 0.220 | 0.938 | 0.213 | 24.740 | 197.960 |
| 11 | 950 | 1.000 | 0.600 | 0.943 | 0.579 | 22.620 | 108.540 |
| 11 | 1000 | 1.000 | 0.440 | 0.949 | 0.423 | 20.380 | 146.820 |
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
| 44 | 550 | 1.000 | 0.480 | 0.943 | 0.466 | 22.640 | 135.780 |
| 44 | 600 | 1.000 | 0.420 | 0.940 | 0.407 | 24.200 | 150.100 |
| 44 | 650 | 1.000 | 0.480 | 0.942 | 0.464 | 23.360 | 136.320 |
| 44 | 700 | 1.000 | 0.420 | 0.940 | 0.407 | 24.040 | 150.400 |
| 44 | 750 | 1.000 | 0.440 | 0.944 | 0.425 | 22.540 | 145.960 |
| 44 | 800 | 1.000 | 0.360 | 0.938 | 0.349 | 24.900 | 164.400 |
| 44 | 850 | 1.000 | 0.320 | 0.943 | 0.308 | 22.780 | 174.720 |
| 44 | 900 | 1.000 | 0.440 | 0.946 | 0.427 | 21.500 | 145.360 |
| 44 | 950 | 1.000 | 0.400 | 0.948 | 0.388 | 20.860 | 154.800 |
| 44 | 1000 | 0.980 | 0.420 | 0.928 | 0.407 | 25.720 | 150.320 |

Training-time best argmax eval:

| Seed | Best Argmax Eval Episode | Best Argmax Eval Success |
|---:|---:|---:|
| 1 | 1000 | 0.680 |
| 11 | 950 | 0.600 |
| 44 | 650 | 0.480 |

## 7. Post Fine-Tuning 100ep Evaluation

| Seed | Checkpoint | Sample Success 100ep | Argmax Success 100ep | Sample Return | Argmax Return | Sample Length | Argmax Length |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | best_sample.pt | 1.000 | 0.730 | 0.955 | 0.708 | 17.820 | 76.440 |
| 1 | best_argmax.pt | 1.000 | 0.730 | 0.955 | 0.708 | 17.820 | 76.440 |
| 1 | last.pt | 1.000 | 0.730 | 0.955 | 0.708 | 17.820 | 76.440 |
| 1 | episode_1000.pt | 1.000 | 0.730 | 0.955 | 0.708 | 17.820 | 76.440 |
| 11 | best_sample.pt | 1.000 | 0.520 | 0.948 | 0.502 | 20.790 | 127.090 |
| 11 | best_argmax.pt | 1.000 | 0.530 | 0.945 | 0.512 | 21.970 | 124.800 |
| 11 | last.pt | 1.000 | 0.520 | 0.948 | 0.502 | 20.790 | 127.090 |
| 11 | episode_1000.pt | 1.000 | 0.520 | 0.948 | 0.502 | 20.790 | 127.090 |
| 44 | best_sample.pt | 1.000 | 0.460 | 0.950 | 0.447 | 19.990 | 140.370 |
| 44 | best_argmax.pt | 1.000 | 0.470 | 0.946 | 0.456 | 21.530 | 138.130 |
| 44 | last.pt | 1.000 | 0.470 | 0.951 | 0.456 | 19.620 | 137.950 |
| 44 | episode_1000.pt | 1.000 | 0.470 | 0.951 | 0.456 | 19.620 | 137.950 |

## 8. Before/After Comparison

Best result is selected by highest 100ep argmax success.

| Seed | 3000 Argmax | Smoke V2 Argmax | FT1000 Best Argmax | Delta vs 3000 | Delta vs Smoke V2 | 3000 Sample | FT1000 Sample | Sample Preserved |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.450 | 0.540 | 0.730 | +0.280 | +0.190 | 1.000 | 1.000 | Yes |
| 11 | 0.040 | 0.370 | 0.530 | +0.490 | +0.160 | 0.990 | 1.000 | Yes |
| 44 | 0.180 | 0.450 | 0.470 | +0.290 | +0.020 | 1.000 | 1.000 | Yes |

## 9. Mean Results

| Stage | Mean Sample Success | Mean Argmax Success | Mean Sample Return | Mean Argmax Return | Mean Argmax Length |
|---|---:|---:|---:|---:|---:|
| 3000 checkpoint | 0.997 | 0.223 | 0.895 | 0.216 | 197.150 |
| 500ep smoke v2 | 1.000 | 0.453 | 0.935 | 0.439 | 142.213 |
| 1000ep fine-tuning | 1.000 | 0.577 | 0.950 | 0.559 | 113.063 |

## 10. Judgment

Success criteria:

- Mean argmax success improves over smoke v2 0.453: passed. FT1000 reached 0.577.
- Mean argmax success reaches at least 0.550: passed.
- Each seed argmax success reaches at least 0.400: passed. Seed 1: 0.730, seed 11: 0.530, seed 44: 0.470.
- Sample success remains at least 0.900 for each seed: passed. All selected FT1000 checkpoints have sample success 1.000.
- Argmax episode length decreases from the 3000 checkpoint: passed for all seeds and on average.

Interpretation:

FT1000 is a clear improvement over both the original 3000 checkpoints and the 500ep smoke v2. The largest absolute gain remains seed 11, which moved from 0.040 to 0.530. Seed 44 improved only slightly over smoke v2, but still improved substantially over the original 3000 checkpoint and stayed above the 0.400 stability threshold.

There is no evidence of policy collapse. Sample success stayed at 1.000 in the selected 100ep evaluations.

`best_argmax.pt` tracking is useful but not perfectly aligned with 100ep re-evaluation because interval eval uses 50 episodes. It selected the best 100ep checkpoint for seed 11, tied or nearly tied for seed 1, and was effectively tied with `last.pt` for seed 44. Keeping it is still important for reporting deterministic-policy performance.

## 11. Next Step Recommendation

Recommendation: A and D.

Use AblationManager FT1000 as the argmax-improvement final result:

- Mean argmax success is above 0.550.
- Sample success is preserved.
- All seeds are above 0.400 argmax success.
- Deterministic episode length improved strongly.

Next experiment:

- Run only a Baseline seed 11 low-entropy fine-tuning pilot.
- Do not immediately fine-tune Baseline seed 1/44 because their sample success was not consistently solved; first determine whether Baseline seed 11, which had the best sample performance, benefits from the same deterministic-policy fine-tuning.

Temperature evaluation remains a useful follow-up, but it is no longer required to justify AblationManager FT1000 as the improved deterministic-policy result.

## 12. Report Interpretation Draft

1. AblationManager already solved DoorKey-6x6 under stochastic evaluation, but greedy argmax execution exposed that the learned policy remained too stochastic for reliable deterministic control.

2. Low-entropy fine-tuning from the 3000-episode checkpoints substantially improved deterministic execution: mean argmax success increased from 0.223 to 0.577 while mean sample success stayed at 1.000.

3. The improvement was consistent across all three seeds, with seed 11 recovering from 0.040 to 0.530 argmax success and every seed exceeding 0.400 after fine-tuning.

4. The decrease in argmax episode length shows that fine-tuning improved not only success frequency but also the efficiency of deterministic trajectories.

5. These results suggest that the main AblationManager failure mode was not insufficient task learning, but insufficient policy determinism; low-entropy fine-tuning is therefore an effective post-training correction for argmax deployment.
