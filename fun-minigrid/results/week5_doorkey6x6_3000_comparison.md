# Week 5 DoorKey-6x6 3000-Episode Comparison

## 1. Purpose

The existing DoorKey-6x6 1000-episode baseline/ablation runs had low best-checkpoint sample evaluation success. This run checks whether increasing only the training budget to 3000 episodes improves DoorKey-6x6 performance.

The model architecture, training algorithm, reward shaping, curriculum, HER, PPO, and hyperparameter search were not changed.

Important limitation: the six requested 1000-episode DoorKey-6x6 `best.pt` checkpoints were not present in the local workspace. Because of that, the requested 100-episode re-evaluation JSON files for the old 1000-episode checkpoints could not be generated. The old 1000 values below are the existing 20-episode `best_eval_sample.json` reference values.

## 2. Existing 1000 Best Checkpoint Re-Evaluation

Requested 100ep re-evaluation status:

| Model | Seed | Sample Success 100ep | Mean Return | Episode Length |
|---|---:|---:|---:|---:|
| Vanilla FuN | 1 | N/A | N/A | N/A |
| Vanilla FuN | 11 | N/A | N/A | N/A |
| Vanilla FuN | 44 | N/A | N/A | N/A |
| AblationManager | 1 | N/A | N/A | N/A |
| AblationManager | 11 | N/A | N/A | N/A |
| AblationManager | 44 | N/A | N/A | N/A |

Missing checkpoint files:

- `checkpoints/baseline_fun_doorkey6x6/seed_1/best.pt`
- `checkpoints/baseline_fun_doorkey6x6/seed_11/best.pt`
- `checkpoints/baseline_fun_doorkey6x6/seed_44/best.pt`
- `checkpoints/ablation_fun_doorkey6x6/seed_1/best.pt`
- `checkpoints/ablation_fun_doorkey6x6/seed_11/best.pt`
- `checkpoints/ablation_fun_doorkey6x6/seed_44/best.pt`

Existing 20ep sample evaluation reference:

| Model | Seed | Existing Sample Success 20ep | Mean Return | Episode Length | Best Checkpoint Episode |
|---|---:|---:|---:|---:|---:|
| Vanilla FuN | 1 | 0.000 | 0.000 | 250.000 | 500 |
| Vanilla FuN | 11 | 0.050 | 0.025 | 247.700 | 1000 |
| Vanilla FuN | 44 | 0.050 | 0.043 | 240.350 | 850 |
| AblationManager | 1 | 0.050 | 0.020 | 249.450 | 800 |
| AblationManager | 11 | 0.050 | 0.026 | 246.950 | 500 |
| AblationManager | 44 | 0.000 | 0.000 | 250.000 | 900 |

Existing 20ep averages:

- Vanilla FuN: success 0.033, mean return 0.022, episode length 246.017
- AblationManager: success 0.033, mean return 0.016, episode length 248.800

## 3. Seed 1 1000 vs 3000

| Model | Episodes | Sample Success 100ep | Mean Return | Episode Length | Reward Signal Episodes | Best Checkpoint Episode |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla FuN | 1000 | N/A, existing 20ep=0.000 | N/A, existing 20ep=0.000 | N/A, existing 20ep=250.000 | 28 | 500 |
| Vanilla FuN | 3000 | 0.080 | 0.048 | 242.700 | 119 | 2500 |
| AblationManager | 1000 | N/A, existing 20ep=0.050 | N/A, existing 20ep=0.020 | N/A, existing 20ep=249.450 | 22 | 800 |
| AblationManager | 3000 | 1.000 | 0.915 | 33.980 | 584 | 3000 |

Additional 3000 best checkpoint argmax 100ep results:

| Model | Argmax Success 100ep | Mean Return | Episode Length |
|---|---:|---:|---:|
| Vanilla FuN | 0.000 | 0.000 | 250.000 |
| AblationManager | 0.450 | 0.437 | 142.760 |

## 4. Learning Curve Observations

- Vanilla FuN 3000: eval success at episode 1000 was 0.050 and episode 3000 was 0.100. Best checkpoint sample 100ep success was 0.080. Reward signal episodes increased from 28 in the old 1000 run to 119 in the 3000 run, but final 100 train-episode success remained 0.100.
- AblationManager 3000: eval success at episode 1000 was 0.050 and episode 3000 was 1.000. Best checkpoint sample 100ep success was 1.000. Reward signal episodes increased from 22 in the old 1000 run to 584 in the 3000 run, and final 100 train-episode success reached 0.990.
- Episode length improved only slightly for Vanilla FuN best sample evaluation, from the 1000-run reference length 250.000 to 242.700 at 3000. AblationManager improved sharply, from the 1000-run reference length 249.450 to 33.980 at 3000.
- No `NaN` or `inf` strings were found in the 3000-run `train.csv` or `eval.csv` files.
- Loss/gradient stability looked acceptable for both 3000 runs. Vanilla final-100 mean grad norm was 0.058 with max 0.966. Ablation final-100 mean grad norm was 0.288 with max 3.130; this is larger but coincides with successful learning rather than divergence.

## 5. Verification

Pre-run checks passed:

- `python -m py_compile train.py evaluate_checkpoint.py`
- baseline 3000 config loads with `total_episodes: 3000`
- ablation 3000 config loads with `total_episodes: 3000`
- ablation 3000 config includes `manager_type: ablation`
- both 3000 runs use separate `_3000` log/checkpoint paths

3000 output checks passed:

- baseline and ablation `train.csv` each have 3000 rows
- baseline and ablation `summary.json` each have `final_episode: 3000`
- baseline and ablation `eval.csv` each have 30 interval rows
- expected `best.pt`, `last.pt`, and `episode_3000.pt` files exist for both 3000 runs

Generated comparison figures:

- `figures/doorkey6x6_3000/seed1_1000_vs_3000_success.png`
- `figures/doorkey6x6_3000/seed1_1000_vs_3000_return.png`
- `figures/doorkey6x6_3000/seed1_1000_vs_3000_episode_length.png`

## 6. Decision

For seed 1, increasing to 3000 episodes clearly improved AblationManager performance. Vanilla FuN also produced more reward signal and a small sample-success gain, but the improvement is limited and argmax 100ep success remains 0.000.

Recommendation from the seed-1-only stage was to extend seed 11 and seed 44 to 3000 episodes, especially for AblationManager.

## 7. Seed 11 and 44 Extension Results

The remaining two seeds were also trained for 3000 episodes with the same fixed setup and separate `_3000` log/checkpoint paths.

Output checks passed for all four additional runs:

- `train.csv` has 3000 rows
- `summary.json` has `final_episode: 3000`
- `eval.csv` has 30 interval rows
- `best.pt`, `last.pt`, and `episode_3000.pt` exist
- no `NaN` or `inf` strings were found in `train.csv` or `eval.csv`

Best checkpoint sample 100ep results:

| Model | Seed | Sample Success 100ep | Mean Return | Episode Length | Best Checkpoint Episode |
|---|---:|---:|---:|---:|---:|
| Vanilla FuN | 1 | 0.080 | 0.048 | 242.700 | 2500 |
| Vanilla FuN | 11 | 1.000 | 0.899 | 40.380 | 3000 |
| Vanilla FuN | 44 | 0.190 | 0.123 | 229.320 | 3000 |
| AblationManager | 1 | 1.000 | 0.915 | 33.980 | 3000 |
| AblationManager | 11 | 0.980 | 0.861 | 52.440 | 3000 |
| AblationManager | 44 | 1.000 | 0.901 | 39.690 | 3000 |

Best checkpoint argmax 100ep results:

| Model | Seed | Argmax Success 100ep | Mean Return | Episode Length |
|---|---:|---:|---:|---:|
| Vanilla FuN | 1 | 0.000 | 0.000 | 250.000 |
| Vanilla FuN | 11 | 0.340 | 0.329 | 169.440 |
| Vanilla FuN | 44 | 0.000 | 0.000 | 250.000 |
| AblationManager | 1 | 0.450 | 0.437 | 142.760 |
| AblationManager | 11 | 0.040 | 0.038 | 240.880 |
| AblationManager | 44 | 0.180 | 0.173 | 207.810 |

Across seeds:

- Vanilla FuN sample success: mean 0.423, population std 0.410
- Vanilla FuN argmax success: mean 0.113
- AblationManager sample success: mean 0.993, population std 0.009
- AblationManager argmax success: mean 0.223

Updated conclusion: simple episode-budget extension is sufficient for AblationManager on DoorKey-6x6 across seeds 1/11/44 under sample evaluation. Vanilla FuN improves for seed 11 but remains seed-sensitive, with weak seed 1 and seed 44 sample performance and near-zero argmax performance on two of three seeds.
