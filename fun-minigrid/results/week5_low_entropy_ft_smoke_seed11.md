# Week 5 Low-Entropy Fine-Tuning Smoke: AblationManager Seed 11

## 1. Purpose

AblationManager DoorKey-6x6 seed 11 had high stochastic-policy performance but very low deterministic-policy performance after the 3000-episode run:

- sample success 100ep: 0.990
- argmax success 100ep: 0.040

This smoke test checks whether resuming from the existing 3000 checkpoint and lowering `entropy_coef` can improve argmax success while preserving sample success.

No model architecture, reward shaping, PPO, HER, or TransformerManager changes were made. Existing 3000-run paths were not used as output paths.

## 2. Pre-Check

Confirmed before running:

- Dual eval support exists through `eval_action_modes`.
- `best_sample.pt` and `best_argmax.pt` checkpoint saving is supported.
- `eval.csv` records sample and argmax metrics in separate columns.
- Existing resume checkpoint exists: `checkpoints/ablation_fun_doorkey6x6_3000/seed_11/best.pt`.
- Existing config exists: `configs/train_fun_ablation_doorkey6x6_seed11_3000.yaml`.
- `python -m py_compile train.py evaluate_checkpoint.py` passed.

The smoke run was executed on the GCP server in a separate workspace. Results were copied back into the local repository under the requested `_argmax_ft_smoke` paths.

## 3. Before Fine-Tuning

Existing 3000 checkpoint performance:

| Model | Seed | Before Sample Success 100ep | Before Argmax Success 100ep | Before Sample Return | Before Argmax Return |
|---|---:|---:|---:|---:|---:|
| AblationManager | 11 | 0.990 | 0.040 | 0.880 | 0.038 |

## 4. Fine-Tuning Settings

Config:

- `configs/smoke_train_fun_ablation_doorkey6x6_seed11_argmax_ft.yaml`

Settings:

- resume checkpoint: `checkpoints/ablation_fun_doorkey6x6_3000/seed_11/best.pt`
- resume checkpoint episode: 3000
- total episodes: 300
- learning rate: 0.0001
- entropy coef: 0.001
- value loss coef: 0.5
- manager loss coef: 0.1
- grad clip norm: 1.0
- eval interval: 50
- eval episodes: 20
- eval action modes: sample, argmax
- log dir: `logs/ablation_fun_doorkey6x6_argmax_ft_smoke/seed_11`
- checkpoint dir: `checkpoints/ablation_fun_doorkey6x6_argmax_ft_smoke/seed_11`

Resume behavior:

- model state was loaded from the checkpoint.
- optimizer state was not resumed.

Reason: the fine-tuning config intentionally changes `learning_rate` from 0.0003 to 0.0001 and `entropy_coef` from 0.01 to 0.001. Reusing the old Adam state is less controlled for this smoke test, so the run used a fresh optimizer with the resumed model weights.

## 5. Fine-Tuning Verification

Smoke command:

```bash
python train.py --config configs/smoke_train_fun_ablation_doorkey6x6_seed11_argmax_ft.yaml
```

Verification:

- `train.csv` rows: 300
- `eval.csv` rows: 6
- `summary.json` `final_episode`: 300
- `eval.csv` contains both sample and argmax columns.
- `train.csv` and `eval.csv` contain no `NaN` or `inf` strings.
- `best.pt`, `best_sample.pt`, `best_argmax.pt`, and `last.pt` were generated in the new smoke checkpoint directory.
- Existing 3000-run output paths were not used by this smoke run.

Mean training entropy during the 300-episode fine-tune was 1.304, lower than the previous 3000-run seed 11 mean entropy of about 1.898.

## 6. Fine-Tuning Eval Curve

Training-time eval used 20 episodes per interval:

| Episode | Sample Success | Argmax Success | Sample Return | Argmax Return | Sample Length | Argmax Length |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1.000 | 0.050 | 0.862 | 0.048 | 55.300 | 238.300 |
| 100 | 1.000 | 0.200 | 0.893 | 0.190 | 42.750 | 204.000 |
| 150 | 1.000 | 0.000 | 0.901 | 0.000 | 39.750 | 250.000 |
| 200 | 1.000 | 0.200 | 0.913 | 0.194 | 34.750 | 202.250 |
| 250 | 1.000 | 0.050 | 0.913 | 0.048 | 34.850 | 238.400 |
| 300 | 1.000 | 0.050 | 0.923 | 0.049 | 30.650 | 238.050 |

Training-time best checkpoints:

- `best_sample.pt`: episode 300, sample eval success 1.000
- `best_argmax.pt`: episode 200, argmax eval success 0.200

## 7. Post Fine-Tuning 100ep Evaluation

Each fine-tuned checkpoint was re-evaluated for 100 episodes with sample and argmax action selection:

| Checkpoint | Sample Success 100ep | Argmax Success 100ep | Sample Return | Argmax Return | Sample Length | Argmax Length |
|---|---:|---:|---:|---:|---:|---:|
| best_sample.pt | 1.000 | 0.180 | 0.918 | 0.175 | 32.870 | 207.200 |
| best_argmax.pt | 0.990 | 0.160 | 0.904 | 0.155 | 36.750 | 212.010 |
| last.pt | 1.000 | 0.180 | 0.918 | 0.175 | 32.870 | 207.200 |

Compared with the before-checkpoint:

- argmax success improved from 0.040 to 0.180 for `best_sample.pt` / `last.pt`.
- sample success stayed at 0.990-1.000.
- argmax episode length improved from 240.880 before fine-tuning to about 207.200 for `best_sample.pt` / `last.pt`.

## 8. Judgment

Success criteria:

- Argmax success should improve from 0.040 to at least 0.200.
- Sample success should remain at least 0.800.
- `best_argmax.pt` should be clearly better than the old `best.pt`.

Result:

- Sample preservation: passed. Sample success remained 0.990-1.000.
- Argmax improvement: partial. 100ep argmax improved from 0.040 to 0.180, but did not reach the 0.200 success threshold.
- `best_argmax.pt`: mixed. It reached 0.200 during 20ep training-time eval, but 100ep re-evaluation was 0.160, lower than `best_sample.pt` / `last.pt` at 0.180.

Interpretation:

Low-entropy fine-tuning helped deterministic behavior, but the effect is not yet strong enough to count as a clear smoke success under the requested threshold. There is no evidence of policy collapse: sample success stayed high and episode length improved. The result suggests entropy reduction is directionally useful, but this exact 300-episode / `entropy_coef=0.001` smoke setting is not yet enough.

## 9. Recommendation

Recommended next step: B-style retry with a less aggressive entropy reduction before expanding to all seeds.

Specifically:

- Try AblationManager seed 11 again with `entropy_coef: 0.003` or `0.005`.
- Keep dual eval and `best_argmax.pt` tracking enabled.
- Keep the same separate output-path rule.

Reasoning:

- Argmax improved substantially from 0.040 to 0.180, so low-entropy fine-tuning is not ineffective.
- It narrowly missed the 0.200 success criterion on 100ep evaluation.
- Sample success was preserved, so the approach did not collapse the policy.
- `entropy_coef=0.001` may be too sharp or too unstable across eval seeds; a milder value may preserve stochastic robustness while still making greedy actions more reliable.

Do not yet expand to seed 1 and seed 44 full fine-tuning. First run one more seed 11 smoke with `entropy_coef` 0.003 or 0.005 and compare against this result.
