# Week 5 Dual Eval Code Update

## 1. Modified Files

Code/config files changed in this step:

- `train.py`
- `src/utils/config.py`
- `src/utils/logger.py`
- `configs/smoke_train_fun_ablation_doorkey6x6_seed1_dual_eval.yaml`

Generated verification outputs:

- `logs/smoke_dual_eval/ablation_doorkey6x6_seed1/eval.csv`
- `logs/smoke_dual_eval/ablation_doorkey6x6_seed1/summary.json`
- `logs/smoke_dual_eval/ablation_doorkey6x6_seed1/train.csv`
- `checkpoints/smoke_dual_eval/ablation_doorkey6x6_seed1/best.pt`
- `checkpoints/smoke_dual_eval/ablation_doorkey6x6_seed1/best_sample.pt`
- `checkpoints/smoke_dual_eval/ablation_doorkey6x6_seed1/best_argmax.pt`
- `checkpoints/smoke_dual_eval/ablation_doorkey6x6_seed1/last.pt`
- `checkpoints/smoke_dual_eval/ablation_doorkey6x6_seed1/episode_30.pt`

## 2. Config Compatibility

`src/utils/config.py` now supports simple YAML list syntax:

```yaml
eval_action_modes:
  - sample
  - argmax
```

Backward compatibility is preserved:

- If `eval_action_modes` is absent, training falls back to `[eval_action_mode]`.
- If both are absent, the previous default behavior is preserved through `eval_action_mode: argmax`.
- Existing flat key-value configs still parse as before.

The primary eval mode is:

- `eval_action_mode` if it exists and is included in `eval_action_modes`
- otherwise the first item in `eval_action_modes`

The primary mode controls the legacy `eval_*` CSV columns and legacy `best.pt`.

## 3. Eval CSV Columns

Existing eval columns are kept:

- `eval_success_rate`
- `eval_mean_return`
- `eval_std_return`
- `eval_mean_episode_length`
- `eval_std_episode_length`
- `eval_episode_seeds`

These columns now refer to the primary eval mode.

New mode-specific columns were added:

- `eval_sample_success_rate`
- `eval_sample_mean_return`
- `eval_sample_std_return`
- `eval_sample_mean_episode_length`
- `eval_sample_std_episode_length`
- `eval_sample_episode_seeds`
- `eval_argmax_success_rate`
- `eval_argmax_mean_return`
- `eval_argmax_std_return`
- `eval_argmax_mean_episode_length`
- `eval_argmax_std_episode_length`
- `eval_argmax_episode_seeds`

If only one eval mode is configured, only that mode's columns are populated; the other mode columns remain empty.

## 4. Checkpoint Save Policy

The checkpoint policy is now:

- `best.pt`: best checkpoint for the primary eval mode, preserving legacy behavior.
- `best_sample.pt`: best checkpoint by sample eval success, when sample eval is enabled.
- `best_argmax.pt`: best checkpoint by argmax eval success, when argmax eval is enabled.
- `last.pt`: still updated at every eval interval.
- `episode_{total_episodes}.pt`: still saved at the end of training.

For the new dual eval smoke config:

- primary eval mode: `sample`
- `best.pt`: sample-primary compatibility checkpoint
- `best_sample.pt`: sample success best checkpoint
- `best_argmax.pt`: argmax success best checkpoint

## 5. Summary JSON Extensions

`summary.json` now includes:

- `eval_action_modes`
- `primary_eval_action_mode`
- `best_sample_success_rate`
- `best_sample_checkpoint_path`
- `best_argmax_success_rate`
- `best_argmax_checkpoint_path`
- `pre_evals`
- `post_evals`
- `final_evals`

Existing fields such as `best_success_rate`, `best_checkpoint_path`, `final_eval`, `pre_eval`, `post_eval`, and `interval_evals` are preserved.

## 6. Smoke Test

Smoke config:

- `configs/smoke_train_fun_ablation_doorkey6x6_seed1_dual_eval.yaml`

Command:

```bash
python train.py --config configs/smoke_train_fun_ablation_doorkey6x6_seed1_dual_eval.yaml
```

Smoke settings:

- env: `MiniGrid-DoorKey-6x6-v0`
- seed: `1`
- total episodes: `30`
- eval interval: `10`
- eval episodes: `5`
- training action mode: `sample`
- eval action modes: `sample`, `argmax`
- manager type: `ablation`

Smoke run completed successfully on CPU.

Observed summary fields:

| Field | Value |
|---|---:|
| `primary_eval_action_mode` | sample |
| `best_success_rate` | 0.0 |
| `best_sample_success_rate` | 0.0 |
| `best_argmax_success_rate` | 0.0 |

The low success values are expected for a 30-episode smoke run and are not a performance result.

## 7. Smoke Output Existence

| Output | Exists |
|---|---:|
| `logs/smoke_dual_eval/ablation_doorkey6x6_seed1/eval.csv` | Yes |
| `logs/smoke_dual_eval/ablation_doorkey6x6_seed1/summary.json` | Yes |
| `checkpoints/smoke_dual_eval/ablation_doorkey6x6_seed1/best_sample.pt` | Yes |
| `checkpoints/smoke_dual_eval/ablation_doorkey6x6_seed1/best_argmax.pt` | Yes |
| `checkpoints/smoke_dual_eval/ablation_doorkey6x6_seed1/best.pt` | Yes |

## 8. Verification

Commands run:

```bash
python -m py_compile train.py evaluate_checkpoint.py
python -m pytest tests/test_config.py tests/test_train_utils.py tests/test_logger.py tests/test_evaluate_checkpoint.py -q
python -m pytest -q
```

Results:

- `py_compile`: passed
- focused tests: 7 passed
- full test suite: 48 passed

`evaluate_checkpoint.py --action-mode argmax` was not modified, so the existing checkpoint-evaluation CLI remains compatible.

## 9. Notes

No model architecture, loss calculation, reward shaping, PPO, HER, or TransformerManager changes were made.

No long DoorKey-6x6 3000/5000 episode experiment was run in this step.
