# MultiRoom-N3-S5 Checkpoint Evaluation Analysis

## Evaluation Setup

All checkpoints were evaluated with the same config-defined environment and max-step limit.
The evaluation used 100 episodes per checkpoint/policy pair with seed offset 9000.
Both stochastic `sample` action selection and deterministic `argmax` action selection were evaluated.

## Overall Comparison

- Best baseline sample result: `best_argmax` with success 0.970, return 0.625, length 24.780.
- Best ablation sample result: `best` with success 1.000, return 0.575, length 28.340.
- Best baseline argmax result: `best_argmax` with success 1.000, return 0.784, length 14.400.
- Best ablation argmax result: `best` with success 1.000, return 0.790, length 14.000.

## Full Results

| model | checkpoint | eval_policy | success_rate | avg_return | avg_episode_length | std_return | std_episode_length |
|---|---|---|---:|---:|---:|---:|---:|
| baseline | best | sample | 0.960 | 0.603 | 26.210 | 0.166 | 10.116 |
| baseline | best | argmax | 0.990 | 0.784 | 14.350 | 0.087 | 5.210 |
| baseline | best_argmax | sample | 0.970 | 0.625 | 24.780 | 0.169 | 10.537 |
| baseline | best_argmax | argmax | 1.000 | 0.784 | 14.400 | 0.045 | 3.000 |
| baseline | best_sample | sample | 0.960 | 0.603 | 26.210 | 0.166 | 10.116 |
| baseline | best_sample | argmax | 0.990 | 0.784 | 14.350 | 0.087 | 5.210 |
| baseline | episode_5000 | sample | 0.960 | 0.603 | 26.210 | 0.166 | 10.116 |
| baseline | episode_5000 | argmax | 0.990 | 0.784 | 14.350 | 0.087 | 5.210 |
| baseline | last | sample | 0.960 | 0.603 | 26.210 | 0.166 | 10.116 |
| baseline | last | argmax | 0.990 | 0.784 | 14.350 | 0.087 | 5.210 |
| ablation | best | sample | 1.000 | 0.575 | 28.340 | 0.141 | 9.377 |
| ablation | best | argmax | 1.000 | 0.790 | 14.000 | 0.038 | 2.553 |
| ablation | best_argmax | sample | 1.000 | 0.575 | 28.340 | 0.141 | 9.377 |
| ablation | best_argmax | argmax | 1.000 | 0.790 | 14.000 | 0.038 | 2.553 |
| ablation | best_sample | sample | 1.000 | 0.575 | 28.340 | 0.141 | 9.377 |
| ablation | best_sample | argmax | 1.000 | 0.790 | 14.000 | 0.038 | 2.553 |
| ablation | episode_5000 | sample | 1.000 | 0.575 | 28.340 | 0.141 | 9.377 |
| ablation | episode_5000 | argmax | 1.000 | 0.790 | 14.000 | 0.038 | 2.553 |
| ablation | last | sample | 1.000 | 0.575 | 28.340 | 0.141 | 9.377 |
| ablation | last | argmax | 1.000 | 0.790 | 14.000 | 0.038 | 2.553 |

## Checkpoint Selection Notes

- `baseline` under `sample` evaluation: `best_argmax` with success 0.970, return 0.625, length 24.780.
- `baseline` under `argmax` evaluation: `best_argmax` with success 1.000, return 0.784, length 14.400.
- `ablation` under `sample` evaluation: `best` with success 1.000, return 0.575, length 28.340.
- `ablation` under `argmax` evaluation: `best` with success 1.000, return 0.790, length 14.000.

## best.pt / best_argmax.pt / best_sample.pt Consistency

- `baseline`: `best_sample.pt` sample=0.960 / return=0.603; `best_argmax.pt` argmax=1.000 / return=0.784; `best.pt` sample=0.960, argmax=0.990.
- `ablation`: `best_sample.pt` sample=1.000 / return=0.575; `best_argmax.pt` argmax=1.000 / return=0.790; `best.pt` sample=1.000, argmax=1.000.

## last.pt vs best checkpoints

- `baseline` `sample`: `last.pt` success 0.960, return 0.603, length 26.210; best observed checkpoint `best_argmax` success 0.970, return 0.625, length 24.780.
- `baseline` `argmax`: `last.pt` success 0.990, return 0.784, length 14.350; best observed checkpoint `best_argmax` success 1.000, return 0.784, length 14.400.
- `ablation` `sample`: `last.pt` success 1.000, return 0.575, length 28.340; best observed checkpoint `best` success 1.000, return 0.575, length 28.340.
- `ablation` `argmax`: `last.pt` success 1.000, return 0.790, length 14.000; best observed checkpoint `best` success 1.000, return 0.790, length 14.000.

## Recommendation

For a deterministic headline metric, use `best_argmax.pt` evaluated with `argmax` for each model, and report the full checkpoint table as a robustness check. Under the best observed argmax comparison, `ablation` is ahead by the success/return/length tie-break rule.
If the paper needs a single pre-registered checkpoint rule independent of evaluation outcomes, prefer reporting `best_argmax.pt` for argmax evaluation and `best_sample.pt` for sample evaluation. If the goal is final-training-state performance, report `episode_5000.pt` or `last.pt` separately rather than mixing it with best checkpoint selection.

Summary CSV: `results/multiroom_n3_s5_checkpoint_eval_summary.csv`
Raw eval logs: `logs/multiroom_n3s5_seed1/checkpoint_eval_100`
