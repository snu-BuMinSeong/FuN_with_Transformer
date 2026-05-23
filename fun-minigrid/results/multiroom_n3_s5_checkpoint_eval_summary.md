# MultiRoom-N3-S5 Checkpoint Evaluation Summary

Evaluation conditions:

- Environment: `MiniGrid-MultiRoom-N3-S5-v0`
- Seed: training config seed `1`
- Evaluation episodes: `100`
- Evaluation seed offset: `9000`
- Max steps: config `max_steps=1000`
- Policies: `sample` and `argmax`

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

Raw JSON files are under `logs/multiroom_n3s5_seed1/checkpoint_eval_100`.
