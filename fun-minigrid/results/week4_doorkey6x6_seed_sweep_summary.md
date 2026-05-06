# Week 4 DoorKey-6x6 Seed Sweep Summary

Scope: vanilla FuN baseline and AblationManager on MiniGrid-DoorKey-6x6-v0, seeds 1/11/44, 1000 episodes, sample training/eval action mode.

| family | seed | train best_success | final eval_success | best sample_success | best argmax_success | best sample_return | best argmax_return |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 0.100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| baseline | 11 | 0.050 | 0.050 | 0.050 | 0.000 | 0.025 | 0.000 |
| baseline | 44 | 0.100 | 0.000 | 0.050 | 0.000 | 0.043 | 0.000 |
| ablation | 1 | 0.100 | 0.000 | 0.050 | 0.000 | 0.020 | 0.000 |
| ablation | 11 | 0.100 | 0.050 | 0.050 | 0.000 | 0.026 | 0.000 |
| ablation | 44 | 0.100 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 |

Generated files:
- logs/*_fun_doorkey6x6/seed_{1,11,44}/train.csv
- logs/*_fun_doorkey6x6/seed_{1,11,44}/eval.csv
- logs/*_fun_doorkey6x6/seed_{1,11,44}/best_eval_sample.json
- logs/*_fun_doorkey6x6/seed_{1,11,44}/best_eval_argmax.json
- checkpoints/*_fun_doorkey6x6/seed_{1,11,44}/best.pt
- checkpoints/*_fun_doorkey6x6/seed_{1,11,44}/last.pt
