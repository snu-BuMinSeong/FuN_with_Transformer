# MultiRoom N4-S4 Three-Seed Process Summary

AUC is computed with trapezoidal integration over evaluation episode and divided by 5000 episodes.

## File Matching
| model | seed | env_id | manager_type | train_csv | eval_csv | config | summary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1 | MiniGrid-MultiRoom-N4-S4-v0 | recurrent | logs\multiroom_n4s4_seed1\baseline\baseline_train.csv | logs\multiroom_n4s4_seed1\baseline\baseline_eval.csv | configs\train_fun_baseline_multiroom_n4s4_ms1000_seed1.yaml | logs\multiroom_n4s4_seed1\baseline\summary.json |
| Baseline | 11 | MiniGrid-MultiRoom-N4-S4-v0 | recurrent | logs\multiroom_n4s4_seed11\baseline\train.csv | logs\multiroom_n4s4_seed11\baseline\eval.csv | configs\train_fun_baseline_multiroom_n4s4_ms1000_seed11.yaml | logs\multiroom_n4s4_seed11\baseline\summary.json |
| Baseline | 44 | MiniGrid-MultiRoom-N4-S4-v0 | recurrent | logs\multiroom_n4s4_seed44\baseline\train.csv | logs\multiroom_n4s4_seed44\baseline\eval.csv | configs\train_fun_baseline_multiroom_n4s4_ms1000_seed44.yaml | logs\multiroom_n4s4_seed44\baseline\summary.json |
| Memory Ablation | 1 | MiniGrid-MultiRoom-N4-S4-v0 | ablation | logs\multiroom_n4s4_seed1\ablation\ablation_train.csv | logs\multiroom_n4s4_seed1\ablation\ablation_eval.csv | configs\train_fun_ablation_multiroom_n4s4_ms1000_seed1.yaml | logs\multiroom_n4s4_seed1\ablation\summary.json |
| Memory Ablation | 11 | MiniGrid-MultiRoom-N4-S4-v0 | ablation | logs\multiroom_n4s4_seed11\ablation\train.csv | logs\multiroom_n4s4_seed11\ablation\eval.csv | configs\train_fun_ablation_multiroom_n4s4_ms1000_seed11.yaml | logs\multiroom_n4s4_seed11\ablation\summary.json |
| Memory Ablation | 44 | MiniGrid-MultiRoom-N4-S4-v0 | ablation | logs\multiroom_n4s4_seed44\ablation\train.csv | logs\multiroom_n4s4_seed44\ablation\eval.csv | configs\train_fun_ablation_multiroom_n4s4_ms1000_seed44.yaml | logs\multiroom_n4s4_seed44\ablation\summary.json |

## Final Eval Performance
| model | seed | eval_policy | final_success | final_return | final_episode_length |
| --- | --- | --- | --- | --- | --- |
| Baseline | 1 | sample | 1.000 | 0.970 | 33.520 |
| Baseline | 1 | argmax | 1.000 | 0.976 | 26.760 |
| Baseline | 11 | sample | 1.000 | 0.969 | 34.480 |
| Baseline | 11 | argmax | 1.000 | 0.983 | 18.520 |
| Baseline | 44 | sample | 1.000 | 0.969 | 34.220 |
| Baseline | 44 | argmax | 1.000 | 0.983 | 18.520 |
| Memory Ablation | 1 | sample | 1.000 | 0.964 | 39.720 |
| Memory Ablation | 1 | argmax | 1.000 | 0.983 | 18.680 |
| Memory Ablation | 11 | sample | 1.000 | 0.956 | 49.400 |
| Memory Ablation | 11 | argmax | 1.000 | 0.983 | 19.200 |
| Memory Ablation | 44 | sample | 1.000 | 0.969 | 34.980 |
| Memory Ablation | 44 | argmax | 1.000 | 0.983 | 18.360 |

## Success Threshold Episodes
| model | seed | eval_policy | first_positive_ep | success_0_5_ep | success_0_8_ep | success_0_9_ep | success_1_0_ep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1 | sample | 100 | 1600 | 1800 | 1800 | 2100 |
| Baseline | 1 | argmax | 1900 | 2100 | 2100 | 2100 | 2800 |
| Baseline | 11 | sample | 100 | 2400 | 2600 | 2600 | 2800 |
| Baseline | 11 | argmax | 2900 | 2900 | 2900 | 2900 | 3400 |
| Baseline | 44 | sample | 100 | 1600 | 1800 | 1900 | 2200 |
| Baseline | 44 | argmax | 2200 | 2200 | 2500 | 2600 | 2900 |
| Memory Ablation | 1 | sample | 100 | 1600 | 1800 | 1900 | 2000 |
| Memory Ablation | 1 | argmax | 2100 | 2400 | 2400 | 2400 | 2700 |
| Memory Ablation | 11 | sample | 100 | 1400 | 1800 | 1900 | 2100 |
| Memory Ablation | 11 | argmax | 1800 | 2000 | 2000 | 2200 | 2500 |
| Memory Ablation | 44 | sample | 100 | 1200 | 1500 | 1700 | 2200 |
| Memory Ablation | 44 | argmax | 1800 | 2000 | 2000 | 2300 | 2300 |

## Curve And Stability Metrics
| model | seed | eval_policy | success_auc | return_auc | episode_length_slope | drop_below_0_9_after_reach | drop_below_1_0_after_reach |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1 | sample | 0.747 | 0.660 | -0.236 | 0 | 2 |
| Baseline | 1 | argmax | 0.558 | 0.546 | -0.277 | 2 | 2 |
| Baseline | 11 | sample | 0.604 | 0.506 | -0.267 | 0 | 2 |
| Baseline | 11 | argmax | 0.410 | 0.401 | -0.282 | 2 | 3 |
| Baseline | 44 | sample | 0.718 | 0.627 | -0.246 | 1 | 7 |
| Baseline | 44 | argmax | 0.525 | 0.516 | -0.281 | 2 | 2 |
| Memory Ablation | 1 | sample | 0.720 | 0.634 | -0.246 | 0 | 4 |
| Memory Ablation | 1 | argmax | 0.519 | 0.510 | -0.288 | 1 | 2 |
| Memory Ablation | 11 | sample | 0.763 | 0.658 | -0.230 | 0 | 3 |
| Memory Ablation | 11 | argmax | 0.541 | 0.531 | -0.272 | 3 | 7 |
| Memory Ablation | 44 | sample | 0.775 | 0.687 | -0.225 | 0 | 2 |
| Memory Ablation | 44 | argmax | 0.563 | 0.554 | -0.272 | 4 | 5 |

## Train Log Metrics
| model | seed | total_train_success | last1000_success | last500_success | last100_success | mean_reward | late_reward | early_length | late_length | length_delta_late_minus_early |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1 | 0.749 | 0.997 | 0.996 | 1.000 | 0.663 | 0.963 | 938.584 | 41.232 | -897.352 |
| Baseline | 11 | 0.599 | 1.000 | 1.000 | 1.000 | 0.505 | 0.961 | 962.277 | 43.596 | -918.681 |
| Baseline | 44 | 0.710 | 0.997 | 1.000 | 1.000 | 0.626 | 0.955 | 953.704 | 49.381 | -904.323 |
| Memory Ablation | 1 | 0.715 | 0.996 | 0.996 | 0.990 | 0.628 | 0.960 | 945.947 | 43.536 | -902.411 |
| Memory Ablation | 11 | 0.760 | 0.996 | 1.000 | 1.000 | 0.655 | 0.957 | 932.349 | 47.722 | -884.627 |
| Memory Ablation | 44 | 0.775 | 0.995 | 1.000 | 1.000 | 0.688 | 0.959 | 934.427 | 44.909 | -889.518 |

## Three-Seed Final Performance Mean
| model | eval_policy | mean_final_success | std_final_success | mean_final_return | std_final_return | mean_final_length | std_final_length |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | sample | 1.000 | 0.000 | 0.969 | 0.000 | 34.073 | 0.405 |
| Baseline | argmax | 1.000 | 0.000 | 0.981 | 0.003 | 21.267 | 3.884 |
| Memory Ablation | sample | 1.000 | 0.000 | 0.963 | 0.005 | 41.367 | 6.001 |
| Memory Ablation | argmax | 1.000 | 0.000 | 0.983 | 0.000 | 18.747 | 0.346 |

## Three-Seed Learning Speed Mean
| model | eval_policy | mean_ep_to_0_5 | mean_ep_to_0_8 | mean_ep_to_0_9 | mean_ep_to_1_0 |
| --- | --- | --- | --- | --- | --- |
| Baseline | sample | 1866.667 | 2066.667 | 2100 | 2366.667 |
| Baseline | argmax | 2400 | 2500 | 2533.333 | 3033.333 |
| Memory Ablation | sample | 1400 | 1700 | 1833.333 | 2100 |
| Memory Ablation | argmax | 2133.333 | 2133.333 | 2300 | 2500 |

## Three-Seed Curve Area Mean
| model | eval_policy | mean_success_auc | mean_return_auc | mean_length_auc | mean_length_slope |
| --- | --- | --- | --- | --- | --- |
| Baseline | sample | 0.690 | 0.598 | 392.498 | -0.250 |
| Baseline | argmax | 0.497 | 0.488 | 493.328 | -0.280 |
| Memory Ablation | sample | 0.753 | 0.660 | 330.760 | -0.234 |
| Memory Ablation | argmax | 0.541 | 0.532 | 449.243 | -0.277 |

## Figures
- `results\multiroom_n4_s4_three_seed_figures\seed_eval_success_rate_sample.png`
- `results\multiroom_n4_s4_three_seed_figures\seed_eval_mean_return_sample.png`
- `results\multiroom_n4_s4_three_seed_figures\seed_eval_mean_episode_length_sample.png`
- `results\multiroom_n4_s4_three_seed_figures\mean_eval_success_rate_sample.png`
- `results\multiroom_n4_s4_three_seed_figures\mean_eval_mean_return_sample.png`
- `results\multiroom_n4_s4_three_seed_figures\seed_eval_success_rate_argmax.png`
- `results\multiroom_n4_s4_three_seed_figures\seed_eval_mean_return_argmax.png`
- `results\multiroom_n4_s4_three_seed_figures\seed_eval_mean_episode_length_argmax.png`
- `results\multiroom_n4_s4_three_seed_figures\mean_eval_success_rate_argmax.png`
- `results\multiroom_n4_s4_three_seed_figures\mean_eval_mean_return_argmax.png`
