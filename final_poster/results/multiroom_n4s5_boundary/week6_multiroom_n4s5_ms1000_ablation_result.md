# Week 6 MultiRoom-N4-S5 AblationManager ms1000 Follow-up

| Model | Seed | Max Steps | Train Success Count | Train Success Rate | Final100 Train Success | Final Sample Success | Final Argmax Success | Final Sample Return | Final Argmax Return | Final Sample Length | Final Argmax Length | Reward Signal Episodes | Best Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AblationManager | 1 | 750 | 6 | 0.001200 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 750.000000 | 750.000000 | 6 | checkpoints/multiroom_n4s5_maxsteps_sweep/ablation_ms750_seed_1/best.pt |
| AblationManager | 1 | 1000 | 16 | 0.003200 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1000.000000 | 1000.000000 | 16 | checkpoints/multiroom_n4s5_maxsteps_sweep/ablation_ms1000_seed_1/best.pt |

## Integrity
- max_steps=750: train_rows=5000, eval_rows=50, summary_final_episode=5000, best.pt=True, last.pt=True, nonfinite=False, final100_time_limit_bound=True
- max_steps=1000: train_rows=5000, eval_rows=50, summary_final_episode=5000, best.pt=True, last.pt=True, nonfinite=False, final100_time_limit_bound=True

## Answers

A. Train reward signal increased from 750 to 1000: Yes. ms750=6, ms1000=16, delta=10
B. Sample eval success became > 0: No.
C. Argmax eval success became > 0: No.
D. Episode length remains tied to max_steps: Yes.
E. Continue N4-S5: Not as the main path.
F. Lower difficulty: Yes, prefer N3-S5 or N4-S4.

## Recommendation

N4-S5 still has only train reward signal. Consider one max_steps=1250 probe only if compute budget is acceptable; otherwise lower difficulty to N3-S5 or N4-S4.
