# Week 6 MultiRoom-N4-S5 Max Steps Sweep Results

| Model | Seed | Max Steps | Train Success Count | Train Success Rate | Final100 Train Success | Final Sample Success | Final Argmax Success | Final Sample Return | Final Argmax Return | Final Sample Length | Final Argmax Length | Reward Signal Episodes | Best Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla FuN | 1 | 250 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 250.000000 | 250.000000 | 0 | checkpoints/multiroom_n4s5_maxsteps_sweep/baseline_ms250_seed_1/best.pt |
| Vanilla FuN | 1 | 500 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 500.000000 | 500.000000 | 0 | checkpoints/multiroom_n4s5_maxsteps_sweep/baseline_ms500_seed_1/best.pt |
| Vanilla FuN | 1 | 750 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 750.000000 | 750.000000 | 0 | checkpoints/multiroom_n4s5_maxsteps_sweep/baseline_ms750_seed_1/best.pt |
| AblationManager | 1 | 250 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 250.000000 | 250.000000 | 0 | checkpoints/multiroom_n4s5_maxsteps_sweep/ablation_ms250_seed_1/best.pt |
| AblationManager | 1 | 500 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 500.000000 | 500.000000 | 0 | checkpoints/multiroom_n4s5_maxsteps_sweep/ablation_ms500_seed_1/best.pt |
| AblationManager | 1 | 750 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 750.000000 | 750.000000 | 0 | checkpoints/multiroom_n4s5_maxsteps_sweep/ablation_ms750_seed_1/best.pt |

## Run Integrity
- Vanilla FuN max_steps=250: train_rows=1300, eval_rows=12, summary_final_episode=0, best.pt=True, last.pt=True, nonfinite=False, final100_time_limit_bound=True
- Vanilla FuN max_steps=500: train_rows=664, eval_rows=6, summary_final_episode=0, best.pt=True, last.pt=True, nonfinite=False, final100_time_limit_bound=True
- Vanilla FuN max_steps=750: train_rows=426, eval_rows=4, summary_final_episode=0, best.pt=True, last.pt=True, nonfinite=False, final100_time_limit_bound=True
- AblationManager max_steps=250: train_rows=1300, eval_rows=12, summary_final_episode=0, best.pt=True, last.pt=True, nonfinite=False, final100_time_limit_bound=True
- AblationManager max_steps=500: train_rows=659, eval_rows=6, summary_final_episode=0, best.pt=True, last.pt=True, nonfinite=False, final100_time_limit_bound=True
- AblationManager max_steps=750: train_rows=417, eval_rows=4, summary_final_episode=0, best.pt=True, last.pt=True, nonfinite=False, final100_time_limit_bound=True

## Analysis

1. First reward signal condition: None.
2. Vanilla vs Ablation first reward signal: No reward signal in either model.
3. Eval sample success > 0: None
4. Eval argmax success > 0: None
5. Episode length tied to max_steps: Vanilla FuN max_steps=250=True, Vanilla FuN max_steps=500=True, Vanilla FuN max_steps=750=True, AblationManager max_steps=250=True, AblationManager max_steps=500=True, AblationManager max_steps=750=True
6. Continue N4-S5 or lower to N3: Wait for all runs to complete before deciding.
7. Next max_steps candidate: No N4-S5 from-scratch candidate under current settings; lower to N3 or change exploration/training stability.

## Incomplete Runs

The table above is partial because at least one run has not reached 5000 episodes.
