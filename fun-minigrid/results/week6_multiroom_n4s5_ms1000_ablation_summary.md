# Week 6 MultiRoom-N4-S5 AblationManager ms1000 Summary

- Complete: True
- Train reward signal 750 -> 1000: Yes. ms750=6, ms1000=16, delta=10
- ms1000 sample eval success > 0: False
- ms1000 argmax eval success > 0: False
- ms1000 final100 episode length tied to max_steps: True
- Recommendation: N4-S5 still has only train reward signal. Consider one max_steps=1250 probe only if compute budget is acceptable; otherwise lower difficulty to N3-S5 or N4-S4.

| Model | Seed | Max Steps | Train Success Count | Train Success Rate | Final100 Train Success | Final Sample Success | Final Argmax Success | Final Sample Return | Final Argmax Return | Final Sample Length | Final Argmax Length | Reward Signal Episodes | Best Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AblationManager | 1 | 750 | 6 | 0.001200 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 750.000000 | 750.000000 | 6 | checkpoints/multiroom_n4s5_maxsteps_sweep/ablation_ms750_seed_1/best.pt |
| AblationManager | 1 | 1000 | 16 | 0.003200 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1000.000000 | 1000.000000 | 16 | checkpoints/multiroom_n4s5_maxsteps_sweep/ablation_ms1000_seed_1/best.pt |
