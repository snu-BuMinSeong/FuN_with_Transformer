# Week 6 MultiRoom-N4-S5 Smoke Test

## Environment

- Environment: `MiniGrid-MultiRoom-N4-S5-v0`
- Seed: 1
- Device: CPU

## Compatibility Checks

| Check | Result |
|---|---|
| Environment creation | Pass |
| `ImgObsWrapper` observation | Pass |
| Raw observation shape | `(7, 7, 3)` |
| Preprocessed observation shape | `(3, 7, 7)` |
| Action space | 7 |
| Max steps | 120 |
| `FuNModel` forward | Pass |
| Logits shape | `(1, 7)` |
| Value shape | `(1,)` |
| Goal shape | `(1, 16)` |

## 20-Episode Baselines

| Policy | Episodes | Success Rate | Mean Return | Mean Episode Length | Terminated | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| Random | 20 | 0.000 | 0.000 | 120.000 | 0 | 20 |
| Untrained FuN sample | 20 | 0.000 | 0.000 | 120.000 | 0 | 20 |
| Untrained FuN argmax | 20 | 0.000 | 0.000 | 120.000 | 0 | 20 |

## Interpretation

`MiniGrid-MultiRoom-N4-S5-v0` is compatible with the current wrapper, preprocessing, model, and rollout loop.

Compared with `MiniGrid-MultiRoom-N2-S4-v0`, this environment is much sparser. No random or untrained policy success was observed in 20 episodes, and all episodes reached the 120-step truncation limit. This makes N4-S5 a better candidate for testing whether recurrent Manager memory helps, but it also increases the risk that both models fail under the current REINFORCE-style training setup.

The next configuration should be treated as a seed 1 pilot before launching a full seed sweep.
