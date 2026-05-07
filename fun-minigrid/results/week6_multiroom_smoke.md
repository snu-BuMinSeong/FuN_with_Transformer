# Week 6 MultiRoom Smoke Test

## Environment

- Environment: `MiniGrid-MultiRoom-N2-S4-v0`
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
| Max steps | 40 |
| `FuNModel` forward | Pass |
| Logits shape | `(1, 7)` |
| Value shape | `(1,)` |
| Goal shape | `(1, 16)` |

## 10-Episode Baselines

| Policy | Episodes | Success Rate | Mean Return | Mean Episode Length | Terminated | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| Random | 10 | 0.100 | 0.019 | 39.600 | 1 | 9 |
| Untrained FuN sample | 10 | 0.000 | 0.000 | 40.000 | 0 | 10 |
| Untrained FuN argmax | 10 | 0.000 | 0.000 | 40.000 | 0 | 10 |

## Interpretation

`MiniGrid-MultiRoom-N2-S4-v0` is compatible with the current environment wrapper, preprocessing, FuN model, and policy rollout loop.

The task is sparse but not obviously broken: random policy found one success in 10 episodes, while both untrained FuN sample and argmax policies failed all 10 episodes and truncated at the 40-step limit. This is a reasonable baseline before seed 1 training.

## Phase 2 Status

The Week 6 environment smoke test is complete. The next step is to create MultiRoom seed 1 Stage 1 and Stage 2 configs by adapting the DoorKey-6x6 two-stage configs.
