# DoorKey-6x6 Experiment Summary

## Experiment Goal

This experiment evaluates Vanilla FuN and AblationManager on `MiniGrid-DoorKey-6x6-v0`. The main question is whether removing recurrent memory from the FuN manager hurts performance in a slightly harder DoorKey environment, and whether the learned policies can be deployed deterministically with argmax action selection.

The experiment was conducted in three stages:

1. Train Vanilla FuN and AblationManager for 3000 episodes.
2. Diagnose the gap between stochastic sample evaluation and deterministic argmax evaluation.
3. Apply low-entropy fine-tuning or additional training where needed.

No model architecture change, reward shaping, PPO, HER, or TransformerManager was introduced during these experiments.

## Evaluation Protocol

Two action-selection modes were evaluated:

- `sample`: actions are sampled from the learned policy distribution.
- `argmax`: the action with the largest policy logit is selected deterministically.

The original training setup used stochastic sampling, so sample evaluation reflects the training-time action-selection regime. Argmax evaluation is stricter and tests whether the learned policy can be deployed as a deterministic controller.

During Week 5, the training code was extended to support dual evaluation. Each run can now record both sample and argmax metrics and save separate checkpoints:

- `best_sample.pt`: best checkpoint by sample success.
- `best_argmax.pt`: best checkpoint by argmax success.
- `best.pt`: backward-compatible primary checkpoint.

This was necessary because selecting `best.pt` only by sample success can hide weak deterministic behavior.

## 3000-Episode Results

At 3000 episodes, AblationManager was clearly stronger and more stable under sample evaluation.

| Model | Seed | Sample Success | Argmax Success | Sample Return | Argmax Return |
|---|---:|---:|---:|---:|---:|
| Vanilla FuN | 1 | 0.080 | 0.000 | 0.048 | 0.000 |
| Vanilla FuN | 11 | 1.000 | 0.340 | 0.899 | 0.329 |
| Vanilla FuN | 44 | 0.190 | 0.000 | 0.123 | 0.000 |
| AblationManager | 1 | 1.000 | 0.450 | 0.915 | 0.437 |
| AblationManager | 11 | 0.980 | 0.040 | 0.861 | 0.038 |
| AblationManager | 44 | 1.000 | 0.180 | 0.901 | 0.173 |

Mean performance:

| Model | Mean Sample Success | Mean Argmax Success |
|---|---:|---:|
| Vanilla FuN | 0.423 | 0.113 |
| AblationManager | 0.993 | 0.223 |

The sample-evaluation result shows that AblationManager learned the DoorKey-6x6 task reliably across all three seeds. Vanilla FuN solved the task only for seed 11, while seed 1 and seed 44 remained weak even after 3000 episodes.

However, AblationManager still had a large sample-argmax gap. In particular, seed 11 reached near-perfect sample success but only 0.040 argmax success. This indicates that the policy had learned successful stochastic behavior, but the highest-probability action alone did not always produce a reliable trajectory.

## Checkpoint Selection Diagnosis

A separate re-evaluation compared `best.pt`, `last.pt`, and `episode_3000.pt` for both models. This showed that the low argmax success was not simply caused by choosing the wrong saved checkpoint among the existing 3000-episode checkpoints.

For AblationManager, the sample policy was already strong, but deterministic argmax execution remained weak in some seeds. Therefore, the main issue was policy determinism rather than task acquisition.

This motivated low-entropy fine-tuning: keep the learned task behavior, but reduce unnecessary stochasticity so the greedy action sequence becomes more reliable.

## AblationManager Low-Entropy Fine-Tuning

AblationManager checkpoints from the 3000-episode runs were fine-tuned for 1000 additional episodes with:

- `learning_rate = 0.0001`
- `entropy_coef = 0.003`
- dual sample/argmax evaluation
- fresh optimizer state

The fine-tuning preserved sample success and substantially improved argmax success.

| Seed | 3000 Argmax | FT1000 Argmax | Delta | 3000 Sample | FT1000 Sample |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.450 | 0.730 | +0.280 | 1.000 | 1.000 |
| 11 | 0.040 | 0.530 | +0.490 | 0.990 | 1.000 |
| 44 | 0.180 | 0.470 | +0.290 | 1.000 | 1.000 |

Mean result:

| Stage | Mean Sample Success | Mean Argmax Success | Mean Argmax Length |
|---|---:|---:|---:|
| 3000 checkpoint | 0.997 | 0.223 | 197.150 |
| 500ep smoke fine-tuning | 1.000 | 0.453 | 142.213 |
| 1000ep fine-tuning | 1.000 | 0.577 | 113.063 |

This is the strongest AblationManager result. Mean argmax success improved from 0.223 to 0.577 while sample success stayed at 1.000. The shorter argmax episode length also shows that deterministic trajectories became more efficient, not merely more successful.

## Vanilla FuN Follow-Up Results

Vanilla FuN had two different failure modes:

- Seed 11 had already learned the task under sample evaluation but had moderate argmax performance.
- Seeds 1 and 44 had weak sample performance at 3000 episodes, meaning they had not reliably learned the task yet.

For seed 11, low-entropy fine-tuning worked well:

| Model | Seed | Stage | Sample Success | Argmax Success | Argmax Length |
|---|---:|---|---:|---:|---:|
| Vanilla FuN | 11 | 3000 | 1.000 | 0.340 | 169.440 |
| Vanilla FuN | 11 | low-entropy FT pilot | 1.000 | 0.580 | 113.260 |

This confirms that low-entropy fine-tuning can also help Vanilla FuN when the starting policy has already learned the task.

For seeds 1 and 44, low-entropy fine-tuning was not applied immediately. Instead, the 3000 checkpoints were resumed for 2000 additional episodes, giving a cumulative 5000-episode budget. This tested whether their weakness was insufficient task learning rather than insufficient determinism.

| Seed | 3000 Sample | 5000 Best Sample | 3000 Argmax | 5000 Best Argmax |
|---:|---:|---:|---:|---:|
| 1 | 0.080 | 1.000 | 0.000 | 0.660 |
| 44 | 0.190 | 1.000 | 0.000 | 0.920 |

The longer training recovered both weak Vanilla FuN seeds. This means the 3000-episode Vanilla FuN failures were not permanent, but they required more training budget than AblationManager.

## Main Interpretation

The DoorKey-6x6 results suggest that AblationManager is more sample-efficient and more stable across seeds than Vanilla FuN at the 3000-episode budget. AblationManager solved the task under sample evaluation for all three seeds, while Vanilla FuN solved it only for seed 11.

The argmax results reveal a separate issue. A high sample success rate does not guarantee that the greedy deterministic policy will succeed. AblationManager seed 11 is the clearest example: sample success was near 1.000, but argmax success was only 0.040 before fine-tuning. This shows that the policy distribution still relied on stochastic action sampling.

Low-entropy fine-tuning is an effective correction when the task has already been learned. For AblationManager, it improved mean argmax success from 0.223 to 0.577 without reducing sample success. For Vanilla FuN seed 11, it improved argmax success from 0.340 to 0.580 while preserving sample success.

Vanilla FuN can eventually reach strong performance, as shown by the 5000-episode seed 1 and seed 44 runs. However, this required a larger training budget and showed stronger seed sensitivity. Therefore, the main advantage of AblationManager in this experiment is not that Vanilla FuN cannot solve DoorKey-6x6, but that AblationManager solves it more consistently within the 3000-episode budget.

## Report-Ready Summary

In DoorKey-6x6, AblationManager achieved much more stable task learning than Vanilla FuN at the 3000-episode budget. Across three seeds, AblationManager reached 0.993 mean sample success, whereas Vanilla FuN reached only 0.423 because two seeds failed to solve the task reliably. This indicates that removing manager recurrence did not hurt performance in this environment; instead, the simpler manager produced more stable learning under the tested budget.

Deterministic argmax evaluation exposed an additional limitation. Even when sample success was high, argmax success could remain low, showing that the learned policy still depended on stochastic action sampling. To address this, low-entropy fine-tuning was applied after task learning. For AblationManager, mean argmax success improved from 0.223 to 0.577 while sample success remained 1.000, demonstrating that deterministic execution can be improved without sacrificing the learned stochastic policy.

For Vanilla FuN, the results were seed-dependent. Seed 11 benefited from the same low-entropy fine-tuning, improving argmax success from 0.340 to 0.580. Seeds 1 and 44, however, first required longer training because their 3000-episode sample success was low. After extending them to a 5000-episode cumulative budget, both reached sample success 1.000 and strong argmax performance. This suggests that Vanilla FuN can solve DoorKey-6x6, but it requires more training and is less stable across seeds than AblationManager.

Overall, the DoorKey-6x6 experiments support two conclusions. First, AblationManager is the more stable and sample-efficient model under the 3000-episode setting. Second, argmax deployment requires explicit deterministic-policy evaluation and, when sample performance is already high, low-entropy fine-tuning is a practical way to reduce the sample-argmax performance gap.
