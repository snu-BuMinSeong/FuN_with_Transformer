# MultiRoom N4-S4 Three-Seed Process Analysis

## Scope And File Matching
This analysis uses the six completed `MiniGrid-MultiRoom-N4-S4-v0` runs for Baseline Manager and Memory Ablation Manager at seeds 1, 11, and 44. The file mapping is listed in the summary document and was checked against config `env_id`, `seed`, `manager_type`, and log directory name.

The seed1 local files use legacy names such as `baseline_train.csv` and `ablation_eval.csv`; seed11 and seed44 use `train.csv` and `eval.csv`. The schemas are equivalent for the metrics used here.

## Final Performance
At the last periodic evaluation point, both models solve N4-S4 at high success rates. For sample evaluation, the three-seed mean final success is 1.000 for Baseline and 1.000 for Memory Ablation. For argmax evaluation, the means are 1.000 and 1.000, respectively.

Final return is also close. Sample final return averages 0.969 for Baseline and 0.963 for Memory Ablation. Argmax final return averages 0.981 for Baseline and 0.983 for Memory Ablation.

Final episode length shows the remaining behavioral-efficiency difference more directly. Sample final length averages 34.073 for Baseline and 41.367 for Memory Ablation; argmax final length averages 21.267 and 18.747. Lower values indicate shorter successful trajectories.

## Learning Speed
sample evaluation에서 0.5/0.8/0.9/1.0 도달 평균 episode는 Baseline 1866.7/2066.7/2100/2366.7, Memory Ablation 1400/1700/1833.3/2100이다.
argmax evaluation에서 0.5/0.8/0.9/1.0 도달 평균 episode는 Baseline 2400/2500/2533.3/3033.3, Memory Ablation 2133.3/2133.3/2300/2500이다.

Averaged over the three seeds, Memory Ablation reaches every listed success threshold earlier than Baseline in both sample and argmax evaluation. This does not mean every individual seed is identical: seed1 sample has a slightly earlier Baseline 0.9 crossing, and seed1 argmax has an earlier Baseline 0.5 crossing. However, seeds11 and 44 reverse that pattern, so the three-seed result is that Memory Ablation learns the N4-S4 task faster on average.

The success AUC reinforces the process-level comparison. In sample evaluation, mean success AUC is 0.690 for Baseline and 0.753 for Memory Ablation. In argmax evaluation, it is 0.497 versus 0.541. Because AUC is normalized by total training episodes, larger values mean that a model maintained higher success earlier and for more of training.

## Stability After Improvement
After reaching high success, instability was measured as drops below 0.9 or below 1.0 after the threshold had first been reached.
- Baseline sample: after >=0.9 drops mean 0.33, after 1.0 drops mean 3.67
- Baseline argmax: after >=0.9 drops mean 2.00, after 1.0 drops mean 2.33
- Memory Ablation sample: after >=0.9 drops mean 0.00, after 1.0 drops mean 3.00
- Memory Ablation argmax: after >=0.9 drops mean 2.67, after 1.0 drops mean 4.67

These drops matter because N4-S4 is eventually solvable by both models. A model that reaches 1.0 but repeatedly falls below it has learned a usable policy, but the evaluation curve indicates less stable policy consolidation.

The stability result is mixed rather than one-sided. In sample evaluation, Memory Ablation has no drops below 0.9 after first reaching that threshold, while Baseline has a small mean drop count. In argmax evaluation, Baseline has fewer drops after reaching 0.9 and 1.0 than Memory Ablation, indicating that the deterministic policy is somewhat more stable for Baseline once it has consolidated.

## Behavioral Efficiency
Final success is identical, but episode length separates the policies. Under sample evaluation, Baseline has shorter final trajectories on average (34.073 steps) than Memory Ablation (41.367 steps). Under argmax evaluation, Memory Ablation is slightly shorter (18.747 steps) than Baseline (21.267 steps). Therefore, N4-S4 should not be summarized only by success; the same success rate can correspond to different trajectory efficiency depending on whether the stochastic or deterministic policy is evaluated.

## Train Log Interpretation
The training logs show on-policy sample behavior, while eval logs report held-out evaluation episodes under sample and argmax policies. Last-window train success is therefore useful as a behavioral-process signal but should not replace the eval curves. In all six runs the final training windows are much stronger than the early windows, and episode length decreases from early to late training, indicating that learning is not only a binary success transition but also a shift toward shorter trajectories.

## Sample Versus Argmax
Sample evaluation reflects the stochastic policy that is also used during training. Argmax evaluation tests whether the policy has formed a deterministic high-probability action sequence. A gap where sample succeeds earlier than argmax indicates that useful behavior exists in the stochastic policy before it has become the dominant deterministic path. A later argmax catch-up indicates policy consolidation.

In this experiment, both modes eventually become strong, so the sample/argmax comparison is most useful for reading when the policy starts to solve the environment versus when the solution becomes deterministically preferred.

## Answers To The Core Questions
1. Both Baseline and Memory Ablation ultimately solve N4-S4 at high success rates across the three seeds.
2. Final success rate alone is not an adequate discriminator, because both models reach the same high-success regime.
3. The meaningful difference is in the timing of first success and threshold crossings. First positive sample success appears at episode 100 for all runs, but argmax first positive success appears later and is generally earlier for Memory Ablation in seeds11 and 44.
4. In the three-seed mean, Memory Ablation reaches 0.5, 0.8, 0.9, and 1.0 earlier than Baseline for both sample and argmax evaluation.
5. Stability is policy-dependent. Memory Ablation is more stable in sample success after reaching 0.9, while Baseline is more stable in argmax after reaching high success.
6. Return and episode length remain important even when success is equal. Sample evaluation favors Baseline in final trajectory length, while argmax evaluation slightly favors Memory Ablation.
7. Sample success before argmax success means that the stochastic policy can sometimes solve the task before the deterministic policy has stabilized. This is visible in both models, especially before the argmax curve catches up.
8. The seed1-only trend should not be overclaimed. Some seed1 crossings favor Baseline, but seed11 and seed44 generally favor Memory Ablation in learning speed and AUC, so the 3-seed result is more balanced and more informative.
9. In a paper, N4-S4 is best presented as a process-level and policy-formation experiment: both models can solve the environment, so the comparison should emphasize learning speed, stability after improvement, and behavioral efficiency.

## Recommended Paper Framing
N4-S4에서는 Baseline과 Memory Ablation이 모두 최종적으로 높은 성공률을 달성하였다. 따라서 이 환경에서는 단순한 최종 성공 여부보다, 학습 속도와 안정성, 그리고 동일한 성공률을 달성한 이후의 행동 효율성 차이가 더 중요한 비교 기준이 된다.

The quantitative support for this framing is in `multiroom_n4_s4_three_seed_process_summary.md` and the generated figures.
