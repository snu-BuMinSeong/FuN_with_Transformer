# Manager Goal Analysis

Q1. Is Ablation more goal-stable than Baseline?

- sample: Ablation-Baseline goal_delta=-0.0011, goal_cosine=-0.0004.
- argmax: Ablation-Baseline goal_delta=-0.0071, goal_cosine=0.0000.

Q2. Does a goal update change the Worker action distribution?

- baseline sample: update KL=0.6461, top1 change rate=0.3880.
- baseline argmax: update KL=1.0891, top1 change rate=0.6015.
- ablation sample: update KL=0.6461, top1 change rate=0.3685.
- ablation argmax: update KL=1.1935, top1 change rate=0.5924.

Q3. Which model connects goal updates to stronger policy changes?

- sample: baseline has the larger mean update KL.
- argmax: ablation has the larger mean update KL.

Q4. Is goal instability linked to longer episodes?

- baseline sample: corr_len_delta=-0.1369, corr_len_cosine=0.1144.
- ablation sample: corr_len_delta=-0.0861, corr_len_cosine=0.0558.
- baseline argmax: corr_len_delta=-0.1537, corr_len_cosine=0.1599.
- ablation argmax: corr_len_delta=0.2783, corr_len_cosine=-0.0745.

Q5. What does this suggest about faster Ablation learning?

These final-checkpoint diagnostics can suggest whether Ablation learned a simpler and more stable goal interface, but they do not by themselves prove the learning-time mechanism because intermediate checkpoints are unavailable.
