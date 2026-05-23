# Manager Goal Results

## 3-seed aggregate table

| model | action_mode | success | return | ep_len | goal_delta | goal_cosine | entropy | top1_prob | update_action_kl |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | sample | 0.9967 | 0.9643 | 39.3367 | 0.0457 | 0.9987 | 1.3236 | 0.5714 | 0.7406 |
| baseline | argmax | 1.0000 | 0.9828 | 19.1467 | 0.0473 | 0.9984 | 1.2835 | 0.5908 | 1.0175 |
| ablation | sample | 0.9967 | 0.9653 | 38.2400 | 0.0446 | 0.9983 | 1.3226 | 0.5729 | 0.7071 |
| ablation | argmax | 1.0000 | 0.9836 | 18.1733 | 0.0402 | 0.9984 | 1.2878 | 0.5899 | 1.0328 |

## Baseline vs Ablation differences

| action_mode | metric | baseline | ablation | difference |
|---|---|---:|---:|---:|
| sample | success | 0.9967 | 0.9967 | 0.0000 |
| sample | return | 0.9643 | 0.9653 | 0.0010 |
| sample | ep_len | 39.3367 | 38.2400 | -1.0967 |
| sample | goal_delta | 0.0457 | 0.0446 | -0.0011 |
| sample | goal_cosine | 0.9987 | 0.9983 | -0.0004 |
| sample | entropy | 1.3236 | 1.3226 | -0.0010 |
| sample | top1_prob | 0.5714 | 0.5729 | 0.0014 |
| sample | update_action_kl | 0.7406 | 0.7071 | -0.0335 |
| argmax | success | 1.0000 | 1.0000 | 0.0000 |
| argmax | return | 0.9828 | 0.9836 | 0.0009 |
| argmax | ep_len | 19.1467 | 18.1733 | -0.9733 |
| argmax | goal_delta | 0.0473 | 0.0402 | -0.0071 |
| argmax | goal_cosine | 0.9984 | 0.9984 | 0.0000 |
| argmax | entropy | 1.2835 | 1.2878 | 0.0043 |
| argmax | top1_prob | 0.5908 | 0.5899 | -0.0009 |
| argmax | update_action_kl | 1.0175 | 1.0328 | 0.0153 |

## Episode length correlation table

| action_mode | model | corr(ep_len, goal_delta) | corr(ep_len, goal_cosine) |
|---|---|---:|---:|
| sample | baseline | -0.1369 | 0.1144 |
| sample | ablation | -0.0861 | 0.0558 |
| argmax | baseline | -0.1537 | 0.1599 |
| argmax | ablation | 0.2783 | -0.0745 |

## Goal update event summary table

| model | action_mode | mean_goal_delta_on_update | mean_action_kl_on_update | top1_change_rate |
|---|---|---:|---:|---:|
| baseline | sample | 0.4812 | 0.6461 | 0.3880 |
| baseline | argmax | 0.6398 | 1.0891 | 0.6015 |
| ablation | sample | 0.4824 | 0.6461 | 0.3685 |
| ablation | argmax | 0.5756 | 1.1935 | 0.5924 |
