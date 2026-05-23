# Hidden Intervention Analysis

This diagnostic evaluates only final Baseline recurrent-manager checkpoints; no retraining was run.

## Normal Check

- argmax: 3-seed mean success=1.0000, mean_return=0.9828, mean_ep_len=19.1467
- sample: 3-seed mean success=0.9967, mean_return=0.9643, mean_ep_len=39.3367

## Drops From Normal

- argmax reset_goal: success_drop=0.0300, return_drop=0.0284, ep_len_change=28.1800
- argmax reset_step: success_drop=0.0300, return_drop=0.0284, ep_len_change=28.1800
- sample reset_goal: success_drop=0.0033, return_drop=0.0043, ep_len_change=4.3867
- sample reset_step: success_drop=0.0033, return_drop=0.0043, ep_len_change=4.3867

## First-pass Interpretation

The final Baseline policy does not appear to rely strongly on recurrent hidden state.
