# MultiRoom N4-S4 Hidden Intervention Summary

## Seed-level results

| seed | action_mode | intervention | success | return | ep_len | success_ep_len |
|---:|---|---|---:|---:|---:|---:|
| 1 | argmax | normal | 1.0000 | 0.9808 | 21.3700 | 21.3700 |
| 1 | argmax | reset_goal | 0.9500 | 0.9347 | 66.9800 | 17.8737 |
| 1 | argmax | reset_step | 0.9500 | 0.9347 | 66.9800 | 17.8737 |
| 1 | sample | normal | 1.0000 | 0.9669 | 36.7400 | 36.7400 |
| 1 | sample | reset_goal | 1.0000 | 0.9680 | 35.5600 | 35.5600 |
| 1 | sample | reset_step | 1.0000 | 0.9680 | 35.5600 | 35.5600 |
| 11 | argmax | normal | 1.0000 | 0.9838 | 17.9500 | 17.9500 |
| 11 | argmax | reset_goal | 1.0000 | 0.9838 | 17.9500 | 17.9500 |
| 11 | argmax | reset_step | 1.0000 | 0.9838 | 17.9500 | 17.9500 |
| 11 | sample | normal | 0.9900 | 0.9592 | 44.2600 | 34.6061 |
| 11 | sample | reset_goal | 0.9800 | 0.9514 | 51.8200 | 32.4694 |
| 11 | sample | reset_step | 0.9800 | 0.9514 | 51.8200 | 32.4694 |
| 44 | argmax | normal | 1.0000 | 0.9837 | 18.1200 | 18.1200 |
| 44 | argmax | reset_goal | 0.9600 | 0.9447 | 57.0500 | 17.7604 |
| 44 | argmax | reset_step | 0.9600 | 0.9447 | 57.0500 | 17.7604 |
| 44 | sample | normal | 1.0000 | 0.9667 | 37.0100 | 37.0100 |
| 44 | sample | reset_goal | 1.0000 | 0.9606 | 43.7900 | 43.7900 |
| 44 | sample | reset_step | 1.0000 | 0.9606 | 43.7900 | 43.7900 |

## Three-seed means by action mode

| action_mode | intervention | mean_success | mean_return | mean_ep_len | mean_success_ep_len |
|---|---|---:|---:|---:|---:|
| argmax | normal | 1.0000 | 0.9828 | 19.1467 | 19.1467 |
| argmax | reset_goal | 0.9700 | 0.9544 | 47.3267 | 17.8614 |
| argmax | reset_step | 0.9700 | 0.9544 | 47.3267 | 17.8614 |
| sample | normal | 0.9967 | 0.9643 | 39.3367 | 36.1187 |
| sample | reset_goal | 0.9933 | 0.9600 | 43.7233 | 37.2731 |
| sample | reset_step | 0.9933 | 0.9600 | 43.7233 | 37.2731 |

## Intervention drops from normal

| action_mode | intervention | success_drop | return_drop | ep_len_change |
|---|---|---:|---:|---:|
| argmax | reset_goal | 0.0300 | 0.0284 | 28.1800 |
| argmax | reset_step | 0.0300 | 0.0284 | 28.1800 |
| sample | reset_goal | 0.0033 | 0.0043 | 4.3867 |
| sample | reset_step | 0.0033 | 0.0043 | 4.3867 |
