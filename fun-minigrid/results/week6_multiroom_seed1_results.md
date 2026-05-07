# Week 6 MultiRoom-N2-S4 Seed 1 Results

## Run Status

The Week 6 MultiRoom seed 1 experiment completed on the GCP server.

GCP instance:

- `instance-20260502-111554`
- zone: `asia-east2-c`
- GPU: Tesla T4

Remote workspace:

- `/home/fumin0193/FuN_with_Transformer/fun-minigrid`

Local copied logs:

- `logs/two_stage_multiroom_n2s4`
- `logs/gcp_run_logs/week6_multiroom_seed1`

Execution:

- Vanilla FuN Stage 1 5000 episodes
- AblationManager Stage 1 5000 episodes
- Vanilla FuN Stage 2 low-entropy fine-tuning 1000 episodes
- AblationManager Stage 2 low-entropy fine-tuning 1000 episodes
- Final 100-episode evaluation for `best_sample.pt`, `best_argmax.pt`, and `last.pt`

## Output Checks

| Model | Stage | Train Rows | Eval Rows | Summary Final Episode |
|---|---|---:|---:|---:|
| Vanilla FuN | Stage 1 5000 | 5000 | 50 | 5000 |
| Vanilla FuN | Stage 2 1000 | 1000 | 20 | 1000 |
| AblationManager | Stage 1 5000 | 5000 | 50 | 5000 |
| AblationManager | Stage 2 1000 | 1000 | 20 | 1000 |

## Stage Summary

| Model | Stage | Best Sample Success | Best Argmax Success | Final Sample Success | Final Argmax Success | Final Sample Return | Final Argmax Return | Final Sample Length | Final Argmax Length |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla FuN | Stage 1 5000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.828 | 0.852 | 7.640 | 6.560 |
| Vanilla FuN | Stage 2 1000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.847 | 0.851 | 6.800 | 6.640 |
| AblationManager | Stage 1 5000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.829 | 0.852 | 7.600 | 6.560 |
| AblationManager | Stage 2 1000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.849 | 0.852 | 6.720 | 6.560 |

## Final 100-Episode Evaluation

Stage 2 checkpoints were evaluated with sample and argmax action selection for 100 episodes using seed offset 9000.

| Model | Checkpoint | Sample Success | Argmax Success | Sample Return | Argmax Return | Sample Length | Argmax Length |
|---|---|---:|---:|---:|---:|---:|---:|
| Vanilla FuN | best_sample.pt | 1.000 | 1.000 | 0.830 | 0.835 | 7.560 | 7.350 |
| Vanilla FuN | best_argmax.pt | 1.000 | 1.000 | 0.830 | 0.835 | 7.560 | 7.350 |
| Vanilla FuN | last.pt | 1.000 | 1.000 | 0.830 | 0.835 | 7.560 | 7.350 |
| AblationManager | best_sample.pt | 1.000 | 1.000 | 0.831 | 0.836 | 7.510 | 7.270 |
| AblationManager | best_argmax.pt | 1.000 | 1.000 | 0.831 | 0.836 | 7.510 | 7.270 |
| AblationManager | last.pt | 1.000 | 1.000 | 0.831 | 0.836 | 7.510 | 7.270 |

## Interpretation

MultiRoom-N2-S4 seed 1 is solved by both Vanilla FuN and AblationManager under the two-stage protocol.

The final 100-episode evaluation does not show a meaningful recurrent-memory advantage for Vanilla FuN. Both models reach sample success 1.000 and argmax success 1.000. AblationManager has a slightly shorter mean episode length, but the difference is very small.

For the current Week 6 research question, this result suggests that `MiniGrid-MultiRoom-N2-S4-v0` is still too easy to expose a strong Manager recurrent memory effect. The next decision is whether to extend seeds 11 and 44 for variance checking, or move to a harder MultiRoom candidate such as `MiniGrid-MultiRoom-N4-S5-v0`.
