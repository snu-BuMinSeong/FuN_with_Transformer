# Week 5 Checkpoint Argmax Recheck

## 1. Scope

This step re-evaluated existing DoorKey-6x6 3000-episode checkpoints with both sample and argmax action selection.

No new training was run. No existing checkpoint or log file was deleted or overwritten. New evaluation JSON files were saved under:

- `logs/argmax_diagnosis/`

The evaluation was executed on the GCP server in a separate evaluation workspace:

- `/home/fumin0193/FuN_with_Transformer/argmax_recheck_workspace/fun-minigrid`

The active GCP instance did not already contain the `_3000` checkpoint directories, so the existing local checkpoint/config/code files were copied into the separate GCP evaluation workspace before running evaluation. The original remote project directory was not overwritten.

Temperature evaluation: not implemented. `evaluate_checkpoint.py` currently supports only `--action-mode sample` and `--action-mode argmax`.

## 2. Evaluation Setup

Each checkpoint was evaluated for 100 episodes with:

- `--action-mode sample`
- `--action-mode argmax`

Example command pattern:

```bash
python evaluate_checkpoint.py \
  --config configs/train_fun_ablation_doorkey6x6_seed1_3000.yaml \
  --checkpoint ablation_fun_doorkey6x6_3000/seed_1/best.pt \
  --output logs/argmax_diagnosis/ablation_seed1_best_argmax_100ep.json \
  --episodes 100 \
  --action-mode argmax
```

Generated JSON count:

- 18 checkpoints x 2 action modes = 36 JSON files

## 3. Checkpoint Results

| Model | Seed | Checkpoint | Sample Success | Argmax Success | Sample Return | Argmax Return | Sample Length | Argmax Length |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Vanilla FuN | 1 | best | 0.080 | 0.000 | 0.049 | 0.000 | 242.360 | 250.000 |
| Vanilla FuN | 1 | last | 0.110 | 0.000 | 0.063 | 0.000 | 241.190 | 250.000 |
| Vanilla FuN | 1 | episode_3000 | 0.110 | 0.000 | 0.063 | 0.000 | 241.190 | 250.000 |
| Vanilla FuN | 11 | best | 1.000 | 0.340 | 0.907 | 0.329 | 37.130 | 169.440 |
| Vanilla FuN | 11 | last | 1.000 | 0.340 | 0.907 | 0.329 | 37.130 | 169.440 |
| Vanilla FuN | 11 | episode_3000 | 1.000 | 0.340 | 0.907 | 0.329 | 37.130 | 169.440 |
| Vanilla FuN | 44 | best | 0.150 | 0.000 | 0.093 | 0.000 | 235.210 | 250.000 |
| Vanilla FuN | 44 | last | 0.150 | 0.000 | 0.093 | 0.000 | 235.210 | 250.000 |
| Vanilla FuN | 44 | episode_3000 | 0.150 | 0.000 | 0.093 | 0.000 | 235.210 | 250.000 |
| AblationManager | 1 | best | 1.000 | 0.450 | 0.912 | 0.437 | 35.310 | 142.760 |
| AblationManager | 1 | last | 1.000 | 0.450 | 0.912 | 0.437 | 35.310 | 142.760 |
| AblationManager | 1 | episode_3000 | 1.000 | 0.450 | 0.912 | 0.437 | 35.310 | 142.760 |
| AblationManager | 11 | best | 0.990 | 0.040 | 0.880 | 0.038 | 46.370 | 240.880 |
| AblationManager | 11 | last | 0.990 | 0.040 | 0.880 | 0.038 | 46.370 | 240.880 |
| AblationManager | 11 | episode_3000 | 0.990 | 0.040 | 0.880 | 0.038 | 46.370 | 240.880 |
| AblationManager | 44 | best | 1.000 | 0.180 | 0.894 | 0.173 | 42.330 | 207.810 |
| AblationManager | 44 | last | 1.000 | 0.180 | 0.894 | 0.173 | 42.330 | 207.810 |
| AblationManager | 44 | episode_3000 | 1.000 | 0.180 | 0.894 | 0.173 | 42.330 | 207.810 |

## 4. Seed-Level Best Argmax Checkpoint

| Model | Seed | Best Argmax Checkpoint | Argmax Success | Sample Success |
|---|---:|---|---:|---:|
| Vanilla FuN | 1 | last / episode_3000 tie | 0.000 | 0.110 |
| Vanilla FuN | 11 | best / last / episode_3000 tie | 0.340 | 1.000 |
| Vanilla FuN | 44 | best / last / episode_3000 tie | 0.000 | 0.150 |
| AblationManager | 1 | best / last / episode_3000 tie | 0.450 | 1.000 |
| AblationManager | 11 | best / last / episode_3000 tie | 0.040 | 0.990 |
| AblationManager | 44 | best / last / episode_3000 tie | 0.180 | 1.000 |

## 5. Judgment

### Was the existing `best.pt` only good for sample?

Partly, but this recheck shows the issue is not simply that `best.pt` is a bad checkpoint compared with `last.pt` or `episode_3000.pt`.

For five of six runs, `best.pt`, `last.pt`, and `episode_3000.pt` have identical results because the sample-selected best checkpoint is already the final 3000-episode checkpoint. The exception is Vanilla FuN seed 1, where `best.pt` is episode 2500 and `last.pt` / `episode_3000.pt` are episode 3000. Even there, argmax success remains 0.000.

So `best.pt` is sample-selected, but replacing it with `last.pt` or `episode_3000.pt` does not fix argmax performance in the saved checkpoint set.

### Is `last.pt` or `episode_3000.pt` better for argmax?

No meaningful improvement was found.

- Vanilla FuN seed 1: `last.pt` and `episode_3000.pt` improve sample success from 0.080 to 0.110, but argmax stays 0.000.
- All other seeds: `best.pt`, `last.pt`, and `episode_3000.pt` are equivalent under both sample and argmax evaluation.

### Is checkpoint selection alone sufficient?

Checkpoint selection among the currently saved `best.pt`, `last.pt`, and `episode_3000.pt` is not sufficient.

However, the previous training runs did not log argmax eval at every interval. It is still possible that an intermediate checkpoint not saved in the current set had better argmax behavior. That is exactly why the newly added `best_argmax.pt` tracking is necessary for future runs.

The current evidence says:

- argmax-aware checkpoint selection is necessary for diagnosis and fair reporting
- but checkpoint selection alone is unlikely to be sufficient for robust argmax success

### Is low-entropy fine-tuning or longer training needed?

Likely yes, especially for AblationManager.

AblationManager is already near solved under sample evaluation:

- seed 1 sample success: 1.000
- seed 11 sample success: 0.990
- seed 44 sample success: 1.000

But argmax remains much lower:

- seed 1 argmax success: 0.450
- seed 11 argmax success: 0.040
- seed 44 argmax success: 0.180

This pattern is consistent with a high-entropy stochastic policy that succeeds through sampling but does not have a reliable greedy trajectory. Longer training may help, but without reducing stochasticity or selecting by argmax, it may keep producing policies that are good under sample and weak under argmax.

## 6. Recommendation

Priority:

1. A. Re-train with `best_argmax.pt` checkpoint selection enabled
2. B. AblationManager low entropy fine-tuning
3. D. Add entropy decay
4. C. Extend to 5000 episodes

Reasoning:

- A should be first because future runs need `best_argmax.pt` to answer the checkpoint-selection question correctly. This is now implemented, but it has not yet been used in a full DoorKey-6x6 run.
- B is the most targeted next experiment because AblationManager already solves sample evaluation. The remaining problem is deterministic action selection, so a low-entropy fine-tuning phase is more directly aligned than simply increasing the episode budget.
- D is promising but changes the training schedule more broadly. It should be considered after the simpler low-entropy fine-tuning check.
- C should not be first. The current 3000-episode results already show excellent sample performance for AblationManager but weak argmax performance. More episodes alone may not solve the deterministic-policy issue.

Recommended next concrete experiment:

- Start from AblationManager DoorKey-6x6 3000 checkpoints.
- Run a short low-entropy fine-tuning experiment with dual eval enabled.
- Save and compare `best_sample.pt` and `best_argmax.pt`.
- Prioritize seed 11 first because it has the largest sample/argmax gap: sample 0.990 vs argmax 0.040.

## 7. Verification Notes

Evaluation was run on GCP only. Local work after evaluation was limited to copying the new JSON outputs back into `logs/argmax_diagnosis/` and writing this summary.

No long 3000/5000 episode training was run in this step.
