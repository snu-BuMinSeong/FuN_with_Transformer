# Week 5 DoorKey-6x6 Two-Stage V2 Results

## 1. Run Status

The two-stage DoorKey-6x6 experiment completed on the GCP server.

Workspace:

- `/home/fumin0193/FuN_with_Transformer/two_stage_doorkey6x6_workspace`

Local copied logs:

- `logs/two_stage_doorkey6x6_v2`
- `logs/gcp_run_logs/two_stage_doorkey6x6_v2_gpu`

Execution note:

- Stage 1 baseline seed 1 and seed 11 completed before GPU driver recovery.
- The remaining Stage 1 runs, all Stage 2 fine-tuning runs, and all final 100ep evaluations completed after the Tesla T4 GPU was enabled.
- GPU was unavailable initially because the NVIDIA kernel module was missing for the active GCP cloud kernel. Installing `linux-headers-cloud-amd64` rebuilt the NVIDIA DKMS module and restored CUDA.

Final orchestrator status:

- `ALL_DONE 2026-05-03T20:16:02+00:00`

## 2. Output Checks

Expected training/eval rows were present after copying logs locally:

| Model | Seed | Stage | Train Rows | Eval Rows | Summary Final Episode |
|---|---:|---|---:|---:|---:|
| Vanilla FuN | 1 | Stage 1 5000 | 5000 | 100 | 5000 |
| Vanilla FuN | 1 | Stage 2 1000 | 1000 | 20 | 1000 |
| Vanilla FuN | 11 | Stage 1 5000 | 5000 | 100 | 5000 |
| Vanilla FuN | 11 | Stage 2 1000 | 1000 | 20 | 1000 |
| Vanilla FuN | 44 | Stage 1 5000 | 5000 | 100 | 5000 |
| Vanilla FuN | 44 | Stage 2 1000 | 1000 | 20 | 1000 |
| AblationManager | 1 | Stage 1 5000 | 5000 | 100 | 5000 |
| AblationManager | 1 | Stage 2 1000 | 1000 | 20 | 1000 |
| AblationManager | 11 | Stage 1 5000 | 5000 | 100 | 5000 |
| AblationManager | 11 | Stage 2 1000 | 1000 | 20 | 1000 |
| AblationManager | 44 | Stage 1 5000 | 5000 | 100 | 5000 |
| AblationManager | 44 | Stage 2 1000 | 1000 | 20 | 1000 |

Other checks:

- Final 100ep evaluation JSON files: 36 present.
- No `nan` or `inf` strings were found in copied CSV logs.
- Required checkpoint files are present on the GCP server under `checkpoints/two_stage_doorkey6x6_v2`.

## 3. Final 100ep Evaluation

Stage 2 checkpoints were evaluated with sample and argmax action selection for 100 episodes.

| Model | Seed | Checkpoint | Sample Success | Argmax Success | Sample Return | Argmax Return | Sample Length | Argmax Length |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Vanilla FuN | 1 | best_sample.pt | 1.000 | 0.910 | 0.962 | 0.875 | 15.010 | 36.330 |
| Vanilla FuN | 1 | best_argmax.pt | 1.000 | 0.940 | 0.956 | 0.911 | 17.710 | 26.570 |
| Vanilla FuN | 1 | last.pt | 1.000 | 0.910 | 0.962 | 0.875 | 15.010 | 36.330 |
| Vanilla FuN | 11 | best_sample.pt | 1.000 | 0.940 | 0.961 | 0.911 | 15.790 | 26.500 |
| Vanilla FuN | 11 | best_argmax.pt | 1.000 | 0.940 | 0.959 | 0.911 | 16.580 | 26.570 |
| Vanilla FuN | 11 | last.pt | 1.000 | 0.940 | 0.961 | 0.911 | 15.790 | 26.500 |
| Vanilla FuN | 44 | best_sample.pt | 1.000 | 0.950 | 0.960 | 0.913 | 16.090 | 27.440 |
| Vanilla FuN | 44 | best_argmax.pt | 1.000 | 0.950 | 0.960 | 0.913 | 16.090 | 27.440 |
| Vanilla FuN | 44 | last.pt | 1.000 | 0.950 | 0.960 | 0.913 | 16.090 | 27.440 |
| AblationManager | 1 | best_sample.pt | 1.000 | 0.940 | 0.961 | 0.911 | 15.580 | 26.530 |
| AblationManager | 1 | best_argmax.pt | 1.000 | 0.980 | 0.962 | 0.950 | 15.360 | 17.080 |
| AblationManager | 1 | last.pt | 1.000 | 0.940 | 0.961 | 0.911 | 15.580 | 26.530 |
| AblationManager | 11 | best_sample.pt | 1.000 | 0.980 | 0.958 | 0.946 | 16.900 | 18.580 |
| AblationManager | 11 | best_argmax.pt | 1.000 | 1.000 | 0.958 | 0.965 | 16.700 | 13.910 |
| AblationManager | 11 | last.pt | 1.000 | 0.980 | 0.958 | 0.946 | 16.900 | 18.580 |
| AblationManager | 44 | best_sample.pt | 1.000 | 0.850 | 0.960 | 0.823 | 16.140 | 48.330 |
| AblationManager | 44 | best_argmax.pt | 1.000 | 0.850 | 0.960 | 0.824 | 15.850 | 47.750 |
| AblationManager | 44 | last.pt | 1.000 | 0.850 | 0.960 | 0.823 | 16.140 | 48.330 |

## 4. Best-by-Argmax Summary

For each seed, the best result is selected by highest 100ep argmax success among `best_sample.pt`, `best_argmax.pt`, and `last.pt`.

| Model | Seed | Best Checkpoint | Sample Success | Argmax Success | Sample Return | Argmax Return | Argmax Length |
|---|---:|---|---:|---:|---:|---:|---:|
| Vanilla FuN | 1 | best_argmax.pt | 1.000 | 0.940 | 0.956 | 0.911 | 26.570 |
| Vanilla FuN | 11 | best_sample.pt | 1.000 | 0.940 | 0.961 | 0.911 | 26.500 |
| Vanilla FuN | 44 | best_sample.pt | 1.000 | 0.950 | 0.960 | 0.913 | 27.440 |
| AblationManager | 1 | best_argmax.pt | 1.000 | 0.980 | 0.962 | 0.950 | 17.080 |
| AblationManager | 11 | best_argmax.pt | 1.000 | 1.000 | 0.958 | 0.965 | 13.910 |
| AblationManager | 44 | best_sample.pt | 1.000 | 0.850 | 0.960 | 0.823 | 48.330 |

Mean best-by-argmax results:

| Model | Mean Sample Success | Mean Argmax Success | Mean Sample Return | Mean Argmax Return | Mean Argmax Length |
|---|---:|---:|---:|---:|---:|
| Vanilla FuN | 1.000 | 0.943 | 0.959 | 0.912 | 26.837 |
| AblationManager | 1.000 | 0.943 | 0.960 | 0.913 | 26.440 |

## 5. Interpretation

The standardized two-stage protocol substantially raises deterministic DoorKey-6x6 performance for both models.

Compared with the earlier 3000-episode results, Vanilla FuN no longer shows the same cross-seed failure after Stage 1 5000 plus Stage 2 low-entropy fine-tuning. All three Vanilla FuN seeds reached sample success 1.000 and argmax success at least 0.940.

AblationManager also reached sample success 1.000 across all seeds. Its best seed results are very strong, including seed 11 argmax success 1.000. Seed 44 remains the weakest AblationManager deterministic case at 0.850, but still well above the earlier FT1000 result.

The final mean argmax success is tied at 0.943 for Vanilla FuN and AblationManager under this two-stage protocol. AblationManager has a slightly shorter mean argmax episode length, but the difference is small in this final setting.

The key conclusion changes from the earlier 3000-budget comparison: AblationManager is more sample-efficient and stable at 3000 episodes, but under a larger standardized 5000 + 1000 two-stage budget, both models can reach strong deterministic DoorKey-6x6 performance.
