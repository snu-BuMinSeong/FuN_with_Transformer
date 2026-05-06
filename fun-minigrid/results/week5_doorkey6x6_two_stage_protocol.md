# DoorKey-6x6 Two-Stage Full Protocol

## 1. Scope

Models:

- Vanilla FuN (`manager_type: recurrent`)
- AblationManager (`manager_type: ablation`)

Environment:

- `MiniGrid-DoorKey-6x6-v0`

Seeds:

- 1, 11, 44

This protocol standardizes the final DoorKey-6x6 experiment into two stages for both models and all seeds.

## 2. Stage 1: Task Learning

Stage 1 trains each model from scratch for 5000 episodes with the original stochastic training setting.

Settings:

- `total_episodes: 5000`
- `entropy_coef: 0.01`
- `learning_rate: 0.0003`
- `action_mode: sample`
- `eval_action_modes: sample, argmax`
- `eval_interval: 50`
- `eval_episodes: 50`

Stage 1 configs:

| Model | Seed | Config |
|---|---:|---|
| Vanilla FuN | 1 | `configs/train_fun_baseline_doorkey6x6_seed1_stage1_5000.yaml` |
| Vanilla FuN | 11 | `configs/train_fun_baseline_doorkey6x6_seed11_stage1_5000.yaml` |
| Vanilla FuN | 44 | `configs/train_fun_baseline_doorkey6x6_seed44_stage1_5000.yaml` |
| AblationManager | 1 | `configs/train_fun_ablation_doorkey6x6_seed1_stage1_5000.yaml` |
| AblationManager | 11 | `configs/train_fun_ablation_doorkey6x6_seed11_stage1_5000.yaml` |
| AblationManager | 44 | `configs/train_fun_ablation_doorkey6x6_seed44_stage1_5000.yaml` |

Stage 1 output paths:

| Model | Seed | Log Dir | Checkpoint Dir |
|---|---:|---|---|
| Vanilla FuN | 1 | `logs/two_stage_doorkey6x6/baseline/seed_1/stage1_5000` | `checkpoints/two_stage_doorkey6x6/baseline/seed_1/stage1_5000` |
| Vanilla FuN | 11 | `logs/two_stage_doorkey6x6/baseline/seed_11/stage1_5000` | `checkpoints/two_stage_doorkey6x6/baseline/seed_11/stage1_5000` |
| Vanilla FuN | 44 | `logs/two_stage_doorkey6x6/baseline/seed_44/stage1_5000` | `checkpoints/two_stage_doorkey6x6/baseline/seed_44/stage1_5000` |
| AblationManager | 1 | `logs/two_stage_doorkey6x6/ablation/seed_1/stage1_5000` | `checkpoints/two_stage_doorkey6x6/ablation/seed_1/stage1_5000` |
| AblationManager | 11 | `logs/two_stage_doorkey6x6/ablation/seed_11/stage1_5000` | `checkpoints/two_stage_doorkey6x6/ablation/seed_11/stage1_5000` |
| AblationManager | 44 | `logs/two_stage_doorkey6x6/ablation/seed_44/stage1_5000` | `checkpoints/two_stage_doorkey6x6/ablation/seed_44/stage1_5000` |

## 3. Stage 2: Low-Entropy Fine-Tuning

Stage 2 resumes from the Stage 1 `best_sample.pt` checkpoint and fine-tunes for deterministic policy improvement.

Settings:

- resume source: Stage 1 `best_sample.pt`
- `total_episodes: 1000`
- `entropy_coef: 0.003`
- `learning_rate: 0.0001`
- `action_mode: sample`
- `eval_action_modes: sample, argmax`
- `eval_interval: 50`
- `eval_episodes: 50`
- `resume_optimizer_state: false`

Stage 2 configs:

| Model | Seed | Config | Resume Checkpoint |
|---|---:|---|---|
| Vanilla FuN | 1 | `configs/train_fun_baseline_doorkey6x6_seed1_stage2_argmax_ft_1000.yaml` | `checkpoints/two_stage_doorkey6x6/baseline/seed_1/stage1_5000/best_sample.pt` |
| Vanilla FuN | 11 | `configs/train_fun_baseline_doorkey6x6_seed11_stage2_argmax_ft_1000.yaml` | `checkpoints/two_stage_doorkey6x6/baseline/seed_11/stage1_5000/best_sample.pt` |
| Vanilla FuN | 44 | `configs/train_fun_baseline_doorkey6x6_seed44_stage2_argmax_ft_1000.yaml` | `checkpoints/two_stage_doorkey6x6/baseline/seed_44/stage1_5000/best_sample.pt` |
| AblationManager | 1 | `configs/train_fun_ablation_doorkey6x6_seed1_stage2_argmax_ft_1000.yaml` | `checkpoints/two_stage_doorkey6x6/ablation/seed_1/stage1_5000/best_sample.pt` |
| AblationManager | 11 | `configs/train_fun_ablation_doorkey6x6_seed11_stage2_argmax_ft_1000.yaml` | `checkpoints/two_stage_doorkey6x6/ablation/seed_11/stage1_5000/best_sample.pt` |
| AblationManager | 44 | `configs/train_fun_ablation_doorkey6x6_seed44_stage2_argmax_ft_1000.yaml` | `checkpoints/two_stage_doorkey6x6/ablation/seed_44/stage1_5000/best_sample.pt` |

Stage 2 output paths:

| Model | Seed | Log Dir | Checkpoint Dir |
|---|---:|---|---|
| Vanilla FuN | 1 | `logs/two_stage_doorkey6x6/baseline/seed_1/stage2_argmax_ft_1000` | `checkpoints/two_stage_doorkey6x6/baseline/seed_1/stage2_argmax_ft_1000` |
| Vanilla FuN | 11 | `logs/two_stage_doorkey6x6/baseline/seed_11/stage2_argmax_ft_1000` | `checkpoints/two_stage_doorkey6x6/baseline/seed_11/stage2_argmax_ft_1000` |
| Vanilla FuN | 44 | `logs/two_stage_doorkey6x6/baseline/seed_44/stage2_argmax_ft_1000` | `checkpoints/two_stage_doorkey6x6/baseline/seed_44/stage2_argmax_ft_1000` |
| AblationManager | 1 | `logs/two_stage_doorkey6x6/ablation/seed_1/stage2_argmax_ft_1000` | `checkpoints/two_stage_doorkey6x6/ablation/seed_1/stage2_argmax_ft_1000` |
| AblationManager | 11 | `logs/two_stage_doorkey6x6/ablation/seed_11/stage2_argmax_ft_1000` | `checkpoints/two_stage_doorkey6x6/ablation/seed_11/stage2_argmax_ft_1000` |
| AblationManager | 44 | `logs/two_stage_doorkey6x6/ablation/seed_44/stage2_argmax_ft_1000` | `checkpoints/two_stage_doorkey6x6/ablation/seed_44/stage2_argmax_ft_1000` |

## 4. GCP Run Commands

Stage 1 can be launched in parallel:

```bash
mkdir -p logs/gcp_run_logs/two_stage_doorkey6x6

nohup python train.py --config configs/train_fun_baseline_doorkey6x6_seed1_stage1_5000.yaml > logs/gcp_run_logs/two_stage_doorkey6x6/baseline_seed1_stage1.out 2>&1 &
nohup python train.py --config configs/train_fun_baseline_doorkey6x6_seed11_stage1_5000.yaml > logs/gcp_run_logs/two_stage_doorkey6x6/baseline_seed11_stage1.out 2>&1 &
nohup python train.py --config configs/train_fun_baseline_doorkey6x6_seed44_stage1_5000.yaml > logs/gcp_run_logs/two_stage_doorkey6x6/baseline_seed44_stage1.out 2>&1 &
nohup python train.py --config configs/train_fun_ablation_doorkey6x6_seed1_stage1_5000.yaml > logs/gcp_run_logs/two_stage_doorkey6x6/ablation_seed1_stage1.out 2>&1 &
nohup python train.py --config configs/train_fun_ablation_doorkey6x6_seed11_stage1_5000.yaml > logs/gcp_run_logs/two_stage_doorkey6x6/ablation_seed11_stage1.out 2>&1 &
nohup python train.py --config configs/train_fun_ablation_doorkey6x6_seed44_stage1_5000.yaml > logs/gcp_run_logs/two_stage_doorkey6x6/ablation_seed44_stage1.out 2>&1 &
wait
```

Stage 2 should be launched only after all Stage 1 `best_sample.pt` checkpoints exist:

```bash
nohup python train.py --config configs/train_fun_baseline_doorkey6x6_seed1_stage2_argmax_ft_1000.yaml > logs/gcp_run_logs/two_stage_doorkey6x6/baseline_seed1_stage2.out 2>&1 &
nohup python train.py --config configs/train_fun_baseline_doorkey6x6_seed11_stage2_argmax_ft_1000.yaml > logs/gcp_run_logs/two_stage_doorkey6x6/baseline_seed11_stage2.out 2>&1 &
nohup python train.py --config configs/train_fun_baseline_doorkey6x6_seed44_stage2_argmax_ft_1000.yaml > logs/gcp_run_logs/two_stage_doorkey6x6/baseline_seed44_stage2.out 2>&1 &
nohup python train.py --config configs/train_fun_ablation_doorkey6x6_seed1_stage2_argmax_ft_1000.yaml > logs/gcp_run_logs/two_stage_doorkey6x6/ablation_seed1_stage2.out 2>&1 &
nohup python train.py --config configs/train_fun_ablation_doorkey6x6_seed11_stage2_argmax_ft_1000.yaml > logs/gcp_run_logs/two_stage_doorkey6x6/ablation_seed11_stage2.out 2>&1 &
nohup python train.py --config configs/train_fun_ablation_doorkey6x6_seed44_stage2_argmax_ft_1000.yaml > logs/gcp_run_logs/two_stage_doorkey6x6/ablation_seed44_stage2.out 2>&1 &
wait
```

If CPU or GPU memory is insufficient, run Stage 1 and Stage 2 sequentially by seed while keeping the same output paths.

## 5. Final Evaluation

For each Stage 2 run, evaluate:

- `best_sample.pt`
- `best_argmax.pt`
- `last.pt`

with:

- sample 100 episodes
- argmax 100 episodes

Example for Vanilla FuN seed 1:

```bash
python evaluate_checkpoint.py --config configs/train_fun_baseline_doorkey6x6_seed1_stage2_argmax_ft_1000.yaml --checkpoint checkpoints/two_stage_doorkey6x6/baseline/seed_1/stage2_argmax_ft_1000/best_sample.pt --output logs/two_stage_doorkey6x6/baseline/seed_1/stage2_argmax_ft_1000/eval_best_sample_sample_100ep.json --episodes 100 --action-mode sample

python evaluate_checkpoint.py --config configs/train_fun_baseline_doorkey6x6_seed1_stage2_argmax_ft_1000.yaml --checkpoint checkpoints/two_stage_doorkey6x6/baseline/seed_1/stage2_argmax_ft_1000/best_sample.pt --output logs/two_stage_doorkey6x6/baseline/seed_1/stage2_argmax_ft_1000/eval_best_sample_argmax_100ep.json --episodes 100 --action-mode argmax
```

The same pattern should be repeated for `best_argmax.pt` and `last.pt`, for all models and seeds.

## 6. Notes Relative to Existing Week 5 Results

This protocol is stricter and more standardized than the existing Week 5 runs.

Already completed results do not exactly match this protocol:

- AblationManager FT1000 resumed from 3000-episode checkpoints, not from Stage 1 5000 `best_sample.pt`.
- Baseline seed 1/44 5000 runs resumed from 3000 checkpoints for an additional 2000 episodes, not from scratch for 5000 episodes.
- Baseline seed 11 FT pilot resumed from its 3000 checkpoint, not from Stage 1 5000 `best_sample.pt`.

Therefore, the two-stage protocol should be treated as a new final standardized experiment if it is executed.
