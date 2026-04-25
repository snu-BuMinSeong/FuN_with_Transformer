# Week 3 Baseline Summary

## 실험 목적

Vanilla FuN을 memory ablation 실험의 비교 기준으로 고정한다.

## 실험 환경

- MiniGrid-DoorKey-5x5-v0
- total episodes: 1000
- eval interval: 50
- eval episodes: 20
- evaluation action mode: argmax

## 모델 구조

- Observation Encoder
- GRU-based Manager
- Worker
- Value head
- Manager value head

## 사용 Config

- `configs/train_fun_baseline_seed1.yaml`
- `configs/train_fun_baseline_seed11.yaml`
- `configs/train_fun_baseline_seed44.yaml`

## Seed별 결과

| Seed | Final Success Rate | Final Mean Return | Final Episode Length | Best Success Rate | Best Checkpoint |
|---:|---:|---:|---:|---:|---|
| 1 | 0.000000 | 0.000000 | 250.000000 | 0.000000 | checkpoints/baseline_fun/seed_1/best.pt |
| 11 | 0.000000 | 0.000000 | 250.000000 | 0.000000 | checkpoints/baseline_fun/seed_11/best.pt |
| 44 | 0.000000 | 0.000000 | 250.000000 | 0.000000 | checkpoints/baseline_fun/seed_44/best.pt |
| Mean | 0.000000 | 0.000000 | 250.000000 | 0.000000 | - |

## 그래프

- `figures/baseline_fun/eval_success_rate.png`
- `figures/baseline_fun/eval_mean_return.png`
- `figures/baseline_fun/eval_episode_length.png`

## 관찰

- 학습 안정성:
- sparse reward 한계:
- seed별 편차:
- 4주차 memory ablation 비교 기준:
