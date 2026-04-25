# Week 3 Baseline Summary

## 실험 목적

Vanilla FuN을 memory ablation 실험의 비교 기준으로 고정한다.

## 실험 환경

- MiniGrid-DoorKey-5x5-v0
- total episodes: TBD
- eval interval: TBD
- eval episodes: TBD
- action mode for evaluation: TBD

## 모델 구조

- Encoder: TBD
- GRU-based Manager: TBD
- Worker: TBD
- Value head: TBD
- Manager value head: TBD

## 사용 config

- seed 1 config path: `configs/train_fun_baseline_seed1.yaml`
- seed 11 config path: `configs/train_fun_baseline_seed11.yaml`
- seed 44 config path: `configs/train_fun_baseline_seed44.yaml`

## Seed별 결과

| Seed | Success Rate | Average Return | Episode Length | Best Checkpoint |
|---:|---:|---:|---:|---|
| 1 | TBD | TBD | TBD | `checkpoints/baseline_fun/seed_1/best.pt` |
| 11 | TBD | TBD | TBD | `checkpoints/baseline_fun/seed_11/best.pt` |
| 44 | TBD | TBD | TBD | `checkpoints/baseline_fun/seed_44/best.pt` |
| Mean | TBD | TBD | TBD | - |

## 그래프 경로

- success rate: TBD
- mean return: TBD
- episode length: TBD

## 관찰

- 학습 안정성:
- sparse reward 한계:
- seed별 편차:
- 4주차 memory ablation 비교 기준:
