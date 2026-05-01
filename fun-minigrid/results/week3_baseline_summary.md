# Week 3 Baseline Summary

## 실험 목적

Vanilla FuN을 memory ablation 실험의 비교 기준으로 고정한다.

## 중요한 수정 사항

기존 argmax evaluation은 모든 seed에서 success가 0에 가깝게 나왔지만, sample evaluation에서는 학습 성능이 확인되었다.
따라서 본 프로젝트에서는 stochastic policy 특성을 고려하여 sample evaluation을 공식 baseline 평가 기준으로 사용하고, argmax evaluation은 참고 지표로 둔다.

## 실험 환경

- MiniGrid-DoorKey-5x5-v0
- total episodes: 1000
- eval interval: 50
- eval episodes: 20
- official evaluation action mode: sample
- reference evaluation action mode: argmax

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

| Seed | Sample Success Rate | Sample Mean Return | Sample Episode Length | Argmax Success Rate | Argmax Mean Return | Argmax Episode Length | Best Checkpoint |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.320000 | 0.127364 | 223.510000 | 0.000000 | 0.000000 | 250.000000 | checkpoints/baseline_fun/seed_1/best.pt |
| 11 | 0.480000 | 0.243948 | 195.570000 | 0.000000 | 0.000000 | 250.000000 | checkpoints/baseline_fun/seed_11/best.pt |
| 44 | 0.450000 | 0.237780 | 196.450000 | 0.000000 | 0.000000 | 250.000000 | checkpoints/baseline_fun/seed_44/best.pt |
| Mean | 0.416667 | 0.203031 | 205.176667 | 0.000000 | 0.000000 | 250.000000 | - |

## 해석

- 학습은 완전히 실패한 것이 아니다.
- train success와 sample checkpoint evaluation success가 확인되므로 policy distribution은 task-relevant behavior를 일부 학습했다.
- argmax는 특정 action 반복 또는 행동 다양성 제거로 인해 DoorKey task에서 실패하는 것으로 해석된다.
- 따라서 4주차 memory ablation 비교에서도 sample evaluation을 primary metric으로 사용해야 한다.
- seed별 편차가 존재하므로 평균뿐 아니라 seed별 결과를 함께 보고해야 한다.

## 그래프

- `figures/baseline_fun/sample_success_rate_by_seed.png`
- `figures/baseline_fun/sample_mean_return_by_seed.png`
- `figures/baseline_fun/sample_episode_length_by_seed.png`
- `figures/baseline_fun/argmax_vs_sample_success_rate.png`
- `figures/baseline_fun/argmax_eval_success_rate_curve.png`

## 한계

- seed별 편차가 크다.
- DoorKey sparse reward 특성상 안정적인 학습은 아직 어렵다.
- argmax deterministic policy 성능은 낮거나 0이다.
- sample evaluation은 stochastic하므로 평가 episode 수에 따른 분산이 있다.

## 4주차 비교 기준

primary:

- sample success rate
- sample mean return
- sample mean episode length

secondary:

- argmax success rate
- train stability
- seed variance
