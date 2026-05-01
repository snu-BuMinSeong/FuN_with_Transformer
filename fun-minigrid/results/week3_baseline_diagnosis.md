# Week 3 Baseline Diagnosis

## 요약
- train에서는 성공/reward가 있었고 sample checkpoint eval도 성공하지만 argmax eval은 0입니다.
- trained sample 평균 success=0.417, untrained sample success=0.140.
- 최종 판단: 2. evaluation argmax 방식 문제가 큼
- baseline 공식 평가는 sample 기준으로 전환하고, argmax는 참고 지표로 유지합니다.
- 4주차 memory ablation도 sample success/return/episode length를 primary metric으로 비교해야 합니다.

## Baseline Eval 결과

| seed | eval action mode | eval best success | eval final success | final return | final length |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | sample | 0.4000 | 0.3500 | 0.1297 | 223.7000 |
| 11 | sample | 0.5000 | 0.4500 | 0.2565 | 191.2500 |
| 44 | sample | 0.6500 | 0.5000 | 0.3022 | 179.9500 |

## Train Log 분석

| seed | rows | train success | success rate | final50 success | reward min/max/mean | entropy mean | grad mean | action coverage mean |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 1000 | 164 | 0.1640 | 0.5000 | 0.0000/0.9424/0.0771 | 1.9340 | 0.0902 | 0.9997 |
- seed 1 action distribution: dominant=2, dominant_fraction=0.1846, collapsed=False, counts=[30903, 31131, 43049, 32226, 31909, 34228, 29698]
| 11 | 1000 | 205 | 0.2050 | 0.3600 | 0.0000/0.9532/0.0921 | 1.9341 | 0.1113 | 0.9997 |
- seed 11 action distribution: dominant=2, dominant_fraction=0.1749, collapsed=False, counts=[29063, 33957, 40238, 34764, 31955, 30081, 30067]
| 44 | 1000 | 181 | 0.1810 | 0.3800 | 0.0000/0.9532/0.0881 | 1.9299 | 0.1129 | 0.9996 |
- seed 44 action distribution: dominant=2, dominant_fraction=0.1642, collapsed=False, counts=[32966, 32259, 37859, 33440, 30299, 33231, 30501]

## Sample Evaluation 결과

| seed | argmax checkpoint success | sample success | sample return | sample length |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.0000 | 0.3200 | 0.1274 | 223.5100 |
| 11 | 0.0000 | 0.4800 | 0.2439 | 195.5700 |
| 44 | 0.0000 | 0.4500 | 0.2378 | 196.4500 |

## Random / Untrained Baseline

| policy | success | mean return | mean length |
| --- | ---: | ---: | ---: |
| random_policy | 0.0700 | 0.0323 | 242.9700 |
| untrained_fun_sample | 0.1400 | 0.0527 | 239.2400 |
| untrained_fun_argmax | 0.0000 | 0.0000 | 250.0000 |

## Empty-5x5 Sanity Check
- summary best_success_rate: 0.0000
- argmax best checkpoint: {"success_rate": 0.0, "mean_return": 0.0, "mean_episode_length": 100.0, "min_return": 0.0, "max_return": 0.0, "episodes": 100}
- sample best checkpoint: {"success_rate": 0.73, "mean_return": 0.46251999999999993, "mean_episode_length": 56.72, "min_return": 0.0, "max_return": 0.946, "episodes": 100}

## Warning 목록
- 자동 감지 warning 없음

## 결론
- 2. evaluation argmax 방식 문제가 큼
- 근거: train에서는 성공/reward가 있었고 sample checkpoint eval도 성공하지만 argmax eval은 0입니다.
- 근거: trained sample 평균 success=0.417, untrained sample success=0.140.
- baseline 공식 평가는 sample 기준으로 전환합니다.
- argmax evaluation은 참고 지표로 유지합니다.
- 4주차 memory ablation 비교도 sample 기준으로 수행해야 합니다.

## 다음 액션 제안
- DoorKey는 argmax와 sample 평가를 계속 병행해서 기록
- Empty-5x5 sanity 결과가 양호하면 DoorKey curriculum 또는 reward shaping 검토
- sample도 실패하면 checkpoint 저장 시점, recurrent reset, loss/return 연결을 우선 점검
- sparse reward 대응으로 intrinsic reward 또는 FuN worker intrinsic reward는 별도 실험으로 분리
