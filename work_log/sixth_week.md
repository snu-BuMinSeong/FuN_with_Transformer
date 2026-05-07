# 6주차 작업 로그

## 작업 일자

- 주 작업일: 2026-05-06
- 결과 정리 및 로그 회수: 2026-05-07

## 작업 목표

DoorKey-6x6 two-stage 실험에서는 Vanilla FuN과 AblationManager가 모두 강한 성능에 도달했다. 따라서 recurrent Manager memory의 효과를 더 잘 확인하기 위해 MultiRoom 계열 환경으로 확장했다.

핵심 질문:

```text
긴 탐색 경로와 여러 방/문 구조를 갖는 MultiRoom 환경에서
Manager recurrent memory가 AblationManager 대비 유의미한 이점을 주는가?
```

## 1. MultiRoom 확장 동기 문서화

작성한 문서:

- `fun-minigrid/results/week6_multiroom_motivation.md`

정리한 내용:

- DoorKey-6x6 two-stage V2에서는 두 모델 모두 평균 sample success `1.000`, 평균 argmax success `0.943`에 도달했다.
- DoorKey-6x6만으로는 recurrent Manager memory의 필요성을 강하게 주장하기 어렵다.
- MultiRoom은 여러 방과 문을 순차적으로 통과해야 하므로 더 긴 horizon과 더 강한 partial history dependency를 제공한다.

## 2. MultiRoom-N2-S4 smoke test

대상 환경:

- `MiniGrid-MultiRoom-N2-S4-v0`

작성한 문서:

- `fun-minigrid/results/week6_multiroom_smoke.md`

확인 결과:

| 항목 | 값 |
|---|---:|
| raw observation shape | `(7, 7, 3)` |
| preprocessed observation shape | `(3, 7, 7)` |
| action space | 7 |
| max steps | 40 |
| FuNModel forward | Pass |

10 episode baseline:

| Policy | Success Rate | Mean Return | Mean Length |
|---|---:|---:|---:|
| Random | 0.100 | 0.019 | 39.600 |
| Untrained FuN sample | 0.000 | 0.000 | 40.000 |
| Untrained FuN argmax | 0.000 | 0.000 | 40.000 |

해석:

- 기존 wrapper, preprocessing, FuN model, rollout loop와 호환된다.
- random policy에서 10 episode 중 1회 성공이 나와 reward signal 자체는 막혀 있지 않다.

## 3. MultiRoom-N2-S4 seed 1 config 구성

생성한 Stage 1 config:

- `fun-minigrid/configs/train_fun_baseline_multiroom_n2s4_seed1.yaml`
- `fun-minigrid/configs/train_fun_ablation_multiroom_n2s4_seed1.yaml`

생성한 Stage 2 config:

- `fun-minigrid/configs/finetune_fun_baseline_multiroom_n2s4_seed1.yaml`
- `fun-minigrid/configs/finetune_fun_ablation_multiroom_n2s4_seed1.yaml`

공통 설정:

| 항목 | Stage 1 | Stage 2 |
|---|---:|---:|
| total episodes | 5000 | 1000 |
| eval interval | 100 | 50 |
| eval episodes | 50 | 50 |
| learning rate | 0.0003 | 0.0001 |
| entropy coef | 0.01 | 0.003 |
| eval modes | sample, argmax | sample, argmax |

## 4. GCP MultiRoom-N2-S4 seed 1 실행

GCP 인스턴스:

- `instance-20260502-111554`
- zone: `asia-east2-c`
- GPU: Tesla T4

작성한 실행 스크립트:

- `fun-minigrid/scripts/run_week6_multiroom_seed1_gcp.sh`
- `fun-minigrid/scripts/evaluate_week6_multiroom_seed1_gcp.sh`

로컬로 회수한 로그:

- `fun-minigrid/logs/two_stage_multiroom_n2s4`
- `fun-minigrid/logs/gcp_run_logs/week6_multiroom_seed1`

실행 완료 항목:

- Vanilla FuN Stage 1 5000 episodes
- AblationManager Stage 1 5000 episodes
- Vanilla FuN Stage 2 1000 episodes
- AblationManager Stage 2 1000 episodes
- Stage 2 checkpoint final 100-episode evaluation

row 수 확인:

| Model | Stage | Train Rows | Eval Rows | Final Episode |
|---|---|---:|---:|---:|
| Vanilla FuN | Stage 1 | 5000 | 50 | 5000 |
| Vanilla FuN | Stage 2 | 1000 | 20 | 1000 |
| AblationManager | Stage 1 | 5000 | 50 | 5000 |
| AblationManager | Stage 2 | 1000 | 20 | 1000 |

Stage summary:

| Model | Stage | Reward Signal Episodes | Best Sample Success | Best Argmax Success | Final Sample Success | Final Argmax Success |
|---|---|---:|---:|---:|---:|---:|
| Vanilla FuN | Stage 1 | 3843 | 1.000 | 1.000 | 1.000 | 1.000 |
| Vanilla FuN | Stage 2 | 1000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AblationManager | Stage 1 | 3556 | 1.000 | 1.000 | 1.000 | 1.000 |
| AblationManager | Stage 2 | 1000 | 1.000 | 1.000 | 1.000 | 1.000 |

Stage 2 final 100-episode evaluation:

| Model | Checkpoint | Sample Success | Argmax Success | Sample Return | Argmax Return | Sample Length | Argmax Length |
|---|---|---:|---:|---:|---:|---:|---:|
| Vanilla FuN | best_sample.pt | 1.000 | 1.000 | 0.830 | 0.835 | 7.560 | 7.350 |
| Vanilla FuN | best_argmax.pt | 1.000 | 1.000 | 0.830 | 0.835 | 7.560 | 7.350 |
| Vanilla FuN | last.pt | 1.000 | 1.000 | 0.830 | 0.835 | 7.560 | 7.350 |
| AblationManager | best_sample.pt | 1.000 | 1.000 | 0.831 | 0.836 | 7.510 | 7.270 |
| AblationManager | best_argmax.pt | 1.000 | 1.000 | 0.831 | 0.836 | 7.510 | 7.270 |
| AblationManager | last.pt | 1.000 | 1.000 | 0.831 | 0.836 | 7.510 | 7.270 |

해석:

- MultiRoom-N2-S4는 두 모델 모두 쉽게 해결했다.
- Vanilla FuN의 recurrent Manager memory 우위는 관찰되지 않았다.
- AblationManager가 mean length에서 아주 근소하게 짧지만, 실질적인 차이라고 보기 어렵다.
- N2-S4는 DoorKey-6x6과 마찬가지로 memory 효과를 검증하기에는 쉬운 환경으로 판단했다.

## 5. 다음 환경: MultiRoom-N4-S5 구성

N2-S4가 너무 쉬웠기 때문에 더 어려운 후보로 `MiniGrid-MultiRoom-N4-S5-v0`를 선택했다.

작성한 smoke 문서:

- `fun-minigrid/results/week6_multiroom_n4s5_smoke.md`

N4-S5 smoke 결과:

| 항목 | 값 |
|---|---:|
| raw observation shape | `(7, 7, 3)` |
| preprocessed observation shape | `(3, 7, 7)` |
| action space | 7 |
| max steps | 120 |
| FuNModel forward | Pass |

20 episode baseline:

| Policy | Success Rate | Mean Return | Mean Length |
|---|---:|---:|---:|
| Random | 0.000 | 0.000 | 120.000 |
| Untrained FuN sample | 0.000 | 0.000 | 120.000 |
| Untrained FuN argmax | 0.000 | 0.000 | 120.000 |

해석:

- N4-S5는 현재 코드와 호환된다.
- N2-S4보다 훨씬 sparse하다.
- seed 1 pilot에서 reward signal이 나오는지 먼저 확인하고, signal이 있을 때만 Stage 2로 진행하는 gate를 두기로 했다.

## 6. MultiRoom-N4-S5 seed 1 pilot config 및 GCP 스크립트

생성한 Stage 1 config:

- `fun-minigrid/configs/train_fun_baseline_multiroom_n4s5_seed1.yaml`
- `fun-minigrid/configs/train_fun_ablation_multiroom_n4s5_seed1.yaml`

생성한 Stage 2 config:

- `fun-minigrid/configs/finetune_fun_baseline_multiroom_n4s5_seed1.yaml`
- `fun-minigrid/configs/finetune_fun_ablation_multiroom_n4s5_seed1.yaml`

생성한 GCP 스크립트:

- `fun-minigrid/scripts/run_week6_multiroom_n4s5_seed1_pilot_gcp.sh`
- `fun-minigrid/scripts/evaluate_week6_multiroom_n4s5_seed1_gcp.sh`

pilot 실행 정책:

- baseline Stage 1 실행
- ablation Stage 1 실행
- 각 Stage 1 `summary.json`에서 reward signal 확인
- reward signal이 있는 모델만 Stage 2 실행
- Stage 2가 실행된 경우에만 final 100-episode evaluation 실행

## 7. GCP MultiRoom-N4-S5 seed 1 pilot 결과

로컬로 회수한 로그:

- `fun-minigrid/logs/two_stage_multiroom_n4s5`
- `fun-minigrid/logs/gcp_run_logs/week6_multiroom_n4s5_seed1_pilot`

실행 상태:

- 2026-05-07 확인 시 GCP tmux 세션은 종료되어 있었다.
- Stage 1 summary는 baseline/ablation 모두 생성되었다.
- Stage 2 summary는 생성되지 않았다.
- final 100-episode evaluation 파일도 생성되지 않았다.

row 수 확인:

| Model | Stage | Train Rows | Eval Rows | Final Episode |
|---|---|---:|---:|---:|
| Vanilla FuN | Stage 1 | 5000 | 50 | 5000 |
| AblationManager | Stage 1 | 5000 | 50 | 5000 |

Stage 1 결과:

| Model | Reward Signal Episodes | Best Reward | Best Success | Best Sample Success | Best Argmax Success | Final Sample Success | Final Argmax Success | Final Length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla FuN | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 120.000 |
| AblationManager | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 120.000 |

해석:

- N4-S5 seed 1에서는 두 모델 모두 5000 episode 동안 reward signal을 전혀 얻지 못했다.
- gate 조건을 만족하지 못했으므로 Stage 2는 실행하지 않았다.
- 이 결과는 recurrent memory 비교 이전에, 현재 REINFORCE-style FuN 학습이 sparse N4-S5 exploration을 해결하지 못한다는 증거에 가깝다.

## 8. 현재 결론

N2-S4:

- 두 모델 모두 성공률 1.000으로 해결했다.
- memory 효과 검증 환경으로는 너무 쉽다.

N4-S5:

- 두 모델 모두 Stage 1 5000 episode에서 reward signal 0이다.
- memory 효과를 비교하기 전에 exploration 자체가 실패한다.

따라서 현재 결과만으로는 recurrent Manager memory의 우위를 주장하기 어렵다. Week 6의 의미 있는 결론은 다음에 가깝다.

```text
N2-S4는 너무 쉬워서 Vanilla FuN과 AblationManager 차이를 드러내지 못했고,
N4-S5는 현재 sparse-reward REINFORCE-style FuN으로 reward signal을 얻지 못했다.
따라서 memory ablation 효과를 보기 위해서는 중간 난이도 환경, curriculum, 또는 exploration 보조 전략이 필요하다.
```

## 9. 다음 작업 후보

바로 seed 11/44 확장으로 가는 것은 권장하지 않는다. N4-S5 seed 1에서 reward signal이 0이었기 때문에 seed 확장은 계산비만 늘릴 가능성이 크다.

우선순위:

1. `MiniGrid-MultiRoom-N3-S4-v0` 또는 사용 가능한 중간 난이도 MultiRoom 후보 확인
2. N4-S5보다 쉬운 중간 환경에서 seed 1 reward signal 확인
3. reward signal이 나오면 Stage 2와 seed 확장 진행
4. reward signal이 계속 0이면 curriculum 또는 exploration 보조 전략 검토

보류:

- N4-S5 seed 11/44 확장
- N6 직접 실행
- PPO/A2C 전환
- TransformerManager 추가
- reward shaping 추가
