# 3주차 체크리스트 - Baseline 안정화 및 결과 확보

## 이번 주 한 줄 목표

**Vanilla FuN을 졸업프로젝트의 비교 기준으로 사용할 수 있도록 안정화하고, seed별 baseline 결과와 그래프를 확보하기**

---

## 현재 출발점

2주차까지 다음 항목은 완료된 상태로 본다.

- vanilla FuN 학습 코드 실행 가능
- rollout 수집, return/advantage 계산 가능
- worker loss, value loss, manager 최소 loss 연결 완료
- optimizer step 및 gradient clipping 연결 완료
- 학습 로그 CSV / summary JSON 저장 가능
- pre/post evaluation 코드 추가 완료
- reward, success, episode length, loss plot 스크립트 추가 완료

하지만 아직 부족한 점은 다음과 같다.

- sparse reward 환경에서 reward 개선이 확인되지 않음
- success rate 개선이 확인되지 않음
- reward signal 자체가 거의 들어오지 않음
- checkpoint 저장/로드가 아직 없음
- 장기 실험 및 seed 반복 실험이 아직 부족함

따라서 3주차는 새 모델을 추가하는 주차가 아니라, **현재 vanilla FuN을 안정화하고 비교 가능한 baseline으로 만드는 주차**로 진행한다.

---

## 1. 학습 정상 동작 점검

- [ ] 현재 `train.py`를 다시 실행해 vanilla FuN 학습이 에러 없이 도는지 확인
- [ ] `total_loss`가 NaN 없이 기록되는지 확인
- [ ] `worker_loss`가 NaN 없이 기록되는지 확인
- [ ] `value_loss`가 NaN 없이 기록되는지 확인
- [ ] `manager_loss`가 NaN 없이 기록되는지 확인
- [ ] `entropy_mean`이 너무 빠르게 0에 가까워지지 않는지 확인
- [ ] `grad_norm`이 폭발하지 않는지 확인
- [ ] encoder / manager / worker / value head에 gradient가 흐르는지 확인
- [ ] action이 특정 action 하나로만 고정되지 않는지 확인
- [ ] goal vector norm이 계속 0이거나 고정되지 않는지 확인

### 완료 기준

- [ ] 100 episode 이상 학습이 중단 없이 실행됨
- [ ] loss, entropy, grad norm이 로그에 정상 저장됨
- [ ] NaN 또는 inf가 발생하지 않음

---

## 2. DoorKey 환경 난이도 및 reward signal 점검

- [ ] random policy 성능 측정
- [ ] 학습 전 untrained FuNPolicy 성능 측정
- [ ] 짧게 학습한 FuNPolicy 성능 측정
- [ ] episode length가 항상 max step에 걸리는지 확인
- [ ] reward가 대부분 0인지 확인
- [ ] success가 한 번이라도 발생하는지 확인
- [ ] `nonzero_reward_fraction` 확인
- [ ] `nonzero_return_fraction` 확인
- [ ] `has_reward_signal`이 true가 되는 episode 수 확인
- [ ] `has_return_signal`이 true가 되는 episode 수 확인

### 완료 기준

- [ ] 현재 환경에서 학습 신호가 실제로 얼마나 희소한지 수치로 정리됨
- [ ] DoorKey-5x5를 그대로 사용할지, 쉬운 sanity-check 환경을 병행할지 판단 가능함

---

## 3. 실험 설정 파일 정리

- [ ] `configs/train_fun.yaml`을 3주차 baseline 실험용으로 정리
- [ ] `env_id` 명시
- [ ] `total_episodes` 명시
- [ ] `eval_interval` 명시
- [ ] `eval_episodes` 명시
- [ ] `seed` 명시
- [ ] `learning_rate` 명시
- [ ] `gamma` 명시
- [ ] `goal_update_interval` 명시
- [ ] `hidden_dim` 명시
- [ ] `goal_size` 또는 `goal_dim` 명시
- [ ] `entropy_coef` 명시
- [ ] `value_loss_coef` 명시
- [ ] `manager_loss_coef` 명시
- [ ] `grad_clip_norm` 명시
- [ ] `log_dir` 명시
- [ ] `checkpoint_dir` 명시

### 권장 기본값

```yaml
env_id: MiniGrid-DoorKey-5x5-v0
seed: 1

total_episodes: 1000
eval_interval: 50
eval_episodes: 20

gamma: 0.99
learning_rate: 0.0003

goal_update_interval: 10
hidden_dim: 64
goal_dim: 16

entropy_coef: 0.01
value_loss_coef: 0.5
manager_loss_coef: 0.1
grad_clip_norm: 1.0

log_dir: logs/baseline_fun
checkpoint_dir: checkpoints/baseline_fun
```

### 완료 기준

- [ ] 같은 config로 실험을 재현할 수 있음
- [ ] 4주차 memory ablation 실험에서 같은 config를 재사용할 수 있음

---

## 4. 하이퍼파라미터 최소 튜닝

3주차에서는 큰 튜닝을 하지 않는다. 비교 기준을 만들기 위한 최소 튜닝만 진행한다.

### learning rate 비교

- [ ] `1e-4` 실험
- [ ] `3e-4` 실험
- [ ] `1e-3` 실험
- [ ] loss 안정성 비교
- [ ] reward / success / episode length 비교

### goal update interval 비교

- [ ] `goal_update_interval = 5` 실험
- [ ] `goal_update_interval = 10` 실험
- [ ] goal norm 변화 확인
- [ ] success rate 또는 return 변화 확인

### entropy coefficient 확인

- [ ] `entropy_coef = 0.01` 기본값 확인
- [ ] action coverage가 너무 낮으면 entropy coefficient 증가 검토
- [ ] entropy가 너무 높아 학습이 불안정하면 entropy coefficient 감소 검토

### 완료 기준

- [ ] baseline 실험에 사용할 기본 hyperparameter 조합 1개 확정
- [ ] 확정 이유를 짧게 기록

---

## 5. checkpoint 저장/로드 구현

2주차에서 아직 완료되지 않은 항목이므로 3주차 필수 작업으로 처리한다.

- [ ] `src/utils/checkpoint.py` 또는 유사 파일 추가
- [ ] `save_checkpoint()` 구현
- [ ] `load_checkpoint()` 구현
- [ ] latest checkpoint 저장
- [ ] best checkpoint 저장
- [ ] 일정 episode마다 checkpoint 저장
- [ ] checkpoint에 model state 저장
- [ ] checkpoint에 optimizer state 저장
- [ ] checkpoint에 episode 저장
- [ ] checkpoint에 config 저장
- [ ] checkpoint에 best metric 저장
- [ ] 저장된 checkpoint로 evaluation 실행 가능하게 연결

### 권장 저장 구조

```text
checkpoints/
└─ baseline_fun/
   ├─ last.pt
   ├─ best.pt
   └─ episode_1000.pt
```

### 저장할 내용 예시

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "episode": episode,
    "config": config,
    "best_success_rate": best_success_rate,
}
```

### 완료 기준

- [ ] 학습 중 checkpoint가 생성됨
- [ ] `last.pt`를 불러와 evaluation 가능
- [ ] `best.pt`를 불러와 evaluation 가능

---

## 6. evaluation loop 정리

- [ ] 학습 중 evaluation을 일정 interval마다 실행
- [ ] evaluation에서는 deterministic action 사용 여부 결정
- [ ] `action_mode="argmax"` 평가 가능하게 확인
- [ ] evaluation seed 고정
- [ ] evaluation episode 수 고정
- [ ] train log와 eval log를 분리 저장
- [ ] `eval_success_rate` 저장
- [ ] `eval_mean_return` 저장
- [ ] `eval_std_return` 저장
- [ ] `eval_mean_episode_length` 저장
- [ ] `eval_std_episode_length` 저장
- [ ] `eval_episode_seeds` 저장

### 완료 기준

- [ ] 학습 로그와 평가 로그가 분리되어 저장됨
- [ ] 4주차 ablation 모델과 같은 평가 코드로 비교 가능함

---

## 7. seed 반복 실험

6주 연구 일정 기준으로 3주차에는 2~3개 seed 반복 실험을 확보해야 한다.

### 권장 seed

```text
1, 11, 44
```

### 해야 할 일

- [ ] seed 1로 baseline 학습 실행
- [ ] seed 11로 baseline 학습 실행
- [ ] seed 44로 baseline 학습 실행
- [ ] 각 seed별 train log 저장
- [ ] 각 seed별 eval log 저장
- [ ] 각 seed별 checkpoint 저장
- [ ] seed별 summary JSON 저장
- [ ] seed별 최종 success rate 정리
- [ ] seed별 최종 average return 정리
- [ ] seed별 최종 episode length 정리

### 권장 로그 구조

```text
logs/
└─ baseline_fun/
   ├─ seed_1/
   │  ├─ train.csv
   │  ├─ eval.csv
   │  └─ summary.json
   ├─ seed_11/
   │  ├─ train.csv
   │  ├─ eval.csv
   │  └─ summary.json
   └─ seed_44/
      ├─ train.csv
      ├─ eval.csv
      └─ summary.json
```

### 완료 기준

- [ ] 최소 2개 seed 결과 확보
- [ ] 가능하면 3개 seed 결과 확보
- [ ] seed별 결과 편차를 확인할 수 있음

---

## 8. 그래프 생성

### 필수 그래프

- [ ] training return curve
- [ ] evaluation success rate curve
- [ ] evaluation mean return curve
- [ ] evaluation mean episode length curve
- [ ] total loss curve
- [ ] worker loss curve
- [ ] value loss curve
- [ ] manager loss curve

### 있으면 좋은 그래프

- [ ] entropy curve
- [ ] grad norm curve
- [ ] action coverage curve
- [ ] goal norm curve
- [ ] value prediction mean curve
- [ ] nonzero reward fraction curve
- [ ] nonzero return fraction curve

### 보고서용 우선 그래프

- [ ] success rate
- [ ] average return
- [ ] average episode length

### 완료 기준

- [ ] 그래프 파일이 저장됨
- [ ] 보고서에 바로 넣을 수 있는 baseline 그래프가 최소 3개 있음

---

## 9. baseline 결과 요약 문서 작성

- [ ] `results/week3_baseline_summary.md` 작성
- [ ] 실험 환경 정리
- [ ] 모델 구조 요약
- [ ] 사용 config 정리
- [ ] seed별 결과 표 작성
- [ ] 최종 평균 success rate 작성
- [ ] 최종 평균 return 작성
- [ ] 최종 평균 episode length 작성
- [ ] 학습이 잘 된 부분과 안 된 부분 구분
- [ ] sparse reward로 인한 한계 기록
- [ ] 4주차 memory ablation에서 비교할 기준 metric 명시

### 결과 표 예시

| Seed | Success Rate | Average Return | Episode Length | Best Checkpoint |
|---:|---:|---:|---:|---|
| 1 | TBD | TBD | TBD | TBD |
| 11 | TBD | TBD | TBD | TBD |
| 44 | TBD | TBD | TBD | TBD |
| Mean | TBD | TBD | TBD | TBD |

### 완료 기준

- [ ] 4주차 시작 전에 baseline 결과를 한눈에 볼 수 있음
- [ ] 보고서 결과 섹션 초안으로 재사용 가능함

---

## 10. 4주차 memory ablation 준비

- [ ] vanilla FuN에서 Manager RNN이 사용되는 부분 정확히 표시
- [ ] memory 제거 시 수정해야 할 파일 목록 작성
- [ ] `AblationManager` 설계 메모 작성
- [ ] 기존 training loop를 그대로 재사용할 수 있는지 확인
- [ ] baseline config를 ablation config로 복사할 준비
- [ ] 비교 metric 확정

### 4주차에서 비교할 metric

- success rate
- average return
- average episode length
- 학습 안정성
- seed별 편차

### 완료 기준

- [ ] 4주차에 바로 `AblationManager` 구현으로 넘어갈 수 있음

---

# 3주차 종료 기준

3주차가 끝났을 때 아래가 만족되면 성공이다.

- [ ] Vanilla FuN 학습이 재현 가능하게 실행된다.
- [ ] baseline config가 고정되어 있다.
- [ ] checkpoint 저장/로드가 가능하다.
- [ ] 학습 로그와 평가 로그가 분리되어 저장된다.
- [ ] 최소 2개 seed, 가능하면 3개 seed 결과가 있다.
- [ ] success rate, average return, episode length 그래프가 있다.
- [ ] sparse reward로 인한 한계가 정리되어 있다.
- [ ] 4주차 memory ablation과 비교할 기준 결과가 확보되어 있다.

---

# 우선순위

## 꼭 해야 하는 것

- [ ] vanilla FuN 재실행 및 안정성 점검
- [ ] config 정리
- [ ] checkpoint 저장/로드
- [ ] evaluation loop 분리
- [ ] seed 반복 실험
- [ ] success rate / return / episode length 그래프 생성

## 되면 좋은 것

- [ ] learning rate 최소 비교
- [ ] goal update interval 최소 비교
- [ ] entropy coefficient 조정
- [ ] action distribution 분석
- [ ] goal norm 분석
- [ ] value prediction 분석

## 이번 주에는 하지 않는 것

- [ ] Transformer Manager 구현
- [ ] Memory Ablation 본실험
- [ ] PPO 등 다른 알고리즘 전환
- [ ] 논문식 Manager objective 완전 재현
- [ ] 대규모 hyperparameter search

---

# 최종 정리

3주차의 핵심은 성능을 크게 올리는 것이 아니라, **Vanilla FuN을 비교 가능한 baseline으로 고정하는 것**이다.

따라서 가장 중요한 산출물은 다음 세 가지다.

1. 재현 가능한 baseline config
2. seed별 baseline 실험 결과
3. success rate / return / episode length 그래프

이 세 가지가 확보되면 4주차에 memory ablation을 같은 조건에서 비교할 수 있다.
