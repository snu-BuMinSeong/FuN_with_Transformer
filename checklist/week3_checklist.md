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

## GCP 학습 실행 계획

3주차 baseline 실험은 로컬에서 짧은 정상 동작 점검을 먼저 수행하고, 이후 장기 학습과 seed 반복 실험은 GCP VM에서 실행한다.

- [x] Google Cloud SDK 설치 및 기본 프로젝트 설정 확인
- [x] VS Code Remote SSH 확장 설치 확인
- [x] GCP VM SSH 접속 설정 완료
- [x] SSH config 권한 및 사용자명 문제 수정
- [x] VS Code에서 GCP VM 원격 폴더 접속 가능 상태 확인
- [x] GCP VM에서 저장소 코드 준비
- [x] GCP VM Python 가상환경 구성
- [x] GCP VM에서 필요한 패키지 import 가능 확인
- [x] GCP VM에서 `nvidia-smi` 및 CUDA/GPU 실행 환경 확인
- [x] GCP VM에서 짧은 smoke test 및 문법/체크포인트 테스트 실행
- [x] GCP VM에서 seed별 baseline 장기 학습 실행
- [x] GCP VM 학습 로그와 checkpoint 저장 경로 정리

### 현재 접속 정보

```text
project: project-ed2a3aec-0315-4f0f-95a
zone: asia-northeast3-b
host: instance-20260425-090526.asia-northeast3-b.project-ed2a3aec-0315-4f0f-95a
remote user: fumin0193
remote workspace: /home/fumin0193
```

### 운영 원칙

- 로컬에서는 코드 수정, 짧은 실행 검증, 로그 형식 점검을 우선 수행한다.
- GCP에서는 시간이 오래 걸리는 baseline 학습, seed 반복 실험, checkpoint 생성 작업을 수행한다.
- GCP 학습은 SSH 연결이 끊겨도 유지되도록 `tmux` 또는 `nohup` 기반으로 실행한다.
- 학습 결과는 seed별로 `logs/`와 `checkpoints/`를 분리해 저장한다.

---

## 1. 학습 정상 동작 점검

- [x] 현재 `train.py`를 다시 실행해 vanilla FuN 학습이 에러 없이 도는지 확인
- [x] `total_loss`가 NaN 없이 기록되는지 확인
- [x] `worker_loss`가 NaN 없이 기록되는지 확인
- [x] `value_loss`가 NaN 없이 기록되는지 확인
- [x] `manager_loss`가 NaN 없이 기록되는지 확인
- [x] `entropy_mean`이 너무 빠르게 0에 가까워지지 않는지 확인
- [x] `grad_norm`이 폭발하지 않는지 확인
- [x] encoder / manager / worker / value head에 gradient가 흐르는지 확인
- [x] action이 특정 action 하나로만 고정되지 않는지 확인
- [x] goal vector norm이 계속 0이거나 고정되지 않는지 확인

### 완료 기준

- [x] 100 episode 이상 학습이 중단 없이 실행됨
- [x] loss, entropy, grad norm이 로그에 정상 저장됨
- [x] NaN 또는 inf가 발생하지 않음

---

## 2. DoorKey 환경 난이도 및 reward signal 점검

- [x] random policy 성능 측정
- [x] 학습 전 untrained FuNPolicy 성능 측정
- [x] 짧게 학습한 FuNPolicy 성능 측정
- [x] episode length가 항상 max step에 걸리는지 확인
- [x] reward가 대부분 0인지 확인
- [x] success가 한 번이라도 발생하는지 확인
- [x] `nonzero_reward_fraction` 확인
- [x] `nonzero_return_fraction` 확인
- [x] `has_reward_signal`이 true가 되는 episode 수 확인
- [x] `has_return_signal`이 true가 되는 episode 수 확인

### 완료 기준

- [x] 현재 환경에서 학습 신호가 실제로 얼마나 희소한지 수치로 정리됨
- [x] DoorKey-5x5를 그대로 사용할지, 쉬운 sanity-check 환경을 병행할지 판단 가능함

---

## 3. 실험 설정 파일 정리

- [x] `configs/train_fun_baseline_seed*.yaml`을 3주차 baseline 실험용으로 정리
- [x] `env_id` 명시
- [x] `total_episodes` 명시
- [x] `eval_interval` 명시
- [x] `eval_episodes` 명시
- [x] `seed` 명시
- [x] `learning_rate` 명시
- [x] `gamma` 명시
- [x] `goal_update_interval` 명시
- [x] `hidden_dim` 명시
- [x] `goal_size` 또는 `goal_dim` 명시
- [x] `entropy_coef` 명시
- [x] `value_loss_coef` 명시
- [x] `manager_loss_coef` 명시
- [x] `grad_clip_norm` 명시
- [x] `log_dir` 명시
- [x] `checkpoint_dir` 명시

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

- [x] 같은 config로 실험을 재현할 수 있음
- [x] 4주차 memory ablation 실험에서 같은 config를 재사용할 수 있음

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

- [x] `entropy_coef = 0.01` 기본값 확인
- [ ] action coverage가 너무 낮으면 entropy coefficient 증가 검토
- [ ] entropy가 너무 높아 학습이 불안정하면 entropy coefficient 감소 검토

### 완료 기준

- [x] baseline 실험에 사용할 기본 hyperparameter 조합 1개 확정
- [x] 확정 이유를 짧게 기록

확정 조합은 `configs/train_fun_baseline_seed1.yaml`, `configs/train_fun_baseline_seed11.yaml`, `configs/train_fun_baseline_seed44.yaml`에 기록된 `learning_rate=0.0003`, `goal_update_interval=10`, `entropy_coef=0.01` 기준이다. 별도 grid search는 하지 않았고, 3주차 목표가 성능 최적화가 아니라 4주차 비교용 baseline 고정이었기 때문에 안정적으로 실행되고 seed별 결과를 확보한 조합을 baseline으로 사용한다.

---

## 5. checkpoint 저장/로드 구현

2주차에서 아직 완료되지 않은 항목이므로 3주차 필수 작업으로 처리한다.

- [x] `src/utils/checkpoint.py` 또는 유사 파일 추가
- [x] `save_checkpoint()` 구현
- [x] `load_checkpoint()` 구현
- [x] latest checkpoint 저장
- [x] best checkpoint 저장
- [x] 일정 episode마다 checkpoint 저장
- [x] checkpoint에 model state 저장
- [x] checkpoint에 optimizer state 저장
- [x] checkpoint에 episode 저장
- [x] checkpoint에 config 저장
- [x] checkpoint에 best metric 저장
- [x] 저장된 checkpoint로 evaluation 실행 가능하게 연결

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

- [x] 학습 중 checkpoint가 생성됨
- [x] `last.pt`를 불러와 evaluation 가능
- [x] `best.pt`를 불러와 evaluation 가능

---

## 6. evaluation loop 정리

- [x] 학습 중 evaluation을 일정 interval마다 실행
- [x] evaluation에서는 deterministic/sample action 사용 여부 결정
- [x] `action_mode="argmax"` 평가 가능하게 확인
- [x] `action_mode="sample"` 평가 가능하게 확인
- [x] evaluation seed 고정
- [x] evaluation episode 수 고정
- [x] train log와 eval log를 분리 저장
- [x] `eval_success_rate` 저장
- [x] `eval_mean_return` 저장
- [x] `eval_std_return` 저장
- [x] `eval_mean_episode_length` 저장
- [x] `eval_std_episode_length` 저장
- [x] `eval_episode_seeds` 저장

### 완료 기준

- [x] 학습 로그와 평가 로그가 분리되어 저장됨
- [x] 4주차 ablation 모델과 같은 평가 코드로 비교 가능함

---

## 7. seed 반복 실험

6주 연구 일정 기준으로 3주차에는 2~3개 seed 반복 실험을 확보해야 한다.

### 권장 seed

```text
1, 11, 44
```

### 해야 할 일

- [x] seed 1, 11, 44 baseline config 생성
- [x] seed 1, 11, 44 실행 스크립트 생성
- [x] GCP/tmux 실행 명령 문서화
- [x] seed 1로 baseline 학습 실행
- [x] seed 11로 baseline 학습 실행
- [x] seed 44로 baseline 학습 실행
- [x] 각 seed별 train log 저장
- [x] 각 seed별 eval log 저장
- [x] 각 seed별 checkpoint 저장
- [x] seed별 summary JSON 저장
- [x] seed별 최종 success rate 정리
- [x] seed별 최종 average return 정리
- [x] seed별 최종 episode length 정리

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

- [x] 최소 2개 seed 결과 확보
- [x] 가능하면 3개 seed 결과 확보
- [x] seed별 결과 편차를 확인할 수 있음

---

## 8. 그래프 생성

### 필수 그래프

- [ ] training return curve
- [x] evaluation success rate curve
- [x] evaluation mean return curve
- [x] evaluation mean episode length curve
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

- [x] success rate
- [x] average return
- [x] average episode length

### 완료 기준

- [x] 그래프 파일이 저장됨
- [x] 보고서에 바로 넣을 수 있는 baseline 그래프가 최소 3개 있음

---

## 9. baseline 결과 요약 문서 작성

- [x] `results/week3_baseline_summary.md` 작성
- [x] 실험 환경 정리
- [x] 모델 구조 요약
- [x] 사용 config 정리
- [x] seed별 결과 표 작성
- [x] 최종 평균 success rate 작성
- [x] 최종 평균 return 작성
- [x] 최종 평균 episode length 작성
- [x] 학습이 잘 된 부분과 안 된 부분 구분
- [x] sparse reward로 인한 한계 기록
- [x] 4주차 memory ablation에서 비교할 기준 metric 명시

### 결과 표 예시

| Seed | Success Rate | Average Return | Episode Length | Best Checkpoint |
|---:|---:|---:|---:|---|
| 1 | TBD | TBD | TBD | TBD |
| 11 | TBD | TBD | TBD | TBD |
| 44 | TBD | TBD | TBD | TBD |
| Mean | TBD | TBD | TBD | TBD |

### 완료 기준

- [x] 4주차 시작 전에 baseline 결과를 한눈에 볼 수 있음
- [x] 보고서 결과 섹션 초안으로 재사용 가능함

---

## 10. 4주차 memory ablation 준비

- [x] vanilla FuN에서 Manager RNN이 사용되는 부분 정확히 표시
- [x] memory 제거 시 수정해야 할 파일 목록 작성
- [x] `AblationManager` 설계 메모 작성
- [x] 기존 training loop를 그대로 재사용할 수 있는지 확인
- [x] baseline config를 ablation config로 복사할 준비
- [x] 비교 metric 확정

코드 점검 결과, Manager 메모리는 `src/models/manager.py`의 `nn.GRUCell`과 `src/models/fun.py`의 `hidden_state`, `goal_update_interval` 갱신 로직에 집중되어 있다. Memory ablation에서 우선 확인하거나 수정할 파일은 `src/models/manager.py`, `src/models/fun.py`, `src/policies/fun_policy.py`, `configs/train_fun_baseline_seed*.yaml`, 관련 shape/model 테스트다. `src/training/trainer.py`, `rollout.py`, `losses.py`, `evaluation.py`는 policy/model 출력 schema가 유지되면 그대로 재사용 가능하다.

### 4주차에서 비교할 metric

- sample success rate
- sample mean return
- sample mean episode length
- 학습 안정성
- seed별 편차

### 완료 기준

- [x] 4주차에 바로 `AblationManager` 구현으로 넘어갈 수 있음

---

# 3주차 종료 기준

3주차가 끝났을 때 아래가 만족되면 성공이다.

- [x] Vanilla FuN 학습이 재현 가능하게 실행된다.
- [x] baseline config가 고정되어 있다.
- [x] checkpoint 저장/로드가 가능하다.
- [x] 학습 로그와 평가 로그가 분리되어 저장된다.
- [x] 최소 2개 seed, 가능하면 3개 seed 결과가 있다.
- [x] success rate, average return, episode length 그래프가 있다.
- [x] sparse reward로 인한 한계가 정리되어 있다.
- [x] 4주차 memory ablation과 비교할 기준 결과가 확보되어 있다.

---

# 우선순위

## 꼭 해야 하는 것

- [x] vanilla FuN 재실행 및 안정성 점검
- [x] config 정리
- [x] checkpoint 저장/로드
- [x] evaluation loop 분리
- [x] seed 반복 실험
- [x] success rate / return / episode length 그래프 생성

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

---

# 2026-04-25 업데이트: Week 3 Baseline 실행 및 결과 정리

## 완료된 핵심 작업

- [x] `evaluate_checkpoint.py` 추가
- [x] 저장된 `last.pt`, `best.pt`를 불러와 evaluation 가능하도록 구현
- [x] `tests/test_evaluate_checkpoint.py` 추가
- [x] checkpoint 저장/로드 테스트 통과 확인
- [x] seed별 실행 스크립트 정리
  - `fun-minigrid/scripts/run_seed1.sh`
  - `fun-minigrid/scripts/run_seed11.sh`
  - `fun-minigrid/scripts/run_seed44.sh`
  - `fun-minigrid/scripts/run_baseline_seeds.sh`
- [x] GCP 실행 문서 보강
  - `fun-minigrid/results/week3_baseline_run_commands.md`
  - `fun-minigrid/results/week3_checkpoint_eval_commands.md`
  - `fun-minigrid/results/week3_result_collection_commands.md`
- [x] seed별 결과 수집 스크립트 추가: `fun-minigrid/aggregate_baseline_results.py`
- [x] baseline evaluation curve 생성 스크립트 추가: `fun-minigrid/plot_baseline_results.py`
- [x] baseline summary template 및 실제 summary 생성
  - `fun-minigrid/results/week3_baseline_summary_template.md`
  - `fun-minigrid/results/week3_baseline_summary.md`

## GCP 실행 상태

- [x] GCP VM 접속 확인
  - host: `instance-20260425-090526.asia-northeast3-b.project-ed2a3aec-0315-4f0f-95a`
  - hostname: `instance-20260425-090526`
- [x] GCP boot disk를 100GB로 확장
  - `/dev/sda1`: 약 99GB
  - 사용량 확인 시 약 85GB 여유 공간 확보
- [x] 기존 원격 `FuN_with_Transformer` 폴더 삭제 후 현재 로컬 `fun-minigrid` 코드 재업로드
- [x] GCP Python/CUDA 환경 확인
  - Python: `3.13.5`
  - torch: `2.6.0+cu124`
  - CUDA available: `True`
  - GPU: Tesla T4
- [x] 학습 전 문법 검사 실행
  - `python -m py_compile train.py evaluate_checkpoint.py aggregate_baseline_results.py plot_baseline_results.py`
- [x] checkpoint 테스트 실행
  - `python -m pytest tests/test_checkpoint.py`

## Seed별 장기 학습

- [x] seed 1 장기 학습 실행 완료
- [x] seed 11 장기 학습 실행 완료
- [x] seed 44 장기 학습 실행 완료
- [x] 각 seed별 `train.csv` 생성
- [x] 각 seed별 `eval.csv` 생성
- [x] 각 seed별 `summary.json` 생성
- [x] 각 seed별 `best.pt`, `last.pt`, `episode_1000.pt` 생성

## 로컬로 가져온 결과 파일

- [x] `fun-minigrid/logs/baseline_fun/seed_1/`
- [x] `fun-minigrid/logs/baseline_fun/seed_11/`
- [x] `fun-minigrid/logs/baseline_fun/seed_44/`
- [x] `fun-minigrid/checkpoints/baseline_fun/seed_1/`
- [x] `fun-minigrid/checkpoints/baseline_fun/seed_11/`
- [x] `fun-minigrid/checkpoints/baseline_fun/seed_44/`
- [x] `fun-minigrid/results/week3_baseline_results.csv`
- [x] `fun-minigrid/results/week3_baseline_results.md`
- [x] `fun-minigrid/results/week3_baseline_summary.md`
- [x] `fun-minigrid/figures/baseline_fun/eval_success_rate.png`
- [x] `fun-minigrid/figures/baseline_fun/eval_mean_return.png`
- [x] `fun-minigrid/figures/baseline_fun/eval_episode_length.png`

## 최종 Baseline 결과

| Seed | Final Success Rate | Final Mean Return | Final Episode Length | Best Success Rate |
|---:|---:|---:|---:|---:|
| 1 | 0.000000 | 0.000000 | 250.000000 | 0.000000 |
| 11 | 0.000000 | 0.000000 | 250.000000 | 0.000000 |
| 44 | 0.000000 | 0.000000 | 250.000000 | 0.000000 |
| Mean | 0.000000 | 0.000000 | 250.000000 | 0.000000 |

## 다음 작업

- [x] Week 3 summary의 관찰 섹션에 sparse reward 한계와 실패 양상 해석 추가
- [x] baseline 결과를 기준으로 Week 4 memory ablation 실험 config 준비
- [x] ablation 구현 전 vanilla FuN Manager/hidden state 사용 지점 재확인

---

# 2026-04-25 업데이트: Sample 기준 Baseline 재정리 완료

## 평가 기준 수정

- [x] 기존 argmax evaluation 기준만으로 baseline을 판단하면 학습 실패처럼 보이는 문제를 확인했다.
- [x] train log와 sample checkpoint evaluation에서 성공/reward signal이 있음을 확인했다.
- [x] baseline 공식 평가 기준을 `sample` evaluation으로 전환했다.
- [x] `argmax` evaluation은 참고 지표로 유지했다.
- [x] 4주차 memory ablation 비교 기준을 sample metric으로 확정했다.

## 재학습 및 로그 정리

- [x] seed 1의 mixed log 문제를 해결하기 위해 기존 seed 1 결과를 백업하고 재학습했다.
- [x] seed 1 재학습 결과 `train.csv` rows와 `summary.json` final episode가 모두 1000으로 일치함을 확인했다.
- [x] seed 11, seed 44도 기존 argmax 기준 결과를 백업하고 sample eval config 기준으로 재학습했다.
- [x] seed 11, seed 44의 `train.csv` rows와 `summary.json` final episode가 모두 1000으로 일치함을 확인했다.

## 최종 Sample 기준 결과

| Seed | Sample Success Rate | Sample Mean Return | Sample Episode Length | Argmax Success Rate |
|---:|---:|---:|---:|---:|
| 1 | 0.320000 | 0.127364 | 223.510000 | 0.000000 |
| 11 | 0.480000 | 0.243948 | 195.570000 | 0.000000 |
| 44 | 0.450000 | 0.237780 | 196.450000 | 0.000000 |
| Mean | 0.416667 | 0.203031 | 205.176667 | 0.000000 |

## 생성/갱신된 산출물

- [x] `fun-minigrid/results/week3_baseline_results.csv`
- [x] `fun-minigrid/results/week3_baseline_results.md`
- [x] `fun-minigrid/results/week3_baseline_summary.md`
- [x] `fun-minigrid/results/week3_baseline_diagnosis.md`
- [x] `fun-minigrid/results/week3_sample_eval_commands.md`
- [x] `fun-minigrid/figures/baseline_fun/sample_success_rate_by_seed.png`
- [x] `fun-minigrid/figures/baseline_fun/sample_mean_return_by_seed.png`
- [x] `fun-minigrid/figures/baseline_fun/sample_episode_length_by_seed.png`
- [x] `fun-minigrid/figures/baseline_fun/argmax_vs_sample_success_rate.png`

## 4주차 비교 기준

primary:

- sample success rate
- sample mean return
- sample mean episode length

secondary:

- argmax success rate
- train stability
- seed variance

---

# 2026-05-02 업데이트: Memory Ablation 구현 및 GCP 실행 완료

## 구현 완료

- [x] `src/models/manager.py`에 `AblationManager` 추가
- [x] `AblationManager`를 `Linear(state_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, goal_dim)` 구조로 구현
- [x] `AblationManager`가 `hidden_state`를 호환성 인자로 받되 goal 계산에는 사용하지 않도록 구현
- [x] `AblationManager.init_hidden()`이 기존 policy/training loop와 호환되도록 zero hidden state를 반환
- [x] `FuNModel`에 `manager_type` 옵션 추가
- [x] 기본값을 `manager_type: recurrent`로 유지해 기존 baseline config 호환성 보존
- [x] `manager_type: ablation` 또는 `feedforward`일 때 `AblationManager` 사용
- [x] `train.py`와 `evaluate_checkpoint.py`가 config의 `manager_type`을 읽도록 수정
- [x] training loop, rollout, loss, evaluation 코드는 변경하지 않고 재사용

## 추가한 Ablation Config

- [x] `fun-minigrid/configs/train_fun_ablation_seed1.yaml`
- [x] `fun-minigrid/configs/train_fun_ablation_seed11.yaml`
- [x] `fun-minigrid/configs/train_fun_ablation_seed44.yaml`
- [x] `fun-minigrid/configs/train_fun_ablation_smoke.yaml`

세 seed config는 baseline과 동일한 환경, episode 수, learning rate, gamma, goal update interval, hidden/goal dimension, loss coefficient, action mode를 사용하고, `manager_type: ablation`, `logs/ablation_fun/seed_*`, `checkpoints/ablation_fun/seed_*`만 다르게 설정했다.

## 테스트 및 Smoke Test

- [x] AblationManager shape 테스트 추가
- [x] AblationManager가 hidden_state를 goal 계산에 사용하지 않는지 테스트 추가
- [x] `FuNModel(manager_type="ablation")` 출력 schema 테스트 추가
- [x] ablation goal update interval 유지 테스트 추가
- [x] ablation trainer 1 episode smoke 테스트 추가
- [x] 로컬 테스트 통과
  - `python -m pytest tests/test_model_shapes.py`
  - `python -m pytest tests/test_fun_policy.py`
  - `python -m pytest tests/test_trainer.py`
- [x] GCP 원격 테스트 통과
  - `tests/test_model_shapes.py`: 9 passed
  - `tests/test_fun_policy.py`: 6 passed
  - `tests/test_trainer.py`: 4 passed

## GCP 실행 및 결과 회수

- [x] 새 GCP VM `fun-minigrid-2` 접속 설정 완료
  - zone: `asia-east1-c`
  - GPU: Tesla T4
  - Python: `3.13.5`
  - torch: `2.6.0+cu124`
- [x] 최신 ablation 코드와 config를 GCP VM으로 업로드
- [x] GCP에서 ablation smoke test 실행
- [x] seed 1, 11, 44 ablation 장기 학습 실행
- [x] 세 seed 모두 1000 episode 완료
- [x] 각 seed별 `train.csv`, `eval.csv`, `summary.json` 생성
- [x] 각 seed별 `best.pt`, `last.pt`, `episode_1000.pt` 생성
- [x] 각 seed별 `checkpoint_eval_best_sample.json`, `checkpoint_eval_best_argmax.json` 생성
- [x] 결과 파일, 로그, checkpoint, 그래프를 로컬로 회수
- [x] GCP VM `fun-minigrid-2` 중지 완료

## Ablation 최종 결과

| Seed | Sample Success Rate | Sample Mean Return | Sample Episode Length | Argmax Success Rate |
|---:|---:|---:|---:|---:|
| 1 | 0.350000 | 0.160604 | 215.110000 | 0.000000 |
| 11 | 0.820000 | 0.496936 | 134.740000 | 0.000000 |
| 44 | 0.640000 | 0.362080 | 167.200000 | 0.000000 |
| Mean | 0.603333 | 0.339873 | 172.350000 | 0.000000 |

## Baseline 대비 차이

| Metric | Ablation - Baseline |
|---|---:|
| Sample success rate | +0.186667 |
| Sample mean return | +0.136843 |
| Sample episode length | -32.826667 |
| Argmax success rate | 0.000000 |

## 생성/회수된 산출물

- [x] `fun-minigrid/results/week4_ablation_results.csv`
- [x] `fun-minigrid/results/week4_ablation_results.md`
- [x] `fun-minigrid/results/week4_ablation_summary.md`
- [x] `fun-minigrid/logs/ablation_fun/seed_1/`
- [x] `fun-minigrid/logs/ablation_fun/seed_11/`
- [x] `fun-minigrid/logs/ablation_fun/seed_44/`
- [x] `fun-minigrid/checkpoints/ablation_fun/seed_1/`
- [x] `fun-minigrid/checkpoints/ablation_fun/seed_11/`
- [x] `fun-minigrid/checkpoints/ablation_fun/seed_44/`
- [x] `fun-minigrid/figures/ablation_fun/eval_success_rate.png`
- [x] `fun-minigrid/figures/ablation_fun/eval_mean_return.png`
- [x] `fun-minigrid/figures/ablation_fun/eval_episode_length.png`

## 해석 메모

- MiniGrid DoorKey-5x5의 현재 1000 episode 조건에서는 recurrent Manager memory를 제거한 ablation이 sample 기준 평균 성능에서 baseline보다 높았다.
- 이는 이 작은 DoorKey 환경에서 GRU Manager memory가 필수적이지 않거나, feedforward Manager가 더 단순해 짧은 학습 조건에서 더 안정적으로 최적화되었을 가능성을 시사한다.
- 다만 baseline과 ablation 모두 argmax success rate는 0이므로, 현재 정책은 deterministic argmax path보다 stochastic sampling에 의존한다.
- seed 3개 결과이므로 강한 통계적 결론보다는 4주차 보고서의 1차 ablation 결과로 해석한다.
