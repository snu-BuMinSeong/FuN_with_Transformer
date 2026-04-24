# 2주차 작업 정리

## 현재 목표

2주차 목표는 1주차에 만든 구조 확인용 FuN을 실제로 학습 가능한 vanilla FuN 형태로 확장하는 것이었다.

처음에는 environment rollout과 최소 forward-pass만 가능한 상태였고, 이후 2주차 동안 rollout 수집, return 계산, worker loss, trainer, optimizer, config, logging, evaluation을 먼저 붙였다. 그 뒤 sparse reward 환경에서 신호가 거의 보이지 않는 문제를 확인하면서 value baseline, manager 최소 loss, 디버깅 summary, plotting, evaluation 보강까지 진행했다.

이번 주 실제 진행 범위는 다음과 같다.

- 학습용 모델 출력 정리
- action distribution, `log_prob`, entropy 처리 추가
- 학습용 rollout 수집 함수 작성
- reward-to-go return / advantage 계산 추가
- worker loss, value loss, manager 최소 loss 구현
- `trainer.py` 학습 루프 작성
- optimizer 및 config 연결
- `train.py` 실행 진입점 추가
- 디버깅용 gradient / entropy / action / value summary 추가
- 학습 로그 CSV / JSON 저장
- 학습 로그 plotting 스크립트 추가
- 학습 전후 평가와 비교 문장 보강

이번 주에 일부러 최소 버전으로 둔 범위는 다음과 같다.

- 논문식 manager objective 완전 재현
- PPO 등 다른 알고리즘 전환
- checkpoint save/load
- 성능 개선 실험
- reward shaping

## 공통 작업 기준

- Python으로 작성했다.
- 기존 `fun-minigrid/` 구조를 유지했다.
- 구조를 크게 바꾸기보다 현재 vanilla FuN 파이프라인을 점진적으로 확장했다.
- worker policy gradient가 실제로 동작하도록 유지하면서 baseline과 manager 학습 경로를 추가했다.
- manager는 논문 재현 대신 “완전히 고정되지 않도록” 최소 supervision을 주는 방향으로 구현했다.
- rollout, returns, losses, trainer, evaluation, logger, plotting 책임을 파일별로 분리했다.
- shape, dtype, device 흐름이 꼬이지 않도록 테스트를 계속 추가했다.
- seed를 고정해 재현 가능한 짧은 학습/평가 실행을 확인했다.
- 로그는 CSV와 JSON으로 남기고, plotting은 별도 스크립트로 분리했다.

## Step별 진행 내용

### Step 1. 학습용 모델 출력 정리

진행 내용:

- `src/models/fun.py`가 기존 `logits` 외에 `action_dist`, `action_probs`, `goal_updated`를 함께 반환하도록 수정했다.
- 이후 value baseline을 위해 `value_head`를 추가하고 `value`를 함께 반환하게 했다.
- manager 최소 학습을 위해 `manager_value_head`를 추가하고 `manager_value`를 반환하게 했다.
- `src/policies/fun_policy.py`는 모델 출력의 distribution을 재사용하고, 학습용 `log_prob`, `entropy`, `value`, `manager_value`를 꺼낼 수 있게 정리했다.

확인 결과:

- 모델 forward 결과만으로 action 선택, `log_prob`, entropy, state value, manager용 value를 모두 계산할 수 있게 됐다.
- 모델 shape 테스트를 통과했다.

### Step 2. 학습용 rollout 수집 함수 작성

진행 내용:

- `src/training/rollout.py`에 `collect_training_rollout()`을 추가했다.
- timestep마다 `observation`, `action`, `reward`, `done`, `terminated`, `truncated`, `log_prob`, `entropy`, `goal`, `step_index`, `goal_updated`, `value`, `manager_value`를 저장하게 했다.
- evaluation에서 seed별 결과를 볼 수 있도록 `run_episode()` 반환값에도 `seed`를 넣었다.

확인 결과:

- 학습용 trajectory dict를 안정적으로 반환할 수 있게 됐다.
- seed 기준 재현성 테스트를 통과했다.

### Step 3. return / advantage 계산 정리

진행 내용:

- `src/training/returns.py`를 추가했다.
- `compute_reward_to_go()`로 reward-to-go return을 계산하게 했다.
- 초반에는 baseline 없는 advantage를 사용했고, 이후 `value_predictions`를 baseline으로 사용하는 `advantage = return - value` 구조로 확장했다.
- `attach_returns_and_advantages()`는 trajectory에 `returns`, `advantages`, `value_predictions`, `value_targets`를 붙이게 했다.

확인 결과:

- 할인/무할인 return 계산 테스트를 통과했다.
- baseline 유무에 따른 advantage shape 검증을 통과했다.

### Step 4. loss 함수 구현

진행 내용:

- `src/training/losses.py`를 추가했다.
- `compute_worker_loss()`로 `-log_prob * advantage` 형태의 worker loss를 구현했다.
- `compute_entropy_bonus()`를 추가했다.
- `compute_value_loss()`로 `MSE(value, return)` 형태의 critic loss를 구현했다.
- 처음엔 manager loss를 placeholder 0 loss로 두었고, 이후 `goal_updated=True`인 step에서 `manager_value`가 return을 맞추도록 하는 최소 MSE loss를 붙였다.
- `compute_total_loss()`는 현재 `worker_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus + manager_loss_coef * manager_loss` 구조다.

확인 결과:

- `total_loss.backward()`까지 실제로 수행되는 것을 테스트로 확인했다.
- worker, value, manager 경로 모두 gradient를 받을 수 있게 됐다.

### Step 5. trainer 작성

진행 내용:

- `src/training/trainer.py`를 추가했다.
- `train_one_episode()`에서 rollout 수집, return/advantage 계산, loss 계산, `zero_grad()`, `backward()`, gradient clipping, `optimizer.step()`까지 연결했다.
- `train()` 함수로 여러 episode를 반복 학습할 수 있게 했다.
- 이후 디버깅용으로 다음 summary를 추가했다.
- `worker_loss`, `value_loss`, `manager_loss`, `entropy_mean`
- `returns_mean/min/max`
- `advantages_mean`, `advantages_abs_mean`
- `values_mean/min/max`
- `manager_values_mean/min/max`
- `log_prob_mean/min/max`
- `action_coverage`, `action_histogram`
- `nonzero_reward_fraction`, `nonzero_return_fraction`
- encoder / manager / worker / value_head / manager_value_head grad norm

확인 결과:

- 1 episode 학습에서 실제 파라미터가 업데이트되는 것을 확인했다.
- short run에서도 summary가 안정적으로 반환됐다.

### Step 6. optimizer 및 config 연결

진행 내용:

- `configs/train_fun.yaml`을 실제 학습 설정 파일로 바꿨다.
- `src/utils/config.py`를 추가해 외부 의존성 없이 flat YAML을 읽게 했다.
- `train.py`를 추가해 config를 읽고 env, model, policy, optimizer, trainer를 연결했다.
- `value_loss_coef`, `manager_loss_coef`, `eval_episodes`, `eval_seed_offset`, `log_interval` 등을 config로 제어할 수 있게 했다.
- `FuNModel`, `ObservationEncoder`, `Manager`, `Worker`는 `hidden_dim`, `goal_size`, `num_actions`, `goal_update_interval`을 설정에서 받게 했다.

확인 결과:

- config 기반 학습 실행과 설정 변경이 가능해졌다.

### Step 7. 실행 및 디버깅

진행 내용:

- 1 episode 디버그 테스트와 10 episode 짧은 실행 테스트를 만들었다.
- 추가로 reward signal 부재와 gradient 흐름 여부를 더 분명히 보기 위해 다음 체크 포인트를 넣었다.
- `has_reward_signal`
- `has_return_signal`
- `nonzero_reward_steps`
- `nonzero_return_steps`
- 모듈별 grad norm과 grad 존재 여부
- entropy와 action coverage

확인 결과:

- action 범위가 `0~6` 안에 있는 것을 확인했다.
- hidden state와 goal shape가 `(1, 64)`, `(1, 16)`으로 유지되는 것을 확인했다.
- `goal_update_interval`이 의도대로 작동하는 것을 확인했다.
- 현재 짧은 실행에서는 reward와 return signal이 거의 없다는 것도 확인했다.

### Step 8. 로그 및 결과 저장

진행 내용:

- `src/utils/logger.py`에 `append_training_log()`와 `write_json_summary()`를 추가했다.
- training CSV에 reward, loss, grad norm, action min/max, action coverage, entropy, value range, manager value range, reward signal 관련 필드를 저장하게 했다.
- `train.py`에서 `reward_moving_avg`, `success_moving_avg`, `loss_moving_avg`를 계산해 CSV에 기록하게 했다.
- summary JSON에는 학습 전체의 평균 grad norm, 평균 entropy, reward/return signal이 있던 episode 수 등을 저장하게 했다.

확인 결과:

- CSV/JSON 저장 테스트를 통과했다.
- 로그만 봐도 어느 부분이 병목인지 추론 가능한 수준이 됐다.

### Step 9. 최소 평가

진행 내용:

- `src/training/evaluation.py`를 추가했다.
- `evaluate_policy()`로 학습 전후 같은 조건의 evaluation rollout을 수행하게 했다.
- 이후 평가를 더 믿을 수 있게 하기 위해 다음 지표를 추가했다.
- `std_reward`
- `std_success_rate`
- `mean_episode_length`
- `std_episode_length`
- `min_reward`, `max_reward`
- `min_episode_length`, `max_episode_length`
- `episode_seeds`
- `build_eval_comparison()`으로 `reward_delta`, `success_rate_delta`, `episode_length_delta`를 계산하게 했다.
- `build_eval_comparison_text()`는 reward, success rate, episode length를 함께 반영하게 했다.
- `train.py`는 `pre_eval`, `post_eval`, `eval_comparison`, `comparison_text`를 summary JSON에 저장하게 했다.

확인 결과:

- pre/post evaluation의 평균뿐 아니라 표준편차와 seed별 결과까지 확인할 수 있게 됐다.
- 현재 짧은 실행에서는 pre/post 모두 reward와 success rate 개선이 거의 없고 episode length도 차이가 없는 상태다.

### 추가 작업 1. 최소 plotting 스크립트

진행 내용:

- `plot_training.py`를 추가했다.
- `logs/week2_train.csv` 같은 학습 로그 CSV를 읽어 다음 그래프를 그리게 했다.
- episode vs reward
- episode vs success 또는 success moving average
- episode vs loss
- moving average 컬럼이 있으면 그 값을 우선 사용하고, 없으면 내부에서 trailing moving average를 계산하게 했다.

확인 결과:

- 코드 자체는 추가되었고 문법 검증을 통과했다.
- 현재 환경에는 `matplotlib`가 설치되지 않아 실제 실행 전 설치가 필요하다.

## 실제 생성 및 수정된 주요 파일

```text
fun-minigrid/
├─ train.py
├─ plot_training.py
├─ configs/
│  └─ train_fun.yaml
├─ src/
│  ├─ models/
│  │  ├─ encoder.py
│  │  ├─ manager.py
│  │  ├─ worker.py
│  │  └─ fun.py
│  ├─ policies/
│  │  └─ fun_policy.py
│  ├─ training/
│  │  ├─ rollout.py
│  │  ├─ returns.py
│  │  ├─ losses.py
│  │  ├─ trainer.py
│  │  └─ evaluation.py
│  └─ utils/
│     ├─ config.py
│     └─ logger.py
└─ tests/
   ├─ test_config.py
   ├─ test_evaluation.py
   ├─ test_fun_policy.py
   ├─ test_logger.py
   ├─ test_losses.py
   ├─ test_model_shapes.py
   ├─ test_returns.py
   ├─ test_trainer.py
   ├─ test_training_debug.py
   └─ test_training_rollout.py
```

## 생성된 주요 로그와 결과 파일

- `fun-minigrid/logs/week2_train.csv`
- `fun-minigrid/logs/week2_train_summary.json`
- `fun-minigrid/logs/test_training_log.csv`
- `fun-minigrid/logs/test_training_summary.json`

## 현재 확인된 상태

완료한 것:

- vanilla FuN 학습 코드 실행 가능
- optimizer step 연결 완료
- config 기반 실행 진입점 추가
- worker, value, manager 최소 loss 연결
- 학습 로그 CSV 저장
- summary JSON 저장
- moving average 저장
- 학습 전후 평가 저장
- 평가 std / delta / seed 정보 저장
- 비교 문장 자동 생성
- plotting 스크립트 추가
- gradient / entropy / action / value 디버깅 summary 추가

아직 부족한 것:

- DoorKey sparse reward 환경에서 reward 개선 없음
- success rate 개선 없음
- reward signal 자체가 거의 들어오지 않음
- worker loss가 실제 reward보다 entropy나 baseline 오차에 더 많이 영향을 받는 구간이 있음
- manager loss는 최소 auxiliary loss 수준이지 논문 재현은 아님
- checkpoint와 장기 실험 자동화는 아직 없음
- plotting 실행을 위해서는 `matplotlib` 설치가 필요함

## 현재 가장 의심되는 병목

현재 가장 의심되는 병목은 코드 연결보다 sparse reward 자체다.

디버깅 summary 기준으로 보면, gradient는 encoder / manager / worker / value head / manager value head 모두에 실제로 흐르고 있다. 즉 "학습이 전혀 안 돌고 있다"기보다는 "유효한 reward/return signal이 거의 없다"는 해석이 더 맞다.

현재 상태를 요약하면 다음과 같다.

- gradient 흐름: 있음
- optimizer step: 정상
- logging / evaluation / plotting 경로: 있음
- reward / success 개선 신호: 없음

## 2주차 종료 평가

이번 주 목표였던 "구조 확인용 FuN을 학습 가능한 vanilla FuN으로 바꾸기"는 최소 구현 기준으로는 달성했다.

초기에는 rollout 수집, return 계산, worker loss, trainer, optimizer, config, logging, evaluation을 연결하는 데 집중했고, 이후 value baseline과 manager 최소 loss까지 확장했다. 추가로 디버깅 summary와 plotting, evaluation 보강까지 붙여서 다음 단계의 안정화 작업을 위한 기반을 마련했다.

즉, 2주차 종료 시점 평가는 다음과 같다.

- 학습 파이프라인 구축: 완료
- baseline / manager 최소 학습 경로 연결: 완료
- 로그 / 평가 / 디버깅 / 시각화 기반 구축: 완료
- 실제 성능 개선 신호 확인: 미달

다음 단계에서는 sparse reward 대응과 baseline 안정화, 더 긴 학습, exploration 또는 reward shaping 검토가 필요하다.
