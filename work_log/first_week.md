# 1주차 작업 정리

## 현재 목표

1주차 목표는 MiniGrid DoorKey 환경을 실행하고 episode 결과를 기록할 수 있는 최소 실행 구조를 만드는 것이다.

이후 최소 FuN-style forward-pass 구조까지 확장했다. 단, 아직 학습 루프나 optimizer를 포함한 실제 training은 구현하지 않는다.

현재 작업 범위는 다음과 같다.

- MiniGrid DoorKey 환경 생성
- observation 전처리
- 랜덤 정책 또는 외부 policy callable 기반 episode 실행
- episode 결과 CSV 저장
- 최소 실행 진입점 `main.py` 작성
- seed 및 success 판정 유틸리티 작성
- 환경 및 향후 학습 설정 파일 작성
- observation, action space, preprocessing 결과 확인 노트북 작성
- Encoder / Manager / Worker / FuNModel의 최소 forward-pass 구조 작성
- 더미 입력으로 모델 shape 검증
- 실제 env observation으로 FuNModel forward 검증
- FuNPolicy wrapper 작성 및 rollout 연결
- FuNPolicy 기반 `main.py` 실행 확인
- 모델 및 policy 테스트 스크립트 작성

아직 구현하지 않는 범위는 다음과 같다.

- training loop
- optimizer
- replay buffer
- loss 계산
- checkpoint 저장 및 로드
- 성능 개선 실험

## 공통 코드 작성 기준

- Python으로 작성한다.
- `gymnasium`과 `minigrid`를 사용한다.
- 시작 환경은 `MiniGrid-DoorKey-5x5-v0`이다.
- 1주차는 최소 실행 구조와 forward-pass 검증까지만 작성한다.
- FuN-style 모델은 학습 가능한 완성 모델이 아니라 구조 확인용 최소 구현으로 둔다.
- 랜덤 정책 또는 외부 policy callable로 episode를 실행할 수 있으면 충분하다.
- 파일 간 책임을 분리한다.
- 불필요하게 클래스를 만들지 않고 함수 단위로 단순하게 작성한다.
- 타입 힌트와 docstring을 포함한다.
- `src/` 기준 모듈 구조를 유지해 import 경로가 꼬이지 않게 한다.
- seed 재현 가능성을 고려한다.
- 로그는 CSV 또는 dict append 수준으로 단순하게 구현한다.

## 작업 파일별 정리

### 1. `src/envs/make_env.py`

환경 생성 책임을 담당한다.

구현 내용:

- `make_env(...)` 함수 작성
- 기본 환경을 `MiniGrid-DoorKey-5x5-v0`로 설정
- `gymnasium`과 `minigrid` 사용
- environment registration 문제를 피하기 위해 `minigrid` import 포함
- `ImgObsWrapper`를 기본 적용해 image observation을 반환
- `seed`가 주어지면 `env.reset(seed=seed)`를 한 번 호출

포함하지 않은 것:

- rollout
- logging
- training
- model 관련 코드
- unrelated wrapper

### 2. `src/envs/preprocess.py`

MiniGrid observation을 이후 코드에서 일관되게 다룰 수 있도록 전처리한다.

구현 내용:

- `preprocess_obs(obs) -> np.ndarray` 함수 작성
- observation을 `np.float32` 배열로 변환
- image observation이 `HWC` 형식이면 `CHW` 형식으로 변환
- batch dimension은 추가하지 않음
- 값 정규화는 아직 하지 않음

포함하지 않은 것:

- PyTorch 의존성
- tensor 변환
- embedding
- model-specific preprocessing
- replay buffer 또는 training logic

### 3. `src/training/rollout.py`

한 episode를 실행하고 episode-level 통계를 수집한다.

구현 내용:

- `run_episode(env, policy, preprocess_fn=None, max_steps=None, seed=None) -> dict` 함수 작성
- Gymnasium API 기준으로 `reset()`과 `step()` 처리
- 기존 callable policy와 stateful policy object를 모두 지원
- callable policy는 `policy(processed_obs) -> int` 형태로 사용
- stateful policy는 `policy.reset(batch_size=1)`과 `policy.act(obs) -> int` 형태로 사용
- callable policy일 때만 `preprocess_fn`을 policy 호출 전에 적용
- stateful policy일 때는 raw observation을 넘기고 policy 내부에서 전처리
- `max_steps`가 있으면 해당 step 수에서 중단
- `random_policy(env)` helper 작성
- `env.action_space.sample()`로 valid random action 생성

수집하는 정보:

- `total_reward`
- `episode_length`
- `terminated`
- `truncated`
- `success`
- `actions`

현재 success 기준:

- `info["success"]`가 있으면 우선 반영
- 없으면 `terminated and total_reward > 0.0`

주의:

- `src/utils/metrics.py`의 `compute_success`는 별도 helper로 만들었지만, 현재 `rollout.py`에서는 아직 직접 import해서 사용하지 않는다.
- `FuNPolicy`처럼 `act()`가 있는 policy는 `run_episode()`에서 `preprocess_fn`을 넘기지 않는 방식으로 사용한다.

포함하지 않은 것:

- gradient update
- optimizer
- replay buffer
- neural network training

### 4. `src/utils/logger.py`

episode 결과를 CSV로 저장한다.

구현 내용:

- `ensure_log_dir(log_path: str) -> None` 함수 작성
- `append_episode_log(log_path: str, episode_result: dict) -> None` 함수 작성
- CSV 파일이 없거나 비어 있으면 header 생성
- episode마다 결과를 한 줄씩 append
- 누락된 key는 빈 값으로 기록

CSV 필드:

- `episode`
- `total_reward`
- `episode_length`
- `success`
- `terminated`
- `truncated`

포함하지 않은 것:

- TensorBoard
- wandb
- plotting
- 복잡한 통계 aggregation

### 5. `src/utils/seed.py`

재현성을 위한 seed 설정을 중앙화한다.

구현 내용:

- `set_seed(seed: int) -> None` 함수 작성
- Python `random` seed 설정
- `numpy` seed 설정
- `torch`가 설치되어 있으면 optional import 후 seed 설정
- CUDA가 사용 가능하면 CUDA seed도 설정

포함하지 않은 것:

- 환경 생성
- training logic

### 6. `src/utils/metrics.py`

episode-level metric helper를 제공한다.

구현 내용:

- `compute_success(terminated, truncated, total_reward, info=None) -> bool` 함수 작성
- `info["success"]`가 있으면 우선 사용
- 없으면 `terminated and not truncated and total_reward > 0.0` 기준 사용

주의:

- 현재는 helper만 작성되어 있으며, `rollout.py`에 아직 연결되어 있지는 않다.

포함하지 않은 것:

- plotting
- 여러 run에 대한 평균 계산
- training-specific metric

### 7. `main.py`

FuNPolicy를 실제 episode rollout에 붙여 보는 최소 실행 진입점이다.

구현 내용:

- `main()` 함수 작성
- `if __name__ == "__main__":` guard 추가
- `--run-id` 인자 추가
- seed 설정
- `make_env`로 환경 생성
- CUDA 사용 가능 여부에 따라 device 자동 선택
- `FuNModel(goal_update_interval=10)` 생성
- `FuNPolicy(action_mode="sample")` 생성
- FuNPolicy로 3개 episode 실행
- episode summary 출력
- episode 후 `get_debug_info()` 출력
- FuNPolicy 실행 회차별 CSV 로그 저장
- 실행 후 `env.close()` 호출

실행 확인:

```powershell
.\.venv\Scripts\python.exe main.py
```

또는 실행 회차를 지정한다.

```powershell
.\.venv\Scripts\python.exe main.py --run-id 1
```

생성된 로그:

```text
logs/week1_random.csv
logs/week1_random_run_1.csv
logs/week1_random_run_2.csv
logs/week1_random_run_3.csv
logs/week1_fun_policy_run_1.csv
```

주의:

- `main.py`의 seed 기본값은 현재 `123`이다.
- `configs/env_doorkey.yaml`의 seed 값은 `42`지만, 현재 `main.py`가 YAML 설정을 읽지는 않는다.
- 같은 `--run-id`로 다시 실행하면 기존 CSV 파일 뒤에 append된다.
- 현재 `main.py`는 random policy가 아니라 FuNPolicy rollout 확인용으로 동작한다.

포함하지 않은 것:

- 학습 루프
- optimizer
- loss
- replay buffer
- checkpoint 처리

### 8. `configs/env_doorkey.yaml`

DoorKey 환경 설정을 코드 밖으로 분리하기 위한 최소 YAML 파일이다.

현재 내용:

```yaml
env_id: MiniGrid-DoorKey-5x5-v0
render_mode: null
seed: 42
max_steps: null
```

주의:

- 현재는 설정 파일만 존재한다.
- `main.py`에서는 아직 이 YAML 파일을 읽지 않는다.

### 9. `configs/train_fun.yaml`

향후 FuN 학습 설정을 위한 placeholder 파일이다.

현재 내용:

```yaml
# Placeholder for future FuN training settings.
# Week 1 does not implement model training yet.
total_episodes: 10
learning_rate: null
goal_interval: null
hidden_dim: null
```

주의:

- 현재 실제 학습에는 사용하지 않는다.
- 이후 FuN 학습 구조를 만들 때 확장할 자리만 마련해 둔 상태이다.

### 10. `notebooks/inspect_observation.ipynb`

MiniGrid observation과 action space, preprocessing 결과를 확인하기 위한 노트북이다.

구현 내용:

- `make_env(seed=42)`로 환경 생성
- 원본 observation shape 확인
- 원본 observation dtype 확인
- action space와 action 개수 확인
- `preprocess_obs()` 적용 후 shape, dtype, min/max 확인

실행 결과:

```text
observation shape: (7, 7, 3)
observation dtype: uint8
action space: Discrete(7)
action space size: 7
```

전처리 결과:

```text
raw shape: (7, 7, 3)
raw dtype: uint8
processed shape: (3, 7, 7)
processed dtype: float32
processed min: 0.0
processed max: 5.0
```

주의:

- 원본 observation은 `ImgObsWrapper` 결과라서 `HWC` 형식이다.
- `preprocess_obs()` 이후에는 모델 입력에 맞게 `CHW` 형식이 된다.
- 현재는 정규화하지 않는다.

### 11. `src/models/encoder.py`

MiniGrid image observation을 compact state embedding으로 변환한다.

구현 내용:

- `ObservationEncoder(nn.Module)` 클래스 작성
- 입력 shape: `(B, 3, 7, 7)`
- 출력 shape: `(B, 64)`
- PyTorch 사용

구조:

```text
Conv2d(3, 16, kernel_size=3, padding=1)
ReLU
Conv2d(16, 32, kernel_size=3, padding=1)
ReLU
Flatten
Linear(32 * 7 * 7, 64)
ReLU
```

포함한 것:

- 입력 shape 검증
- 잘못된 shape에 대한 명확한 `ValueError`

포함하지 않은 것:

- batch normalization
- dropout
- environment logic
- training logic

### 12. `src/models/manager.py`

FuN-style 상위 정책 역할을 하는 최소 recurrent Manager이다.

구현 내용:

- `Manager(nn.Module)` 클래스 작성
- 입력 state embedding size: `64`
- hidden size: `64`
- goal size: `16`
- `GRUCell(input_size=64, hidden_size=64)` 사용
- `Linear(64, 16)`으로 goal vector 생성
- `init_hidden(batch_size, device=None)` helper 작성

입출력:

```text
state_emb: (B, 64)
hidden_state: (B, 64)
goal: (B, 16)
next_hidden_state: (B, 64)
```

포함한 것:

- 입력 shape 검증
- batch size mismatch 검증

포함하지 않은 것:

- LSTM
- goal normalization
- training logic
- environment interaction logic

### 13. `src/models/worker.py`

state embedding과 manager goal을 받아 MiniGrid action logits를 출력한다.

구현 내용:

- `Worker(nn.Module)` 클래스 작성
- state embedding size: `64`
- goal size: `16`
- action logits size: `7`

구조:

```text
Concatenate(state_emb, goal)
Linear(64 + 16, 64)
ReLU
Linear(64, 7)
```

입출력:

```text
state_emb: (B, 64)
goal: (B, 16)
logits: (B, 7)
```

포함하지 않은 것:

- softmax
- action sampling
- training code

### 14. `src/models/fun.py`

Encoder, Manager, Worker를 결합한 최소 FuN-like forward-pass 모델이다.

구현 내용:

- `FuNModel(nn.Module)` 클래스 작성
- 내부에서 `ObservationEncoder`, `Manager`, `Worker` 생성
- `goal_update_interval: int = 10` 인자 추가
- `init_hidden(batch_size, device=None)` 작성
- `init_goal(batch_size, device=None)` 작성
- `forward(obs, hidden_state, current_goal, step_count)` 작성

동작:

- observation을 state embedding으로 encode
- `step_count % goal_update_interval == 0`이면 Manager로 goal과 hidden state 업데이트
- update step이 아니면 기존 goal과 hidden state 유지
- Worker가 state embedding과 active goal을 받아 action logits 출력

반환값:

```python
{
    "state_emb": ...,
    "goal": ...,
    "hidden_state": ...,
    "logits": ...
}
```

포함하지 않은 것:

- optimizer
- loss function
- rollout logic
- replay buffer
- action sampling

### 15. 더미 입력 모델 실행 확인

Encoder, Manager, Worker, FuNModel을 순서대로 더미 입력으로 실행했다.

실행 조건:

```text
batch_size = 2
dummy obs shape = (2, 3, 7, 7)
dtype = torch.float32
torch.manual_seed(0)
```

확인 결과:

```text
dummy obs: torch.Size([2, 3, 7, 7]) torch.float32
encoder state_emb: torch.Size([2, 64]) torch.float32
manager goal: torch.Size([2, 16]) torch.float32
manager next_hidden: torch.Size([2, 64]) torch.float32
worker logits: torch.Size([2, 7]) torch.float32
fun step 0 state_emb: torch.Size([2, 64])
fun step 0 goal: torch.Size([2, 16])
fun step 0 hidden_state: torch.Size([2, 64])
fun step 0 logits: torch.Size([2, 7])
fun step 1 goal unchanged: True
fun step 1 hidden unchanged: True
fun step 1 logits: torch.Size([2, 7])
```

의미:

- Encoder는 observation을 `(B, 64)` embedding으로 변환한다.
- Manager는 `(B, 16)` goal과 `(B, 64)` hidden state를 반환한다.
- Worker는 7개 discrete action에 대한 logits를 반환한다.
- FuNModel은 update step에서는 goal을 갱신하고, interval 사이 step에서는 goal과 hidden state를 유지한다.

### 16. `src/policies/fun_policy.py`

`FuNModel`을 episode rollout에서 사용할 수 있도록 감싼 inference 전용 policy wrapper이다.

구현 내용:

- `FuNPolicy` 클래스 작성
- 내부에서 model, device, hidden state, current goal, step count 관리
- 생성자에서 `model.eval()` 호출
- `reset(batch_size=1)` 작성
- `_obs_to_tensor(obs)` 작성
- `act(obs) -> int` 작성
- `get_debug_info() -> dict` 작성
- `action_mode="sample"`과 `action_mode="argmax"` 지원

동작:

- raw observation을 받아 내부에서 `preprocess_obs()` 적용
- `(3, 7, 7)` observation을 `(1, 3, 7, 7)` tensor로 변환
- `torch.no_grad()` 안에서 `FuNModel` forward 실행
- model output에서 hidden state와 current goal 갱신
- logits로부터 action 선택
- action을 Python `int`로 반환
- action 선택 후 `step_count` 증가

debug 정보:

```text
step_count
hidden_norm
goal_norm
```

포함하지 않은 것:

- optimizer
- loss
- replay buffer
- rollout loop
- checkpoint loading
- batch training support

### 17. 실제 env observation 기반 FuNModel 테스트

`tests/test_real_obs.py`를 작성해 실제 MiniGrid observation이 모델에 들어갈 수 있는지 확인했다.

확인 결과:

```text
obs_tensor shape  : torch.Size([1, 3, 7, 7])
state_emb shape   : torch.Size([1, 64])
goal shape        : torch.Size([1, 16])
hidden_state shape: torch.Size([1, 64])
logits shape      : torch.Size([1, 7])
Real observation test passed.
```

의미:

- `make_env()`가 만든 실제 observation이 `preprocess_obs()`를 거쳐 모델 입력 shape에 맞게 들어간다.
- `FuNModel`이 실제 observation으로 forward 가능하다.

### 18. 모델 shape 테스트

`tests/test_model_shapes.py`를 작성했다.

확인 대상:

- `ObservationEncoder`
- `Manager`
- `Worker`
- `FuNModel`
- FuNModel goal update behavior

확인 결과:

```text
All shape tests passed.
```

의미:

- 각 모델 모듈의 입력/출력 shape이 현재 설계와 일치한다.
- goal update interval 사이에는 goal과 hidden state가 유지된다.

### 19. FuNPolicy 테스트

`tests/test_fun_policy.py`를 작성했다.

확인 항목:

- `policy.reset()` 후 `step_count == 0`
- reset 후 hidden state shape: `(1, 64)`
- reset 후 goal shape: `(1, 16)`
- `policy.act(obs)`가 `0~6` 범위 action 반환
- `policy.act(obs)` 호출 후 `step_count`가 1 증가
- `goal_update_interval=10` 기준으로 goal이 일정 step 동안 유지되다가 갱신됨
- `get_debug_info()`에서 goal norm 변화 확인

확인 결과:

```text
All FuNPolicy tests passed.
```

주의:

- 현재 `FuNPolicy.act()`는 model 호출 후 `step_count`를 증가시킨다.
- 따라서 내부 `step_count == 10`에서 goal update가 일어나며, 외부 호출 기준으로는 11번째 `act()` 후 변화가 확인된다.

### 20. FuNPolicy rollout 실행 확인

`run_episode()`를 수정해 FuNPolicy를 직접 꽂아 episode를 실행할 수 있게 했다.

확인 내용:

- random callable policy 기존 동작 유지
- FuNPolicy stateful policy 동작 확인
- `main.py`에서 FuNPolicy로 3 episode 실행
- episode 결과 CSV 저장 확인

`main.py --run-id 1` 실행 결과:

```text
device=cpu
run=1 episode=1 reward=0.000 length=250 success=False
debug={'step_count': 250, 'hidden_norm': 0.8343132734298706, 'goal_norm': 0.5022020936012268}
run=1 episode=2 reward=0.000 length=250 success=False
debug={'step_count': 250, 'hidden_norm': 0.7664127349853516, 'goal_norm': 0.4769981801509857}
run=1 episode=3 reward=0.000 length=250 success=False
debug={'step_count': 250, 'hidden_norm': 0.7818857431411743, 'goal_norm': 0.46780428290367126}
```

생성된 로그:

```text
logs/week1_fun_policy_run_1.csv
```

주의:

- 현재 FuNPolicy는 학습된 정책이 아니므로 reward가 낮거나 success가 나오지 않는 것이 정상이다.
- 이 단계의 목적은 성능이 아니라 rollout 연결과 shape/state 흐름 확인이다.

## 실제 생성된 주요 파일

```text
fun-minigrid/
├─ main.py
├─ configs/
│  ├─ env_doorkey.yaml
│  └─ train_fun.yaml
├─ src/
│  ├─ envs/
│  │  ├─ make_env.py
│  │  ├─ preprocess.py
│  │  └─ wrappers.py
│  ├─ models/
│  │  ├─ encoder.py
│  │  ├─ manager.py
│  │  ├─ worker.py
│  │  └─ fun.py
│  ├─ policies/
│  │  └─ fun_policy.py
│  ├─ training/
│  │  ├─ rollout.py
│  │  ├─ trainer.py
│  │  └─ losses.py
│  └─ utils/
│     ├─ logger.py
│     ├─ seed.py
│     └─ metrics.py
├─ tests/
│  ├─ test_model_shapes.py
│  ├─ test_real_obs.py
│  └─ test_fun_policy.py
└─ logs/
   ├─ week1_random.csv
   ├─ week1_random_run_1.csv
   ├─ week1_random_run_2.csv
   ├─ week1_random_run_3.csv
   ├─ week1_random_run_4.csv
   └─ week1_fun_policy_run_1.csv
```

주의:

- `models/encoder.py`, `models/manager.py`, `models/worker.py`, `models/fun.py`는 최소 forward-pass 구현이 완료되었다.
- `src/policies/fun_policy.py`는 inference/testing용 policy wrapper로 구현 완료되었다.
- `tests/` 아래 모델 shape, 실제 observation, policy 동작 테스트가 작성되었다.
- `trainer.py`, `losses.py`, `wrappers.py`는 현재 파일만 만들어진 상태이며, 실질 구현은 아직 없다.
- `logs/test_episode_log.csv`는 logger 동작 확인용으로 생성된 테스트 로그이다.
- `logs/week1_random_run_4.csv`는 실행 중 중단되어 일부 결과만 들어 있을 수 있다.
- `__pycache__/` 파일들은 Python 실행 과정에서 자동 생성된 캐시 파일이다.

## 보류 중인 작업

다음 작업은 아직 보류한다.

- `trainer.py`
- `losses.py`
- replay buffer
- optimizer / gradient update
- Transformer manager
- memory ablation model
- checkpoint save/load
- 실제 FuN 학습 루프
- policy sampling과 loss 연결
- config YAML을 main 실행에 연결
- 학습 성능 개선 실험
- 학습된 checkpoint 저장/로드

## 확인한 수정 사항

기존 정리에서 보완하거나 바로잡은 점:

- `seed.py`, `metrics.py`는 "필요하면 작성"이 아니라 실제로 작성 완료된 파일이다.
- `metrics.py`의 `compute_success`는 작성되었지만 현재 `rollout.py`와 연결되어 있지는 않다.
- `main.py`는 YAML 설정 파일을 아직 읽지 않고, 내부 고정값으로 실행한다.
- `env_doorkey.yaml`의 seed는 `42`, `main.py`의 seed는 `123`으로 서로 다르다.
- `encoder.py`, `manager.py`, `worker.py`, `fun.py`는 placeholder가 아니라 최소 forward-pass 구현이 완료된 상태이다.
- `fun_policy.py`는 작성 완료되었고, `run_episode()`와 `main.py`에 연결되었다.
- `run_episode()`는 기존 callable policy와 FuNPolicy 같은 stateful policy를 모두 지원한다.
- `main.py`는 현재 random baseline이 아니라 FuNPolicy rollout 확인용으로 변경되었다.
- `trainer.py`, `losses.py`, `wrappers.py`는 아직 placeholder 상태이다.

## 1주차 완료 평가

완료한 것:

- MiniGrid 환경 설치 및 실행
- 사용할 환경을 `MiniGrid-DoorKey-5x5-v0`로 선정
- `ImgObsWrapper` 기반 환경 생성
- observation 전처리
- raw / processed observation shape 확인
- action space 크기 확인
- episode rollout loop 작성
- random policy baseline 실행
- reward, success, episode length, terminated, truncated 기록
- CSV logging
- seed 설정 유틸리티
- success metric helper
- Encoder / Manager / Worker / FuNModel 최소 구현
- 더미 입력 shape 테스트
- 실제 env observation forward 테스트
- FuNPolicy wrapper 구현
- FuNPolicy reset / act / goal update 테스트
- FuNPolicy를 `run_episode()`와 `main.py`에 연결

아직 안 한 것:

- training loop
- loss 설계
- optimizer 연결
- replay buffer
- 실제 학습
- checkpoint 저장/로드
- config YAML을 실제 실행 설정으로 연결
- 성능 개선 실험

결론:

1주차 목표였던 "MiniGrid DoorKey 환경에서 episode를 실행하고 로그를 남기는 최소 구조"는 완료했다.

추가로 2주차 초반에 해당하는 "FuN-style forward-pass 모델과 policy rollout 연결"까지 진행했다. 현재 상태는 학습 전 모델을 environment rollout에 붙여서 실행할 수 있는 단계이다.
