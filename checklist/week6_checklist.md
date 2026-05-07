# Week 6 Checklist — MultiRoom 확장 실험 설계

## 이번 주 한 줄 목표

**DoorKey-6x6보다 어려운 MultiRoom 환경에서 Vanilla FuN과 AblationManager를 비교할 수 있도록, 환경 검증부터 최소 본실험까지 진행한다.**

---

## 현재 출발점

DoorKey-6x6 two-stage 실험까지 완료된 상태로 본다.

완료된 것:

- Vanilla FuN DoorKey-6x6 two-stage 결과 확보
- AblationManager DoorKey-6x6 two-stage 결과 확보
- 두 모델 모두 최종 sample success 1.000 도달
- 두 모델 모두 최종 mean argmax success 0.943 도달
- low-entropy fine-tuning이 argmax 성능 개선에 효과적임을 확인

현재 해석:

- DoorKey-6x6에서는 충분한 학습 budget과 low-entropy fine-tuning 후 두 모델 모두 강한 성능에 도달했다.
- 따라서 DoorKey-6x6만으로는 recurrent memory의 필요성을 강하게 주장하기 어렵다.
- 더 긴 시간 의존성, 부분 관측성, 탐색 난이도가 있는 MultiRoom 환경으로 확장해 memory 효과를 다시 확인한다.

---

## 0. MultiRoom 확장 목적 정리

- [ ] MultiRoom을 추가하는 이유를 짧게 문서화한다.
- [ ] DoorKey-6x6과 MultiRoom의 차이를 정리한다.
- [ ] 이번 확장의 연구 질문을 명확히 쓴다.

### 연구 질문

```text
RQ. 더 긴 탐색 경로와 복수 방 구조를 갖는 MultiRoom 환경에서는
Manager recurrent memory가 Vanilla FuN 성능에 유의미한 이점을 주는가?
```

### 기대되는 비교 포인트

- DoorKey는 비교적 짧은 horizon과 단일 key-door 구조다.
- MultiRoom은 여러 방과 문을 순차적으로 통과해야 하므로 더 긴 탐색과 과거 정보 활용이 필요할 수 있다.
- 따라서 recurrent Manager가 AblationManager보다 유리해질 가능성이 있다.
- 반대로 AblationManager가 여전히 비슷하거나 더 좋다면, 현재 FuN 구현에서 recurrent memory의 실질적 기여가 제한적이라는 결론을 강화할 수 있다.

### 완료 기준

- [ ] `results/week6_multiroom_motivation.md` 작성

---

## 1. MultiRoom 환경 후보 선정

MiniGrid의 MultiRoom 계열 중 너무 어려운 환경부터 시작하지 않는다.

### 후보 환경

- [ ] `MiniGrid-MultiRoom-N2-S4-v0`
- [ ] `MiniGrid-MultiRoom-N4-S5-v0`
- [ ] `MiniGrid-MultiRoom-N6-v0`

### 권장 시작점

```yaml
env_id: MiniGrid-MultiRoom-N2-S4-v0
```

처음부터 `N6`로 가지 않는다. 먼저 N2-S4에서 코드 호환성과 reward signal을 확인한다.

### 확인할 것

- [ ] 환경 생성 가능 여부
- [ ] observation shape 확인
- [ ] action space 확인
- [ ] max_steps 확인
- [ ] random policy success rate 측정
- [ ] untrained FuN sample success rate 측정
- [ ] untrained FuN argmax success rate 측정

### 완료 기준

- [ ] MultiRoom 후보 중 최소 1개 환경이 현재 wrapper/model/training loop와 호환됨
- [ ] random/untrained baseline 수치 확보

---

## 2. 환경 smoke test

### 해야 할 일

- [ ] `make_env()`에서 MultiRoom env_id를 config로 받을 수 있는지 확인
- [ ] 기존 `ImgObsWrapper`가 그대로 작동하는지 확인
- [ ] `preprocess_obs()` 결과 shape가 `(3, 7, 7)`인지 확인
- [ ] `FuNModel` forward가 에러 없이 도는지 확인
- [ ] `FuNPolicy` sample rollout이 에러 없이 도는지 확인
- [ ] `FuNPolicy` argmax rollout이 에러 없이 도는지 확인

### 실행 예시

```bash
python main.py --env-id MiniGrid-MultiRoom-N2-S4-v0 --run-id multiroom_smoke
```

또는 현재 `main.py`가 `--env-id`를 지원하지 않으면 먼저 추가한다.

### 완료 기준

- [ ] sample policy로 10 episode 이상 rollout 가능
- [ ] argmax policy로 10 episode 이상 rollout 가능
- [ ] observation/action/step loop 관련 에러 없음

---

## 3. MultiRoom config 추가

DoorKey-6x6 two-stage config를 복사해서 MultiRoom용으로 분리한다.

### 추가할 config

#### Stage 1

- [ ] `configs/train_fun_baseline_multiroom_n2s4_seed1.yaml`
- [ ] `configs/train_fun_ablation_multiroom_n2s4_seed1.yaml`

#### Stage 2

- [ ] `configs/finetune_fun_baseline_multiroom_n2s4_seed1.yaml`
- [ ] `configs/finetune_fun_ablation_multiroom_n2s4_seed1.yaml`

### Stage 1 권장 설정

```yaml
env_id: MiniGrid-MultiRoom-N2-S4-v0
seed: 1

total_episodes: 5000
eval_interval: 100
eval_episodes: 50

gamma: 0.99
learning_rate: 0.0003
entropy_coef: 0.01
value_loss_coef: 0.5
manager_loss_coef: 0.1
grad_clip_norm: 1.0

goal_update_interval: 10
hidden_dim: 64
goal_dim: 16

eval_action_modes:
  - sample
  - argmax
```

### Stage 2 low-entropy fine-tuning 권장 설정

```yaml
env_id: MiniGrid-MultiRoom-N2-S4-v0
seed: 1

total_episodes: 1000
eval_interval: 50
eval_episodes: 50

gamma: 0.99
learning_rate: 0.0001
entropy_coef: 0.003
value_loss_coef: 0.5
manager_loss_coef: 0.1
grad_clip_norm: 1.0

goal_update_interval: 10
hidden_dim: 64
goal_dim: 16

eval_action_modes:
  - sample
  - argmax
```

### 로그 경로 권장

```text
logs/baseline_fun_multiroom_n2s4/seed_1/
logs/ablation_fun_multiroom_n2s4/seed_1/
logs/baseline_fun_multiroom_n2s4_ft/seed_1/
logs/ablation_fun_multiroom_n2s4_ft/seed_1/
```

### 체크포인트 경로 권장

```text
checkpoints/baseline_fun_multiroom_n2s4/seed_1/
checkpoints/ablation_fun_multiroom_n2s4/seed_1/
checkpoints/baseline_fun_multiroom_n2s4_ft/seed_1/
checkpoints/ablation_fun_multiroom_n2s4_ft/seed_1/
```

### 완료 기준

- [ ] baseline/ablation 각각 Stage 1 config 생성
- [ ] baseline/ablation 각각 Stage 2 config 생성
- [ ] config loader로 정상 로드 확인

---

## 4. Seed 1 Stage 1 smoke 학습

MultiRoom은 난이도가 올라가기 때문에 바로 3 seed를 돌리지 않는다.

### 먼저 실행할 것

```text
Vanilla FuN MultiRoom-N2-S4 seed 1 Stage 1
AblationManager MultiRoom-N2-S4 seed 1 Stage 1
```

### 확인할 것

- [ ] train.csv rows가 기대 episode 수와 일치하는지 확인
- [ ] eval.csv rows가 기대 eval 횟수와 일치하는지 확인
- [ ] summary.json final_episode 확인
- [ ] best.pt 생성 확인
- [ ] last.pt 생성 확인
- [ ] episode checkpoint 생성 확인
- [ ] NaN/inf 없음 확인
- [ ] reward signal 발생 여부 확인
- [ ] sample success가 0만 나오는지 확인
- [ ] argmax success가 0만 나오는지 확인

### 중간 판단 기준

| 결과 | 판단 |
|---|---|
| 둘 다 train success 0 | 환경이 너무 어렵거나 exploration 부족 |
| sample만 조금 성공 | Stage 2 fine-tuning 진행 가능 |
| sample success 0.5 이상 | 매우 긍정적, seed 확장 후보 |
| argmax도 상승 | 보고서용 핵심 결과 후보 |
| Ablation만 성공 | memory가 오히려 방해될 가능성 |
| Vanilla만 성공 | recurrent memory 효과 가능성 |

### 완료 기준

- [ ] seed 1 Stage 1 결과 확보
- [ ] Stage 2로 넘어갈 가치가 있는지 판단

---

## 5. Seed 1 Stage 2 low-entropy fine-tuning

Stage 1에서 sample success 또는 reward signal이 확인되면 Stage 2를 진행한다.

### 해야 할 일

- [ ] Stage 1 best checkpoint 선택 기준 확정
- [ ] sample 기준 best.pt로 resume할지, argmax 기준 best_argmax.pt로 resume할지 결정
- [ ] Vanilla FuN Stage 2 실행
- [ ] AblationManager Stage 2 실행
- [ ] sample/argmax eval curve 확인
- [ ] final 100ep checkpoint evaluation 수행

### checkpoint 선택 권장

- Stage 1에서 argmax가 낮으면 `best_sample.pt`에서 fine-tuning 시작
- Stage 1에서 argmax가 이미 높으면 `best_argmax.pt` 또는 `last.pt`도 비교 가능

### final evaluation 대상

- [ ] `best_sample.pt`
- [ ] `best_argmax.pt`
- [ ] `last.pt`

각 checkpoint를 sample/argmax 각각 100 episode로 평가한다.

### 완료 기준

- [ ] seed 1 Stage 2 final 100ep evaluation JSON 생성
- [ ] sample/argmax 기준 best checkpoint 선택 가능

---

## 6. Seed 확장 여부 판단

Seed 1 결과를 보고 seed 11, 44로 확장할지 결정한다.

### seed 확장 권장 조건

다음 중 하나라도 만족하면 seed 11, 44 확장한다.

- [ ] Vanilla 또는 Ablation 중 하나라도 sample success >= 0.5
- [ ] Vanilla 또는 Ablation 중 하나라도 argmax success >= 0.3
- [ ] 두 모델 간 차이가 명확하게 관찰됨
- [ ] Stage 2 fine-tuning 후 성능 개선이 뚜렷함

### seed 확장 보류 조건

- [ ] 둘 다 sample success가 거의 0
- [ ] reward signal이 거의 없음
- [ ] train success가 거의 없음
- [ ] Stage 2에서도 개선 없음

### 확장 시 추가할 config

- [ ] `configs/train_fun_baseline_multiroom_n2s4_seed11.yaml`
- [ ] `configs/train_fun_baseline_multiroom_n2s4_seed44.yaml`
- [ ] `configs/train_fun_ablation_multiroom_n2s4_seed11.yaml`
- [ ] `configs/train_fun_ablation_multiroom_n2s4_seed44.yaml`
- [ ] `configs/finetune_fun_baseline_multiroom_n2s4_seed11.yaml`
- [ ] `configs/finetune_fun_baseline_multiroom_n2s4_seed44.yaml`
- [ ] `configs/finetune_fun_ablation_multiroom_n2s4_seed11.yaml`
- [ ] `configs/finetune_fun_ablation_multiroom_n2s4_seed44.yaml`

### 완료 기준

- [ ] seed 1 결과 기반으로 seed 확장 여부 결정
- [ ] 확장한다면 seed 11, 44 config 준비 완료

---

## 7. MultiRoom 결과 집계 스크립트 추가

DoorKey-6x6 집계 스크립트를 복사해서 MultiRoom용으로 수정한다.

### 추가할 파일

- [ ] `aggregate_multiroom_results.py`
- [ ] `plot_multiroom_results.py`

### 집계할 지표

- [ ] sample success
- [ ] argmax success
- [ ] sample mean return
- [ ] argmax mean return
- [ ] sample episode length
- [ ] argmax episode length
- [ ] best-by-sample checkpoint
- [ ] best-by-argmax checkpoint
- [ ] train final100 success
- [ ] train final100 return

### 생성할 결과 파일

- [ ] `results/week6_multiroom_results.csv`
- [ ] `results/week6_multiroom_results.md`
- [ ] `results/week6_multiroom_summary.md`

### 생성할 그래프

- [ ] `figures/multiroom/sample_success_by_seed.png`
- [ ] `figures/multiroom/argmax_success_by_seed.png`
- [ ] `figures/multiroom/argmax_vs_sample_success.png`
- [ ] `figures/multiroom/mean_return_by_seed.png`
- [ ] `figures/multiroom/episode_length_by_seed.png`
- [ ] `figures/multiroom/stage1_vs_stage2_argmax.png`

### 완료 기준

- [ ] MultiRoom 결과를 DoorKey-6x6 결과와 같은 형식으로 비교 가능

---

## 8. MultiRoom 최종 해석 기준

### 가능한 결론 A — Vanilla FuN 우세

```text
MultiRoom에서는 Vanilla FuN이 AblationManager보다 높은 success rate 또는 짧은 episode length를 보였다.
이는 복수 방 구조와 긴 탐색 horizon에서 Manager recurrent memory가 유리하게 작용할 수 있음을 시사한다.
```

### 가능한 결론 B — AblationManager 우세

```text
MultiRoom에서도 AblationManager가 Vanilla FuN과 같거나 더 높은 성능을 보였다.
이는 현재 실험 범위에서는 Manager recurrent memory가 필수적이지 않으며,
단순 feed-forward goal generation만으로도 충분한 성능을 낼 수 있음을 시사한다.
```

### 가능한 결론 C — 둘 다 실패

```text
MultiRoom에서는 두 모델 모두 충분한 성공률에 도달하지 못했다.
이는 현재 REINFORCE-style FuN 구현과 sparse reward만으로는 더 복잡한 탐색 환경을 해결하기 어렵다는 한계를 보여준다.
```

### 가능한 결론 D — sample은 성공, argmax는 실패

```text
MultiRoom에서도 sample policy는 성공 행동 분포를 학습했지만,
argmax policy는 안정적으로 성공하지 못했다.
이는 stochastic policy와 deterministic execution 사이의 gap이 더 어려운 환경에서 커질 수 있음을 보여준다.
```

---

## 9. DoorKey-6x6와 MultiRoom 비교표 준비

보고서에는 환경별 최종 결과를 하나의 표로 정리한다.

| Env | Model | Stage | Seed | Sample Success | Argmax Success | Sample Return | Argmax Return | Argmax Length |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DoorKey-6x6 | Vanilla FuN | Two-stage | 1 |  |  |  |  |  |
| DoorKey-6x6 | AblationManager | Two-stage | 1 |  |  |  |  |  |
| MultiRoom-N2-S4 | Vanilla FuN | Two-stage | 1 |  |  |  |  |  |
| MultiRoom-N2-S4 | AblationManager | Two-stage | 1 |  |  |  |  |  |

### 해석 포인트

- [ ] 환경이 어려워졌을 때 두 모델의 성능 차이가 커지는가?
- [ ] DoorKey에서는 차이가 작았지만 MultiRoom에서는 차이가 나타나는가?
- [ ] recurrent memory가 최종 성능에 기여하는가?
- [ ] recurrent memory가 sample efficiency에 기여하는가?
- [ ] low-entropy fine-tuning 효과가 MultiRoom에서도 유지되는가?

---

## 10. Week 6 종료 기준

### 최소 성공 기준

- [ ] MultiRoom 환경 smoke test 완료
- [ ] seed 1 기준 Vanilla FuN Stage 1 실행 완료
- [ ] seed 1 기준 AblationManager Stage 1 실행 완료
- [ ] reward signal 또는 실패 원인 정리
- [ ] `results/week6_multiroom_summary.md` 작성

### 권장 성공 기준

- [ ] seed 1 기준 Vanilla/Ablation Stage 2까지 완료
- [ ] final 100ep sample/argmax evaluation 완료
- [ ] DoorKey-6x6와 MultiRoom 비교표 작성

### 매우 좋은 성공 기준

- [ ] seed 1, 11, 44 전체 two-stage 실험 완료
- [ ] 평균 및 seed variance 정리
- [ ] 최종 보고서 결과 섹션에 바로 넣을 그래프 생성

---

## 이번 주에는 하지 않는 게 좋은 것

- [ ] 처음부터 `MiniGrid-MultiRoom-N6-v0` 3 seed 본실험
- [ ] TransformerManager 본실험
- [ ] PPO / A2C 등 알고리즘 전환
- [ ] reward shaping 대규모 추가
- [ ] hyperparameter grid search
- [ ] DoorKey-8x8과 MultiRoom을 동시에 확장

---

## 추천 진행 순서

1. `MiniGrid-MultiRoom-N2-S4-v0` 환경 smoke test
2. random/untrained baseline 측정
3. seed 1 Vanilla FuN Stage 1 실행
4. seed 1 AblationManager Stage 1 실행
5. reward signal과 sample success 확인
6. 가능하면 seed 1 Stage 2 low-entropy fine-tuning
7. seed 1 결과가 의미 있으면 seed 11, 44 확장
8. 결과 집계 및 DoorKey-6x6와 비교
9. 보고서 결론 수정

---

## 최종 메모

MultiRoom 확장은 연구적으로 의미가 있다. DoorKey-6x6에서는 두 모델 모두 높은 성능에 도달했기 때문에, recurrent memory의 필요성을 판단하기에는 환경이 너무 쉬웠을 수 있다.

다만 MultiRoom은 난이도가 갑자기 올라갈 수 있으므로, 이번 주 목표는 무리하게 3 seed 전체 결과를 확보하는 것이 아니라 **MultiRoom에서 memory 효과를 검증할 수 있는 실험 프로토콜을 세우고 seed 1 결과를 확보하는 것**으로 잡는 것이 안전하다.
