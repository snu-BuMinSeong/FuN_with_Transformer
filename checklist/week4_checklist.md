# Week 4 Checklist - DoorKey-5x5 결과 확정 및 DoorKey-6x6 확장 준비

## 이번 주 한 줄 목표

**DoorKey-5x5 baseline vs memory ablation 결과를 최종 확정하고, DoorKey-6x6 확장 실험이 바로 돌아가도록 설정·검증·실행 준비를 끝낸다.**

---

## 현재 출발점

이미 완료된 것으로 보는 항목:

- [x] Vanilla FuN 구현
- [x] AblationManager 구현
- [x] DoorKey-5x5 baseline 결과 확보
- [x] DoorKey-5x5 memory ablation 결과 확보
- [x] sample 기준 평가를 primary metric으로 확정
- [x] argmax 기준 평가는 secondary/reference metric으로 유지
- [x] baseline 결과 집계 자동화 일부 완료

이번 주에는 새 알고리즘을 추가하기보다, **이미 나온 DoorKey-5x5 결과를 보고서에 넣을 수 있는 수준으로 확정하고 DoorKey-6x6 실험을 시작할 수 있는 상태**를 만드는 데 집중한다.

---

## 1. DoorKey-5x5 결과 파일 확인

### 확인할 파일

- [x] `results/week4_ablation_results.md` 존재 여부 확인
- [x] `results/week4_ablation_summary.md` 존재 여부 확인
- [x] `results/week4_ablation_results.csv` 또는 equivalent CSV 존재 여부 확인
- [x] `figures/ablation_fun/` 아래 그래프 파일 존재 여부 확인
- [x] baseline seed별 로그 존재 여부 확인
- [x] ablation seed별 로그 존재 여부 확인
- [x] baseline seed별 checkpoint 존재 여부 확인
- [x] ablation seed별 checkpoint 존재 여부 확인

확인 메모:

- 기준 경로: `fun-minigrid/`
- 결과 파일: `fun-minigrid/results/week4_ablation_results.md`, `fun-minigrid/results/week4_ablation_summary.md`, `fun-minigrid/results/week4_ablation_results.csv`
- 그래프: `fun-minigrid/figures/ablation_fun/` 아래 PNG 10개 확인
- 로그: baseline/ablation 모두 seed 1, 11, 44 디렉터리 확인
- checkpoint: baseline/ablation 모두 seed 1, 11, 44의 `best.pt` 확인

### 완료 기준

- [x] DoorKey-5x5 결과 파일이 누락 없이 존재한다.
- [x] 결과 표와 그래프를 보고서에 바로 가져갈 수 있다.

---

## 2. DoorKey-5x5 baseline vs ablation 결과 검증

### 핵심 metric

Primary metric:

- [x] sample success rate
- [x] sample mean return
- [x] sample mean episode length

Secondary metric:

- [x] argmax success rate
- [x] train stability
- [x] seed variance
- [x] reward signal 발생 여부

### 점검할 것

- [x] baseline과 ablation이 같은 seed set을 사용했는지 확인
- [x] baseline과 ablation이 같은 total episode 수를 사용했는지 확인
- [x] baseline과 ablation이 같은 evaluation episode 수를 사용했는지 확인
- [x] baseline과 ablation이 같은 `action_mode=sample` 기준으로 평가되었는지 확인
- [x] argmax 결과가 primary metric에 섞이지 않았는지 확인
- [x] best checkpoint evaluation 결과를 사용했는지 확인
- [x] train/eval 로그가 오염되거나 append 문제를 일으키지 않았는지 확인

검증 메모:

- 비교 대상은 `MiniGrid-DoorKey-5x5-v0`, seed 1/11/44로 동일하다.
- 모든 baseline/ablation run은 `train.csv` 1000 rows, `eval.csv` 20 rows, `summary.json` final episode 1000이다.
- 학습/주기 평가 설정은 `action_mode=sample`, `eval_action_mode=sample`, `eval_episodes=20`으로 동일하다.
- 최종 비교 표는 `best.pt`를 `eval_episodes=100`으로 재평가한 `checkpoint_eval_best_sample.json`을 primary로 사용하고, `checkpoint_eval_best_argmax.json`은 secondary로 분리되어 있다.
- baseline mean: sample success 0.416667, mean return 0.203031, episode length 205.176667, argmax success 0.000000.
- ablation mean: sample success 0.603333, mean return 0.339873, episode length 172.350000, argmax success 0.000000.
- ablation - baseline: success +0.186667, return +0.136843, episode length -32.826667, argmax success +0.000000.
- reward signal은 baseline 183.33 episodes 평균, ablation 222.33 episodes 평균으로 둘 다 발생했으며 ablation 쪽이 더 많다.
- baseline 쪽에 과거 `seed_*_old_*` 백업 디렉터리가 있으나 현재 집계 대상 seed 디렉터리는 별도로 1000 rows만 있어 append 오염 증거는 없다.

### 완료 기준

- [x] DoorKey-5x5에서 AblationManager와 Vanilla FuN의 차이를 수치로 설명할 수 있다.
- [x] “memory 제거 모델이 왜 더 좋게 나왔는지 / recurrent memory 이점이 왜 명확하지 않았는지”에 대한 해석 초안이 있다.

---

## 3. DoorKey-5x5 최종 해석 문장 작성

### 작성할 문장 초안

- [x] 실험 환경 설명 문장
- [x] Vanilla FuN 구조 설명 문장
- [x] AblationManager 구조 설명 문장
- [x] sample evaluation을 primary metric으로 둔 이유
- [x] argmax evaluation이 실패 또는 낮게 나온 이유
- [x] baseline 대비 ablation 결과 요약
- [x] recurrent memory의 명확한 이점이 관찰되지 않았다는 해석
- [x] 단순한 DoorKey-5x5 환경에서는 현재 state 기반 goal 생성만으로 충분했을 가능성
- [x] 더 복잡한 DoorKey-6x6에서 memory 효과를 다시 확인해야 한다는 연결 문장

초안 파일:

- `report/week4_doorkey5x5_interpretation_draft.md`

### 보고서용 핵심 해석 구조

```text
DoorKey-5x5에서는 AblationManager가 Vanilla FuN보다 같거나 더 나은 성능을 보였다.
이는 현재 환경에서는 Manager의 recurrent memory가 성능 향상에 필수적이지 않을 수 있음을 시사한다.
다만 DoorKey-5x5는 비교적 작은 환경이므로, 환경 복잡도가 증가했을 때 recurrent memory의 효과가 나타나는지 추가 확인이 필요하다.
따라서 다음 실험은 DoorKey-6x6으로 확장한다.
```

### 완료 기준

- [x] 결과 표 아래에 붙일 수 있는 해석 문단이 완성된다.

---

## 4. DoorKey-6x6 실험 config 준비

### 만들 config

Baseline:

- [x] `configs/train_fun_baseline_doorkey6x6_seed1.yaml`
- [x] `configs/train_fun_baseline_doorkey6x6_seed11.yaml`
- [x] `configs/train_fun_baseline_doorkey6x6_seed44.yaml`

Ablation:

- [x] `configs/train_fun_ablation_doorkey6x6_seed1.yaml`
- [x] `configs/train_fun_ablation_doorkey6x6_seed11.yaml`
- [x] `configs/train_fun_ablation_doorkey6x6_seed44.yaml`

### config 수정 기준

기존 DoorKey-5x5 config를 복사한 뒤 최소한 다음 항목을 수정한다.

```yaml
env_id: MiniGrid-DoorKey-6x6-v0
seed: 1  # 11, 44도 각각 생성

total_episodes: 1000
eval_interval: 50
eval_episodes: 20

eval_action_mode: sample
train_action_mode: sample

log_dir: logs/baseline_fun_doorkey6x6/seed_1
checkpoint_dir: checkpoints/baseline_fun_doorkey6x6/seed_1
```

Ablation config는 log/checkpoint 경로를 다음처럼 분리한다.

```yaml
log_dir: logs/ablation_fun_doorkey6x6/seed_1
checkpoint_dir: checkpoints/ablation_fun_doorkey6x6/seed_1
```

### 완료 기준

- [x] seed 1 baseline/ablation config가 먼저 준비된다.
- [x] seed 11, 44 config는 seed 1 smoke test가 통과한 뒤 확장한다.

확인 메모:

- DoorKey-6x6 baseline/ablation config 6개를 모두 생성했다.
- 공통 설정은 `env_id=MiniGrid-DoorKey-6x6-v0`, `total_episodes=1000`, `eval_interval=50`, `eval_episodes=20`, `action_mode=sample`, `eval_action_mode=sample`이다.
- 로그와 checkpoint 경로는 `logs/*_doorkey6x6/seed_*`, `checkpoints/*_doorkey6x6/seed_*`로 기존 DoorKey-5x5 결과와 섞이지 않게 분리했다.
- seed 11, 44 config도 준비했지만 실행은 seed 1 smoke test 통과 후 진행한다.

---

## 5. DoorKey-6x6 smoke test 실행

### 먼저 실행할 것

- [x] Vanilla FuN DoorKey-6x6 seed 1 smoke test
- [x] AblationManager DoorKey-6x6 seed 1 smoke test

### 권장 smoke test 조건

- `total_episodes`: 20 또는 50
- `eval_interval`: 10
- `eval_episodes`: 5 또는 10
- 목적: 성능 확인이 아니라 실행 경로 확인

### 확인할 것

- [x] `train.py`가 에러 없이 실행되는지 확인
- [x] observation shape 문제가 없는지 확인
- [x] action range 문제가 없는지 확인
- [x] loss NaN/inf가 없는지 확인
- [x] grad norm NaN/inf가 없는지 확인
- [x] `train.csv` 생성 확인
- [x] `eval.csv` 생성 확인
- [x] `summary.json` 생성 확인
- [x] `last.pt` 또는 checkpoint 생성 확인

실행 메모:

- Smoke config:
  - `configs/train_fun_baseline_doorkey6x6_smoke.yaml`
  - `configs/train_fun_ablation_doorkey6x6_smoke.yaml`
- Smoke 조건: `total_episodes=20`, `eval_interval=10`, `eval_episodes=5`, seed 1.
- Baseline smoke 로그: `logs/baseline_fun_doorkey6x6/smoke`, checkpoint: `checkpoints/baseline_fun_doorkey6x6/smoke`.
- Ablation smoke 로그: `logs/ablation_fun_doorkey6x6/smoke`, checkpoint: `checkpoints/ablation_fun_doorkey6x6/smoke`.
- 두 run 모두 `train.csv` 20 rows, `eval.csv` 2 rows, `summary.json` final episode 20, `best.pt`/`last.pt`/`episode_20.pt` 생성을 확인했다.
- `train.csv`와 `eval.csv`에서 NaN/inf 문자열은 발견되지 않았다.
- Baseline smoke post-eval: success 0.200000, mean return 0.123500, mean episode length 230.600000.
- Ablation smoke post-eval: success 0.000000, mean return 0.000000, mean episode length 250.000000.

### 완료 기준

- [x] DoorKey-6x6 baseline/ablation 모두 짧은 실행이 에러 없이 끝난다.
- [x] smoke test 로그 경로가 본실험 로그와 섞이지 않는다.

---

## 6. DoorKey-6x6 seed 1 본실험 실행

### 실행 대상

- [ ] Vanilla FuN DoorKey-6x6 seed 1
- [ ] AblationManager DoorKey-6x6 seed 1

### 확인할 것

- [ ] `train.csv` rows = 1000
- [ ] `summary.json` final episode = 1000
- [ ] `best.pt` 생성
- [ ] `last.pt` 생성
- [ ] `episode_1000.pt` 생성
- [ ] `checkpoint_eval_best_sample.json` 생성
- [ ] `checkpoint_eval_best_argmax.json` 생성
- [ ] sample success가 0만 나오는지 확인
- [ ] reward signal이 있는지 확인
- [ ] episode length가 계속 max step인지 확인

### 완료 기준

- [ ] DoorKey-6x6 seed 1에서 baseline과 ablation을 최소 1회 비교할 수 있다.

---

## 7. DoorKey-6x6 seed 11, 44 확장 여부 판단

### seed 확장 조건

다음 중 하나라도 만족하면 seed 11, 44로 확장한다.

- [ ] seed 1에서 baseline 또는 ablation 중 하나라도 sample success가 0보다 크다.
- [ ] success는 낮지만 reward signal 또는 episode length 개선이 보인다.
- [ ] 학습은 실패했지만 baseline과 ablation의 안정성 차이가 보인다.
- [ ] 실행 시간이 충분하고 GCP 자원이 가능하다.

### 확장 실행 대상

- [ ] Vanilla FuN DoorKey-6x6 seed 11
- [ ] Vanilla FuN DoorKey-6x6 seed 44
- [ ] AblationManager DoorKey-6x6 seed 11
- [ ] AblationManager DoorKey-6x6 seed 44

### 완료 기준

- [ ] 가능하면 DoorKey-6x6도 seed 1, 11, 44 평균 결과를 확보한다.
- [ ] 시간이 부족하면 seed 1 결과만 pilot으로 정리한다.

---

## 8. DoorKey-6x6 결과 집계 자동화 준비

### 만들거나 수정할 스크립트

- [ ] `aggregate_doorkey6x6_results.py` 추가 또는 기존 aggregate script 확장
- [ ] `plot_doorkey6x6_results.py` 추가 또는 기존 plot script 확장

### 집계할 파일

- [ ] `logs/baseline_fun_doorkey6x6/seed_*/checkpoint_eval_best_sample.json`
- [ ] `logs/ablation_fun_doorkey6x6/seed_*/checkpoint_eval_best_sample.json`
- [ ] `logs/baseline_fun_doorkey6x6/seed_*/checkpoint_eval_best_argmax.json`
- [ ] `logs/ablation_fun_doorkey6x6/seed_*/checkpoint_eval_best_argmax.json`

### 생성할 산출물

- [ ] `results/week5_doorkey6x6_results.csv`
- [ ] `results/week5_doorkey6x6_results.md`
- [ ] `results/week5_doorkey6x6_summary.md`
- [ ] `figures/doorkey6x6/sample_success_rate_by_model.png`
- [ ] `figures/doorkey6x6/sample_mean_return_by_model.png`
- [ ] `figures/doorkey6x6/sample_episode_length_by_model.png`
- [ ] `figures/doorkey6x6/argmax_vs_sample_success_rate.png`

### 완료 기준

- [ ] DoorKey-6x6 결과를 DoorKey-5x5와 같은 형식으로 비교할 수 있다.

---

## 9. 보고서 결과 섹션 초안 시작

### 이번 주에 최소로 작성할 것

- [ ] 연구 질문 재정리
- [ ] 실험 환경 표 초안
- [ ] 모델 비교 표 초안
- [ ] DoorKey-5x5 결과 표 삽입
- [ ] DoorKey-5x5 결과 해석 문단 작성
- [ ] DoorKey-6x6 실험 목적 문단 작성

### 최종 결과표 형태

| Env | Model | Seed | Sample Success | Mean Return | Episode Length | Argmax Success |
|---|---|---:|---:|---:|---:|---:|
| DoorKey-5x5 | Vanilla | 1 | TBD | TBD | TBD | TBD |
| DoorKey-5x5 | Ablation | 1 | TBD | TBD | TBD | TBD |
| DoorKey-6x6 | Vanilla | 1 | TBD | TBD | TBD | TBD |
| DoorKey-6x6 | Ablation | 1 | TBD | TBD | TBD | TBD |

### 완료 기준

- [ ] 보고서 결과 섹션의 뼈대가 생긴다.
- [ ] 이후 DoorKey-6x6 결과만 채우면 되는 상태가 된다.

---

## 10. 이번 주에는 하지 않는 것

아래 작업은 이번 주 범위에서 제외한다.

- [ ] TransformerManager 본실험
- [ ] DoorKey-8x8 3 seed 본실험
- [ ] MultiRoom 본실험
- [ ] KeyCorridor 본실험
- [ ] PPO / HER / reward shaping 추가
- [ ] 대규모 hyperparameter search
- [ ] FuN 논문식 Manager objective 완전 재현

이유:

- 현재 핵심 연구 질문은 “Manager recurrent memory가 필요한가?”이다.
- 이미 DoorKey-5x5에서는 baseline과 ablation 비교가 가능하다.
- 다음에 필요한 것은 새 알고리즘이 아니라 **환경 난이도 증가에 따른 비교 재검증**이다.

---

## 4주차 종료 기준

이번 주가 끝날 때 아래가 만족되면 성공이다.

- [ ] DoorKey-5x5 baseline vs ablation 결과가 최종 확인되어 있다.
- [ ] DoorKey-5x5 결과 해석 문단이 있다.
- [ ] DoorKey-6x6 baseline/ablation seed 1 config가 있다.
- [ ] DoorKey-6x6 smoke test가 통과했다.
- [ ] 가능하면 DoorKey-6x6 seed 1 본실험 결과가 있다.
- [ ] 결과 집계와 그래프 생성 방향이 정해져 있다.
- [ ] 보고서 결과 섹션 초안이 시작되어 있다.

---

## 우선순위

### 1순위 - 반드시 하기

- [ ] DoorKey-5x5 결과 확정
- [ ] DoorKey-5x5 해석 문단 작성
- [ ] DoorKey-6x6 seed 1 baseline config 생성
- [ ] DoorKey-6x6 seed 1 ablation config 생성
- [ ] DoorKey-6x6 smoke test 실행

### 2순위 - 가능하면 하기

- [ ] DoorKey-6x6 seed 1 baseline/ablation 본실험
- [ ] DoorKey-6x6 결과 집계 스크립트 준비
- [ ] DoorKey-6x6 결과 그래프 스크립트 준비

### 3순위 - 여유가 있으면 하기

- [ ] DoorKey-6x6 seed 11, 44 config 생성
- [ ] DoorKey-6x6 seed 11, 44 본실험 실행
- [ ] DoorKey-8x8 seed 1 pilot config만 준비

---

## 바로 다음 액션

1. `results/week4_ablation_summary.md`와 `results/week4_ablation_results.md`를 확인한다.
2. DoorKey-5x5 baseline vs ablation의 sample metric 표를 확정한다.
3. DoorKey-5x5 최종 해석 문단을 작성한다.
4. DoorKey-6x6 seed 1 baseline/ablation config를 만든다.
5. DoorKey-6x6 smoke test를 실행한다.
