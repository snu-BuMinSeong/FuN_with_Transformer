# 5주차 작업 로그

## 작업 목적

DoorKey-6x6 환경에서 기존 1000 episode baseline/ablation 결과의 best checkpoint sample evaluation success rate가 낮게 나왔다. 다만 학습 중 reward signal은 발생했으므로, 이번 작업에서는 모델 구조나 학습 알고리즘을 바꾸지 않고 학습 episode 수를 3000으로 늘렸을 때 성능이 개선되는지 확인했다.

이번 작업에서 하지 않은 것:

- TransformerManager 추가
- PPO 전환
- HER 추가
- reward shaping 추가
- curriculum 추가
- hyperparameter search
- 기존 모델 구조 변경

핵심 질문은 다음 하나였다.

```text
DoorKey-6x6에서 단순히 학습 episode 수를 1000에서 3000으로 늘리면 성능이 개선되는가?
```

## 기존 1000 episode checkpoint 재평가 시도

먼저 기존 DoorKey-6x6 1000 episode best checkpoint를 100 episode 기준으로 재평가하려고 했다.

대상 checkpoint:

- `checkpoints/baseline_fun_doorkey6x6/seed_1/best.pt`
- `checkpoints/baseline_fun_doorkey6x6/seed_11/best.pt`
- `checkpoints/baseline_fun_doorkey6x6/seed_44/best.pt`
- `checkpoints/ablation_fun_doorkey6x6/seed_1/best.pt`
- `checkpoints/ablation_fun_doorkey6x6/seed_11/best.pt`
- `checkpoints/ablation_fun_doorkey6x6/seed_44/best.pt`

하지만 로컬 저장소에는 위 6개 `best.pt` 파일이 없었다. 따라서 기존 1000 episode checkpoint의 100 episode 재평가 JSON은 생성하지 못했다.

대신 기존 로그에 남아 있던 20 episode sample evaluation 결과를 참고값으로 정리했다.

| Model | Seed | Existing Sample Success 20ep | Mean Return | Episode Length | Best Checkpoint Episode |
|---|---:|---:|---:|---:|---:|
| Vanilla FuN | 1 | 0.000 | 0.000 | 250.000 | 500 |
| Vanilla FuN | 11 | 0.050 | 0.025 | 247.700 | 1000 |
| Vanilla FuN | 44 | 0.050 | 0.043 | 240.350 | 850 |
| AblationManager | 1 | 0.050 | 0.020 | 249.450 | 800 |
| AblationManager | 11 | 0.050 | 0.026 | 246.950 | 500 |
| AblationManager | 44 | 0.000 | 0.000 | 250.000 | 900 |

평균:

- Vanilla FuN: success `0.033`, mean return `0.022`, episode length `246.017`
- AblationManager: success `0.033`, mean return `0.016`, episode length `248.800`

## 3000 episode config 생성

DoorKey-6x6 seed 1, 11, 44에 대해 baseline/ablation 3000 episode config를 새로 만들었다. 기존 1000 episode 로그와 checkpoint를 덮어쓰지 않기 위해 모든 새 경로에는 `_3000`을 포함했다.

생성한 baseline config:

- `fun-minigrid/configs/train_fun_baseline_doorkey6x6_seed1_3000.yaml`
- `fun-minigrid/configs/train_fun_baseline_doorkey6x6_seed11_3000.yaml`
- `fun-minigrid/configs/train_fun_baseline_doorkey6x6_seed44_3000.yaml`

생성한 ablation config:

- `fun-minigrid/configs/train_fun_ablation_doorkey6x6_seed1_3000.yaml`
- `fun-minigrid/configs/train_fun_ablation_doorkey6x6_seed11_3000.yaml`
- `fun-minigrid/configs/train_fun_ablation_doorkey6x6_seed44_3000.yaml`

공통 설정:

```yaml
env_id: MiniGrid-DoorKey-6x6-v0
total_episodes: 3000
eval_interval: 100
eval_episodes: 20
action_mode: sample
eval_action_mode: sample
gamma: 0.99
learning_rate: 0.0003
goal_update_interval: 10
hidden_dim: 64
goal_size: 16
goal_dim: 16
entropy_coef: 0.01
value_loss_coef: 0.5
manager_loss_coef: 0.1
grad_clip_norm: 1.0
```

ablation config에는 모두 다음 설정을 명시했다.

```yaml
manager_type: ablation
```

로그/checkpoint 경로 예시:

```text
logs/baseline_fun_doorkey6x6_3000/seed_1
checkpoints/baseline_fun_doorkey6x6_3000/seed_1
logs/ablation_fun_doorkey6x6_3000/seed_1
checkpoints/ablation_fun_doorkey6x6_3000/seed_1
```

## 실행 전 점검

실행 전에 다음을 확인했다.

```bash
python -m py_compile train.py evaluate_checkpoint.py
```

확인한 항목:

- 새 config가 정상적으로 로드됨
- baseline/ablation 모두 `total_episodes: 3000`
- ablation config에 `manager_type: ablation` 존재
- 3000 episode 로그/checkpoint 경로가 기존 1000 episode 경로와 분리됨
- `_3000` 경로를 사용하므로 기존 smoke test와 1000 episode 결과를 덮어쓰지 않음

## 3000 episode 학습 실행

다음 6개 실험을 로컬 CPU에서 실행했다.

```bash
python train.py --config configs/train_fun_baseline_doorkey6x6_seed1_3000.yaml
python train.py --config configs/train_fun_baseline_doorkey6x6_seed11_3000.yaml
python train.py --config configs/train_fun_baseline_doorkey6x6_seed44_3000.yaml

python train.py --config configs/train_fun_ablation_doorkey6x6_seed1_3000.yaml
python train.py --config configs/train_fun_ablation_doorkey6x6_seed11_3000.yaml
python train.py --config configs/train_fun_ablation_doorkey6x6_seed44_3000.yaml
```

각 run의 stdout은 해당 log directory의 `run_stdout.log`에 저장했다.

예:

- `logs/baseline_fun_doorkey6x6_3000/seed_1/run_stdout.log`
- `logs/ablation_fun_doorkey6x6_3000/seed_44/run_stdout.log`

## 산출물 검증

각 run에서 다음 파일이 생성되었는지 확인했다.

Baseline:

- `logs/baseline_fun_doorkey6x6_3000/seed_*/train.csv`
- `logs/baseline_fun_doorkey6x6_3000/seed_*/eval.csv`
- `logs/baseline_fun_doorkey6x6_3000/seed_*/summary.json`
- `checkpoints/baseline_fun_doorkey6x6_3000/seed_*/best.pt`
- `checkpoints/baseline_fun_doorkey6x6_3000/seed_*/last.pt`
- `checkpoints/baseline_fun_doorkey6x6_3000/seed_*/episode_3000.pt`

Ablation:

- `logs/ablation_fun_doorkey6x6_3000/seed_*/train.csv`
- `logs/ablation_fun_doorkey6x6_3000/seed_*/eval.csv`
- `logs/ablation_fun_doorkey6x6_3000/seed_*/summary.json`
- `checkpoints/ablation_fun_doorkey6x6_3000/seed_*/best.pt`
- `checkpoints/ablation_fun_doorkey6x6_3000/seed_*/last.pt`
- `checkpoints/ablation_fun_doorkey6x6_3000/seed_*/episode_3000.pt`

검증 결과:

- 6개 run 모두 `train.csv` row 수가 3000
- 6개 run 모두 `summary.json`의 `final_episode`가 3000
- 6개 run 모두 `eval.csv`가 100 episode 간격으로 30 rows 기록됨
- 6개 run 모두 `best.pt`, `last.pt`, `episode_3000.pt` 생성됨
- 6개 run 모두 `train.csv`, `eval.csv`에서 `NaN` 또는 `inf` 문자열 없음

## 3000 best checkpoint 100 episode 평가

각 3000 episode run의 `best.pt`를 sample 기준 100 episode로 평가했다. 추가로 argmax 기준 100 episode 평가도 수행했다.

생성한 sample 평가 JSON:

- `logs/baseline_fun_doorkey6x6_3000/seed_1/checkpoint_eval_best_sample_100ep.json`
- `logs/baseline_fun_doorkey6x6_3000/seed_11/checkpoint_eval_best_sample_100ep.json`
- `logs/baseline_fun_doorkey6x6_3000/seed_44/checkpoint_eval_best_sample_100ep.json`
- `logs/ablation_fun_doorkey6x6_3000/seed_1/checkpoint_eval_best_sample_100ep.json`
- `logs/ablation_fun_doorkey6x6_3000/seed_11/checkpoint_eval_best_sample_100ep.json`
- `logs/ablation_fun_doorkey6x6_3000/seed_44/checkpoint_eval_best_sample_100ep.json`

생성한 argmax 평가 JSON:

- `logs/baseline_fun_doorkey6x6_3000/seed_1/checkpoint_eval_best_argmax_100ep.json`
- `logs/baseline_fun_doorkey6x6_3000/seed_11/checkpoint_eval_best_argmax_100ep.json`
- `logs/baseline_fun_doorkey6x6_3000/seed_44/checkpoint_eval_best_argmax_100ep.json`
- `logs/ablation_fun_doorkey6x6_3000/seed_1/checkpoint_eval_best_argmax_100ep.json`
- `logs/ablation_fun_doorkey6x6_3000/seed_11/checkpoint_eval_best_argmax_100ep.json`
- `logs/ablation_fun_doorkey6x6_3000/seed_44/checkpoint_eval_best_argmax_100ep.json`

## 3000 episode sample 평가 결과

Best checkpoint sample 100 episode 결과:

| Model | Seed | Sample Success 100ep | Mean Return | Episode Length | Best Checkpoint Episode |
|---|---:|---:|---:|---:|---:|
| Vanilla FuN | 1 | 0.080 | 0.048 | 242.700 | 2500 |
| Vanilla FuN | 11 | 1.000 | 0.899 | 40.380 | 3000 |
| Vanilla FuN | 44 | 0.190 | 0.123 | 229.320 | 3000 |
| AblationManager | 1 | 1.000 | 0.915 | 33.980 | 3000 |
| AblationManager | 11 | 0.980 | 0.861 | 52.440 | 3000 |
| AblationManager | 44 | 1.000 | 0.901 | 39.690 | 3000 |

seed 평균:

| Model | Mean Sample Success | Sample Success Std | Mean Return | Mean Episode Length |
|---|---:|---:|---:|---:|
| Vanilla FuN | 0.423 | 0.410 | 0.357 | 170.800 |
| AblationManager | 0.993 | 0.009 | 0.892 | 42.037 |

## 3000 episode argmax 평가 결과

Best checkpoint argmax 100 episode 결과:

| Model | Seed | Argmax Success 100ep | Mean Return | Episode Length |
|---|---:|---:|---:|---:|
| Vanilla FuN | 1 | 0.000 | 0.000 | 250.000 |
| Vanilla FuN | 11 | 0.340 | 0.329 | 169.440 |
| Vanilla FuN | 44 | 0.000 | 0.000 | 250.000 |
| AblationManager | 1 | 0.450 | 0.437 | 142.760 |
| AblationManager | 11 | 0.040 | 0.038 | 240.880 |
| AblationManager | 44 | 0.180 | 0.173 | 207.810 |

seed 평균:

- Vanilla FuN argmax success mean: `0.113`
- AblationManager argmax success mean: `0.223`

argmax 성능은 sample 성능보다 훨씬 낮았다. 따라서 DoorKey-6x6에서도 기존 5x5와 마찬가지로 deterministic argmax 평가만 보면 학습 성능을 과소평가할 수 있다.

## 학습 중 reward signal

3000 episode run에서 reward signal episode 수는 다음과 같았다.

| Model | Seed | Reward Signal Episodes | Final 100 Train Success |
|---|---:|---:|---:|
| Vanilla FuN | 1 | 119 | 0.100 |
| Vanilla FuN | 11 | 539 | 1.000 |
| Vanilla FuN | 44 | 101 | 0.160 |
| AblationManager | 1 | 584 | 0.990 |
| AblationManager | 11 | 576 | 0.990 |
| AblationManager | 44 | 725 | 1.000 |

관찰:

- AblationManager는 세 seed 모두 reward signal이 크게 증가했고 final 100 train success도 거의 1.0에 도달했다.
- Vanilla FuN은 seed 11에서는 성공했지만 seed 1, 44에서는 reward signal과 final success가 제한적이었다.
- Vanilla FuN은 seed sensitivity가 크고, AblationManager는 sample 평가 기준으로 매우 안정적이었다.

## 그래프 생성

seed 1에 대해 1000 vs 3000 비교 그래프를 생성했다.

- `fun-minigrid/figures/doorkey6x6_3000/seed1_1000_vs_3000_success.png`
- `fun-minigrid/figures/doorkey6x6_3000/seed1_1000_vs_3000_return.png`
- `fun-minigrid/figures/doorkey6x6_3000/seed1_1000_vs_3000_episode_length.png`

처음에는 matplotlib가 Tk backend를 잡으면서 `init.tcl` 문제로 실패했다. 이후 `MPLBACKEND=Agg`와 `MPLCONFIGDIR`를 지정해서 PNG 생성만 수행하도록 수정해 성공했다.

## 결과 문서

이번 실험 결과는 다음 문서에 정리했다.

- `fun-minigrid/results/week5_doorkey6x6_3000_comparison.md`

문서에는 다음 내용이 포함되어 있다.

- 기존 1000 checkpoint 100ep 재평가 실패 사유
- 기존 20ep 평가 참고값
- seed 1 기준 1000 vs 3000 비교
- seed 11, 44까지 확장한 3000 결과
- sample/argmax 100ep 평가 결과
- reward signal, final train success, NaN/inf 검증
- 최종 해석

## 최종 해석

DoorKey-6x6에서 단순 episode 수 증가의 효과는 모델 종류에 따라 다르게 나타났다.

AblationManager:

- seed 1, 11, 44 모두 sample 100ep success가 `0.98 ~ 1.00`
- 평균 sample success `0.993`
- 평균 episode length `42.037`
- reward signal episode도 세 seed 모두 크게 증가
- 3000 episode 확장만으로 DoorKey-6x6을 안정적으로 해결했다.

Vanilla FuN:

- seed 11에서는 sample success `1.000`까지 상승
- seed 1은 `0.080`, seed 44는 `0.190`으로 낮음
- 평균 sample success `0.423`, 표준편차 `0.410`
- seed sensitivity가 매우 큼
- argmax 평가에서는 seed 1, 44가 여전히 `0.000`

따라서 이번 결과는 다음과 같이 정리할 수 있다.

```text
DoorKey-6x6에서 3000 episode 확장은 AblationManager에는 충분히 효과적이었다.
Vanilla FuN도 일부 seed에서는 성공하지만, seed variance가 커서 안정적인 결론을 내리기 어렵다.
```

5x5 결과와 연결하면, 현재 구현과 MiniGrid DoorKey 계열에서는 recurrent Manager memory가 항상 유리하지 않으며, 더 단순한 memory-free manager가 오히려 안정적으로 최적화될 수 있다는 해석이 가능하다.

다만 argmax 성능은 두 모델 모두 sample 성능보다 낮으므로, 공식 비교 지표는 계속 sample 100 episode 평가를 primary metric으로 두고, argmax는 secondary metric으로 해석하는 것이 적절하다.

## 남은 이슈

- 기존 1000 episode DoorKey-6x6 best checkpoint 6개가 로컬에 없어 100ep 재평가를 수행하지 못했다.
- 원격 GCP 인스턴스 SSH 연결도 타임아웃이 발생해 checkpoint 회수를 하지 못했다.
- 따라서 1000 vs 3000 비교에서 1000 값은 기존 20ep evaluation reference를 사용했다.
- 향후 1000 checkpoint를 회수할 수 있으면 동일한 100ep sample 기준으로 재평가해 비교표를 보완해야 한다.

## 다음 작업 제안

- DoorKey-6x6 3000 episode 결과를 보고서 본문용 표와 해석 문단으로 정리한다.
- AblationManager가 6x6에서 안정적으로 성공한 이유를 5x5 결과와 함께 논의한다.
- Vanilla FuN의 seed sensitivity를 recurrent memory 최적화 불안정성 관점에서 분석한다.
- 가능하면 checkpoint 정책과 결과 백업 방식을 정리해, 이후 기존 checkpoint 부재 문제가 반복되지 않게 한다.
