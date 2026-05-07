# 중간발표 정리: MultiRoom-N5 계열 제외

## 발표 핵심 메시지

이번 중간발표의 핵심은 "MiniGrid DoorKey 환경에서 FuN 구조를 직접 구현하고, Manager recurrent memory가 실제 성능에 기여하는지 ablation으로 검증했다"는 흐름으로 잡는 것이 좋다.

결론은 단정적으로 "memory가 필요 없다"가 아니라, 현재 구현과 실험 조건에서는 다음처럼 말하는 것이 안전하다.

```text
DoorKey-5x5와 DoorKey-6x6에서는 recurrent Manager memory가 뚜렷한 우위를 보이지 않았다.
오히려 memory를 제거한 AblationManager가 짧은 budget에서는 더 안정적이고 sample-efficient했다.
다만 충분한 two-stage 학습 budget을 주면 Vanilla FuN과 AblationManager 모두 DoorKey-6x6을 강하게 해결했다.
따라서 recurrent memory의 장점을 보려면 DoorKey보다 더 긴 horizon 또는 더 강한 partial-history dependency가 있는 환경이 필요하다.
```

## 추천 발표 흐름

### 1. 연구 질문

발표에서 먼저 이렇게 말하면 좋다.

```text
FuN은 Manager가 recurrent hidden state를 사용해서 상위 goal을 생성한다.
그런데 MiniGrid DoorKey 같은 sparse reward 환경에서 이 recurrent memory가 실제로 성능에 도움이 되는지 확인하고 싶었다.
그래서 Vanilla FuN과 recurrent memory를 제거한 AblationManager를 같은 조건에서 비교했다.
```

슬라이드에 넣을 것:

- Vanilla FuN 구조: Encoder -> GRU Manager -> Worker
- AblationManager 구조: Encoder -> Feedforward Manager -> Worker
- 비교 대상 표

```text
Vanilla FuN: GRUCell 기반 recurrent Manager
AblationManager: recurrent state 제거, current state embedding만으로 goal 생성
```

### 2. 구현 및 실험 파이프라인

말할 내용:

```text
초기에는 MiniGrid 환경 실행, observation preprocessing, rollout, CSV logging부터 구현했다.
이후 FuN model, policy wrapper, training rollout, return/advantage 계산, policy-gradient loss, value loss, checkpoint, evaluation, plotting까지 단계적으로 붙였다.
```

강조할 구현물:

- `fun-minigrid/src/models/encoder.py`
- `fun-minigrid/src/models/manager.py`
- `fun-minigrid/src/models/fun.py`
- `fun-minigrid/src/policies/fun_policy.py`
- `fun-minigrid/src/training/trainer.py`
- `fun-minigrid/evaluate_checkpoint.py`
- `fun-minigrid/aggregate_baseline_results.py`

슬라이드에 넣을 것:

- 프로젝트 구조 캡처
- 학습/평가 파이프라인 다이어그램
- checkpoint 흐름: `best_sample.pt`, `best_argmax.pt`, `last.pt`

### 3. 중요한 평가 기준: sample vs argmax

말할 내용:

```text
초기 argmax evaluation만 보면 모든 seed가 실패한 것처럼 보였다.
하지만 같은 checkpoint를 sample mode로 평가하면 성공률이 확인되었다.
따라서 현재 정책은 deterministic greedy policy라기보다 stochastic policy distribution으로 task-relevant behavior를 학습한 상태라고 해석했다.
그래서 sample evaluation을 primary metric으로, argmax evaluation을 deterministic deployment 참고 지표로 분리했다.
```

근거 수치: DoorKey-5x5 Vanilla FuN baseline

| Seed | Sample Success | Argmax Success |
|---:|---:|---:|
| 1 | 0.320 | 0.000 |
| 11 | 0.480 | 0.000 |
| 44 | 0.450 | 0.000 |
| Mean | 0.417 | 0.000 |

슬라이드에 넣을 그림:

- `fun-minigrid/figures/baseline_fun/argmax_vs_sample_success_rate.png`
- `fun-minigrid/figures/baseline_fun/sample_success_rate_by_seed.png`

### 4. DoorKey-5x5 memory ablation 결과

말할 내용:

```text
DoorKey-5x5에서 AblationManager는 Vanilla FuN보다 평균 sample success, return, episode length 모두 좋았다.
이는 작은 DoorKey 환경에서는 recurrent memory가 필수적이지 않고, 더 단순한 feedforward Manager가 최적화하기 쉬웠을 가능성을 보여준다.
```

핵심 결과:

| Model | Mean Sample Success | Mean Return | Mean Episode Length | Mean Argmax Success |
|---|---:|---:|---:|---:|
| Vanilla FuN | 0.417 | 0.203 | 205.177 | 0.000 |
| AblationManager | 0.603 | 0.340 | 172.350 | 0.000 |
| Difference | +0.187 | +0.137 | -32.827 | 0.000 |

슬라이드에 넣을 그림:

- `fun-minigrid/figures/ablation_fun/baseline_vs_ablation_success_rate.png`
- `fun-minigrid/figures/ablation_fun/baseline_vs_ablation_mean_return.png`
- `fun-minigrid/figures/ablation_fun/baseline_vs_ablation_episode_length.png`
- `fun-minigrid/figures/ablation_fun/seedwise_success_rate_comparison.png`

### 5. DoorKey-6x6 3000 episode 결과

말할 내용:

```text
DoorKey-6x6으로 확장했을 때 3000 episode budget에서는 차이가 더 크게 나타났다.
AblationManager는 세 seed 모두 거의 완전히 해결했지만, Vanilla FuN은 seed 11만 성공하고 seed 1, 44는 약했다.
즉 AblationManager가 더 sample-efficient하고 seed 안정성이 높았다.
```

핵심 결과:

| Model | Mean Sample Success | Mean Argmax Success |
|---|---:|---:|
| Vanilla FuN | 0.423 | 0.113 |
| AblationManager | 0.993 | 0.223 |

Seed별 sample success:

| Model | Seed 1 | Seed 11 | Seed 44 |
|---|---:|---:|---:|
| Vanilla FuN | 0.080 | 1.000 | 0.190 |
| AblationManager | 1.000 | 0.980 | 1.000 |

슬라이드에 넣을 것:

- 3000 episode 결과 표
- `fun-minigrid/figures/doorkey6x6_3000/seed1_1000_vs_3000_success.png`
- `fun-minigrid/figures/doorkey6x6_3000/seed1_1000_vs_3000_return.png`
- `fun-minigrid/figures/doorkey6x6_3000/seed1_1000_vs_3000_episode_length.png`

주의해서 말할 점:

```text
이 단계에서는 AblationManager가 훨씬 좋았지만, argmax 성공률은 sample 성공률보다 여전히 낮았다.
따라서 task acquisition과 deterministic deployment 문제를 분리해서 봐야 한다.
```

### 6. Low-entropy fine-tuning과 two-stage protocol

말할 내용:

```text
sample 성공률은 높지만 argmax 성공률이 낮은 문제를 줄이기 위해 low-entropy fine-tuning을 적용했다.
Stage 1에서는 충분히 task를 학습시키고, Stage 2에서는 낮은 entropy coefficient와 낮은 learning rate로 deterministic action selection을 안정화했다.
```

Two-stage V2 설정:

- Stage 1: 5000 episodes
- Stage 2: 1000 episodes low-entropy fine-tuning
- sample/argmax dual evaluation
- checkpoint별 final 100 episode evaluation

DoorKey-6x6 two-stage 최종 평균:

| Model | Mean Sample Success | Mean Argmax Success | Mean Argmax Length |
|---|---:|---:|---:|
| Vanilla FuN | 1.000 | 0.943 | 26.837 |
| AblationManager | 1.000 | 0.943 | 26.440 |

말할 결론:

```text
충분한 two-stage budget에서는 두 모델 모두 DoorKey-6x6을 강하게 해결했다.
따라서 DoorKey-6x6만으로는 recurrent memory의 필요성을 강하게 주장하기 어렵다.
AblationManager는 3000 episode에서는 더 빠르고 안정적이었지만, 최종 budget을 충분히 주면 Vanilla FuN도 따라온다.
```

### 7. MultiRoom-N2-S4 결과, N5 계열 제외

발표에서 MultiRoom을 넣는다면 N2-S4까지만 짧게 넣는 것이 좋다. 사용자가 제외하라고 한 `multiroom-n5` 계열 또는 N4-S5/N5 관련 결과는 본문에서 빼고, "더 어려운 환경은 향후 과제" 정도로만 말한다.

말할 내용:

```text
DoorKey-6x6에서 두 모델이 모두 강하게 성공했기 때문에 더 긴 horizon의 MultiRoom으로 확장했다.
다만 발표 범위에서는 MultiRoom-N2-S4 seed 1 결과까지만 포함한다.
N2-S4에서는 두 모델 모두 sample과 argmax success 1.000을 달성해서, 이 환경도 memory 효과를 보기에는 쉬웠다.
```

MultiRoom-N2-S4 final 100 episode evaluation:

| Model | Sample Success | Argmax Success | Sample Return | Argmax Return | Sample Length | Argmax Length |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla FuN | 1.000 | 1.000 | 0.830 | 0.835 | 7.560 | 7.350 |
| AblationManager | 1.000 | 1.000 | 0.831 | 0.836 | 7.510 | 7.270 |

말할 결론:

```text
MultiRoom-N2-S4도 두 모델 모두 쉽게 해결했기 때문에 memory advantage를 보여주지는 못했다.
따라서 다음 단계는 너무 어렵지도, 너무 쉽지도 않은 중간 난이도 환경을 찾거나 exploration 보조 전략을 검토하는 것이다.
```

### 8. 최종 결론

발표 마지막에 그대로 말하기 좋은 문장:

```text
현재까지의 결과는 recurrent Manager memory가 DoorKey 계열에서 필수적이라는 주장보다는,
현재 FuN 구현에서는 memory-free AblationManager가 더 단순하고 안정적으로 최적화될 수 있음을 보여준다.
특히 3000 episode DoorKey-6x6에서는 AblationManager가 Vanilla FuN보다 훨씬 안정적이었다.
하지만 two-stage budget을 충분히 주면 두 모델 모두 높은 deterministic 성능에 도달했다.
따라서 recurrent memory의 효과를 검증하려면 DoorKey보다 긴 horizon과 partial-history dependency가 더 강한 환경에서 추가 실험이 필요하다.
```

## 발표자료 첨부 추천 목록

우선순위 높은 자료:

1. 모델 구조 비교 그림
   - Vanilla FuN: Encoder, GRU Manager, Worker
   - AblationManager: Encoder, Feedforward Manager, Worker

2. Sample vs argmax 차이 그래프
   - `fun-minigrid/figures/baseline_fun/argmax_vs_sample_success_rate.png`

3. DoorKey-5x5 ablation 비교 그래프
   - `fun-minigrid/figures/ablation_fun/baseline_vs_ablation_success_rate.png`
   - `fun-minigrid/figures/ablation_fun/baseline_vs_ablation_mean_return.png`
   - `fun-minigrid/figures/ablation_fun/baseline_vs_ablation_episode_length.png`

4. DoorKey-6x6 3000 결과 표
   - Vanilla mean sample success 0.423
   - Ablation mean sample success 0.993

5. DoorKey-6x6 two-stage 최종 표
   - 두 모델 모두 mean sample success 1.000
   - 두 모델 모두 mean argmax success 0.943

6. MultiRoom-N2-S4 결과 표
   - 두 모델 모두 sample/argmax success 1.000
   - memory 효과 검증에는 쉬운 환경이었다는 결론

보조 자료:

- `fun-minigrid/figures/baseline_fun/sample_success_rate_by_seed.png`
- `fun-minigrid/figures/ablation_fun/seedwise_success_rate_comparison.png`
- `fun-minigrid/figures/doorkey6x6_3000/seed1_1000_vs_3000_success.png`

## 발표 슬라이드 구성안

1. 제목: FuN Manager Memory Ablation in MiniGrid
2. 연구 질문: recurrent Manager memory가 실제로 필요한가?
3. 구현 파이프라인: env, preprocessing, FuN model, training, checkpoint, evaluation
4. 모델 비교: Vanilla FuN vs AblationManager
5. 평가 기준: sample primary, argmax secondary
6. DoorKey-5x5 결과
7. DoorKey-6x6 3000 episode 결과
8. Low-entropy fine-tuning / two-stage protocol
9. DoorKey-6x6 two-stage 최종 결과
10. MultiRoom-N2-S4 확장 결과
11. 한계와 다음 단계
12. 결론

## 발표 때 조심할 표현

피해야 할 표현:

```text
recurrent memory는 필요 없다.
AblationManager가 항상 더 좋다.
FuN이 실패했다.
```

대신 쓸 표현:

```text
현재 DoorKey 실험 범위에서는 recurrent memory의 이점이 뚜렷하지 않았다.
AblationManager가 제한된 budget에서 더 안정적으로 최적화되었다.
Vanilla FuN도 충분한 two-stage budget에서는 강한 성능에 도달했다.
현재 정책은 sample과 argmax 사이의 성능 차이가 있어서 두 평가를 분리해서 해석해야 한다.
```

## 참고한 주요 문서

- `fun-minigrid/results/week3_baseline_summary.md`
- `fun-minigrid/results/week4_ablation_summary.md`
- `report/week4_doorkey5x5_interpretation_draft.md`
- `fun-minigrid/results/week5_doorkey6x6_3000_comparison.md`
- `report/week5_doorkey6x6_results_summary.md`
- `fun-minigrid/results/week5_two_stage_doorkey6x6_v2_results.md`
- `fun-minigrid/results/week6_multiroom_seed1_results.md`
