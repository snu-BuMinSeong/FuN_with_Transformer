# 최종발표 포스터 활용 가이드

## 한 줄 메시지

Vanilla FuN의 recurrent manager memory를 제거한 AblationManager는 tested MiniGrid 환경에서 성능을 망치지 않았고, 제한된 학습 budget에서는 더 빠르고 안정적으로 과제를 획득하는 경우가 많았다. 다만 최종 성공률이 동률인 환경도 있으므로, "항상 우월"이 아니라 "더 단순한 manager가 충분하며 일부 조건에서 sample-efficient"라고 쓰는 것이 안전하다.

## 권장 포스터 구성

### 1. Problem / Model

- 질문: FuN manager의 recurrent hidden state가 MiniGrid navigation에서 꼭 필요한가?
- 비교군: Vanilla FuN(recurrent manager) vs AblationManager(memory-ablated manager).
- 평가 방식: `sample`은 학습 중 사용한 stochastic policy, `argmax`는 deterministic deployment 성능.
- Method 영역에는 모델 구조를 간단히 그려라: observation/encoder -> manager goal -> worker policy -> action. Baseline에는 recurrent hidden state, Ablation에는 hidden-state 제거를 표시하면 된다.

### 2. DoorKey-6x6: limited-budget sample efficiency

사용 파일:

- 메인 그래프: `figures/poster_main/doorkey6x6_3000_mean_success.png`
- 보조 그래프: `figures/doorkey6x6_3000/*.png`
- 근거 문서: `results/doorkey6x6/week5_doorkey6x6_results_summary.md`

핵심 문장:

- 3000 episodes에서 AblationManager는 3 seeds mean sample success `0.993`, Vanilla FuN은 `0.423`.
- 같은 3000 episodes에서 argmax success는 둘 다 낮지만 AblationManager가 더 높다: `0.223` vs `0.113`.
- 더 큰 two-stage budget(5000 + 1000 low-entropy FT)에서는 두 모델 모두 mean argmax success `0.943`까지 올라간다.

포스터 해석:

- AblationManager의 장점은 "Vanilla가 못 푼다"가 아니라 "적은 budget에서 더 일관되게 먼저 푼다"로 표현하라.
- sample과 argmax 차이를 별도 메시지로 두면 deterministic deployment의 중요성을 설명하기 좋다.

### 3. MultiRoom N3-S5: checkpoint robustness

사용 파일:

- 결과표: `results/multiroom_n3s5/multiroom_n3_s5_checkpoint_eval_summary.md`
- CSV: `results/multiroom_n3s5/multiroom_n3_s5_checkpoint_eval_summary.csv`

핵심 문장:

- N3-S5에서는 두 모델 모두 거의 해결한다.
- best argmax 기준으로 Baseline은 success `1.000`, return `0.784`, length `14.400`; AblationManager는 success `1.000`, return `0.790`, length `14.000`.

포스터 해석:

- 이 결과는 AblationManager가 memory 제거에도 성능을 유지한다는 중간 난이도 근거로 쓰면 된다.
- 크게 강조하기보다는 DoorKey와 N4-S4 사이의 bridge result로 배치하라.

### 4. MultiRoom N4-S4: final success tie, learning process difference

사용 파일:

- 메인 그래프: `figures/poster_main/multiroom_n4s4_auc_and_speed.png`
- 효율 그래프: `figures/poster_main/multiroom_n4s4_final_episode_length.png`
- 원본 학습 곡선: `figures/multiroom_n4s4_learning_curves/*.png`
- 근거 문서: `results/multiroom_n4s4/multiroom_n4_s4_three_seed_process_summary.md`

핵심 문장:

- 최종 success는 sample/argmax 모두 두 모델이 3 seeds mean `1.000`.
- 그러나 AblationManager가 더 빠르게 threshold에 도달한다.
- sample success AUC: Baseline `0.690`, Ablation `0.753`.
- argmax success AUC: Baseline `0.497`, Ablation `0.541`.
- 1.0 success 도달 episode 평균: sample `2366.7 -> 2100.0`, argmax `3033.3 -> 2500.0`.

포스터 해석:

- 성공률만 보여주면 차이가 사라진다. 반드시 learning speed/AUC와 threshold crossing을 같이 보여줘야 한다.
- final episode length는 mixed result다. sample에서는 Baseline이 더 짧고, argmax에서는 AblationManager가 더 짧다.

### 5. Mechanism diagnostics: hidden state and goal interface

사용 파일:

- 메인 그래프: `figures/poster_main/manager_goal_policy_coupling.png`
- 메인 그래프: `figures/poster_main/hidden_intervention_success_drop.png`
- 원본 그래프: `figures/manager_goal_analysis/*.png`
- 근거 문서: `results/multiroom_n4s4_manager_goal_analysis/manager_goal_results.md`
- 근거 문서: `results/multiroom_n4s4_hidden_intervention/hidden_intervention_analysis.md`

핵심 문장:

- Baseline final checkpoint에서 hidden reset은 argmax success를 평균 약 `0.03`만 떨어뜨린다.
- manager goal stability 차이는 작다. argmax goal_delta는 Baseline `0.0473`, Ablation `0.0402`.
- goal update 시 worker action distribution은 실제로 바뀐다. update action KL은 sample 약 `0.65~0.74`, argmax 약 `1.02~1.03`.

포스터 해석:

- 이 진단은 "final policy가 recurrent hidden state에 강하게 의존하지 않는 것처럼 보인다"는 근거다.
- 단, final checkpoint 진단이므로 "Ablation이 왜 더 빨리 학습했는지 증명"한다고 과장하지 마라.

### 6. MultiRoom N4-S5: boundary condition

사용 파일:

- 메인 그래프: `figures/poster_main/multiroom_n4s5_reward_signal_boundary.png`
- 결과 문서: `results/multiroom_n4s5_boundary/week6_multiroom_n4s5_maxsteps_sweep_summary.md`

핵심 문장:

- N4-S5에서는 tested runs 모두 final eval success가 `0`.
- max_steps를 늘리면 train reward signal은 증가한다. Ablation ms1000은 reward signal episodes `16`까지 증가했지만 eval success는 여전히 `0`.

포스터 해석:

- "현재 방법의 확장 한계" 또는 "future work" 패널에 넣어라.
- negative result라서 오히려 실험 범위를 솔직하게 보여주는 역할을 한다.

## 발표 시 주의할 표현

- 피해야 할 표현: "AblationManager is universally better."
- 권장 표현: "AblationManager is sufficient and often more sample-efficient under the tested budgets."
- 피해야 할 표현: "Hidden state is useless."
- 권장 표현: "Final Baseline checkpoints show weak dependence on recurrent hidden state under this intervention."
- 피해야 할 표현: "N4-S5 was solved."
- 권장 표현: "N4-S5 produced sparse train reward signals but no evaluation success under the tested sweep."

## 압축본 사용법

`final_poster.zip`에는 이 폴더 전체가 들어 있다. 포스터 제작 시에는 `figures/poster_main`의 PNG를 우선 사용하고, 수치 인용은 `poster_key_numbers.csv` 또는 각 `results/*.md`의 표를 확인하면 된다.
