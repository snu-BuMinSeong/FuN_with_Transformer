# Week 4 Memory Ablation Summary

## 1. Experiment Goal

- Vanilla FuN의 GRU Manager memory를 제거했을 때 성능이 어떻게 변하는지 확인한다.

## 2. Compared Models

- Vanilla FuN: GRUCell 기반 recurrent Manager
- Ablation FuN: feedforward AblationManager, current state embedding만 사용

## 3. Experimental Setup

- Environment: MiniGrid-DoorKey-5x5-v0
- Seeds: 1, 11, 44
- total_episodes: 1000
- eval_episodes: 100 for checkpoint evaluation
- Primary metric: sample success rate, sample mean return, sample episode length
- Secondary metric: argmax success rate

## 4. Main Results

- Baseline mean sample success rate: 0.416667
- Ablation mean sample success rate: 0.603333
- Difference: 0.186667
- Baseline mean return: 0.203031
- Ablation mean return: 0.339873
- Difference: 0.136843
- Baseline mean episode length: 205.176667
- Ablation mean episode length: 172.350000
- Difference: -32.826667
- Baseline mean argmax success rate: 0.000000
- Ablation mean argmax success rate: 0.000000
- Difference: 0.000000

## 5. Interpretation

- AblationManager가 vanilla FuN보다 높거나 비슷한 성능을 보이면, MiniGrid DoorKey-5x5에서는 recurrent memory가 필수적이지 않았다고 해석할 수 있다.
- Feedforward Manager가 더 단순해서 짧은 학습 조건에서 더 안정적으로 최적화되었을 가능성이 있다.
- Sample 평가에서는 성공하지만 argmax 평가는 둘 다 실패하면, 현재 정책은 deterministic path policy라기보다 stochastic sampling에 의존한다고 해석할 수 있다.

## 6. Limitations

- Seed가 3개라 통계적으로 강한 결론은 아니다.
- DoorKey-5x5는 작은 환경이라 recurrent memory의 장점이 잘 드러나지 않을 수 있다.
- Argmax evaluation이 모두 실패하므로 정책 안정성에는 한계가 있다.
- FuN 논문식 Manager objective를 완전히 재현한 것은 아니다.

## 7. Next Step

- 결과 그래프 생성
- 필요하면 더 어려운 DoorKey 환경 또는 longer training에서 재검증
- 이후 TransformerManager는 선택적 확장으로 진행

## Missing File Note

No missing files were detected.
