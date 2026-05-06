# DoorKey-5x5 Baseline vs Memory Ablation Interpretation Draft

## Experiment Setting

본 실험은 `MiniGrid-DoorKey-5x5-v0` 환경에서 Vanilla FuN과 Manager memory ablation 모델을 비교하기 위해 수행하였다. 두 모델은 동일한 seed set인 1, 11, 44를 사용했으며, 각 seed마다 1000 episode 동안 학습했다. 학습 중 주기 평가는 20 episode 기준으로 수행했고, 최종 비교는 각 run의 `best.pt` checkpoint를 100 episode로 재평가한 결과를 사용했다.

Vanilla FuN은 Manager가 GRUCell 기반 recurrent hidden state를 유지하면서 subgoal을 생성하는 구조이다. 반면 AblationManager는 recurrent memory를 제거하고 현재 observation에서 얻은 state embedding만으로 subgoal을 생성한다. 따라서 두 모델의 비교는 DoorKey-5x5 수준의 환경에서 Manager의 recurrent memory가 실제 성능 향상에 기여하는지를 확인하는 ablation 실험으로 볼 수 있다.

## Metric Choice

본 결과에서는 sample evaluation을 primary metric으로 사용했다. 학습과 평가 모두 `action_mode=sample`을 기본 설정으로 사용했기 때문에, sample success rate, sample mean return, sample mean episode length가 학습된 정책의 실제 사용 조건에 가장 가깝다. Argmax evaluation은 deterministic policy로 고정했을 때의 참고 지표로만 사용했다.

Argmax evaluation에서는 Vanilla FuN과 AblationManager 모두 success rate가 0.000000으로 나타났다. 이는 현재 정책이 하나의 안정적인 deterministic action sequence를 학습했다기보다, stochastic sampling 과정에서 성공 경로를 탐색하는 방식에 더 의존하고 있음을 의미한다. 따라서 argmax 결과를 primary metric에 섞으면 학습 설정과 맞지 않는 기준으로 모델을 평가하게 된다.

## Result Summary

DoorKey-5x5에서 AblationManager는 Vanilla FuN보다 높은 평균 sample 성능을 보였다. Vanilla FuN의 평균 sample success rate는 0.416667이고, AblationManager의 평균 sample success rate는 0.603333으로, ablation 모델이 0.186667 높았다. 평균 return도 Vanilla FuN 0.203031에서 AblationManager 0.339873으로 증가했으며, 평균 episode length는 205.176667에서 172.350000으로 감소했다. 이는 AblationManager가 더 자주 성공했을 뿐 아니라, 성공과 실패를 포함한 전체 episode에서도 더 짧은 trajectory를 보였다는 뜻이다.

Seed별 결과를 보면 ablation의 우위는 seed 11과 seed 44에서 특히 뚜렷하다. Seed 1에서는 baseline 0.320000, ablation 0.350000으로 차이가 작았지만, seed 11에서는 0.480000 대비 0.820000, seed 44에서는 0.450000 대비 0.640000으로 ablation이 더 높은 sample success rate를 기록했다. 다만 seed가 3개뿐이므로 통계적으로 강한 결론보다는 현재 실험 조건에서 관찰된 경향으로 해석해야 한다.

## Interpretation

이 결과는 DoorKey-5x5 환경에서는 Manager의 recurrent memory가 성능 향상에 필수적이지 않았을 가능성을 시사한다. DoorKey-5x5는 비교적 작은 환경이고, 문을 열기 위한 key-door-goal 구조도 현재 observation과 짧은 horizon의 행동 선택만으로 어느 정도 해결될 수 있다. 이런 조건에서는 recurrent hidden state가 제공하는 장기 문맥 정보보다, 현재 state embedding을 바로 subgoal 생성에 사용하는 단순한 feedforward Manager가 더 최적화하기 쉬웠을 수 있다.

또한 AblationManager는 recurrent state를 유지하지 않기 때문에 학습해야 할 동역학이 더 단순하다. 짧은 1000 episode 학습 조건에서는 이 단순성이 오히려 안정적인 최적화에 유리하게 작용했을 가능성이 있다. 실제로 reward signal 발생 episode 평균도 baseline 183.33 episode, ablation 222.33 episode로 ablation 쪽이 더 높아, DoorKey-5x5에서는 memory 제거가 reward 획득 기회를 줄이지 않았다.

다만 이 결과를 recurrent memory가 일반적으로 불필요하다는 결론으로 확장해서는 안 된다. DoorKey-5x5는 환경 크기와 부분 관측 복잡도가 제한적이므로, memory의 장점이 드러나기 어려운 조건일 수 있다. 환경 크기가 커지거나 목표까지 필요한 중간 상태 추적이 길어지면 recurrent memory가 더 중요한 역할을 할 수 있다. 따라서 다음 실험은 DoorKey-6x6으로 확장하여, 환경 복잡도가 증가했을 때 Vanilla FuN의 recurrent Manager가 ablation 대비 이점을 보이는지 재검증하는 방향이 적절하다.

## Short Report Paragraph

DoorKey-5x5 실험에서는 Manager의 recurrent memory를 제거한 AblationManager가 Vanilla FuN보다 높은 sample 평가 성능을 보였다. 세 seed 평균 기준으로 sample success rate는 0.416667에서 0.603333으로 증가했고, mean return은 0.203031에서 0.339873으로 증가했으며, mean episode length는 205.176667에서 172.350000으로 감소했다. 반면 argmax success rate는 두 모델 모두 0.000000으로, 현재 정책은 deterministic action selection보다 stochastic sampling 조건에서 성공 경로를 찾는 경향이 강했다. 이 결과는 작은 DoorKey-5x5 환경에서는 Manager의 recurrent memory가 필수적인 성능 요소로 작용하지 않았고, 현재 state 기반 subgoal 생성만으로도 충분했을 가능성을 시사한다. 다만 DoorKey-5x5는 단순한 환경이므로, recurrent memory의 효과가 더 복잡한 DoorKey-6x6에서 다시 나타나는지 추가 검증이 필요하다.
