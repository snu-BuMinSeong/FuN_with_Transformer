# 2주차 체크리스트 - Vanilla FuN 구현

## 1. 학습용 모델 출력 정리
- [x] `FuNModel`이 학습에 필요한 값을 반환하도록 수정
- [x] `Worker` 출력에서 action logits를 안정적으로 받을 수 있게 정리
- [x] action sampling에 필요한 distribution 처리 추가
- [x] 선택한 action의 `log_prob`를 얻을 수 있게 구현
- [x] value head 추가 및 value prediction 반환 연결
- [x] manager의 goal update 여부를 step마다 추적 가능하게 정리

## 2. rollout 수집 코드 작성
- [x] 기존 `run_episode()`와 별도로 학습용 rollout 수집 함수 만들기
- [x] timestep마다 observation 저장
- [x] timestep마다 action 저장
- [x] timestep마다 reward 저장
- [x] timestep마다 done / terminated / truncated 저장
- [x] timestep마다 log_prob 저장
- [x] timestep마다 goal 저장
- [x] timestep마다 step index 저장
- [x] timestep마다 value prediction 저장
- [x] timestep마다 manager 관련 학습값 저장
- [x] episode 종료 후 trajectory 형태로 반환되게 만들기
- [x] seed 넣어서 재현 가능하게 확인

## 3. return / advantage 계산
- [x] episode 단위 return 계산 함수 작성
- [x] reward-to-go 계산 확인
- [x] baseline 없는 advantage 계산 구현
- [x] predicted value를 baseline으로 사용하는 advantage 계산 연결
- [x] shape 안 꼬이는지 확인

## 4. loss 함수 구현
- [x] `losses.py`에 worker loss 작성
- [x] policy gradient 형태로 우선 단순 구현
- [x] entropy term 추가
- [x] value loss 추가
- [x] manager loss를 placeholder가 아닌 최소 실제 loss로 연결
- [x] 최종 `total_loss` 계산 구조 만들기

## 5. trainer 작성
- [x] `trainer.py`에 기본 학습 루프 작성
- [x] 한 episode rollout 수집
- [x] return/advantage 계산
- [x] loss 계산
- [x] `optimizer.zero_grad()` / `backward()` / `step()` 연결
- [x] gradient clipping 추가
- [x] 학습 중 NaN 나는지 체크
- [x] episode별 로그 출력용 summary 반환
- [x] 디버깅용 gradient / entropy / value / action summary 보강

## 6. optimizer 및 학습 설정 연결
- [x] optimizer 생성
- [x] learning rate 설정
- [x] goal update interval 설정 연결
- [x] hidden dimension, episode 수 등 학습 파라미터 정리
- [x] value loss / manager loss 계수 config 연결
- [x] evaluation episode 수 config 연결
- [x] `train.py`에서 config 읽도록 수정

## 7. 실행 및 디버깅
- [x] 1 episode 학습 실행 확인
- [x] 10 episode 정도 짧게 돌려서 에러 없는지 확인
- [x] action 범위가 정상인지 확인
- [x] reward가 정상적으로 기록되는지 확인
- [x] hidden state, goal shape가 안 깨지는지 확인
- [x] goal update interval이 의도대로 작동하는지 확인
- [x] grad norm, entropy, value range, action coverage를 확인할 수 있게 정리

## 8. 로그 및 결과 저장
- [x] episode별 return 저장
- [x] success 여부 저장
- [x] episode length 저장
- [x] 학습 로그 파일(csv/json) 저장
- [x] moving average 계산 가능하게 형식 정리
- [x] gradient / entropy / value / reward signal 관련 디버그 필드 저장

## 9. 최소 평가
- [x] 학습 전 policy 성능 1회 기록
- [x] 학습 후 policy 성능 기록
- [x] reward가 완전 랜덤 수준인지 아닌지 확인
- [x] success rate가 0만 나오는지 확인
- [x] mean_episode_length 포함
- [x] std 및 seed별 편차를 볼 수 있게 정리
- [x] 최소 비교 문장 남기기

## 10. 시각화
- [x] 학습 로그 CSV를 읽는 최소 plot 스크립트 추가
- [x] reward 그래프
- [x] success 또는 success moving average 그래프
- [x] loss 그래프

## 11. 2주차 종료 기준
이번 주 끝날 때 아래가 되면 성공이다.

- [x] vanilla FuN 학습 코드가 실제로 돈다
- [x] optimizer step까지 연결되어 있다
- [x] 학습 로그가 남는다
- [x] value baseline이 붙어 있다
- [x] manager가 완전히 고정되지 않도록 최소 학습 경로가 있다
- [ ] 최소한 reward 또는 success rate가 랜덤과 완전히 같지는 않다
- [x] 다음 주 baseline 안정화로 넘어갈 수 있다

## 우선순위

### 꼭 해야 하는 것
- [x] rollout 수집
- [x] worker loss
- [x] value loss
- [x] manager 최소 loss
- [x] trainer 작성
- [x] optimizer step 연결
- [x] 로그 저장

### 되면 좋은 것
- [x] value head
- [x] manager loss 최소 연결
- [x] config 정리
- [x] plot 코드
- [ ] checkpoint 저장/로드

## 이번 주 한 줄 목표
**구조 확인용 FuN을 학습 가능한 vanilla FuN으로 바꾸고, baseline과 manager까지 최소 학습 가능 상태로 연결하기**

## 현재 상태 한 줄 요약
worker, value, manager까지 포함한 최소 학습 파이프라인과 로그/평가/시각화는 연결됐지만, sparse reward 환경에서 실제 reward와 success 개선은 아직 확인되지 않았다.
