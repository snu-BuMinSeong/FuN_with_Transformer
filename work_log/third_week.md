# 3주차 작업 로그

## 목표

3주차 체크리스트의 1번 항목인 "학습 정상 동작 점검"만 먼저 수행했다.

## 이번에 한 일

- `checklist/week3_checklist.md`를 확인해 이번 턴의 범위를 1번 항목으로 한정했다.
- `fun-minigrid/train.py`와 `fun-minigrid/configs/train_fun.yaml`를 확인해 현재 학습 실행 경로와 로그 저장 필드를 점검했다.
- 기본 `python`으로는 `torch`가 없어 실행되지 않는 것을 확인했고, 저장소 내부 가상환경인 `fun-minigrid/.venv`를 사용해야 한다는 점을 확인했다.
- `.venv`로 `train.py --config configs/train_fun.yaml`를 실행해 300 episode 학습이 중단 없이 끝나는 것을 확인했다.
- 다만 기존 `logs/week2_train.csv`가 예전 6컬럼 episode 로그 파일이라, 이번 training row가 같은 파일에 이어붙으면서 CSV 구조가 깨지는 문제를 발견했다.
- 이 문제를 피해 점검용 설정 파일 `fun-minigrid/configs/train_fun_week3_check.yaml`를 추가했다.
- 점검용 설정으로 `train.py --config configs/train_fun_week3_check.yaml`를 다시 실행해 100 episode 학습이 중단 없이 완료되는 것을 확인했다.
- 새 로그 파일 `logs/week3_train_check.csv`와 `logs/week3_train_check_summary.json`을 기준으로 세부 지표를 확인했다.

## 학습 정상 동작 점검 결과

- 100 episode 학습이 에러 없이 완료됐다.
- `total_loss`, `worker_loss`, `value_loss`, `manager_loss`에서 NaN 또는 inf는 발생하지 않았다.
- `entropy_mean`은 약 `1.9448 ~ 1.9455` 범위로 유지되어 0으로 붕괴하지 않았다.
- `grad_norm`은 약 `0.0020 ~ 0.5077` 범위였고 NaN 또는 inf는 없었다.
- `encoder`, `manager`, `worker`, `value_head`, `manager_value_head` gradient norm이 모두 0보다 큰 episode가 확인되어 gradient가 실제로 흐르고 있었다.
- `action_coverage`는 100 episode 모두 `1.0`이었고, action histogram도 여러 action이 분포되어 있어 특정 action 하나로 고정되지는 않았다.
- `final_goal_norm`은 약 `0.4612 ~ 0.8360` 범위였고 전 episode에서 0으로 고정되지 않았다.
- 100 episode 중 성공 episode는 8회였고, reward signal이 있었던 episode도 8회였다.

## 메모

- 학습 실행 자체는 정상이다.
- 기존 `logs/week2_train.csv`는 과거 형식의 로그가 남아 있어 3주차 점검용으로는 부적절하다.
- 이후 3주차 작업에서는 training log와 episode log를 경로 단위로 분리하거나, 실행 전 로그 파일 형식을 검증하는 정리가 필요하다.
