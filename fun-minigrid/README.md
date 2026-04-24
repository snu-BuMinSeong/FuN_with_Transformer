# fun-minigrid

## File Overview

- `make_env.py`: DoorKey 환경 생성
- `wrappers.py`: ImgObsWrapper 같은 래퍼 적용
- `preprocess.py`: observation tensor 변환
- `encoder.py`: 이미지 인코더
- `manager.py`: 상위 정책
- `worker.py`: 하위 정책
- `fun.py`: Manager+Worker 합친 전체 모델
- `rollout.py`: 한 episode 또는 n-step 수집
- `trainer.py`: 학습 루프
- `logger.py`: reward, success, episode length 기록
