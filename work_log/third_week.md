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

## GCP 서버 연결 설정

로컬에서 짧은 학습 정상 동작 점검은 완료했지만, seed 반복 실험과 baseline 장기 학습은 로컬보다 GCP VM에서 실행하는 방향으로 정리했다.

이번에 GCP 학습 실행을 위해 다음 환경 설정을 완료했다.

- Google Cloud SDK가 설치되어 있는 것을 확인했다.
- 현재 GCP 설정을 확인했다.
  - project: `project-ed2a3aec-0315-4f0f-95a`
  - zone: `asia-northeast3-b`
  - account: `fumin0193@gmail.com`
- 실행 중인 VM을 확인했다.
  - host: `instance-20260425-090526.asia-northeast3-b.project-ed2a3aec-0315-4f0f-95a`
  - external IP: `34.158.208.72`
- `gcloud compute config-ssh`로 GCP VM SSH 설정을 생성했다.
- Windows OpenSSH가 `~/.ssh/config` 권한 문제로 접속을 거부하는 문제를 수정했다.
- 한글 Windows 사용자명 때문에 SSH config에 깨진 경로가 들어간 문제를 수정했다.
  - `IdentityFile`을 `~/.ssh/google_compute_engine` 형식으로 정리했다.
  - `UserKnownHostsFile`을 `~/.ssh/google_compute_known_hosts` 형식으로 정리했다.
  - VM 접속 사용자를 `fumin0193`로 명시했다.
- SSH 접속 테스트를 완료했다.
  - `hostname` 결과: `instance-20260425-090526`
- VS Code Remote SSH 확장이 설치되어 있는 것을 확인했고, VS Code에서 GCP VM의 `/home/fumin0193` 원격 폴더를 열 수 있는 상태로 만들었다.

## 이후 학습 실행 계획

- 앞으로 시간이 오래 걸리는 baseline 학습은 GCP VM에서 실행한다.
- 로컬에서는 코드 수정, 짧은 smoke test, 로그 형식 점검을 중심으로 진행한다.
- GCP VM에서는 가상환경 구성 후 `train.py` 기반 학습을 실행한다.
- 장기 학습은 SSH 연결이 끊겨도 계속 돌도록 `tmux` 또는 `nohup`으로 실행한다.
- seed별 결과 비교를 위해 로그와 checkpoint는 seed별 디렉터리로 분리해 저장한다.
- 다음 단계에서는 GCP VM에서 저장소 준비, Python 환경 구성, 짧은 학습 smoke test를 먼저 수행한 뒤 seed별 baseline 실험을 실행한다.

## baseline 실험 관리 기능 보강

3주차 baseline을 재현 가능한 비교 기준으로 고정하기 위해 학습 알고리즘 자체는 유지하고, 실험 관리 기능만 보강했다.

이번에 수정하거나 추가한 주요 파일은 다음과 같다.

- `fun-minigrid/train.py`
- `fun-minigrid/src/utils/logger.py`
- `fun-minigrid/src/utils/checkpoint.py`
- `fun-minigrid/tests/test_checkpoint.py`
- `fun-minigrid/configs/train_fun_baseline_seed1.yaml`

구현한 내용은 다음과 같다.

- config의 `log_dir`, `checkpoint_dir`를 읽어 seed별 로그와 checkpoint 디렉터리를 만들도록 정리했다.
- `logs/baseline_fun/seed_1/train.csv`, `eval.csv`, `summary.json` 구조로 로그가 분리되도록 했다.
- train log에는 training 지표만 저장하고, eval 결과는 `eval.csv`에 따로 저장되도록 했다.
- evaluation은 `eval_action_mode: argmax`를 사용해 deterministic 평가가 되도록 했다.
- evaluation 결과를 `eval_success_rate`, `eval_mean_return`, `eval_std_return`, `eval_mean_episode_length`, `eval_std_episode_length`, `eval_episode_seeds` 필드로 정리해 저장했다.
- `src/utils/checkpoint.py`에 `save_checkpoint()`와 `load_checkpoint()`를 추가했다.
- checkpoint에는 model state, optimizer state, episode, config, best success rate, extra 정보를 저장하도록 했다.
- 학습 중 `last.pt`, `best.pt`, 마지막 episode checkpoint가 저장되도록 연결했다.
- `summary.json`에 config, final episode, best checkpoint path, last checkpoint path, train/eval log path, final eval, comparison text를 포함하도록 정리했다.

검증 결과는 다음과 같다.

- `python -m pytest tests/test_checkpoint.py` 실행 결과 1개 테스트가 통과했다.
- `python train.py --config configs/train_fun_baseline_seed1.yaml`로 100 episode smoke test를 실행했다.
- smoke test는 CPU에서 중단 없이 완료됐다.
- `train.csv`에는 100개 episode row가 저장됐다.
- `eval.csv`에는 episode 20, 40, 60, 80, 100 평가 결과가 저장됐다.
- `train.csv`에서 NaN 또는 inf는 발견되지 않았다.
- checkpoint 파일 `last.pt`, `best.pt`, `episode_100.pt`가 생성됐다.
- final eval 기준 success rate와 mean return은 모두 0.0이었다. 이는 sparse reward 환경에서 짧은 smoke test를 한 결과이며, 이번 작업의 목표는 성능 향상이 아니라 실험 기록 구조 검증이었다.

## seed 반복 실험 준비

3주차 baseline 장기 실험을 seed 1, 11, 44에서 같은 조건으로 반복 실행할 수 있도록 config와 실행 스크립트를 추가했다.

추가한 seed별 config는 다음과 같다.

- `fun-minigrid/configs/train_fun_baseline_seed1.yaml`
- `fun-minigrid/configs/train_fun_baseline_seed11.yaml`
- `fun-minigrid/configs/train_fun_baseline_seed44.yaml`

세 config는 다음 항목만 다르고 나머지 hyperparameter는 동일하게 맞췄다.

- `seed`
- `log_dir`
- `checkpoint_dir`

공통 baseline 설정은 다음과 같다.

- `env_id: MiniGrid-DoorKey-5x5-v0`
- `total_episodes: 1000`
- `eval_interval: 50`
- `eval_episodes: 20`
- `gamma: 0.99`
- `learning_rate: 0.0003`
- `goal_update_interval: 10`
- `hidden_dim: 64`
- `goal_size: 16`
- `goal_dim: 16`
- `entropy_coef: 0.01`
- `value_loss_coef: 0.5`
- `manager_loss_coef: 0.1`
- `grad_clip_norm: 1.0`

추가한 실행 스크립트는 다음과 같다.

- `fun-minigrid/scripts/run_baseline_seeds.sh`
- `fun-minigrid/scripts/run_seed1.sh`
- `fun-minigrid/scripts/run_seed11.sh`
- `fun-minigrid/scripts/run_seed44.sh`

또한 GCP/tmux에서 실행하기 쉽도록 `fun-minigrid/results/week3_baseline_run_commands.md`에 개별 seed 실행 명령, 전체 seed 실행 명령, tmux 예시, seed별 로그/checkpoint 저장 위치를 정리했다.

config 로더 확인 결과 세 config 모두 정상적으로 읽혔고, 각 config의 `total_episodes`가 1000으로 설정되어 있음을 확인했다.

아직 남은 작업은 GCP VM에서 seed 1, 11, 44 장기 학습을 실제로 실행하고, seed별 `train.csv`, `eval.csv`, `summary.json`, checkpoint 결과를 확보하는 것이다.

---

# 2026-04-25 작업 로그: Baseline 장기 학습 실행 및 결과 회수

## 작업 목적

MiniGrid DoorKey 환경에서 vanilla FuN baseline을 seed 1, 11, 44로 장기 학습하고, 4주차 memory ablation 비교 기준으로 사용할 결과 파일을 확보했다.

이번 작업에서는 학습 알고리즘을 수정하지 않았다. Transformer Manager, Memory Ablation, PPO 등 다른 알고리즘은 구현하지 않았다.

## 로컬 코드 정리

다음 기능을 추가하거나 정리했다.

- `evaluate_checkpoint.py`
  - 저장된 checkpoint를 config 기준으로 불러와 evaluation 수행
  - `--config`, `--checkpoint`, `--output`, `--episodes`, `--seed-offset`, `--action-mode` 지원
- `tests/test_evaluate_checkpoint.py`
  - 임시 checkpoint 저장 후 `evaluate_checkpoint()`가 로드 및 평가를 수행하는지 확인
- `aggregate_baseline_results.py`
  - seed별 `eval.csv`, `summary.json`을 읽어 결과 표 생성
  - `results/week3_baseline_results.csv`
  - `results/week3_baseline_results.md`
  - `results/week3_baseline_summary.md`
- `plot_baseline_results.py`
  - seed별 eval curve 생성
  - success rate, mean return, episode length 그래프 저장
- seed 실행 스크립트 정리
  - `scripts/run_seed1.sh`
  - `scripts/run_seed11.sh`
  - `scripts/run_seed44.sh`
  - `scripts/run_baseline_seeds.sh`
- 실행/수집 문서 정리
  - `results/week3_baseline_run_commands.md`
  - `results/week3_checkpoint_eval_commands.md`
  - `results/week3_result_collection_commands.md`
  - `results/week3_baseline_summary_template.md`

## 로컬 검증

실행한 주요 검증은 다음과 같다.

```bash
python -m pytest tests/test_checkpoint.py
python -m pytest tests/test_evaluate_checkpoint.py
python -m py_compile aggregate_baseline_results.py
python -m py_compile plot_baseline_results.py
```

로컬에서 기존 seed 1 checkpoint를 사용해 `last.pt`, `best.pt` evaluation도 확인했다.

## GCP 디스크 확장

GCP VM의 루트 디스크가 가득 차 있어 학습을 시작할 수 없었다.

확인 당시 상태:

```text
/dev/sda1  9.7G  9.2G  0  100%  /
```

GCP boot disk를 100GB로 확장했다.

- disk name: `instance-20260425-090526`
- zone: `asia-northeast3-b`
- project: `project-ed2a3aec-0315-4f0f-95a`
- size: `100GB`

VM 내부에서 `growpart`와 `resize2fs`로 루트 파티션과 ext4 파일시스템을 확장했다.

확장 후 상태:

```text
/dev/sda1  99G  9.2G  85G  10%  /
```

## GCP 저장소 재업로드

기존 원격 저장소가 오래된 상태였고, 사용자가 삭제를 허용했다.

삭제한 원격 경로:

```text
/home/fumin0193/FuN_with_Transformer
```

이후 현재 로컬의 `fun-minigrid` 코드 중 학습에 필요한 파일을 다시 업로드했다.

원격 프로젝트 경로:

```text
/home/fumin0193/FuN_with_Transformer/fun-minigrid
```

기존 전역 가상환경을 프로젝트 `.venv`로 연결했다.

```bash
ln -sfn /home/fumin0193/.venv .venv
```

확인한 환경:

- Python: `3.13.5`
- torch: `2.6.0+cu124`
- CUDA available: `True`
- GPU: Tesla T4
- `gymnasium`, `minigrid` import 가능

## GCP 학습 실행

학습 전 다음을 확인했다.

```bash
python -m py_compile train.py evaluate_checkpoint.py aggregate_baseline_results.py plot_baseline_results.py
python -m pytest tests/test_checkpoint.py
```

세 seed를 각각 tmux 세션에서 실행했다.

```bash
tmux new -d -s fun_seed1  "cd /home/fumin0193/FuN_with_Transformer/fun-minigrid && source .venv/bin/activate && python train.py --config configs/train_fun_baseline_seed1.yaml 2>&1 | tee logs/baseline_fun/seed_1/run_stdout.log"
tmux new -d -s fun_seed11 "cd /home/fumin0193/FuN_with_Transformer/fun-minigrid && source .venv/bin/activate && python train.py --config configs/train_fun_baseline_seed11.yaml 2>&1 | tee logs/baseline_fun/seed_11/run_stdout.log"
tmux new -d -s fun_seed44 "cd /home/fumin0193/FuN_with_Transformer/fun-minigrid && source .venv/bin/activate && python train.py --config configs/train_fun_baseline_seed44.yaml 2>&1 | tee logs/baseline_fun/seed_44/run_stdout.log"
```

실행 중 GPU에서 Python process 3개가 확인되었고, 각 seed의 `train.csv`가 증가하는 것을 확인했다.

## 학습 완료 및 결과 생성

학습 완료 후 tmux 세션은 종료되어 있었다.

각 seed에서 다음 파일 생성을 확인했다.

- `logs/baseline_fun/seed_x/train.csv`
- `logs/baseline_fun/seed_x/eval.csv`
- `logs/baseline_fun/seed_x/summary.json`
- `logs/baseline_fun/seed_x/run_stdout.log`
- `checkpoints/baseline_fun/seed_x/best.pt`
- `checkpoints/baseline_fun/seed_x/last.pt`
- `checkpoints/baseline_fun/seed_x/episode_1000.pt`

GCP에서 결과 집계와 그래프 생성을 실행했다.

```bash
python aggregate_baseline_results.py
python plot_baseline_results.py
```

또한 각 seed의 `best.pt`를 다시 evaluation했다.

```bash
python evaluate_checkpoint.py --config configs/train_fun_baseline_seed1.yaml  --checkpoint checkpoints/baseline_fun/seed_1/best.pt  --output logs/baseline_fun/seed_1/checkpoint_eval_best.json
python evaluate_checkpoint.py --config configs/train_fun_baseline_seed11.yaml --checkpoint checkpoints/baseline_fun/seed_11/best.pt --output logs/baseline_fun/seed_11/checkpoint_eval_best.json
python evaluate_checkpoint.py --config configs/train_fun_baseline_seed44.yaml --checkpoint checkpoints/baseline_fun/seed_44/best.pt --output logs/baseline_fun/seed_44/checkpoint_eval_best.json
```

## 로컬로 회수한 결과

GCP에서 생성된 결과 파일을 로컬 저장소로 내려받았다.

```text
fun-minigrid/logs/baseline_fun/seed_1/
fun-minigrid/logs/baseline_fun/seed_11/
fun-minigrid/logs/baseline_fun/seed_44/
fun-minigrid/checkpoints/baseline_fun/seed_1/
fun-minigrid/checkpoints/baseline_fun/seed_11/
fun-minigrid/checkpoints/baseline_fun/seed_44/
fun-minigrid/results/
fun-minigrid/figures/baseline_fun/
```

주요 결과 파일:

- `results/week3_baseline_results.csv`
- `results/week3_baseline_results.md`
- `results/week3_baseline_summary.md`
- `figures/baseline_fun/eval_success_rate.png`
- `figures/baseline_fun/eval_mean_return.png`
- `figures/baseline_fun/eval_episode_length.png`

## 최종 결과

장기 학습 결과 집계는 다음과 같다.

| Seed | Final Success Rate | Final Mean Return | Final Episode Length | Best Success Rate |
|---:|---:|---:|---:|---:|
| 1 | 0.000000 | 0.000000 | 250.000000 | 0.000000 |
| 11 | 0.000000 | 0.000000 | 250.000000 | 0.000000 |
| 44 | 0.000000 | 0.000000 | 250.000000 | 0.000000 |
| Mean | 0.000000 | 0.000000 | 250.000000 | 0.000000 |

`best.pt` checkpoint evaluation 결과도 세 seed 모두 다음과 같았다.

- mean reward: `0.000`
- success rate: `0.000`
- mean episode length: `250.000`

## 해석 메모

- vanilla FuN baseline은 1000 episode 기준으로 DoorKey sparse reward 문제를 해결하지 못했다.
- evaluation에서 모든 seed의 final success rate가 0으로 유지되었다.
- episode length가 250으로 고정되어, 평가 episode 대부분이 max step까지 진행된 것으로 보인다.
- 이 결과는 4주차 memory ablation과 비교할 baseline으로 사용할 수 있다.
- 다만 baseline 자체의 성능이 낮으므로, ablation 비교 시 성능 개선/저하보다 학습 안정성, reward signal 빈도, seed별 편차를 함께 봐야 한다.
