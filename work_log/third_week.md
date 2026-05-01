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

이 시점에서 남은 작업은 GCP VM에서 seed 1, 11, 44 장기 학습을 실제로 실행하고, seed별 `train.csv`, `eval.csv`, `summary.json`, checkpoint 결과를 확보하는 것이었다.

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

---

# 2026-04-25 작업 로그: Baseline 실패 원인 진단

## 작업 목적

seed 1, 11, 44 baseline 장기 학습 결과에서 argmax evaluation success rate가 모두 0으로 나온 원인을 진단했다.

이번 작업에서는 모델 구조를 바꾸지 않았다. Transformer Manager, Memory Ablation, PPO, reward shaping은 구현하지 않았다.

## 추가한 진단 파일

- `fun-minigrid/diagnose_baseline.py`
- `fun-minigrid/configs/train_fun_empty5x5_sanity_seed1.yaml`
- `fun-minigrid/results/week3_diagnosis_commands.md`
- `fun-minigrid/results/week3_baseline_diagnosis.md`
- `fun-minigrid/results/week3_baseline_diagnosis.json`

## 실행한 진단

다음 항목을 실제로 실행했다.

- `diagnose_baseline.py` 문법 검사
- seed 1, 11, 44의 `best.pt` checkpoint sample evaluation
- random policy baseline 측정
- untrained FuNPolicy sample / argmax baseline 측정
- `MiniGrid-Empty-5x5-v0` sanity 학습
- Empty-5x5 best checkpoint argmax / sample evaluation
- 최종 diagnosis report 생성

## Train Log 진단 결과

argmax evaluation 결과만 보면 학습이 전혀 안 된 것처럼 보였지만, train log에는 성공과 reward signal이 존재했다.

| Seed | train rows | train success | train success rate | final 50 success | reward max |
|---:|---:|---:|---:|---:|---:|
| 1 | 1100 | 263 | 0.2391 | 0.7600 | 0.9280 |
| 11 | 1000 | 165 | 0.1650 | 0.5800 | 0.9244 |
| 44 | 1000 | 120 | 0.1200 | 0.2200 | 0.8596 |

추가 관찰은 다음과 같다.

- 세 seed 모두 `total_reward` max가 0보다 컸다.
- 세 seed 모두 `has_reward_signal`, `has_return_signal`이 true인 episode가 있었다.
- `action_coverage` 평균은 거의 1.0으로, action 하나로 collapse된 상태는 아니었다.
- action histogram도 특정 action 하나가 90% 이상을 차지하지 않았다.
- loss, advantage, value, log probability 계열 값은 NaN/inf 없이 기록되어 있었다.
- seed 1은 `train.csv` row가 1100인데 `summary.json` final episode는 1000이라, 이전 실행 로그가 append되었을 가능성이 있다.

## Argmax Eval과 Sample Eval 비교

기존 baseline evaluation은 `argmax` 기준이었고, 세 seed 모두 success rate가 0이었다.

하지만 같은 best checkpoint를 `sample` action mode로 평가하면 성공이 나왔다.

| Seed | argmax best success | sample success | sample mean return | sample mean episode length |
|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.8500 | 0.5629 | 117.2500 |
| 11 | 0.0000 | 0.6200 | 0.3351 | 174.1500 |
| 44 | 0.0000 | 0.2000 | 0.0939 | 229.4800 |

이 결과는 policy가 완전히 학습하지 못한 것이 아니라, stochastic sampling에서는 성공 가능한 행동 분포를 가지고 있지만 deterministic argmax trajectory에서는 실패하고 있음을 의미한다.

## Random / Untrained Baseline

DoorKey-5x5에서 100 episode 기준 random 및 untrained baseline도 측정했다.

| Policy | success | mean return | mean episode length |
|---|---:|---:|---:|
| random policy | 0.1200 | 0.0400 | 242.2300 |
| untrained FuN sample | 0.1400 | 0.0527 | 239.2400 |
| untrained FuN argmax | 0.0000 | 0.0000 | 250.0000 |

trained checkpoint의 sample 평균 success는 약 0.557로, untrained sample success 0.140보다 높았다. 따라서 학습이 전혀 없었다고 보기는 어렵다.

## Empty-5x5 Sanity Check

DoorKey sparse reward만의 문제인지 확인하기 위해 `MiniGrid-Empty-5x5-v0`에서 300 episode sanity 학습을 실행했다.

결과는 다음과 같다.

- train 중 success와 reward가 발생했다.
- best checkpoint argmax evaluation success rate는 0.0000이었다.
- best checkpoint sample evaluation success rate는 0.7300이었다.
- sample mean return은 0.4625, mean episode length는 56.72였다.

즉 쉬운 Empty-5x5 환경에서도 argmax는 실패하고 sample은 성공했다. 따라서 현재 현상은 DoorKey sparse reward만으로 설명하기 어렵고, evaluation action mode의 영향이 크다.

## Warning

자동 진단에서 확인된 warning은 다음과 같다.

- eval success가 모든 seed에서 0이다.
- seed 1의 `train.csv` rows가 1100으로, `summary.json`의 final episode 1000과 다르다.

## 결론

현재 가장 가능성 높은 원인은 **evaluation argmax 방식 문제**다.

근거는 다음과 같다.

- train 중에는 성공과 reward signal이 있었다.
- best checkpoint를 sample mode로 평가하면 세 seed 모두 성공이 나온다.
- trained sample 성능은 random/untrained sample보다 높다.
- Empty-5x5에서도 sample은 성공하지만 argmax는 실패한다.
- action distribution은 한 action으로 collapse되지 않았다.

따라서 “vanilla FuN이 DoorKey에서 완전히 학습하지 못했다”라고 단정하기보다, 현재 정책이 deterministic argmax 평가에서 실패하는 형태로 학습되었다고 보는 것이 더 정확하다.

## 다음 액션

- argmax와 sample evaluation을 모두 baseline 지표로 유지한다.
- eval episode별 action sequence와 first-action distribution을 추가로 확인한다.
- checkpoint별 sample success curve를 만들어 학습 진행 중 sample 성능이 언제 개선되는지 본다.
- argmax 실패 원인을 확인하기 전에는 reward shaping, curriculum, intrinsic reward를 바로 넣지 않는다.
- 이후 reward shaping이나 curriculum을 검토하더라도, 이번 진단 결과와 분리된 별도 실험으로 진행한다.

---

# 2026-04-25 작업 로그: Sample 기준 Baseline 재학습 및 최종 정리

## 작업 목적

기존 argmax evaluation 기준에서는 모든 seed의 success가 0으로 나왔지만, sample checkpoint evaluation에서는 학습 성능이 확인되었다.

따라서 3주차 baseline을 4주차 memory ablation의 비교 기준으로 쓰기 위해 공식 평가 기준을 sample evaluation으로 전환하고, seed 1, 11, 44를 같은 기준으로 다시 정리했다.

이번 작업에서도 학습 알고리즘과 모델 구조는 수정하지 않았다. Transformer Manager, Memory Ablation, PPO, reward shaping은 구현하지 않았다.

## 평가 기준 변경

baseline config의 evaluation mode를 sample 기준으로 맞췄다.

- `configs/train_fun_baseline_seed1.yaml`
- `configs/train_fun_baseline_seed11.yaml`
- `configs/train_fun_baseline_seed44.yaml`

공식 평가 기준:

- sample success rate
- sample mean return
- sample mean episode length

참고 평가 기준:

- argmax success rate
- argmax mean return
- argmax episode length

## 로그 오염 정리 및 재학습

seed 1은 이전 smoke test 로그가 같은 `train.csv`에 append되어 row 수가 1100으로 늘어난 문제가 있었다.

이를 해결하기 위해 기존 seed 1 결과를 백업하고 다시 학습했다.

백업 경로:

- `logs/baseline_fun/seed_1_old_mixed_20260425_212418`
- `checkpoints/baseline_fun/seed_1_old_mixed_20260425_212418`

seed 11, seed 44도 기존 argmax 기준 결과를 보존한 뒤 sample eval config 기준으로 재학습했다.

백업 경로:

- `logs/baseline_fun/seed_11_old_argmax_20260425_214945`
- `logs/baseline_fun/seed_44_old_argmax_20260425_214945`
- `checkpoints/baseline_fun/seed_11_old_argmax_20260425_214945`
- `checkpoints/baseline_fun/seed_44_old_argmax_20260425_214945`

재학습 후 세 seed 모두 다음 조건을 만족했다.

- `train.csv` rows: 1000
- `summary.json` final episode: 1000
- `best.pt`, `last.pt`, `episode_1000.pt` 생성
- `checkpoint_eval_best_sample.json` 생성
- `checkpoint_eval_best_argmax.json` 생성

## 재학습 Train Log 결과

| Seed | train success | success rate | final50 success | final100 success | reward max | reward mean | summary best success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 164 / 1000 | 0.1640 | 0.5000 | 0.3900 | 0.9424 | 0.077082 | 0.40 |
| 11 | 205 / 1000 | 0.2050 | 0.3600 | 0.3700 | 0.9532 | 0.092050 | 0.50 |
| 44 | 181 / 1000 | 0.1810 | 0.3800 | 0.4200 | 0.9532 | 0.088102 | 0.65 |

세 seed 모두 train 중 성공과 reward signal이 확인되었다.

## Best Checkpoint Evaluation 결과

각 seed의 `best.pt`를 100 episode로 sample/argmax 각각 평가했다.

| Seed | Sample Success Rate | Sample Mean Return | Sample Episode Length | Argmax Success Rate | Argmax Mean Return | Argmax Episode Length |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.320000 | 0.127364 | 223.510000 | 0.000000 | 0.000000 | 250.000000 |
| 11 | 0.480000 | 0.243948 | 195.570000 | 0.000000 | 0.000000 | 250.000000 |
| 44 | 0.450000 | 0.237780 | 196.450000 | 0.000000 | 0.000000 | 250.000000 |
| Mean | 0.416667 | 0.203031 | 205.176667 | 0.000000 | 0.000000 | 250.000000 |

sample 기준으로는 세 seed 모두 성공이 확인되었다. argmax 기준은 세 seed 모두 여전히 실패했다.

## 갱신한 코드와 문서

수정하거나 갱신한 주요 파일은 다음과 같다.

- `fun-minigrid/aggregate_baseline_results.py`
  - 공식 metric을 `checkpoint_eval_best_sample.json` 기준으로 집계
  - argmax 결과를 보조 metric으로 별도 컬럼에 기록
  - sample JSON이 없으면 eval CSV fallback을 사용하고 note 출력
- `fun-minigrid/plot_baseline_results.py`
  - sample checkpoint evaluation 기준 seed별 bar plot 생성
  - argmax vs sample success 비교 그래프 생성
  - Windows/Tk 문제를 피하기 위해 `Agg` backend 사용
- `fun-minigrid/diagnose_baseline.py`
  - eval action mode를 명시
  - argmax checkpoint evaluation과 sample checkpoint evaluation을 분리해 표시
- `fun-minigrid/results/week3_baseline_summary.md`
  - sample 기준 baseline summary로 갱신
- `fun-minigrid/results/week3_baseline_results.md`
  - sample/argmax metric을 함께 포함하는 최종 결과 표로 갱신

## 생성한 그래프

다음 그래프를 생성했다.

- `figures/baseline_fun/sample_success_rate_by_seed.png`
- `figures/baseline_fun/sample_mean_return_by_seed.png`
- `figures/baseline_fun/sample_episode_length_by_seed.png`
- `figures/baseline_fun/argmax_vs_sample_success_rate.png`
- `figures/baseline_fun/argmax_eval_success_rate_curve.png`
- `figures/baseline_fun/eval_mean_return.png`
- `figures/baseline_fun/eval_episode_length.png`

## 최종 해석

- vanilla FuN baseline은 sample 기준으로 DoorKey에서 완전히 실패한 것은 아니다.
- train log와 sample checkpoint evaluation 모두에서 성공/reward signal이 확인된다.
- argmax deterministic evaluation은 행동 다양성을 제거하면서 DoorKey task에서 실패하는 것으로 보인다.
- 따라서 4주차 memory ablation의 primary metric은 sample 기준으로 둔다.
- argmax 결과는 참고 지표로 남기되, 공식 성능 비교 기준으로 사용하지 않는다.
- seed별 편차가 있으므로 평균뿐 아니라 seed별 결과도 함께 보고해야 한다.

## 4주차 비교 기준

primary:

- sample success rate
- sample mean return
- sample mean episode length

secondary:

- argmax success rate
- train stability
- seed variance

---

# 2026-05-01 작업 로그: 최신 현황 반영 및 코드 전체 점검

## 작업 목적

3주차 문서와 실제 코드/산출물 상태가 맞는지 다시 확인하고, 오래된 체크리스트 항목을 최신 baseline 결과 기준으로 정리했다.

## 체크리스트 최신화

- `checklist/week3_checklist.md` 상단의 GCP 미완료 항목을 최신 실행 결과 기준으로 완료 처리했다.
- GCP 저장소 재업로드, Python/CUDA 환경 확인, 문법 검사, checkpoint 테스트, seed별 장기 학습, 로그/checkpoint 경로 정리가 완료된 상태임을 반영했다.
- `entropy_coef=0.01`, `learning_rate=0.0003`, `goal_update_interval=10` 조합을 3주차 baseline 고정 hyperparameter로 기록했다.
- 별도 grid search는 하지 않았고, 3주차 목표가 성능 최적화가 아니라 4주차 비교용 baseline 고정이었기 때문에 안정적으로 실행되고 seed별 결과를 확보한 조합을 baseline으로 사용한다고 정리했다.
- `week3_baseline_summary.md`에 sparse reward 한계와 argmax/sample 실패 양상 해석이 이미 들어가 있으므로 관련 TODO를 완료 처리했다.
- 누락되어 있던 `fun-minigrid/results/week3_sample_eval_commands.md` 산출물을 체크리스트에 추가했다.

## 코드 전체 점검 결과

현재 코드는 vanilla FuN baseline을 학습, 평가, 진단, 결과 집계까지 수행할 수 있는 상태다.

### 모델

- `src/models/encoder.py`
  - MiniGrid `ImgObsWrapper`의 `7x7x3` observation을 `3x7x7` tensor로 받아 CNN 기반 embedding으로 변환한다.
- `src/models/manager.py`
  - `nn.GRUCell` 기반 Manager memory를 사용한다.
  - state embedding과 hidden state를 입력으로 받아 다음 hidden state와 goal vector를 만든다.
- `src/models/worker.py`
  - state embedding과 goal vector를 concat해서 discrete action logits를 출력한다.
- `src/models/fun.py`
  - encoder, GRU Manager, Worker, value head, manager value head를 묶은 현재 baseline 모델이다.
  - `goal_update_interval`마다만 Manager를 호출하고, 그 외 step에서는 이전 goal과 hidden state를 유지한다.

### 정책과 rollout

- `src/policies/fun_policy.py`
  - recurrent hidden state와 current goal을 episode마다 reset한다.
  - `action_mode`를 `sample` 또는 `argmax`로 선택할 수 있다.
  - 학습용 `act_for_training()`은 action, log_prob, entropy, value, manager_value, goal, goal update flag를 반환한다.
- `src/training/rollout.py`
  - 일반 evaluation episode와 학습 trajectory 수집을 분리한다.
  - 학습 trajectory에는 reward, done, log_prob, entropy, value, manager_value, goal, goal update 여부가 저장된다.

### 학습 로직

- `src/training/returns.py`
  - discounted reward-to-go와 value baseline 기반 advantage를 계산한다.
- `src/training/losses.py`
  - worker policy-gradient loss, value MSE loss, entropy bonus, manager value loss를 계산한다.
  - manager loss는 goal update step에 대해서만 manager value와 return target의 MSE를 계산한다.
- `src/training/trainer.py`
  - episode 단위 rollout, return/advantage 계산, loss 계산, backward, gradient clipping, optimizer step을 수행한다.
  - NaN/inf loss와 non-finite grad norm을 검사한다.
  - train log에 action coverage, reward signal, return signal, goal norm, module별 grad norm 등 진단 지표를 남긴다.

### 학습 실행과 기록

- `train.py`
  - flat YAML config를 읽어 env/model/policy/optimizer를 구성한다.
  - train policy와 eval policy를 분리하며, 현재 baseline config는 둘 다 `sample` 기준이다.
  - interval evaluation을 `eval.csv`에 저장하고, `best.pt`, `last.pt`, `episode_1000.pt` checkpoint를 저장한다.
  - `summary.json`에 config, checkpoint path, final eval, moving average, reward/return signal, grad norm 요약을 저장한다.
- `src/utils/logger.py`
  - train/eval CSV schema를 분리한다.
- `src/utils/checkpoint.py`
  - model state, optimizer state, episode, config, best success rate, extra metadata를 저장/로드한다.

### 평가, 진단, 집계

- `evaluate_checkpoint.py`
  - 저장된 checkpoint를 config 기준으로 복원하고 `sample` 또는 `argmax` 평가를 실행한다.
- `aggregate_baseline_results.py`
  - seed 1, 11, 44의 `checkpoint_eval_best_sample.json`을 공식 metric으로 집계한다.
  - argmax 결과는 보조 metric으로 함께 기록한다.
- `plot_baseline_results.py`
  - sample 기준 seed별 bar plot과 argmax/sample 비교 plot을 생성한다.
  - 기존 `eval.csv` curve는 argmax reference curve로 유지한다.
- `diagnose_baseline.py`
  - train log, eval log, checkpoint sample/argmax 평가, random/untrained baseline, Empty-5x5 sanity 결과를 종합해 argmax 평가 문제를 진단한다.

## 현재 baseline 산출물 상태

- seed 1, 11, 44 모두 sample 기준 재학습 결과가 정리되어 있다.
- 각 seed는 `train.csv` 1000 row와 `summary.json` final episode 1000이 일치한다.
- 각 seed에 `best.pt`, `last.pt`, `episode_1000.pt`가 있다.
- 각 seed에 `checkpoint_eval_best_sample.json`과 `checkpoint_eval_best_argmax.json`이 있다.
- 최종 공식 결과는 sample 기준 평균 success rate `0.416667`, mean return `0.203031`, episode length `205.176667`이다.
- argmax 기준 success rate는 세 seed 모두 `0.000000`이므로 참고 지표로만 유지한다.

## 4주차 memory ablation 관점에서 본 코드 상태

- Manager memory가 있는 핵심 위치는 `src/models/manager.py`의 `nn.GRUCell`과 `src/models/fun.py`의 `hidden_state`, `goal_update_interval` 갱신 로직이다.
- memory 제거 실험에서 우선 수정 후보는 다음 파일이다.
  - `src/models/manager.py`
  - `src/models/fun.py`
  - `src/policies/fun_policy.py`
  - `configs/train_fun_baseline_seed*.yaml`을 복사한 ablation config
  - `tests/test_model_shapes.py`, `tests/test_fun_policy.py`, `tests/test_trainer.py` 등 shape/policy/training 관련 테스트
- `src/training/trainer.py`, `src/training/rollout.py`, `src/training/losses.py`, `src/training/evaluation.py`는 model과 policy의 출력 schema를 유지하면 그대로 재사용 가능하다.
- 따라서 4주차 첫 구현은 training loop 변경보다 `AblationManager` 또는 memory-free Manager variant를 추가하고, 같은 config/metric으로 baseline과 비교하는 방식이 가장 작게 시작할 수 있다.

## 남은 작업

- baseline config를 복사해 ablation seed config를 만든다.
- `AblationManager` 설계 메모를 작성한다.
- memory-free Manager가 기존 `FuNModel` 출력 schema를 유지하도록 구현한다.
- baseline과 같은 sample metric으로 seed별 ablation 결과를 집계한다.

---

# 2026-05-02 작업 로그: AblationManager 구현, GCP 학습 실행, 결과 회수

## 작업 목적

Vanilla FuN baseline에서 Manager의 recurrent memory만 제거한 `AblationManager`를 추가하고, 기존 training loop, rollout, loss, evaluation 코드는 그대로 재사용한 상태에서 baseline과 같은 조건으로 seed 1, 11, 44 ablation 실험을 수행했다.

이번 작업에서는 TransformerManager, PPO, reward shaping, HER, hyperparameter search는 추가하지 않았다.

## 구현 내용

다음 파일을 수정하거나 추가했다.

- `fun-minigrid/src/models/manager.py`
- `fun-minigrid/src/models/fun.py`
- `fun-minigrid/train.py`
- `fun-minigrid/evaluate_checkpoint.py`
- `fun-minigrid/tests/test_model_shapes.py`
- `fun-minigrid/tests/test_fun_policy.py`
- `fun-minigrid/tests/test_trainer.py`
- `fun-minigrid/configs/train_fun_ablation_seed1.yaml`
- `fun-minigrid/configs/train_fun_ablation_seed11.yaml`
- `fun-minigrid/configs/train_fun_ablation_seed44.yaml`
- `fun-minigrid/configs/train_fun_ablation_smoke.yaml`

`AblationManager`는 recurrent memory를 제거한 feedforward Manager다.

구조:

```text
state_emb
 -> Linear(state_dim, hidden_dim)
 -> ReLU
 -> Linear(hidden_dim, goal_dim)
 -> goal
```

동작:

- `hidden_state`는 기존 `Manager`와의 호환성을 위해 인자로 받는다.
- goal 계산에는 `hidden_state`를 사용하지 않는다.
- `hidden_state`가 있으면 그대로 `next_hidden_state`로 반환한다.
- `hidden_state`가 없으면 zero tensor를 만들어 반환한다.
- `init_hidden(batch_size, device=None)`는 기존 policy/training code와 호환되도록 zero tensor를 반환한다.

`FuNModel`에는 `manager_type` 옵션을 추가했다.

- 기본값: `manager_type="recurrent"`
- `manager_type="recurrent"`: 기존 `Manager` 사용
- `manager_type="ablation"` 또는 `"feedforward"`: `AblationManager` 사용

기존 baseline config에는 `manager_type`이 없어도 기존 recurrent Manager로 동작하도록 유지했다.

## 유지한 부분

다음 코드는 구조를 바꾸지 않았다.

- `src/training/trainer.py`
- `src/training/rollout.py`
- `src/training/losses.py`
- `src/training/evaluation.py`
- `src/policies/fun_policy.py`의 출력 schema

`FuNModel.forward()`의 반환 dict도 기존과 동일하게 유지했다.

- `state_emb`
- `goal`
- `hidden_state`
- `logits`
- `action_dist`
- `action_probs`
- `value`
- `manager_value`
- `goal_updated`

`goal_update_interval` 동작도 baseline과 동일하다.

- update step에서만 manager 호출
- update step이 아니면 이전 goal과 hidden_state 유지

## 추가한 Config

baseline seed config를 복사해 다음 ablation config를 추가했다.

- `configs/train_fun_ablation_seed1.yaml`
- `configs/train_fun_ablation_seed11.yaml`
- `configs/train_fun_ablation_seed44.yaml`

baseline과 동일하게 유지한 값:

- `env_id: MiniGrid-DoorKey-5x5-v0`
- `total_episodes: 1000`
- `max_steps: 250`
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
- `action_mode: sample`
- `eval_action_mode: sample`

ablation에서 바꾼 값:

- `manager_type: ablation`
- `log_dir: logs/ablation_fun/seed_*`
- `checkpoint_dir: checkpoints/ablation_fun/seed_*`

짧은 검증용으로 `configs/train_fun_ablation_smoke.yaml`도 추가했다.

## 로컬 테스트

로컬에서 다음 테스트를 실행했고 모두 통과했다.

```bash
python -m pytest tests/test_model_shapes.py
python -m pytest tests/test_fun_policy.py
python -m pytest tests/test_trainer.py
```

결과:

```text
tests/test_model_shapes.py: 9 passed
tests/test_fun_policy.py: 6 passed
tests/test_trainer.py: 4 passed
```

추가로 `python train.py --config configs/train_fun_ablation_smoke.yaml`로 짧은 smoke test를 실행해 ablation 경로가 실제 학습 코드에서 동작함을 확인했다.

## GCP VM 실행

기존 GCP VM `instance-20260425-090526`은 `asia-northeast3-b`에서 `n1-standard-4 + Tesla T4` 리소스 부족으로 시작할 수 없었다.

새 VM을 사용했다.

```text
name: fun-minigrid-2
zone: asia-east1-c
external IP: 35.194.128.10
user: fumin0193
remote path: /home/fumin0193/FuN_with_Transformer/fun-minigrid
GPU: Tesla T4
Python: 3.13.5
torch: 2.6.0+cu124
```

SSH 접속 시 Windows 한글 사용자명 때문에 `gcloud compute config-ssh`가 깨진 key path를 생성해서 public key 인증이 실패했다. SSH config의 `fun-minigrid-2` Host 블록을 다음처럼 정리해 접속 문제를 해결했다.

```text
User fumin0193
IdentityFile ~/.ssh/google_compute_engine
UserKnownHostsFile ~/.ssh/google_compute_known_hosts
```

GCP에 최신 ablation 코드와 config를 업로드한 뒤 원격에서 다음 테스트를 다시 실행했다.

```bash
python -m pytest tests/test_model_shapes.py
python -m pytest tests/test_fun_policy.py
python -m pytest tests/test_trainer.py
```

원격 테스트 결과:

```text
tests/test_model_shapes.py: 9 passed
tests/test_fun_policy.py: 6 passed
tests/test_trainer.py: 4 passed
```

## GCP Ablation 학습

GCP에서 seed 1, 11, 44 ablation 학습을 실행했다.

```bash
python train.py --config configs/train_fun_ablation_seed1.yaml
python train.py --config configs/train_fun_ablation_seed11.yaml
python train.py --config configs/train_fun_ablation_seed44.yaml
```

세 seed 모두 1000 episode까지 완료되었다.

생성된 파일:

- `logs/ablation_fun/seed_1/train.csv`
- `logs/ablation_fun/seed_1/eval.csv`
- `logs/ablation_fun/seed_1/summary.json`
- `logs/ablation_fun/seed_11/train.csv`
- `logs/ablation_fun/seed_11/eval.csv`
- `logs/ablation_fun/seed_11/summary.json`
- `logs/ablation_fun/seed_44/train.csv`
- `logs/ablation_fun/seed_44/eval.csv`
- `logs/ablation_fun/seed_44/summary.json`
- `checkpoints/ablation_fun/seed_1/best.pt`
- `checkpoints/ablation_fun/seed_1/last.pt`
- `checkpoints/ablation_fun/seed_1/episode_1000.pt`
- `checkpoints/ablation_fun/seed_11/best.pt`
- `checkpoints/ablation_fun/seed_11/last.pt`
- `checkpoints/ablation_fun/seed_11/episode_1000.pt`
- `checkpoints/ablation_fun/seed_44/best.pt`
- `checkpoints/ablation_fun/seed_44/last.pt`
- `checkpoints/ablation_fun/seed_44/episode_1000.pt`

## Best Checkpoint Evaluation

각 seed의 `best.pt`를 100 episode로 sample/argmax 각각 평가했다.

생성된 파일:

- `logs/ablation_fun/seed_1/checkpoint_eval_best_sample.json`
- `logs/ablation_fun/seed_1/checkpoint_eval_best_argmax.json`
- `logs/ablation_fun/seed_11/checkpoint_eval_best_sample.json`
- `logs/ablation_fun/seed_11/checkpoint_eval_best_argmax.json`
- `logs/ablation_fun/seed_44/checkpoint_eval_best_sample.json`
- `logs/ablation_fun/seed_44/checkpoint_eval_best_argmax.json`

평가 결과:

| Seed | Sample Success Rate | Sample Mean Return | Sample Episode Length | Argmax Success Rate | Argmax Mean Return | Argmax Episode Length |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.350000 | 0.160604 | 215.110000 | 0.000000 | 0.000000 | 250.000000 |
| 11 | 0.820000 | 0.496936 | 134.740000 | 0.000000 | 0.000000 | 250.000000 |
| 44 | 0.640000 | 0.362080 | 167.200000 | 0.000000 | 0.000000 | 250.000000 |
| Mean | 0.603333 | 0.339873 | 172.350000 | 0.000000 | 0.000000 | 250.000000 |

## Baseline과 Ablation 비교

baseline 결과:

| Model | Mean Sample Success Rate | Mean Sample Return | Mean Episode Length | Mean Argmax Success Rate |
|---|---:|---:|---:|---:|
| baseline | 0.416667 | 0.203031 | 205.176667 | 0.000000 |

ablation 결과:

| Model | Mean Sample Success Rate | Mean Sample Return | Mean Episode Length | Mean Argmax Success Rate |
|---|---:|---:|---:|---:|
| ablation | 0.603333 | 0.339873 | 172.350000 | 0.000000 |

차이:

| Metric | Ablation - Baseline |
|---|---:|
| Sample success rate | +0.186667 |
| Sample mean return | +0.136843 |
| Sample episode length | -32.826667 |
| Argmax success rate | 0.000000 |

## 로컬로 회수한 결과

GCP에서 생성된 결과를 로컬로 내려받았다.

- `fun-minigrid/results/week4_ablation_results.csv`
- `fun-minigrid/results/week4_ablation_results.md`
- `fun-minigrid/results/week4_ablation_summary.md`
- `fun-minigrid/logs/ablation_fun/seed_1/`
- `fun-minigrid/logs/ablation_fun/seed_11/`
- `fun-minigrid/logs/ablation_fun/seed_44/`
- `fun-minigrid/checkpoints/ablation_fun/seed_1/`
- `fun-minigrid/checkpoints/ablation_fun/seed_11/`
- `fun-minigrid/checkpoints/ablation_fun/seed_44/`
- `fun-minigrid/figures/ablation_fun/eval_success_rate.png`
- `fun-minigrid/figures/ablation_fun/eval_mean_return.png`
- `fun-minigrid/figures/ablation_fun/eval_episode_length.png`

결과 회수 후 GCP VM을 중지했다.

```text
fun-minigrid-2: TERMINATED
instance-20260425-090526: TERMINATED
```

## 해석

- 현재 MiniGrid DoorKey-5x5, 1000 episode, 3 seed 조건에서는 recurrent Manager memory를 제거한 ablation이 vanilla FuN baseline보다 sample 기준 평균 성능이 높았다.
- sample success rate는 baseline `0.416667`에서 ablation `0.603333`으로 증가했다.
- sample mean return은 baseline `0.203031`에서 ablation `0.339873`으로 증가했다.
- sample episode length는 baseline `205.176667`에서 ablation `172.350000`으로 감소했다.
- 이 결과는 작은 DoorKey-5x5 환경에서는 GRU Manager memory가 필수적이지 않거나, feedforward Manager가 더 단순해 짧은 학습 조건에서 더 안정적으로 최적화되었을 가능성을 시사한다.
- 다만 baseline과 ablation 모두 argmax success rate는 `0.000000`이므로, 현재 정책은 deterministic argmax path보다 stochastic sampling에 의존한다.
- seed 3개 결과이므로 통계적으로 강한 결론보다는 1차 ablation 결과로 해석해야 한다.

## 다음 작업

- `week4_ablation_summary.md` 내용을 보고서 결과 섹션 초안으로 정리한다.
- 필요하면 DoorKey 더 큰 환경 또는 longer training에서 recurrent Manager의 장점이 나타나는지 재검증한다.
- TransformerManager는 이번 ablation 결과 해석 이후 선택적 확장으로 진행한다.
