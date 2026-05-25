# FuN with Transformer Repository Index

This repository contains a MiniGrid implementation and experiment archive for a Feudal Networks (FuN) style hierarchical reinforcement learning project. The main working code is in `fun-minigrid/`; the rest of the repository is mostly experiment notes, reports, and final-poster artifacts.

Use this file as a navigation index before reading the full tree.

## Top-Level Map

- `fun-minigrid/`: main Python project for MiniGrid environments, FuN models, training, evaluation, analysis scripts, configs, tests, and generated figures/results.
- `final_poster/`: curated final-poster package with selected configs, poster-ready figures, result summaries, lightweight logs, and a poster guide.
- `final_poster.zip`: zip source for the `final_poster/` package.
- `checklist/`: weekly project checklists.
- `work_log/`: weekly progress logs.
- `report/`: presentation and written-result notes.
- `.agents/`, `.codex/`: local assistant/tooling metadata.

## Main Code Index

Primary project directory: `fun-minigrid/`

- `fun-minigrid/train.py`: config-driven training entry point. Reads YAML configs, builds the environment/model/policy, logs train/eval CSVs, and saves checkpoints.
- `fun-minigrid/main.py`: lightweight smoke/demo runner for a few FuNPolicy episodes.
- `fun-minigrid/evaluate_checkpoint.py`: checkpoint evaluation entry point.
- `fun-minigrid/analyze_manager_goals.py`: manager-goal diagnostic analysis.
- `fun-minigrid/diagnose_baseline.py`: baseline diagnostic helper.
- `fun-minigrid/diagnose_hidden_intervention.py`: hidden-state intervention diagnostic helper.
- `fun-minigrid/aggregate_*.py`, `fun-minigrid/plot_*.py`: result aggregation and plotting helpers.
- `fun-minigrid/GCP_ACCESS_RUNBOOK.md`: operational notes for running experiments on GCP.

## Source Package Layout

All importable code lives under `fun-minigrid/src/`.

- `src/envs/`: MiniGrid environment creation, observation preprocessing, and wrappers.
- `src/models/`: FuN model components.
  - `encoder.py`: image encoder.
  - `manager.py`: high-level goal policy.
  - `worker.py`: low-level action policy.
  - `fun.py`: combined Manager + Worker model.
- `src/policies/fun_policy.py`: policy wrapper for action selection (`sample` or `argmax`).
- `src/training/`: rollout, loss, return, evaluation, and training-loop logic.
- `src/utils/`: config loading, checkpointing, logging, metrics, and seeding helpers.
- `src/analysis/`: log plotting utilities.

## Experiments And Configs

Experiment configs are in `fun-minigrid/configs/`.

Common config families:

- `train_fun_*.yaml`: main training runs.
- `smoke_train_*.yaml`: short smoke-test runs.
- `*_baseline_*.yaml`: vanilla FuN baseline settings.
- `*_ablation_*.yaml`: ablation-manager settings.
- `doorkey6x6`: DoorKey-6x6 experiments.
- `multiroom_n*s*`: MultiRoom experiments.
- `*_stage1_*`, `*_stage2_*`, `*_argmax_ft_*`: two-stage or argmax fine-tuning runs.

Shell orchestration scripts are in `fun-minigrid/scripts/`, including week-specific GCP run/evaluation scripts and summarizers.

## Results And Artifacts

- `fun-minigrid/results/`: experiment summaries, CSVs, and markdown analyses from the working project.
- `fun-minigrid/figures/`: generated figures from experiments.
- `final_poster/results/`: curated result summaries copied into the final poster package.
- `final_poster/figures/poster_main/`: recommended poster figures.
- `final_poster/poster_guide.md`: Korean guide for poster framing and figure selection.
- `final_poster/poster_key_numbers.csv`: headline numeric values for the poster.
- `final_poster/manifest.csv`: generated listing of final-poster package contents.

## Tests

Tests are under `fun-minigrid/tests/` and cover model shapes, config parsing, checkpoints, evaluation, losses, rollout/training behavior, logging, hidden intervention, and manager-goal analysis.

From `fun-minigrid/`, the expected test command is:

```bash
python -m pytest tests
```

## Common Commands

Run from `fun-minigrid/` unless noted.

```bash
python main.py --run-id 1
python train.py --config configs/train_fun.yaml
python train.py --config configs/train_fun_ablation_doorkey6x6_smoke.yaml
python evaluate_checkpoint.py --help
python -m pytest tests
```

Dependency note: `fun-minigrid/requirements.txt` is currently empty, so inspect imports and the execution environment before assuming dependencies are fully documented. Core runtime dependencies include Python, PyTorch, MiniGrid/Gymnasium-style environments, NumPy, pandas/Matplotlib for analysis, and pytest for tests.

## Suggested Reading Order For ChatGPT

1. Read this file first.
2. Read `fun-minigrid/README.md` for the original short file overview.
3. Read `fun-minigrid/train.py` to understand the training workflow.
4. Read `fun-minigrid/src/models/fun.py`, then `manager.py`, `worker.py`, and `encoder.py`.
5. Read `fun-minigrid/src/training/trainer.py`, `rollout.py`, `losses.py`, and `evaluation.py`.
6. Inspect the relevant YAML config in `fun-minigrid/configs/`.
7. For final presentation context, read `final_poster/README.md`, `final_poster/poster_guide.md`, and `final_poster/poster_key_numbers.csv`.

