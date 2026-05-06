# Week 5 Argmax Evaluation Diagnosis

## 1. Scope

This document diagnoses why DoorKey-6x6 3000-episode checkpoints show high sample evaluation success but low argmax evaluation success.

No code was modified, no training was run, and no checkpoint/log file was deleted.

## 2. Current Sample vs Argmax Results

The table below uses the existing 100-episode best-checkpoint evaluation JSON files under `logs/*_doorkey6x6_3000/seed_*/checkpoint_eval_best_*_100ep.json`.

| Model | Seed | Sample Success 100ep | Argmax Success 100ep | Gap |
|---|---:|---:|---:|---:|
| Vanilla FuN | 1 | 0.080 | 0.000 | 0.080 |
| Vanilla FuN | 11 | 1.000 | 0.340 | 0.660 |
| Vanilla FuN | 44 | 0.190 | 0.000 | 0.190 |
| AblationManager | 1 | 1.000 | 0.450 | 0.550 |
| AblationManager | 11 | 0.980 | 0.040 | 0.940 |
| AblationManager | 44 | 1.000 | 0.180 | 0.820 |

Across seeds:

- Vanilla FuN sample success mean: 0.423
- Vanilla FuN argmax success mean: 0.113
- AblationManager sample success mean: 0.993
- AblationManager argmax success mean: 0.223

The largest issue is AblationManager: sample evaluation is nearly solved across all three seeds, but argmax success remains low and highly seed-dependent.

## 3. Evaluation Code Path

`evaluate_checkpoint.py` loads the training config, restores the checkpoint into a freshly built `FuNModel`, builds a `FuNPolicy`, creates the same MiniGrid environment via `make_env`, then calls `evaluate_policy`.

Important observations:

- `--action-mode` is parsed with choices `argmax` and `sample`; the default CLI mode is `argmax`.
- `build_policy_from_config(..., action_mode=...)` passes the CLI action mode directly into `FuNPolicy`.
- `evaluate_policy` runs `run_episode` repeatedly with deterministic episode seeds `seed + eval_seed_offset + episode_idx`.
- `run_episode` calls `env.reset(seed=episode_seed)` at the start of each episode.
- `run_episode` calls `policy.reset(batch_size=1)` at the start of each episode when the policy has a reset method.
- `FuNPolicy.reset` reinitializes both recurrent hidden state and current goal through `model.init_hidden(...)` and `model.init_goal(...)`, and resets `step_count` to 0.
- Sample and argmax evaluation therefore share the same observation preprocessing, environment reset path, episode seed path, and manager/worker hidden-state reset path.

Argmax action selection is implemented in `FuNPolicy.act_for_training`:

- `sample`: `action_dist.sample()`
- `argmax`: `torch.argmax(out["logits"], dim=-1)`

So current argmax evaluation really is greedy action selection from the largest policy logit, not sampling from probabilities.

One minor implementation note: `run_episode` stops when `max_steps` is reached, but it only breaks the loop and does not set the returned `truncated` flag to true. This affects metadata clarity, not the action selection itself.

## 4. Training-Time Eval and `best.pt` Selection

All six DoorKey-6x6 3000 config files contain:

- `action_mode: sample`
- `eval_action_mode: sample`

In `train.py`, the training policy uses `action_mode` and the eval policy uses `eval_action_mode`. Since the configs set `eval_action_mode: sample`, every interval eval during training was sample-based.

`best.pt` is selected inside the interval eval block:

- compute interval eval with `eval_policy`
- normalize it into `eval_success_rate`
- if `eval_success_rate >= best_success_rate`, save `best.pt`

Therefore, current `best.pt` is a sample-eval best checkpoint, not an argmax-eval best checkpoint.

There is currently no separate `best_sample.pt` / `best_argmax.pt` checkpoint selection logic. There is also no training-time argmax eval log beside the sample eval log.

Current summary values confirm this:

| Model | Seed | Config Eval Mode | `summary.best_success_rate` | Best Checkpoint Episode | 100ep Sample Success |
|---|---:|---|---:|---:|---:|
| Vanilla FuN | 1 | sample | 0.200 | 2500 | 0.080 |
| Vanilla FuN | 11 | sample | 1.000 | 3000 | 1.000 |
| Vanilla FuN | 44 | sample | 0.300 | 3000 | 0.190 |
| AblationManager | 1 | sample | 1.000 | 3000 | 1.000 |
| AblationManager | 11 | sample | 1.000 | 3000 | 0.980 |
| AblationManager | 44 | sample | 1.000 | 3000 | 1.000 |

Using current `best.pt` for argmax evaluation is valid as a post-hoc diagnostic, but it is not a fair estimate of the best deterministic policy reachable during the run. The checkpoint selection objective and the evaluation objective are mismatched.

If argmax success is a target metric, `best_argmax.pt` should be saved separately.

## 5. Likely Causes of Low Argmax Success

### 5.1 Checkpoint Selection Mismatch

The strongest confirmed issue is checkpoint selection. The training loop saves `best.pt` using sample eval success only. A checkpoint can be excellent under stochastic sampling while having poor greedy behavior if the highest-logit action sequence is brittle.

This is especially visible for AblationManager seed 11:

- sample success: 0.980
- argmax success: 0.040
- gap: 0.940

That gap is too large to treat `best.pt` as an argmax-optimized checkpoint.

### 5.2 High-Entropy Stochastic Policy

The training configs use `entropy_coef: 0.01`, and the run summaries show high mean entropy:

| Model | Seed | Mean Entropy |
|---|---:|---:|
| Vanilla FuN | 1 | 1.944 |
| Vanilla FuN | 11 | 1.887 |
| Vanilla FuN | 44 | 1.944 |
| AblationManager | 1 | 1.883 |
| AblationManager | 11 | 1.898 |
| AblationManager | 44 | 1.856 |

For 7 actions, maximum categorical entropy is about 1.946. Several runs remain very close to maximum entropy. That means the learned policy can still depend heavily on stochastic exploration at evaluation time. In that regime, sample evaluation can succeed because it occasionally takes necessary non-greedy actions, while argmax repeatedly chooses a narrow deterministic path.

### 5.3 Deterministic Local Traps

DoorKey requires a coordinated sequence: navigate to key, pick up, navigate to door, open, then reach goal. If the greedy action at a critical state is slightly wrong, argmax can repeat a turn, toggle, pickup, or movement action that traps the agent until max steps.

The argmax episode lengths support this:

| Model | Seed | Argmax Success | Argmax Episode Length |
|---|---:|---:|---:|
| Vanilla FuN | 1 | 0.000 | 250.000 |
| Vanilla FuN | 44 | 0.000 | 250.000 |
| AblationManager | 11 | 0.040 | 240.880 |

Near-max episode length with near-zero success is consistent with deterministic loops or local traps.

### 5.4 Seed Variance

Seed variance remains important. Vanilla FuN seed 11 reaches sample success 1.000 and argmax success 0.340, while seeds 1 and 44 have argmax success 0.000. AblationManager solves sample evaluation across seeds but argmax varies from 0.040 to 0.450.

This suggests both learning variance and checkpoint-selection variance. It is not just a single bad run.

### 5.5 Evaluation Code Bug Possibility

The main argmax implementation looks correct:

- same preprocessing as sample
- same reset path as sample
- hidden state and goal reset every episode
- argmax chooses the largest policy logit

So a direct evaluation-code bug is not the leading explanation.

Remaining code-level checks worth adding later:

- record per-step logits/probabilities/actions for failed argmax episodes
- compare sample and argmax trajectories on identical episode seeds
- detect repeated action/state loops in argmax rollouts
- log policy entropy during checkpoint evaluation, not only training summaries

## 6. Recommended Next Steps

Priority order:

1. Log both sample eval and argmax eval during training.

   Keep the current sample eval, but add a second eval pass with `action_mode: argmax` at the same interval and seed schedule. This will show whether argmax performance ever becomes good during training or never improves.

2. Save `best_sample.pt` and `best_argmax.pt` separately.

   Current `best.pt` is sample-selected. For deterministic deployment or deterministic reporting, use `best_argmax.pt`. Keep `best_sample.pt` for stochastic-policy analysis.

3. Add temperature evaluation.

   Evaluate with a range such as argmax, temperature 0.25, 0.5, 0.75, 1.0, and pure sample. This will reveal whether performance collapses only at exact greedy decoding or improves with slightly reduced stochasticity.

4. Run a low-entropy fine-tuning experiment later.

   After separate eval/checkpoint logging exists, test whether reducing entropy pressure late in training improves deterministic success without changing architecture. This should be treated as a new controlled experiment, not part of the current diagnosis.

5. Run a 5000-episode extension later.

   Episode-budget extension may help, but it should come after argmax-aware logging. Otherwise another long run may still save only a sample-best checkpoint and leave the deterministic-policy question unresolved.

## 7. Bottom Line

The current low argmax success is most likely not caused by hidden-state reset or preprocessing mismatch. The confirmed structural issue is that training-time checkpoint selection optimizes sample evaluation only. Combined with very high policy entropy, the saved `best.pt` checkpoints can be excellent stochastic policies but poor deterministic policies.

Before changing model structure or training algorithm, the next code change should be evaluation/checkpoint bookkeeping: log both action modes and save separate best checkpoints for sample and argmax.
