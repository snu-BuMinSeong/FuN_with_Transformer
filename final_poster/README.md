# Final Poster Package

This folder contains the result tables, poster-ready figures, selected configs, and lightweight logs needed to prepare the final presentation poster.

## Folder Map

- `figures/poster_main/`: synthesized figures recommended for the main poster panels.
- `figures/multiroom_n4s4_learning_curves/`: original N4-S4 learning-curve figures by seed and mean.
- `figures/manager_goal_analysis/`: original mechanism-diagnostic plots from manager-goal analysis.
- `figures/doorkey6x6_3000/`: original DoorKey-6x6 1000-vs-3000 figures.
- `figures/early_training_diagnostics/`: earlier baseline/ablation and sample/argmax diagnostic figures.
- `results/`: markdown analyses and CSV summaries copied from the completed experiments.
- `configs/`: YAML configs for the experiments used in the poster.
- `logs_light/`: selected evaluation CSV/JSON logs only. Large per-episode summaries and raw train logs were excluded to keep the package small.
- `scripts/make_poster_figures.py`: script used to regenerate `figures/poster_main/*.png`.
- `poster_guide.md`: Korean guide for what to put on the poster and how to frame each result.
- `poster_key_numbers.csv`: headline values to cite in the poster.
- `manifest.csv`: generated file list for checking package contents.

## Recommended Main Figures

1. `figures/poster_main/doorkey6x6_3000_mean_success.png`
2. `figures/poster_main/multiroom_n4s4_auc_and_speed.png`
3. `figures/poster_main/multiroom_n4s4_final_episode_length.png`
4. `figures/poster_main/manager_goal_policy_coupling.png`
5. `figures/poster_main/hidden_intervention_success_drop.png`
6. `figures/poster_main/multiroom_n4s5_reward_signal_boundary.png`

Use the original learning-curve PNGs as secondary figures or appendix material.
