from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG = ROOT / ".mplconfig"
MPLCONFIG.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = ROOT / "figures" / "poster_main"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Baseline": "#4C78A8",
    "Memory Ablation": "#F58518",
    "Vanilla FuN": "#4C78A8",
    "AblationManager": "#F58518",
    "baseline": "#4C78A8",
    "ablation": "#F58518",
}


def read_csv(rel_path: str) -> list[dict[str, str]]:
    with (ROOT / rel_path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def grouped_mean(rows: list[dict[str, str]], keys: tuple[str, ...], metric: str) -> dict[tuple[str, ...], float]:
    buckets: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in keys)].append(float(row[metric]))
    return {key: mean(vals) for key, vals in buckets.items()}


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def plot_multiroom_n4s4_auc_speed() -> None:
    rows = read_csv("results/multiroom_n4s4/multiroom_n4_s4_three_seed_process_summary.csv")
    auc = grouped_mean(rows, ("model", "eval_policy"), "success_auc")
    ep100 = grouped_mean(rows, ("model", "eval_policy"), "success_1_0_ep")
    policies = ["sample", "argmax"]
    models = ["Baseline", "Memory Ablation"]
    width = 0.34
    x = range(len(policies))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for i, model in enumerate(models):
        offset = (i - 0.5) * width
        axes[0].bar([v + offset for v in x], [auc[(model, p)] for p in policies], width, label=model, color=COLORS[model])
        axes[1].bar([v + offset for v in x], [ep100[(model, p)] for p in policies], width, label=model, color=COLORS[model])

    axes[0].set_title("N4-S4 learning AUC")
    axes[0].set_ylabel("Success AUC (higher is better)")
    axes[0].set_xticks(list(x), ["Sample", "Argmax"])
    axes[0].set_ylim(0, 0.85)

    axes[1].set_title("N4-S4 first full-success episode")
    axes[1].set_ylabel("Episode to 1.0 success (lower is better)")
    axes[1].set_xticks(list(x), ["Sample", "Argmax"])
    axes[1].set_ylim(0, 3400)

    for ax in axes:
        style_axes(ax)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("MultiRoom N4-S4: Ablation learns earlier, final success ties", y=1.04, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "multiroom_n4s4_auc_and_speed.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_multiroom_n4s4_final_lengths() -> None:
    rows = read_csv("results/multiroom_n4s4/multiroom_n4_s4_three_seed_process_summary.csv")
    lengths = grouped_mean(rows, ("model", "eval_policy"), "final_episode_length")
    policies = ["sample", "argmax"]
    models = ["Baseline", "Memory Ablation"]
    width = 0.34
    x = range(len(policies))

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, model in enumerate(models):
        offset = (i - 0.5) * width
        ax.bar([v + offset for v in x], [lengths[(model, p)] for p in policies], width, label=model, color=COLORS[model])
    ax.set_title("N4-S4 final trajectory length")
    ax.set_ylabel("Mean final episode length")
    ax.set_xticks(list(x), ["Sample", "Argmax"])
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "multiroom_n4s4_final_episode_length.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_n4s5_boundary() -> None:
    rows = read_csv("results/multiroom_n4s5_boundary/week6_multiroom_n4s5_maxsteps_sweep_results.csv")
    by_model: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        by_model[row["Model"]].append((int(row["Max Steps"]), int(row["Reward Signal Episodes"])))

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for model, points in by_model.items():
        points = sorted(points)
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            linewidth=2.2,
            label=model,
            color=COLORS.get(model, "#666666"),
        )
    ax.set_title("N4-S5 boundary: train signal without eval success")
    ax.set_xlabel("Max steps")
    ax.set_ylabel("Reward-signal episodes")
    ax.text(0.02, 0.92, "Final eval success = 0 for all runs", transform=ax.transAxes, fontsize=9)
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "multiroom_n4s5_reward_signal_boundary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_manager_goal_coupling() -> None:
    rows = read_csv("results/multiroom_n4s4_manager_goal_analysis/summaries/goal_aggregate_summary.csv")
    labels = [f"{r['model_type']}\n{r['action_mode']}" for r in rows]
    colors = [COLORS[r["model_type"]] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axes[0].bar(labels, [float(r["mean_goal_delta_l2"]) for r in rows], color=colors)
    axes[0].set_title("Goal update magnitude")
    axes[0].set_ylabel("Mean goal delta L2")
    axes[1].bar(labels, [float(r["mean_action_kl_on_goal_update"]) for r in rows], color=colors)
    axes[1].set_title("Worker policy change at goal update")
    axes[1].set_ylabel("Mean action KL")

    for ax in axes:
        style_axes(ax)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Final-checkpoint mechanism diagnostics", y=1.04, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "manager_goal_policy_coupling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_hidden_intervention_drop() -> None:
    rows = read_csv("results/multiroom_n4s4_hidden_intervention/hidden_intervention_summary.csv")
    by_action_intervention: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_action_intervention[(row["action_mode"], row["intervention"])].append(float(row["success_rate"]))

    actions = ["sample", "argmax"]
    interventions = ["reset_goal", "reset_step"]
    x = range(len(actions))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, intervention in enumerate(interventions):
        drops = []
        for action in actions:
            normal = mean(by_action_intervention[(action, "normal")])
            intervened = mean(by_action_intervention[(action, intervention)])
            drops.append(normal - intervened)
        ax.bar([v + (i - 0.5) * width for v in x], drops, width, label=intervention, color=["#72B7B2", "#E45756"][i])
    ax.set_title("Baseline hidden-state intervention")
    ax.set_ylabel("Success drop vs normal")
    ax.set_xticks(list(x), ["Sample", "Argmax"])
    ax.set_ylim(0, 0.05)
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "hidden_intervention_success_drop.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_doorkey_budget_success() -> None:
    data = {
        "Vanilla FuN": {"Sample": 0.423, "Argmax": 0.113},
        "AblationManager": {"Sample": 0.993, "Argmax": 0.223},
    }
    modes = ["Sample", "Argmax"]
    models = ["Vanilla FuN", "AblationManager"]
    x = range(len(modes))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, model in enumerate(models):
        offset = (i - 0.5) * width
        ax.bar([v + offset for v in x], [data[model][m] for m in modes], width, label=model, color=COLORS[model])
    ax.set_title("DoorKey-6x6 at 3000 episodes")
    ax.set_ylabel("Mean success across seeds")
    ax.set_xticks(list(x), modes)
    ax.set_ylim(0, 1.05)
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "doorkey6x6_3000_mean_success.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_multiroom_n4s4_auc_speed()
    plot_multiroom_n4s4_final_lengths()
    plot_n4s5_boundary()
    plot_manager_goal_coupling()
    plot_hidden_intervention_drop()
    plot_doorkey_budget_success()


if __name__ == "__main__":
    main()
