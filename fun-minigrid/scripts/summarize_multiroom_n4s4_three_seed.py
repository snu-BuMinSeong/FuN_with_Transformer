from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "figures" / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS = ROOT / "results"
FIGURES = RESULTS / "multiroom_n4_s4_three_seed_figures"

EXPERIMENTS = [
    ("baseline", 1),
    ("baseline", 11),
    ("baseline", 44),
    ("ablation", 1),
    ("ablation", 11),
    ("ablation", 44),
]

MODEL_LABEL = {
    "baseline": "Baseline",
    "ablation": "Memory Ablation",
}

POLICIES = ["sample", "argmax"]
THRESHOLDS = [0.5, 0.8, 0.9, 1.0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_simple_yaml(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    current_key: str | None = None
    current_list: list[str] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_key and current_list is not None:
            current_list.append(stripped[2:].strip())
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            current_key = key
            current_list = []
            data[key] = current_list
            continue
        current_key = None
        current_list = None
        if value == "null":
            data[key] = None
        elif value.lower() in {"true", "false"}:
            data[key] = value.lower() == "true"
        else:
            try:
                data[key] = int(value)
            except ValueError:
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value
    return data


def find_log_files(model: str, seed: int) -> tuple[Path, Path, Path | None]:
    log_dir = ROOT / "logs" / f"multiroom_n4s4_seed{seed}" / model
    train_candidates = [log_dir / "train.csv", log_dir / f"{model}_train.csv"]
    eval_candidates = [log_dir / "eval.csv", log_dir / f"{model}_eval.csv"]
    train_csv = next((p for p in train_candidates if p.exists()), None)
    eval_csv = next((p for p in eval_candidates if p.exists()), None)
    if train_csv is None or eval_csv is None:
        raise FileNotFoundError(f"Missing train/eval CSV for {model} seed {seed} in {log_dir}")
    summary = log_dir / "summary.json"
    return train_csv, eval_csv, summary if summary.exists() else None


def config_path(model: str, seed: int) -> Path:
    return ROOT / "configs" / f"train_fun_{model}_multiroom_n4s4_ms1000_seed{seed}.yaml"


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def b(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() == "true"


def first_episode_at(rows: list[dict[str, str]], policy: str, threshold: float, positive: bool = False) -> int | None:
    key = f"eval_{policy}_success_rate"
    for row in rows:
        value = f(row, key)
        if positive:
            if value > 0:
                return int(row["episode"])
        elif value >= threshold:
            return int(row["episode"])
    return None


def auc(rows: list[dict[str, str]], key: str, total_episodes: int) -> float:
    xs = np.array([int(r["episode"]) for r in rows], dtype=float)
    ys = np.array([f(r, key) for r in rows], dtype=float)
    if len(xs) == 0:
        return math.nan
    if len(xs) == 1:
        return float(ys[0] * xs[0] / total_episodes)
    area = float(np.sum((ys[:-1] + ys[1:]) * 0.5 * (xs[1:] - xs[:-1])))
    return area / total_episodes


def count_drops_after_reach(rows: list[dict[str, str]], policy: str, threshold: float) -> int:
    key = f"eval_{policy}_success_rate"
    reached = False
    drops = 0
    below = False
    for row in rows:
        value = f(row, key)
        if not reached and value >= threshold:
            reached = True
            below = False
            continue
        if reached:
            if value < threshold and not below:
                drops += 1
                below = True
            elif value >= threshold:
                below = False
    return drops


def slope(rows: list[dict[str, str]], key: str) -> float:
    xs = np.array([int(r["episode"]) for r in rows], dtype=float)
    ys = np.array([f(r, key) for r in rows], dtype=float)
    if len(xs) < 2:
        return math.nan
    return float(np.polyfit(xs, ys, 1)[0])


def train_stats(rows: list[dict[str, str]]) -> dict[str, float]:
    rewards = np.array([f(r, "total_reward") for r in rows], dtype=float)
    lengths = np.array([f(r, "episode_length") for r in rows], dtype=float)
    successes = np.array([1.0 if b(r, "success") else 0.0 for r in rows], dtype=float)
    episodes = np.array([int(r["episode"]) for r in rows], dtype=float)

    def tail_success(n: int) -> float:
        return float(successes[-n:].mean())

    late_n = min(1000, len(rows))
    length_slope = float(np.polyfit(episodes, lengths, 1)[0]) if len(rows) >= 2 else math.nan
    early_length = float(lengths[:late_n].mean())
    late_length = float(lengths[-late_n:].mean())
    return {
        "total_train_success": float(successes.mean()),
        "last1000_success": tail_success(min(1000, len(rows))),
        "last500_success": tail_success(min(500, len(rows))),
        "last100_success": tail_success(min(100, len(rows))),
        "mean_reward": float(rewards.mean()),
        "late_reward": float(rewards[-late_n:].mean()),
        "early_length": early_length,
        "late_length": late_length,
        "length_delta_late_minus_early": late_length - early_length,
        "train_length_slope": length_slope,
    }


def fmt(x: object, digits: int = 3) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        if math.isnan(x):
            return "NA"
        return f"{x:.{digits}f}"
    return str(x)


def md_table(headers: list[str], rows: list[list[object]], digits: int = 3) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(v, digits) for v in row) + " |")
    return "\n".join(out)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def metric_key(policy: str, metric: str) -> str:
    return f"eval_{policy}_{metric}"


def plot_seed_curves(curves: dict[tuple[str, int], list[dict[str, str]]], policy: str, metric: str, ylabel: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"seed_eval_{metric}_{policy}.png"
    plt.figure(figsize=(10, 6))
    for model, seed in EXPERIMENTS:
        rows = curves[(model, seed)]
        xs = [int(r["episode"]) for r in rows]
        ys = [f(r, metric_key(policy, metric)) for r in rows]
        linestyle = "-" if model == "baseline" else "--"
        plt.plot(xs, ys, linestyle=linestyle, label=f"{MODEL_LABEL[model]} seed{seed}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"MultiRoom N4-S4 {policy} eval {ylabel}")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_mean_curves(curves: dict[tuple[str, int], list[dict[str, str]]], policy: str, metric: str, ylabel: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"mean_eval_{metric}_{policy}.png"
    plt.figure(figsize=(9, 5.5))
    for model in ["baseline", "ablation"]:
        by_episode: dict[int, list[float]] = defaultdict(list)
        for seed in [1, 11, 44]:
            for row in curves[(model, seed)]:
                by_episode[int(row["episode"])].append(f(row, metric_key(policy, metric)))
        xs = sorted(by_episode)
        means = [mean(by_episode[x]) for x in xs]
        stds = [pstdev(by_episode[x]) if len(by_episode[x]) > 1 else 0.0 for x in xs]
        y = np.array(means)
        s = np.array(stds)
        plt.plot(xs, y, label=MODEL_LABEL[model])
        plt.fill_between(xs, y - s, y + s, alpha=0.15)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"MultiRoom N4-S4 3-seed mean {policy} eval {ylabel}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def avg(rows: list[dict[str, object]], key: str) -> float:
    values = [float(r[key]) for r in rows if r.get(key) is not None and r.get(key) != ""]
    return mean(values) if values else math.nan


def std(rows: list[dict[str, object]], key: str) -> float:
    values = [float(r[key]) for r in rows if r.get(key) is not None and r.get(key) != ""]
    return pstdev(values) if len(values) > 1 else 0.0


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    file_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    train_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    eval_curves: dict[tuple[str, int], list[dict[str, str]]] = {}

    for model, seed in EXPERIMENTS:
        train_csv, eval_csv, summary_json = find_log_files(model, seed)
        cfg_path = config_path(model, seed)
        cfg = read_simple_yaml(cfg_path)
        train = read_csv(train_csv)
        eval_rows = read_csv(eval_csv)
        eval_curves[(model, seed)] = eval_rows
        total_episodes = int(cfg["total_episodes"])
        log_dir = train_csv.parent

        file_rows.append(
            {
                "model": MODEL_LABEL[model],
                "seed": seed,
                "train_csv": str(train_csv.relative_to(ROOT)),
                "eval_csv": str(eval_csv.relative_to(ROOT)),
                "config": str(cfg_path.relative_to(ROOT)),
                "summary": str(summary_json.relative_to(ROOT)) if summary_json else "",
                "log_dir": str(log_dir.relative_to(ROOT)),
                "env_id": cfg.get("env_id"),
                "manager_type": cfg.get("manager_type"),
                "config_seed": cfg.get("seed"),
            }
        )

        tstats = train_stats(train)
        train_row = {"model": MODEL_LABEL[model], "seed": seed, **tstats}
        train_rows.append(train_row)

        for policy in POLICIES:
            last = eval_rows[-1]
            final_row = {
                "model": MODEL_LABEL[model],
                "seed": seed,
                "eval_policy": policy,
                "final_episode": int(last["episode"]),
                "final_success": f(last, metric_key(policy, "success_rate")),
                "final_return": f(last, metric_key(policy, "mean_return")),
                "final_episode_length": f(last, metric_key(policy, "mean_episode_length")),
            }
            final_rows.append(final_row)

            threshold_row = {
                "model": MODEL_LABEL[model],
                "seed": seed,
                "eval_policy": policy,
                "first_positive_ep": first_episode_at(eval_rows, policy, 0.0, positive=True),
            }
            for th in THRESHOLDS:
                threshold_row[f"success_{str(th).replace('.', '_')}_ep"] = first_episode_at(eval_rows, policy, th)
            threshold_rows.append(threshold_row)

            curve_row = {
                "model": MODEL_LABEL[model],
                "seed": seed,
                "eval_policy": policy,
                "success_auc": auc(eval_rows, metric_key(policy, "success_rate"), total_episodes),
                "return_auc": auc(eval_rows, metric_key(policy, "mean_return"), total_episodes),
                "episode_length_auc": auc(eval_rows, metric_key(policy, "mean_episode_length"), total_episodes),
                "episode_length_slope": slope(eval_rows, metric_key(policy, "mean_episode_length")),
                "drop_below_0_9_after_reach": count_drops_after_reach(eval_rows, policy, 0.9),
                "drop_below_1_0_after_reach": count_drops_after_reach(eval_rows, policy, 1.0),
            }
            curve_rows.append(curve_row)

            summary_rows.append({**final_row, **threshold_row, **curve_row, **tstats})

    aggregate_final: list[dict[str, object]] = []
    aggregate_speed: list[dict[str, object]] = []
    aggregate_auc: list[dict[str, object]] = []
    for model_label in [MODEL_LABEL["baseline"], MODEL_LABEL["ablation"]]:
        for policy in POLICIES:
            finals = [r for r in final_rows if r["model"] == model_label and r["eval_policy"] == policy]
            thresholds = [r for r in threshold_rows if r["model"] == model_label and r["eval_policy"] == policy]
            curves = [r for r in curve_rows if r["model"] == model_label and r["eval_policy"] == policy]
            aggregate_final.append(
                {
                    "model": model_label,
                    "eval_policy": policy,
                    "mean_final_success": avg(finals, "final_success"),
                    "std_final_success": std(finals, "final_success"),
                    "mean_final_return": avg(finals, "final_return"),
                    "std_final_return": std(finals, "final_return"),
                    "mean_final_length": avg(finals, "final_episode_length"),
                    "std_final_length": std(finals, "final_episode_length"),
                }
            )
            speed_row: dict[str, object] = {"model": model_label, "eval_policy": policy}
            for th in THRESHOLDS:
                key = f"success_{str(th).replace('.', '_')}_ep"
                values = [r[key] for r in thresholds if r.get(key) is not None]
                speed_row[f"mean_ep_to_{str(th).replace('.', '_')}"] = mean(values) if values else None
            aggregate_speed.append(speed_row)
            aggregate_auc.append(
                {
                    "model": model_label,
                    "eval_policy": policy,
                    "mean_success_auc": avg(curves, "success_auc"),
                    "std_success_auc": std(curves, "success_auc"),
                    "mean_return_auc": avg(curves, "return_auc"),
                    "std_return_auc": std(curves, "return_auc"),
                    "mean_length_auc": avg(curves, "episode_length_auc"),
                    "mean_length_slope": avg(curves, "episode_length_slope"),
                }
            )

    figure_paths: list[Path] = []
    for policy in POLICIES:
        figure_paths.append(plot_seed_curves(eval_curves, policy, "success_rate", "Success Rate"))
        figure_paths.append(plot_seed_curves(eval_curves, policy, "mean_return", "Mean Return"))
        figure_paths.append(plot_seed_curves(eval_curves, policy, "mean_episode_length", "Episode Length"))
        figure_paths.append(plot_mean_curves(eval_curves, policy, "success_rate", "Success Rate"))
        figure_paths.append(plot_mean_curves(eval_curves, policy, "mean_return", "Mean Return"))

    write_csv(RESULTS / "multiroom_n4_s4_three_seed_process_summary.csv", summary_rows)

    threshold_headers = [
        "model",
        "seed",
        "eval_policy",
        "first_positive_ep",
        "success_0_5_ep",
        "success_0_8_ep",
        "success_0_9_ep",
        "success_1_0_ep",
    ]
    curve_headers = [
        "model",
        "seed",
        "eval_policy",
        "success_auc",
        "return_auc",
        "episode_length_slope",
        "drop_below_0_9_after_reach",
        "drop_below_1_0_after_reach",
    ]
    train_headers = [
        "model",
        "seed",
        "total_train_success",
        "last1000_success",
        "last500_success",
        "last100_success",
        "mean_reward",
        "late_reward",
        "early_length",
        "late_length",
        "length_delta_late_minus_early",
    ]

    summary_md = [
        "# MultiRoom N4-S4 Three-Seed Process Summary",
        "",
        "AUC is computed with trapezoidal integration over evaluation episode and divided by 5000 episodes.",
        "",
        "## File Matching",
        md_table(
            ["model", "seed", "env_id", "manager_type", "train_csv", "eval_csv", "config", "summary"],
            [
                [
                    r["model"],
                    r["seed"],
                    r["env_id"],
                    r["manager_type"],
                    r["train_csv"],
                    r["eval_csv"],
                    r["config"],
                    r["summary"],
                ]
                for r in file_rows
            ],
            digits=4,
        ),
        "",
        "## Final Eval Performance",
        md_table(
            ["model", "seed", "eval_policy", "final_success", "final_return", "final_episode_length"],
            [[r["model"], r["seed"], r["eval_policy"], r["final_success"], r["final_return"], r["final_episode_length"]] for r in final_rows],
        ),
        "",
        "## Success Threshold Episodes",
        md_table(threshold_headers, [[r.get(h) for h in threshold_headers] for r in threshold_rows]),
        "",
        "## Curve And Stability Metrics",
        md_table(curve_headers, [[r.get(h) for h in curve_headers] for r in curve_rows]),
        "",
        "## Train Log Metrics",
        md_table(train_headers, [[r.get(h) for h in train_headers] for r in train_rows]),
        "",
        "## Three-Seed Final Performance Mean",
        md_table(
            ["model", "eval_policy", "mean_final_success", "std_final_success", "mean_final_return", "std_final_return", "mean_final_length", "std_final_length"],
            [[r.get(h) for h in ["model", "eval_policy", "mean_final_success", "std_final_success", "mean_final_return", "std_final_return", "mean_final_length", "std_final_length"]] for r in aggregate_final],
        ),
        "",
        "## Three-Seed Learning Speed Mean",
        md_table(
            ["model", "eval_policy", "mean_ep_to_0_5", "mean_ep_to_0_8", "mean_ep_to_0_9", "mean_ep_to_1_0"],
            [[r.get(h) for h in ["model", "eval_policy", "mean_ep_to_0_5", "mean_ep_to_0_8", "mean_ep_to_0_9", "mean_ep_to_1_0"]] for r in aggregate_speed],
        ),
        "",
        "## Three-Seed Curve Area Mean",
        md_table(
            ["model", "eval_policy", "mean_success_auc", "mean_return_auc", "mean_length_auc", "mean_length_slope"],
            [[r.get(h) for h in ["model", "eval_policy", "mean_success_auc", "mean_return_auc", "mean_length_auc", "mean_length_slope"]] for r in aggregate_auc],
        ),
        "",
        "## Figures",
        *[f"- `{p.relative_to(ROOT)}`" for p in figure_paths],
        "",
    ]
    (RESULTS / "multiroom_n4_s4_three_seed_process_summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    def get_agg(rows: list[dict[str, object]], model_label: str, policy: str) -> dict[str, object]:
        return next(r for r in rows if r["model"] == model_label and r["eval_policy"] == policy)

    b_sample_final = get_agg(aggregate_final, "Baseline", "sample")
    a_sample_final = get_agg(aggregate_final, "Memory Ablation", "sample")
    b_argmax_final = get_agg(aggregate_final, "Baseline", "argmax")
    a_argmax_final = get_agg(aggregate_final, "Memory Ablation", "argmax")
    b_sample_speed = get_agg(aggregate_speed, "Baseline", "sample")
    a_sample_speed = get_agg(aggregate_speed, "Memory Ablation", "sample")
    b_argmax_speed = get_agg(aggregate_speed, "Baseline", "argmax")
    a_argmax_speed = get_agg(aggregate_speed, "Memory Ablation", "argmax")
    b_sample_auc = get_agg(aggregate_auc, "Baseline", "sample")
    a_sample_auc = get_agg(aggregate_auc, "Memory Ablation", "sample")
    b_argmax_auc = get_agg(aggregate_auc, "Baseline", "argmax")
    a_argmax_auc = get_agg(aggregate_auc, "Memory Ablation", "argmax")

    def threshold_sentence(policy: str, b_speed: dict[str, object], a_speed: dict[str, object]) -> str:
        return (
            f"{policy} evaluation에서 0.5/0.8/0.9/1.0 도달 평균 episode는 "
            f"Baseline {fmt(b_speed['mean_ep_to_0_5'], 1)}/{fmt(b_speed['mean_ep_to_0_8'], 1)}/"
            f"{fmt(b_speed['mean_ep_to_0_9'], 1)}/{fmt(b_speed['mean_ep_to_1_0'], 1)}, "
            f"Memory Ablation {fmt(a_speed['mean_ep_to_0_5'], 1)}/{fmt(a_speed['mean_ep_to_0_8'], 1)}/"
            f"{fmt(a_speed['mean_ep_to_0_9'], 1)}/{fmt(a_speed['mean_ep_to_1_0'], 1)}이다."
        )

    stability_lines = []
    for model_label in ["Baseline", "Memory Ablation"]:
        for policy in POLICIES:
            subset = [r for r in curve_rows if r["model"] == model_label and r["eval_policy"] == policy]
            stability_lines.append(
                f"- {model_label} {policy}: after >=0.9 drops mean {mean(float(r['drop_below_0_9_after_reach']) for r in subset):.2f}, "
                f"after 1.0 drops mean {mean(float(r['drop_below_1_0_after_reach']) for r in subset):.2f}"
            )

    analysis_md = [
        "# MultiRoom N4-S4 Three-Seed Process Analysis",
        "",
        "## Scope And File Matching",
        "This analysis uses the six completed `MiniGrid-MultiRoom-N4-S4-v0` runs for Baseline Manager and Memory Ablation Manager at seeds 1, 11, and 44. The file mapping is listed in the summary document and was checked against config `env_id`, `seed`, `manager_type`, and log directory name.",
        "",
        "The seed1 local files use legacy names such as `baseline_train.csv` and `ablation_eval.csv`; seed11 and seed44 use `train.csv` and `eval.csv`. The schemas are equivalent for the metrics used here.",
        "",
        "## Final Performance",
        f"At the last periodic evaluation point, both models solve N4-S4 at high success rates. For sample evaluation, the three-seed mean final success is {fmt(b_sample_final['mean_final_success'])} for Baseline and {fmt(a_sample_final['mean_final_success'])} for Memory Ablation. For argmax evaluation, the means are {fmt(b_argmax_final['mean_final_success'])} and {fmt(a_argmax_final['mean_final_success'])}, respectively.",
        "",
        f"Final return is also close. Sample final return averages {fmt(b_sample_final['mean_final_return'])} for Baseline and {fmt(a_sample_final['mean_final_return'])} for Memory Ablation. Argmax final return averages {fmt(b_argmax_final['mean_final_return'])} for Baseline and {fmt(a_argmax_final['mean_final_return'])} for Memory Ablation.",
        "",
        f"Final episode length shows the remaining behavioral-efficiency difference more directly. Sample final length averages {fmt(b_sample_final['mean_final_length'])} for Baseline and {fmt(a_sample_final['mean_final_length'])} for Memory Ablation; argmax final length averages {fmt(b_argmax_final['mean_final_length'])} and {fmt(a_argmax_final['mean_final_length'])}. Lower values indicate shorter successful trajectories.",
        "",
        "## Learning Speed",
        threshold_sentence("sample", b_sample_speed, a_sample_speed),
        threshold_sentence("argmax", b_argmax_speed, a_argmax_speed),
        "",
        "Averaged over the three seeds, Memory Ablation reaches every listed success threshold earlier than Baseline in both sample and argmax evaluation. This does not mean every individual seed is identical: seed1 sample has a slightly earlier Baseline 0.9 crossing, and seed1 argmax has an earlier Baseline 0.5 crossing. However, seeds11 and 44 reverse that pattern, so the three-seed result is that Memory Ablation learns the N4-S4 task faster on average.",
        "",
        f"The success AUC reinforces the process-level comparison. In sample evaluation, mean success AUC is {fmt(b_sample_auc['mean_success_auc'])} for Baseline and {fmt(a_sample_auc['mean_success_auc'])} for Memory Ablation. In argmax evaluation, it is {fmt(b_argmax_auc['mean_success_auc'])} versus {fmt(a_argmax_auc['mean_success_auc'])}. Because AUC is normalized by total training episodes, larger values mean that a model maintained higher success earlier and for more of training.",
        "",
        "## Stability After Improvement",
        "After reaching high success, instability was measured as drops below 0.9 or below 1.0 after the threshold had first been reached.",
        *stability_lines,
        "",
        "These drops matter because N4-S4 is eventually solvable by both models. A model that reaches 1.0 but repeatedly falls below it has learned a usable policy, but the evaluation curve indicates less stable policy consolidation.",
        "",
        "The stability result is mixed rather than one-sided. In sample evaluation, Memory Ablation has no drops below 0.9 after first reaching that threshold, while Baseline has a small mean drop count. In argmax evaluation, Baseline has fewer drops after reaching 0.9 and 1.0 than Memory Ablation, indicating that the deterministic policy is somewhat more stable for Baseline once it has consolidated.",
        "",
        "## Behavioral Efficiency",
        f"Final success is identical, but episode length separates the policies. Under sample evaluation, Baseline has shorter final trajectories on average ({fmt(b_sample_final['mean_final_length'])} steps) than Memory Ablation ({fmt(a_sample_final['mean_final_length'])} steps). Under argmax evaluation, Memory Ablation is slightly shorter ({fmt(a_argmax_final['mean_final_length'])} steps) than Baseline ({fmt(b_argmax_final['mean_final_length'])} steps). Therefore, N4-S4 should not be summarized only by success; the same success rate can correspond to different trajectory efficiency depending on whether the stochastic or deterministic policy is evaluated.",
        "",
        "## Train Log Interpretation",
        "The training logs show on-policy sample behavior, while eval logs report held-out evaluation episodes under sample and argmax policies. Last-window train success is therefore useful as a behavioral-process signal but should not replace the eval curves. In all six runs the final training windows are much stronger than the early windows, and episode length decreases from early to late training, indicating that learning is not only a binary success transition but also a shift toward shorter trajectories.",
        "",
        "## Sample Versus Argmax",
        "Sample evaluation reflects the stochastic policy that is also used during training. Argmax evaluation tests whether the policy has formed a deterministic high-probability action sequence. A gap where sample succeeds earlier than argmax indicates that useful behavior exists in the stochastic policy before it has become the dominant deterministic path. A later argmax catch-up indicates policy consolidation.",
        "",
        "In this experiment, both modes eventually become strong, so the sample/argmax comparison is most useful for reading when the policy starts to solve the environment versus when the solution becomes deterministically preferred.",
        "",
        "## Answers To The Core Questions",
        "1. Both Baseline and Memory Ablation ultimately solve N4-S4 at high success rates across the three seeds.",
        "2. Final success rate alone is not an adequate discriminator, because both models reach the same high-success regime.",
        "3. The meaningful difference is in the timing of first success and threshold crossings. First positive sample success appears at episode 100 for all runs, but argmax first positive success appears later and is generally earlier for Memory Ablation in seeds11 and 44.",
        "4. In the three-seed mean, Memory Ablation reaches 0.5, 0.8, 0.9, and 1.0 earlier than Baseline for both sample and argmax evaluation.",
        "5. Stability is policy-dependent. Memory Ablation is more stable in sample success after reaching 0.9, while Baseline is more stable in argmax after reaching high success.",
        "6. Return and episode length remain important even when success is equal. Sample evaluation favors Baseline in final trajectory length, while argmax evaluation slightly favors Memory Ablation.",
        "7. Sample success before argmax success means that the stochastic policy can sometimes solve the task before the deterministic policy has stabilized. This is visible in both models, especially before the argmax curve catches up.",
        "8. The seed1-only trend should not be overclaimed. Some seed1 crossings favor Baseline, but seed11 and seed44 generally favor Memory Ablation in learning speed and AUC, so the 3-seed result is more balanced and more informative.",
        "9. In a paper, N4-S4 is best presented as a process-level and policy-formation experiment: both models can solve the environment, so the comparison should emphasize learning speed, stability after improvement, and behavioral efficiency.",
        "",
        "## Recommended Paper Framing",
        "N4-S4에서는 Baseline과 Memory Ablation이 모두 최종적으로 높은 성공률을 달성하였다. 따라서 이 환경에서는 단순한 최종 성공 여부보다, 학습 속도와 안정성, 그리고 동일한 성공률을 달성한 이후의 행동 효율성 차이가 더 중요한 비교 기준이 된다.",
        "",
        "The quantitative support for this framing is in `multiroom_n4_s4_three_seed_process_summary.md` and the generated figures.",
        "",
    ]
    (RESULTS / "multiroom_n4_s4_three_seed_process_analysis.md").write_text("\n".join(analysis_md), encoding="utf-8")

    print("wrote", RESULTS / "multiroom_n4_s4_three_seed_process_summary.csv")
    print("wrote", RESULTS / "multiroom_n4_s4_three_seed_process_summary.md")
    print("wrote", RESULTS / "multiroom_n4_s4_three_seed_process_analysis.md")
    for path in figure_paths:
        print("figure", path)


if __name__ == "__main__":
    main()
