from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from evaluate_checkpoint import build_model_from_config, build_policy_from_config
from src.envs.make_env import make_env
from src.training.evaluation import evaluate_policy
from src.training.rollout import random_policy
from src.utils.config import load_simple_yaml
from src.utils.seed import set_seed


BASE_LOG_DIR = Path("logs/baseline_fun")
RESULTS_DIR = Path("results")
SEEDS = [1, 11, 44]
NUM_ACTIONS = 7


def _missing() -> str:
    return "missing"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return out
    return out


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _numeric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _to_float(row.get(column))
        if value is not None:
            values.append(value)
    return values


def _summary_stats(rows: list[dict[str, str]], column: str) -> dict[str, float | str]:
    values = _numeric_values(rows, column)
    if not values:
        return {"status": _missing()}
    finite_values = [v for v in values if math.isfinite(v)]
    if not finite_values:
        return {"min": min(values), "max": max(values), "mean": float("nan")}
    return {
        "min": min(finite_values),
        "max": max(finite_values),
        "mean": sum(finite_values) / len(finite_values),
    }


def _count_true(rows: list[dict[str, str]], column: str) -> int | str:
    if not rows or column not in rows[0]:
        return _missing()
    return sum(1 for row in rows if _to_bool(row.get(column)))


def _mean_column(rows: list[dict[str, str]], column: str) -> float | str:
    values = _numeric_values(rows, column)
    if not values:
        return _missing()
    return sum(values) / len(values)


def _final_n_success_rate(rows: list[dict[str, str]], n: int = 50) -> float | str:
    if not rows or "success" not in rows[0]:
        return _missing()
    tail = rows[-n:]
    return sum(1.0 for row in tail if _to_bool(row.get("success"))) / len(tail)


def _final_n_mean(rows: list[dict[str, str]], column: str, n: int = 50) -> float | str:
    tail = rows[-n:]
    values = _numeric_values(tail, column)
    if not values:
        return _missing()
    return sum(values) / len(values)


def _parse_action_histogram(raw: str | None) -> list[int] | None:
    if raw is None or raw == "":
        return None
    text = raw.strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            values = [int(x) for x in parsed]
            return values if values else None
    except (SyntaxError, ValueError):
        pass
    try:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError:
        return None
    return values if values else None


def _action_analysis(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"status": _missing()}
    result: dict[str, Any] = {"action_coverage": _summary_stats(rows, "action_coverage")}
    if "action_histogram" not in rows[0]:
        result["histogram_status"] = _missing()
        return result

    totals: list[int] = []
    parsed_rows = 0
    dominant_fractions: list[float] = []
    for row in rows:
        hist = _parse_action_histogram(row.get("action_histogram"))
        if hist is None:
            continue
        if len(totals) < len(hist):
            totals.extend([0] * (len(hist) - len(totals)))
        for idx, count in enumerate(hist):
            totals[idx] += count
        total = sum(hist)
        if total > 0:
            dominant_fractions.append(max(hist) / total)
        parsed_rows += 1

    if not totals:
        result["histogram_status"] = "unparseable"
        return result

    total_actions = sum(totals)
    distribution = [count / total_actions if total_actions else 0.0 for count in totals]
    result.update(
        {
            "histogram_rows_parsed": parsed_rows,
            "total_action_counts": totals,
            "action_distribution": distribution,
            "dominant_action": int(max(range(len(totals)), key=lambda idx: totals[idx])),
            "dominant_action_fraction": max(distribution) if distribution else 0.0,
            "mean_episode_dominant_fraction": (
                sum(dominant_fractions) / len(dominant_fractions) if dominant_fractions else _missing()
            ),
            "collapsed": bool(max(distribution) > 0.9) if distribution else False,
        }
    )
    return result


def _load_checkpoint_eval(seed: int, action_mode: str) -> dict[str, Any] | str:
    path = BASE_LOG_DIR / f"seed_{seed}" / f"checkpoint_eval_best_{action_mode}.json"
    if action_mode == "argmax" and not path.exists():
        path = BASE_LOG_DIR / f"seed_{seed}" / "checkpoint_eval_best.json"
    if not path.exists():
        return _missing()
    data = _read_json(path)
    eval_result = data.get("eval_result", {})
    return {
        "path": str(path),
        "success_rate": eval_result.get("mean_success_rate", _missing()),
        "mean_return": eval_result.get("mean_reward", _missing()),
        "mean_episode_length": eval_result.get("mean_episode_length", _missing()),
        "episodes": data.get("eval_episodes", _missing()),
        "checkpoint_episode": data.get("checkpoint_episode", _missing()),
    }


def analyze_seed(seed: int) -> dict[str, Any]:
    seed_dir = BASE_LOG_DIR / f"seed_{seed}"
    train_path = seed_dir / "train.csv"
    eval_path = seed_dir / "eval.csv"
    summary_path = seed_dir / "summary.json"
    train_rows = _read_csv(train_path)
    eval_rows = _read_csv(eval_path)
    summary = _read_json(summary_path)

    train_columns = set(train_rows[0].keys()) if train_rows else set()
    eval_columns = set(eval_rows[0].keys()) if eval_rows else set()
    success_total = _count_true(train_rows, "success")
    train_success_rate = (
        success_total / len(train_rows) if isinstance(success_total, int) and train_rows else _missing()
    )

    eval_success_values = _numeric_values(eval_rows, "eval_success_rate")
    eval_return_values = _numeric_values(eval_rows, "eval_mean_return")

    return {
        "seed": seed,
        "paths": {
            "train_csv": str(train_path) if train_path.exists() else _missing(),
            "eval_csv": str(eval_path) if eval_path.exists() else _missing(),
            "summary_json": str(summary_path) if summary_path.exists() else _missing(),
        },
        "train": {
            "rows": len(train_rows),
            "success_column": "success" in train_columns,
            "success_total": success_total,
            "success_rate": train_success_rate,
            "reward_column": "total_reward" if "total_reward" in train_columns else _missing(),
            "total_reward": _summary_stats(train_rows, "total_reward"),
            "episode_length": _summary_stats(train_rows, "episode_length"),
            "final_50_success_rate": _final_n_success_rate(train_rows),
            "final_50_mean_reward": _final_n_mean(train_rows, "total_reward"),
            "has_reward_signal_true": _count_true(train_rows, "has_reward_signal"),
            "has_return_signal_true": _count_true(train_rows, "has_return_signal"),
            "nonzero_reward_fraction_mean": _mean_column(train_rows, "nonzero_reward_fraction"),
            "nonzero_return_fraction_mean": _mean_column(train_rows, "nonzero_return_fraction"),
            "total_loss": _summary_stats(train_rows, "total_loss"),
            "worker_loss": _summary_stats(train_rows, "worker_loss"),
            "value_loss": _summary_stats(train_rows, "value_loss"),
            "manager_loss": _summary_stats(train_rows, "manager_loss"),
            "entropy_mean": _summary_stats(train_rows, "entropy_mean"),
            "grad_norm": _summary_stats(train_rows, "grad_norm"),
            "action_coverage": _summary_stats(train_rows, "action_coverage"),
            "final_goal_norm": _summary_stats(train_rows, "final_goal_norm"),
            "values_mean": _summary_stats(train_rows, "values_mean"),
            "value_min": _summary_stats(train_rows, "value_min"),
            "value_max": _summary_stats(train_rows, "value_max"),
            "advantages_mean": _summary_stats(train_rows, "advantages_mean"),
            "advantages_abs_mean": _summary_stats(train_rows, "advantages_abs_mean"),
            "log_prob_mean": _summary_stats(train_rows, "log_prob_mean"),
            "log_prob_min": _summary_stats(train_rows, "log_prob_min"),
            "log_prob_max": _summary_stats(train_rows, "log_prob_max"),
            "action_analysis": _action_analysis(train_rows),
        },
        "eval": {
            "action_mode": summary.get("config", {}).get("eval_action_mode", _missing())
            if isinstance(summary.get("config"), dict)
            else _missing(),
            "rows": len(eval_rows),
            "best_success_rate": max(eval_success_values) if eval_success_values else _missing(),
            "final_success_rate": eval_success_values[-1] if eval_success_values else _missing(),
            "best_mean_return": max(eval_return_values) if eval_return_values else _missing(),
            "final_mean_return": eval_return_values[-1] if eval_return_values else _missing(),
            "final_mean_episode_length": (
                _numeric_values(eval_rows, "eval_mean_episode_length")[-1]
                if eval_rows and "eval_mean_episode_length" in eval_columns
                else _missing()
            ),
            "ever_success_gt_zero": any(value > 0.0 for value in eval_success_values),
        },
        "summary": {
            "best_success_rate": summary.get("best_success_rate", _missing()),
            "final_eval": summary.get("final_eval", _missing()),
            "best_checkpoint_path": summary.get("best_checkpoint_path", _missing()),
            "last_checkpoint_path": summary.get("last_checkpoint_path", _missing()),
            "final_episode_checkpoint_path": summary.get("final_episode_checkpoint_path", _missing()),
            "final_reward_moving_avg": summary.get("final_reward_moving_avg", _missing()),
            "final_success_moving_avg": summary.get("final_success_moving_avg", _missing()),
            "best_success": summary.get("best_success", _missing()),
        },
        "sample_evaluation": _load_checkpoint_eval(seed, "sample"),
        "argmax_checkpoint_evaluation": _load_checkpoint_eval(seed, "argmax"),
    }


def evaluate_reference_policies() -> dict[str, Any]:
    cfg = load_simple_yaml("configs/train_fun_baseline_seed1.yaml")
    cfg["env_id"] = "MiniGrid-DoorKey-5x5-v0"
    seed = int(cfg.get("seed", 1))
    max_steps = int(cfg.get("max_steps", 250))
    episodes = 100
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output: dict[str, Any] = {}
    env = make_env(env_id=str(cfg["env_id"]), render_mode=cfg.get("render_mode"), seed=seed)
    try:
        output["random_policy"] = evaluate_policy(
            env=env,
            policy=random_policy(env),
            num_episodes=episodes,
            max_steps=max_steps,
            seed=seed + 5000,
        )
    finally:
        env.close()

    for action_mode in ("sample", "argmax"):
        set_seed(seed)
        env = make_env(env_id=str(cfg["env_id"]), render_mode=cfg.get("render_mode"), seed=seed)
        try:
            model = build_model_from_config(cfg, device=device)
            policy = build_policy_from_config(cfg, model=model, device=device, action_mode=action_mode)
            output[f"untrained_fun_{action_mode}"] = evaluate_policy(
                env=env,
                policy=policy,
                num_episodes=episodes,
                max_steps=max_steps,
                seed=seed + 6000,
            )
        finally:
            env.close()

    return {key: _compact_eval(value) for key, value in output.items()}


def _compact_eval(eval_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "success_rate": eval_result.get("mean_success_rate", _missing()),
        "mean_return": eval_result.get("mean_reward", _missing()),
        "mean_episode_length": eval_result.get("mean_episode_length", _missing()),
        "min_return": eval_result.get("min_reward", _missing()),
        "max_return": eval_result.get("max_reward", _missing()),
        "episodes": eval_result.get("num_episodes", _missing()),
    }


def load_empty_sanity() -> dict[str, Any]:
    base = Path("logs/sanity_empty5x5/seed_1")
    summary = _read_json(base / "summary.json")
    argmax_eval = _read_json(base / "checkpoint_eval_best_argmax.json")
    sample_eval = _read_json(base / "checkpoint_eval_best_sample.json")
    if not summary and not argmax_eval and not sample_eval:
        return {"status": "not_run"}
    return {
        "summary_best_success_rate": summary.get("best_success_rate", _missing()),
        "summary_final_eval": summary.get("final_eval", _missing()),
        "argmax_best_checkpoint": _compact_eval(argmax_eval.get("eval_result", {})) if argmax_eval else _missing(),
        "sample_best_checkpoint": _compact_eval(sample_eval.get("eval_result", {})) if sample_eval else _missing(),
    }


def detect_warnings(seed_reports: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if all(report["train"]["success_total"] == 0 for report in seed_reports if report["train"]["rows"]):
        warnings.append("train success가 모든 seed에서 0입니다.")
    if all(report["eval"]["best_success_rate"] == 0 for report in seed_reports if report["eval"]["rows"]):
        warnings.append("eval success가 모든 seed에서 0입니다.")
    if all(report["train"]["total_reward"].get("max") == 0 for report in seed_reports if report["train"]["rows"]):
        warnings.append("total_reward max가 모든 seed에서 0입니다.")

    for report in seed_reports:
        seed = report["seed"]
        train = report["train"]
        final_episode = report["summary"].get("final_eval", {})
        expected_episodes = None
        if isinstance(final_episode, dict):
            expected_episodes = final_episode.get("episode")
        if isinstance(expected_episodes, (int, float)) and train["rows"] != int(expected_episodes):
            warnings.append(
                f"seed {seed}: train.csv rows({train['rows']})가 summary final episode({int(expected_episodes)})와 다릅니다."
            )
        episode_length = train["episode_length"]
        if episode_length.get("mean") == episode_length.get("max") and episode_length.get("max") == 250:
            warnings.append(f"seed {seed}: episode_length 평균이 max_steps(250)와 같습니다.")
        entropy = train["entropy_mean"]
        if _range_is_tiny(entropy, tolerance=1e-3):
            warnings.append(f"seed {seed}: entropy_mean 변화가 매우 작습니다.")
        action_coverage = train["action_coverage"]
        if isinstance(action_coverage.get("mean"), float) and action_coverage["mean"] < 0.5:
            warnings.append(f"seed {seed}: action_coverage 평균이 낮습니다.")
        grad_norm = train["grad_norm"]
        if isinstance(grad_norm.get("mean"), float) and abs(grad_norm["mean"]) < 1e-8:
            warnings.append(f"seed {seed}: grad_norm이 거의 0입니다.")
        value_loss = train["value_loss"]
        has_reward = train["has_reward_signal_true"]
        if isinstance(has_reward, int) and has_reward == 0 and not _range_is_tiny(value_loss, tolerance=1e-8):
            warnings.append(f"seed {seed}: reward signal 없이 value_loss만 변하고 있습니다.")
        adv_abs = train["advantages_abs_mean"]
        if isinstance(adv_abs.get("mean"), float) and abs(adv_abs["mean"]) < 1e-8:
            warnings.append(f"seed {seed}: advantages_abs_mean이 0에 가깝습니다.")
        for column in ("log_prob_mean", "log_prob_min", "log_prob_max"):
            stats = train[column]
            if any(isinstance(stats.get(k), float) and not math.isfinite(stats[k]) for k in ("min", "max", "mean")):
                warnings.append(f"seed {seed}: {column}에 NaN/inf가 있습니다.")
        for column in ("total_loss", "worker_loss", "value_loss", "manager_loss"):
            stats = train[column]
            if any(isinstance(stats.get(k), float) and not math.isfinite(stats[k]) for k in ("min", "max", "mean")):
                warnings.append(f"seed {seed}: {column}에 NaN/inf가 있습니다.")
        action = train["action_analysis"]
        if action.get("collapsed"):
            warnings.append(f"seed {seed}: action distribution이 한 action으로 collapse된 것으로 보입니다.")
    return warnings


def _range_is_tiny(stats: dict[str, Any], tolerance: float) -> bool:
    return isinstance(stats.get("min"), float) and isinstance(stats.get("max"), float) and (
        abs(stats["max"] - stats["min"]) <= tolerance
    )


def decide_conclusion(
    seed_reports: list[dict[str, Any]],
    reference: dict[str, Any],
    empty_sanity: dict[str, Any],
) -> tuple[str, list[str]]:
    train_success = sum(
        report["train"]["success_total"]
        for report in seed_reports
        if isinstance(report["train"]["success_total"], int)
    )
    argmax_success = [
        report["argmax_checkpoint_evaluation"].get("success_rate")
        for report in seed_reports
        if isinstance(report.get("argmax_checkpoint_evaluation"), dict)
    ]
    if argmax_success:
        eval_any = any(isinstance(value, (int, float)) and value > 0.0 for value in argmax_success)
    else:
        eval_any = any(
            report["eval"]["ever_success_gt_zero"]
            for report in seed_reports
            if report["eval"].get("action_mode") == "argmax"
        )
    sample_success = [
        report["sample_evaluation"].get("success_rate")
        for report in seed_reports
        if isinstance(report["sample_evaluation"], dict)
    ]
    sample_any = any(isinstance(value, (int, float)) and value > 0.0 for value in sample_success)
    reasons: list[str] = []

    if train_success > 0 and not eval_any and sample_any:
        conclusion = "2. evaluation argmax 방식 문제가 큼"
        reasons.append("train에서는 성공/reward가 있었고 sample checkpoint eval도 성공하지만 argmax eval은 0입니다.")
    elif empty_sanity.get("status") != "not_run" and _empty_success(empty_sanity) and train_success > 0 and not eval_any:
        conclusion = "1. DoorKey sparse reward 문제가 가장 큼"
        reasons.append("더 쉬운 Empty-5x5에서는 성공하지만 DoorKey argmax eval은 실패합니다.")
    elif train_success == 0 and not eval_any and not sample_any:
        conclusion = "3. 학습 로직 또는 loss 연결 버그 가능성이 큼"
        reasons.append("train/eval/sample 모두 성공이 없습니다.")
    elif train_success > 0 and not eval_any and not sample_any:
        conclusion = "4. 아직 판단 불가, 추가 확인 필요"
        reasons.append("train에서는 reward가 있으나 checkpoint sample/argmax eval은 모두 0입니다.")
    else:
        conclusion = "4. 아직 판단 불가, 추가 확인 필요"
        reasons.append("현재 신호가 한 가지 원인으로 수렴하지 않습니다.")

    untrained_sample = reference.get("untrained_fun_sample", {}).get("success_rate", _missing())
    if isinstance(untrained_sample, (int, float)) and sample_success:
        trained_mean = sum(float(v) for v in sample_success if isinstance(v, (int, float))) / len(sample_success)
        reasons.append(f"trained sample 평균 success={trained_mean:.3f}, untrained sample success={untrained_sample:.3f}.")
    return conclusion, reasons


def _empty_success(empty_sanity: dict[str, Any]) -> bool:
    for key in ("argmax_best_checkpoint", "sample_best_checkpoint"):
        value = empty_sanity.get(key)
        if isinstance(value, dict):
            success = value.get("success_rate")
            if isinstance(success, (int, float)) and success > 0.0:
                return True
    final_eval = empty_sanity.get("summary_final_eval")
    if isinstance(final_eval, dict):
        success = final_eval.get("eval_success_rate")
        return isinstance(success, (int, float)) and success > 0.0
    return False


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Week 3 Baseline Diagnosis", ""]
    lines += ["## 요약"]
    for reason in report["conclusion_reasons"]:
        lines.append(f"- {reason}")
    lines.append(f"- 최종 판단: {report['conclusion']}")
    lines.append("- baseline 공식 평가는 sample 기준으로 전환하고, argmax는 참고 지표로 유지합니다.")
    lines.append("- 4주차 memory ablation도 sample success/return/episode length를 primary metric으로 비교해야 합니다.")
    lines.append("")

    lines += [
        "## Baseline Eval 결과",
        "",
        "| seed | eval action mode | eval best success | eval final success | final return | final length |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for seed_report in report["seeds"]:
        ev = seed_report["eval"]
        lines.append(
            f"| {seed_report['seed']} | {_fmt(ev['action_mode'])} | {_fmt(ev['best_success_rate'])} | {_fmt(ev['final_success_rate'])} | "
            f"{_fmt(ev['final_mean_return'])} | {_fmt(ev['final_mean_episode_length'])} |"
        )
    lines.append("")

    lines += [
        "## Train Log 분석",
        "",
        "| seed | rows | train success | success rate | final50 success | reward min/max/mean | entropy mean | grad mean | action coverage mean |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for seed_report in report["seeds"]:
        tr = seed_report["train"]
        reward = tr["total_reward"]
        lines.append(
            f"| {seed_report['seed']} | {tr['rows']} | {_fmt(tr['success_total'])} | {_fmt(tr['success_rate'])} | "
            f"{_fmt(tr['final_50_success_rate'])} | {_fmt(reward.get('min'))}/{_fmt(reward.get('max'))}/{_fmt(reward.get('mean'))} | "
            f"{_fmt(tr['entropy_mean'].get('mean'))} | {_fmt(tr['grad_norm'].get('mean'))} | {_fmt(tr['action_coverage'].get('mean'))} |"
        )
        action = tr["action_analysis"]
        lines.append(
            f"- seed {seed_report['seed']} action distribution: dominant={_fmt(action.get('dominant_action', _missing()))}, "
            f"dominant_fraction={_fmt(action.get('dominant_action_fraction', _missing()))}, "
            f"collapsed={_fmt(action.get('collapsed', _missing()))}, counts={_fmt(action.get('total_action_counts', _missing()))}"
        )
    lines.append("")

    lines += [
        "## Sample Evaluation 결과",
        "",
        "| seed | argmax checkpoint success | sample success | sample return | sample length |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for seed_report in report["seeds"]:
        sample = seed_report["sample_evaluation"]
        sample_dict = sample if isinstance(sample, dict) else {}
        argmax = seed_report.get("argmax_checkpoint_evaluation")
        argmax_dict = argmax if isinstance(argmax, dict) else {}
        lines.append(
            f"| {seed_report['seed']} | {_fmt(argmax_dict.get('success_rate', argmax))} | "
            f"{_fmt(sample_dict.get('success_rate', sample))} | {_fmt(sample_dict.get('mean_return', _missing()))} | "
            f"{_fmt(sample_dict.get('mean_episode_length', _missing()))} |"
        )
    lines.append("")

    lines += ["## Random / Untrained Baseline", "", "| policy | success | mean return | mean length |", "| --- | ---: | ---: | ---: |"]
    for name, value in report["reference_policies"].items():
        lines.append(
            f"| {name} | {_fmt(value.get('success_rate'))} | {_fmt(value.get('mean_return'))} | {_fmt(value.get('mean_episode_length'))} |"
        )
    lines.append("")

    lines += ["## Empty-5x5 Sanity Check"]
    empty = report["empty5x5_sanity"]
    if empty.get("status") == "not_run":
        lines.append("- 아직 실행 결과가 없습니다.")
    else:
        lines.append(f"- summary best_success_rate: {_fmt(empty.get('summary_best_success_rate'))}")
        lines.append(f"- argmax best checkpoint: {_fmt(empty.get('argmax_best_checkpoint'))}")
        lines.append(f"- sample best checkpoint: {_fmt(empty.get('sample_best_checkpoint'))}")
    lines.append("")

    lines += ["## Warning 목록"]
    if report["warnings"]:
        lines += [f"- {warning}" for warning in report["warnings"]]
    else:
        lines.append("- 자동 감지 warning 없음")
    lines.append("")

    lines += ["## 결론", f"- {report['conclusion']}"]
    for reason in report["conclusion_reasons"]:
        lines.append(f"- 근거: {reason}")
    lines.append("- baseline 공식 평가는 sample 기준으로 전환합니다.")
    lines.append("- argmax evaluation은 참고 지표로 유지합니다.")
    lines.append("- 4주차 memory ablation 비교도 sample 기준으로 수행해야 합니다.")
    lines.append("")

    lines += [
        "## 다음 액션 제안",
        "- DoorKey는 argmax와 sample 평가를 계속 병행해서 기록",
        "- Empty-5x5 sanity 결과가 양호하면 DoorKey curriculum 또는 reward shaping 검토",
        "- sample도 실패하면 checkpoint 저장 시점, recurrent reset, loss/return 연결을 우선 점검",
        "- sparse reward 대응으로 intrinsic reward 또는 FuN worker intrinsic reward는 별도 실험으로 분리",
    ]
    lines.append("")
    return "\n".join(lines)


def print_console_summary(report: dict[str, Any]) -> None:
    print("# Week 3 Baseline Diagnosis")
    print(f"conclusion: {report['conclusion']}")
    for seed_report in report["seeds"]:
        train = seed_report["train"]
        ev = seed_report["eval"]
        sample = seed_report["sample_evaluation"]
        sample_success = sample.get("success_rate") if isinstance(sample, dict) else sample
        argmax = seed_report.get("argmax_checkpoint_evaluation")
        argmax_success = argmax.get("success_rate") if isinstance(argmax, dict) else argmax
        print(
            f"seed={seed_report['seed']} "
            f"train_success={train['success_total']}/{train['rows']} "
            f"train_reward_max={train['total_reward'].get('max')} "
            f"eval_mode={ev['action_mode']} "
            f"eval_best={ev['best_success_rate']} "
            f"eval_final={ev['final_success_rate']} "
            f"argmax_checkpoint_success={argmax_success} "
            f"sample_success={sample_success}"
        )
    print("warnings:")
    for warning in report["warnings"]:
        print(f"- {warning}")
    print("outputs:")
    print("- results/week3_baseline_diagnosis.md")
    print("- results/week3_baseline_diagnosis.json")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_reports = [analyze_seed(seed) for seed in SEEDS]
    reference = evaluate_reference_policies()
    empty_sanity = load_empty_sanity()
    warnings = detect_warnings(seed_reports)
    conclusion, reasons = decide_conclusion(seed_reports, reference, empty_sanity)
    report = {
        "seeds": seed_reports,
        "reference_policies": reference,
        "empty5x5_sanity": empty_sanity,
        "warnings": warnings,
        "conclusion": conclusion,
        "conclusion_reasons": reasons,
    }

    (RESULTS_DIR / "week3_baseline_diagnosis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / "week3_baseline_diagnosis.md").write_text(
        build_markdown(report),
        encoding="utf-8",
    )
    print_console_summary(report)


if __name__ == "__main__":
    main()
