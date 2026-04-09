"""
Statistics for pairwise secondary evaluation.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import optimize, stats as sp_stats

from .runner import summarise_judge_orders


Record = Dict[str, Any]


def compute_slice_statistics(
    *,
    records: List[Record],
    run_spec,
    case_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    judges = [judge.judge_id for judge in run_spec.judge_manifest.judges]
    per_judge = {
        judge_id: compute_scope_statistics(
            [record for record in records if record["judge_id"] == judge_id]
        )
        for judge_id in judges
    }

    pooled_complete = all(per_judge[judge_id]["summary"]["total"] > 0 for judge_id in judges)
    pooled = compute_scope_statistics(records) if pooled_complete else {
        "status": "incomplete",
        "reason": "all four judges did not complete the same cohort",
        "summary": {
            "total": len(records),
            "invalid": sum(1 for record in records if record["is_invalid"]),
            "ties": sum(1 for record in records if record["is_tie"]),
            "decisive": sum(1 for record in records if not record["is_invalid"] and not record["is_tie"]),
        },
        "win_rates": [],
        "bradley_terry": [],
        "swap_consistency": {},
        "verbosity_bias": {},
    }

    return {
        "run_id": run_spec.config.run_id,
        "layer": run_spec.config.layer,
        "slice_id": run_spec.config.slice_id,
        "run_mode": run_spec.config.run_mode,
        "criteria": run_spec.criteria,
        "judge_panel": [
            {
                "judge_id": judge.judge_id,
                "role": judge.role,
                "hf_source": judge.hf_source,
                "local_model_id": judge.local_model_id,
            }
            for judge in run_spec.judge_manifest.judges
        ],
        "case_manifest_status": case_manifest.get("status", "unknown"),
        "case_count": len(case_manifest.get("cases", [])),
        "candidate_systems": case_manifest.get("systems", []),
        "boundary_notes": case_manifest.get("boundary_notes", []),
        "per_judge": per_judge,
        "pooled_complete": pooled_complete,
        "pooled": pooled,
        "judge_agreement": compute_judge_agreement(records),
        "execution_summary": compute_execution_summary(records, run_spec=run_spec),
    }


def compute_scope_statistics(records: List[Record]) -> Dict[str, Any]:
    return {
        "summary": {
            "total": len(records),
            "invalid": sum(1 for record in records if record["is_invalid"]),
            "ties": sum(1 for record in records if record["is_tie"]),
            "decisive": sum(1 for record in records if not record["is_invalid"] and not record["is_tie"]),
        },
        "win_rates": compute_win_rates(records),
        "bradley_terry": compute_bradley_terry(records),
        "swap_consistency": compute_swap_consistency(records),
        "verbosity_bias": compute_verbosity_bias(records),
    }


def compute_win_rates(records: List[Record]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Record]] = defaultdict(list)
    for record in records:
        grouped[(record["criterion_id"],) + _sorted_pair(record)].append(record)

    rows: List[Dict[str, Any]] = []
    for (criterion_id, system_a, system_b), bucket in sorted(grouped.items()):
        wins_a = 0
        wins_b = 0
        ties = 0
        invalid = 0
        decisive_records: List[Record] = []
        for record in bucket:
            if record["is_invalid"]:
                invalid += 1
                continue
            if record["is_tie"]:
                ties += 1
                continue
            decisive_records.append(record)
            if record["winner"] == system_a:
                wins_a += 1
            elif record["winner"] == system_b:
                wins_b += 1

        decisive_total = wins_a + wins_b
        total = decisive_total + ties + invalid
        ci_a = case_stratified_bootstrap(
            decisive_records,
            lambda sample: _win_rate_for(sample, winner=system_a),
        )
        ci_b = case_stratified_bootstrap(
            decisive_records,
            lambda sample: _win_rate_for(sample, winner=system_b),
        )
        rows.append(
            {
                "criterion_id": criterion_id,
                "system_a": system_a,
                "system_b": system_b,
                "wins_a": wins_a,
                "wins_b": wins_b,
                "ties": ties,
                "invalid": invalid,
                "decisive_total": decisive_total,
                "total": total,
                "win_rate_a": round(wins_a / decisive_total, 6) if decisive_total else 0.0,
                "win_rate_b": round(wins_b / decisive_total, 6) if decisive_total else 0.0,
                "win_rate_a_ci_lower": ci_a[0],
                "win_rate_a_ci_upper": ci_a[1],
                "win_rate_b_ci_lower": ci_b[0],
                "win_rate_b_ci_upper": ci_b[1],
            }
        )
    return rows


def compute_bradley_terry(records: List[Record]) -> List[Dict[str, Any]]:
    decisive = [record for record in records if not record["is_invalid"] and not record["is_tie"]]
    if not decisive:
        return []

    comparisons = [(record["winner"], _loser_for(record)) for record in decisive]
    model_ids = sorted({item for pair in comparisons for item in pair})
    result = fit_bradley_terry(comparisons, model_ids=model_ids)

    counts = Counter(record["winner"] for record in decisive)
    rows: List[Dict[str, Any]] = []
    for system_id in model_ids:
        score = result.get(system_id, {})
        rows.append(
            {
                "system_id": system_id,
                "score": score.get("score", 0.0),
                "se": score.get("se", 0.0),
                "ci_lower": score.get("ci_lower", 0.0),
                "ci_upper": score.get("ci_upper", 0.0),
                "decisive_total": counts.get(system_id, 0),
            }
        )
    return rows


def fit_bradley_terry(
    comparisons: List[Tuple[str, str]],
    *,
    model_ids: List[str],
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    idx = {model_id: position for position, model_id in enumerate(model_ids)}
    count_matrix = np.zeros((len(model_ids), len(model_ids)), dtype=np.float64)
    for winner, loser in comparisons:
        count_matrix[idx[winner], idx[loser]] += 1.0

    def neg_log_likelihood(strengths: np.ndarray) -> float:
        value = 0.0
        for i in range(len(model_ids)):
            for j in range(len(model_ids)):
                wins = count_matrix[i, j]
                if wins == 0:
                    continue
                diff = strengths[i] - strengths[j]
                value += wins * np.log1p(np.exp(-diff))
        return float(value)

    def gradient(strengths: np.ndarray) -> np.ndarray:
        grad = np.zeros(len(model_ids), dtype=np.float64)
        for i in range(len(model_ids)):
            for j in range(len(model_ids)):
                if i == j:
                    continue
                total = count_matrix[i, j] + count_matrix[j, i]
                if total == 0:
                    continue
                diff = strengths[i] - strengths[j]
                prob = 1.0 / (1.0 + np.exp(-diff))
                grad[i] -= count_matrix[i, j] * (1.0 - prob)
                grad[i] += count_matrix[j, i] * prob
        return grad

    optimum = optimize.minimize(
        neg_log_likelihood,
        np.zeros(len(model_ids), dtype=np.float64),
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    strengths = optimum.x - optimum.x.mean()
    standard_errors = _bt_standard_errors(strengths, count_matrix)
    z_value = float(sp_stats.norm.ppf(1 - (1 - confidence) / 2))
    return {
        model_id: {
            "score": round(float(strengths[idx[model_id]]), 6),
            "se": round(float(standard_errors[idx[model_id]]), 6),
            "ci_lower": round(float(strengths[idx[model_id]] - z_value * standard_errors[idx[model_id]]), 6),
            "ci_upper": round(float(strengths[idx[model_id]] + z_value * standard_errors[idx[model_id]]), 6),
        }
        for model_id in model_ids
    }


def _bt_standard_errors(strengths: np.ndarray, count_matrix: np.ndarray) -> np.ndarray:
    size = len(strengths)
    fisher = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            total = count_matrix[i, j] + count_matrix[j, i]
            if total == 0:
                continue
            diff = strengths[i] - strengths[j]
            prob = 1.0 / (1.0 + np.exp(-diff))
            info = total * prob * (1.0 - prob)
            fisher[i, i] += info
            fisher[i, j] -= info
    try:
        inverse = np.linalg.inv(fisher + np.eye(size) * 1e-8)
        return np.sqrt(np.maximum(np.diag(inverse), 0.0))
    except np.linalg.LinAlgError:
        return np.zeros(size, dtype=np.float64)


def compute_swap_consistency(records: List[Record]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Record]] = defaultdict(dict)
    for record in records:
        grouped[
            (
                record["case_id"],
                record["criterion_id"],
                record["judge_id"],
                record["canonical_pair_key"],
            )
        ][record["order"]] = record

    counters = {"consistent": 0, "inconsistent": 0, "tie_involved": 0, "invalid_involved": 0}
    for pair in grouped.values():
        if "AB" not in pair or "BA" not in pair:
            continue
        left = pair["AB"]
        right = pair["BA"]
        if left["is_invalid"] or right["is_invalid"]:
            counters["invalid_involved"] += 1
        elif left["is_tie"] or right["is_tie"]:
            counters["tie_involved"] += 1
        elif left["winner"] == right["winner"]:
            counters["consistent"] += 1
        else:
            counters["inconsistent"] += 1

    decisive_pairs = counters["consistent"] + counters["inconsistent"]
    return {
        **counters,
        "consistency_rate": round(counters["consistent"] / decisive_pairs, 6) if decisive_pairs else 0.0,
    }


def compute_verbosity_bias(records: List[Record]) -> Dict[str, Any]:
    decisive = [record for record in records if not record["is_invalid"] and not record["is_tie"]]
    if not decisive:
        return {
            "n": 0,
            "mean_winner_length": 0.0,
            "mean_loser_length": 0.0,
            "pearson_r": 0.0,
            "p_value": 1.0,
            "winner_longer_rate": 0.0,
        }

    winner_lengths: List[float] = []
    loser_lengths: List[float] = []
    paired_lengths: List[float] = []
    selected: List[float] = []
    winner_longer = 0

    for record in decisive:
        winner_length = record["response_length_a"] if record["winner"] == record["system_a"] else record["response_length_b"]
        loser_length = record["response_length_b"] if record["winner"] == record["system_a"] else record["response_length_a"]
        winner_lengths.append(float(winner_length))
        loser_lengths.append(float(loser_length))
        paired_lengths.extend([float(winner_length), float(loser_length)])
        selected.extend([1.0, 0.0])
        if winner_length > loser_length:
            winner_longer += 1

    if len(set(paired_lengths)) <= 1:
        pearson_r, p_value = 0.0, 1.0
    else:
        pearson_r, p_value = sp_stats.pearsonr(paired_lengths, selected)

    return {
        "n": len(decisive),
        "mean_winner_length": round(float(np.mean(winner_lengths)), 2),
        "mean_loser_length": round(float(np.mean(loser_lengths)), 2),
        "pearson_r": round(float(pearson_r), 6),
        "p_value": round(float(p_value), 6),
        "winner_longer_rate": round(winner_longer / len(decisive), 6),
    }


def compute_judge_agreement(records: List[Record]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, str]] = defaultdict(dict)
    for record in records:
        grouped[
            (
                record["case_id"],
                record["criterion_id"],
                record["order"],
                record["canonical_pair_key"],
            )
        ][record["judge_id"]] = _agreement_label(record)

    judge_ids = sorted({record["judge_id"] for record in records})
    pairwise = {}
    agreement_total = 0
    comparison_total = 0
    for left, right in combinations(judge_ids, 2):
        left_values: List[str] = []
        right_values: List[str] = []
        for bucket in grouped.values():
            if left in bucket and right in bucket:
                left_values.append(bucket[left])
                right_values.append(bucket[right])
        if not left_values:
            continue
        agreement_count = sum(a == b for a, b in zip(left_values, right_values))
        pairwise[f"{left}__vs__{right}"] = {
            "kappa": round(_cohens_kappa(left_values, right_values), 6),
            "agreement_rate": round(agreement_count / len(left_values), 6),
            "n": len(left_values),
        }
        agreement_total += agreement_count
        comparison_total += len(left_values)

    return {
        "pairwise_kappa": pairwise,
        "overall_agreement_rate": round(agreement_total / comparison_total, 6) if comparison_total else 0.0,
    }


def compute_execution_summary(records: List[Record], *, run_spec) -> Dict[str, Any]:
    if run_spec.config.run_mode != "stacked":
        return {
            "mode": run_spec.config.run_mode,
            "routine_two_judge_results": {"count": 0},
            "escalated_four_judge_results": {"count": 0},
            "persistent_disagreement_cases": [],
            "uncertain_case_count": 0,
            "escalation_summary": {},
            "primary_audit_agreement": {"n": 0, "agreement_rate": 0.0},
            "all_judge_agreement": {"n": 0, "agreement_rate": 0.0},
            "comparison_rows": [],
        }

    grouped: Dict[str, List[Record]] = defaultdict(list)
    for record in records:
        grouped[record["comparison_key"]].append(record)

    primary_id = run_spec.judge_manifest.primary_judge().judge_id
    audit_id = run_spec.judge_manifest.audit_judge().judge_id
    escalation_ids = [judge.judge_id for judge in run_spec.judge_manifest.escalation_judges()]
    escalation_counter: Counter[str] = Counter()
    comparison_rows: List[Dict[str, Any]] = []
    persistent_cases: List[Dict[str, Any]] = []
    routine_count = 0
    escalated_count = 0
    resolved_escalated = 0
    uncertain_count = 0
    primary_audit_agree = 0
    primary_audit_total = 0
    all_judge_agree = 0
    all_judge_total = 0

    for comparison_key, bucket in sorted(grouped.items()):
        per_judge_records: Dict[str, List[Record]] = defaultdict(list)
        for record in bucket:
            per_judge_records[record["judge_id"]].append(record)
        judge_summaries = {
            judge_id: summarise_judge_orders(judge_bucket)
            for judge_id, judge_bucket in per_judge_records.items()
        }
        stage = "escalated" if any(record.get("judge_stage") == "escalated" for record in bucket) else "routine"
        outcome = bucket[0].get("comparison_outcome", "unknown")
        reasons = sorted({reason for record in bucket for reason in record.get("escalation_reason", [])})
        for reason in reasons:
            escalation_counter[reason] += 1
        high_risk_forced = any(record.get("high_risk_forced") for record in bucket)
        example = bucket[0]
        judge_winners = {
            judge_id: summary.get("winner") or summary.get("status")
            for judge_id, summary in judge_summaries.items()
        }

        primary_summary = judge_summaries.get(primary_id)
        audit_summary = judge_summaries.get(audit_id)
        if primary_summary and audit_summary:
            if primary_summary["status"] == "decisive" and audit_summary["status"] == "decisive":
                primary_audit_total += 1
                if primary_summary["winner"] == audit_summary["winner"]:
                    primary_audit_agree += 1

        if stage == "routine":
            routine_count += 1
        else:
            escalated_count += 1
            if all(judge_id in judge_summaries for judge_id in [primary_id, audit_id, *escalation_ids]):
                all_judge_total += 1
                decisive = [
                    judge_summaries[judge_id]
                    for judge_id in [primary_id, audit_id, *escalation_ids]
                ]
                if all(summary["status"] == "decisive" for summary in decisive):
                    winners = {summary["winner"] for summary in decisive}
                    if len(winners) == 1:
                        all_judge_agree += 1
            if outcome == "resolved_escalated":
                resolved_escalated += 1
            if outcome == "uncertain":
                uncertain_count += 1
                persistent_cases.append(
                    {
                        "comparison_key": comparison_key,
                        "case_id": example["case_id"],
                        "criterion_id": example["criterion_id"],
                        "canonical_pair_key": example["canonical_pair_key"],
                        "escalation_reason": reasons,
                        "high_risk_forced": high_risk_forced,
                        "judge_winners": judge_winners,
                    }
                )

        comparison_rows.append(
            {
                "comparison_key": comparison_key,
                "case_id": example["case_id"],
                "criterion_id": example["criterion_id"],
                "canonical_pair_key": example["canonical_pair_key"],
                "judge_stage": stage,
                "comparison_outcome": outcome,
                "high_risk_forced": high_risk_forced,
                "escalation_reason": reasons,
                "judge_count": len(judge_summaries),
            }
        )

    return {
        "mode": "stacked",
        "routine_two_judge_results": {
            "count": routine_count,
        },
        "escalated_four_judge_results": {
            "count": escalated_count,
            "resolved_count": resolved_escalated,
            "uncertain_count": uncertain_count,
        },
        "persistent_disagreement_cases": persistent_cases,
        "uncertain_case_count": uncertain_count,
        "escalation_summary": dict(sorted(escalation_counter.items())),
        "primary_audit_agreement": {
            "n": primary_audit_total,
            "agreement_rate": round(primary_audit_agree / primary_audit_total, 6) if primary_audit_total else 0.0,
        },
        "all_judge_agreement": {
            "n": all_judge_total,
            "agreement_rate": round(all_judge_agree / all_judge_total, 6) if all_judge_total else 0.0,
        },
        "comparison_rows": comparison_rows,
    }


def case_stratified_bootstrap(
    records: List[Record],
    statistic,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    if not records:
        return 0.0, 0.0
    case_groups: Dict[str, List[Record]] = defaultdict(list)
    for record in records:
        case_groups[record["case_id"]].append(record)
    case_ids = sorted(case_groups)
    rng = np.random.RandomState(seed)
    estimates = np.zeros(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
        sample: List[Record] = []
        for case_id in sample_ids:
            sample.extend(case_groups[case_id])
        estimates[index] = statistic(sample)
    alpha = 1 - confidence
    return (
        round(float(np.percentile(estimates, 100 * alpha / 2)), 6),
        round(float(np.percentile(estimates, 100 * (1 - alpha / 2))), 6),
    )


def _win_rate_for(records: Sequence[Record], *, winner: str) -> float:
    decisive = [record for record in records if not record["is_invalid"] and not record["is_tie"]]
    if not decisive:
        return 0.0
    wins = sum(1 for record in decisive if record["winner"] == winner)
    return wins / len(decisive)


def _loser_for(record: Record) -> str:
    return record["system_b"] if record["winner"] == record["system_a"] else record["system_a"]


def _sorted_pair(record: Record) -> Tuple[str, str]:
    return tuple(sorted((record["system_a"], record["system_b"])))


def _agreement_label(record: Record) -> str:
    if record["is_invalid"]:
        return "INVALID"
    if record["is_tie"]:
        return "TIE"
    return record["winner"]


def _cohens_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    labels = sorted(set(left) | set(right))
    label_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for label_left, label_right in zip(left, right):
        matrix[label_index[label_left], label_index[label_right]] += 1
    total = matrix.sum()
    if total == 0:
        return 0.0
    observed = float(np.trace(matrix) / total)
    expected = float((matrix.sum(axis=1) * matrix.sum(axis=0)).sum() / (total * total))
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return float((observed - expected) / (1 - expected))
