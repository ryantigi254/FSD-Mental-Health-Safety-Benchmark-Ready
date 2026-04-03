from reliable_clinical_benchmark.pairwise.statistics import (
    case_stratified_bootstrap,
    compute_bradley_terry,
    compute_swap_consistency,
    compute_verbosity_bias,
    compute_win_rates,
)


def _record(
    *,
    case_id: str,
    judge_id: str,
    order: str,
    winner: str,
    system_a: str = "model_a",
    system_b: str = "model_b",
    criterion_id: str = "clarity",
    is_tie: bool = False,
    is_invalid: bool = False,
    len_a: int = 10,
    len_b: int = 5,
):
    return {
        "case_id": case_id,
        "layer": "core",
        "slice_id": "study_a",
        "criterion_id": criterion_id,
        "judge_id": judge_id,
        "order": order,
        "system_a": system_a,
        "system_b": system_b,
        "winner": winner,
        "is_tie": is_tie,
        "is_invalid": is_invalid,
        "response_length_a": len_a,
        "response_length_b": len_b,
        "canonical_pair_key": "__vs__".join(sorted([system_a, system_b])),
    }


def test_compute_win_rates_and_bradley_terry():
    records = [
        _record(case_id="c1", judge_id="j1", order="AB", winner="model_a"),
        _record(case_id="c1", judge_id="j1", order="BA", winner="model_a"),
        _record(case_id="c2", judge_id="j1", order="AB", winner="model_b"),
        _record(case_id="c2", judge_id="j1", order="BA", winner="model_b"),
    ]

    rows = compute_win_rates(records)
    assert len(rows) == 1
    assert rows[0]["wins_a"] == 2
    assert rows[0]["wins_b"] == 2

    bt_rows = compute_bradley_terry(records)
    assert {row["system_id"] for row in bt_rows} == {"model_a", "model_b"}


def test_compute_swap_consistency_and_verbosity_bias():
    records = [
        _record(case_id="c1", judge_id="j1", order="AB", winner="model_a", len_a=12, len_b=4),
        _record(case_id="c1", judge_id="j1", order="BA", winner="model_a", len_a=12, len_b=4),
        _record(case_id="c2", judge_id="j1", order="AB", winner="model_a", len_a=7, len_b=8),
        _record(case_id="c2", judge_id="j1", order="BA", winner="model_b", len_a=7, len_b=8),
    ]

    swap = compute_swap_consistency(records)
    assert swap["consistent"] == 1
    assert swap["inconsistent"] == 1

    verbosity = compute_verbosity_bias(records)
    assert verbosity["n"] == 4
    assert 0.0 <= verbosity["winner_longer_rate"] <= 1.0


def test_case_stratified_bootstrap_returns_interval():
    records = [
        _record(case_id="c1", judge_id="j1", order="AB", winner="model_a"),
        _record(case_id="c2", judge_id="j1", order="AB", winner="model_a"),
        _record(case_id="c3", judge_id="j1", order="AB", winner="model_b"),
        _record(case_id="c4", judge_id="j1", order="AB", winner="model_b"),
    ]

    interval = case_stratified_bootstrap(
        records,
        lambda sample: sum(1 for row in sample if row["winner"] == "model_a") / len(sample),
        n_bootstrap=50,
    )
    assert interval[0] <= interval[1]
