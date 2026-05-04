from reliable_clinical_benchmark.pairwise.parser import PairwiseParser


def test_parser_handles_a_b_tie_and_invalid():
    parser = PairwiseParser()

    parsed_a = parser.parse(
        case_id="case_1",
        layer="core",
        slice_id="study_a",
        criterion_id="clarity",
        judge_id="judge_1",
        order="AB",
        system_a="model_a",
        system_b="model_b",
        response_length_a=10,
        response_length_b=12,
        raw_response="Short reason.\n[[A]]",
    )
    assert parsed_a.winner == "model_a"
    assert not parsed_a.is_invalid
    assert not parsed_a.is_tie

    parsed_ba = parser.parse(
        case_id="case_1",
        layer="core",
        slice_id="study_a",
        criterion_id="clarity",
        judge_id="judge_1",
        order="BA",
        system_a="model_a",
        system_b="model_b",
        response_length_a=10,
        response_length_b=12,
        raw_response="Short reason.\n[[A]]",
    )
    assert parsed_ba.winner == "model_b"

    parsed_tie = parser.parse(
        case_id="case_1",
        layer="core",
        slice_id="study_a",
        criterion_id="clarity",
        judge_id="judge_1",
        order="AB",
        system_a="model_a",
        system_b="model_b",
        response_length_a=10,
        response_length_b=12,
        raw_response="Roughly equal.\n[[TIE]]",
    )
    assert parsed_tie.is_tie
    assert parsed_tie.winner == "TIE"

    parsed_invalid = parser.parse(
        case_id="case_1",
        layer="core",
        slice_id="study_a",
        criterion_id="clarity",
        judge_id="judge_1",
        order="AB",
        system_a="model_a",
        system_b="model_b",
        response_length_a=10,
        response_length_b=12,
        raw_response="I refuse to choose.",
    )
    assert parsed_invalid.is_invalid
    assert parsed_invalid.winner == "INVALID"
    assert parsed_invalid.canonical_pair_key == "model_a__vs__model_b"


def test_parser_uses_terminal_verdict_line_not_in_text_markers():
    parser = PairwiseParser()

    parsed = parser.parse(
        case_id="case_1",
        layer="core",
        slice_id="study_a",
        criterion_id="clarity",
        judge_id="judge_1",
        order="AB",
        system_a="model_a",
        system_b="model_b",
        response_length_a=10,
        response_length_b=12,
        raw_response=(
            "Response B quotes the instruction token [[A]], but is still worse.\n"
            "The actual choice is based on the final judgement only.\n"
            "Verdict: [[B]]"
        ),
    )

    assert parsed.verdict == "B"
    assert parsed.winner == "model_b"
    assert "[[A]]" in parsed.reasoning_excerpt


def test_parser_rejects_non_terminal_verdict_markers():
    parser = PairwiseParser()

    parsed = parser.parse(
        case_id="case_1",
        layer="core",
        slice_id="study_a",
        criterion_id="clarity",
        judge_id="judge_1",
        order="AB",
        system_a="model_a",
        system_b="model_b",
        response_length_a=10,
        response_length_b=12,
        raw_response="Verdict: [[A]]\nAdditional explanation after the verdict.",
    )

    assert parsed.is_invalid
