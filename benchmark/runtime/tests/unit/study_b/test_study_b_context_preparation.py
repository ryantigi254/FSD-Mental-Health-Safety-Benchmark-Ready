from reliable_clinical_benchmark.pipelines.study_b import _prepare_response_for_context


def test_prepare_response_for_context_strips_think_blocks():
    raw = "<think>hidden reasoning</think>\nFinal guidance for the user."
    prepared = _prepare_response_for_context(raw)
    assert "hidden reasoning" not in prepared
    assert "Final guidance" in prepared


def test_prepare_response_for_context_truncates_long_context_only():
    raw = "A" * 2500
    prepared = _prepare_response_for_context(raw, max_chars=100)
    assert len(prepared) < len(raw)
    assert "[Context truncated for stability]" in prepared
