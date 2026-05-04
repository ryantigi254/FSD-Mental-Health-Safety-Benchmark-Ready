import json
from pathlib import Path

import pytest

from reliable_clinical_benchmark.pairwise.aggregator import PairwiseAggregator
from reliable_clinical_benchmark.pairwise.config import load_pairwise_run_spec
from reliable_clinical_benchmark.pairwise.report import PairwiseReportBuilder
from reliable_clinical_benchmark.pairwise.runner import PairwiseRunner


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _winner_marker_for(prompt: str, winner_text: str, loser_text: str) -> str:
    if f"### Response A\n{winner_text}" in prompt:
        return "[[A]]"
    if f"### Response B\n{winner_text}" in prompt:
        return "[[B]]"
    raise AssertionError(f"Could not locate expected winner text in prompt. Loser text={loser_text!r}")


@pytest.mark.integration
def test_pairwise_smoke_run_writes_stacked_outputs(tmp_path: Path, monkeypatch):
    case_manifest = {
        "manifest_version": "pairwise.cases.v1",
        "slice_id": "study_a",
        "layer": "core",
        "status": "ready",
        "systems": ["model_a", "model_b"],
        "boundary_notes": [],
        "source_paths": [],
        "cases": [
            {
                "case_id": "routine_case",
                "context": "Routine case context.",
                "tags": [],
                "responses": [
                    {"system_id": "model_a", "response_id": "routine::model_a", "text": "routine_model_a", "token_count": 8, "word_count": 2},
                    {"system_id": "model_b", "response_id": "routine::model_b", "text": "routine_model_b", "token_count": 8, "word_count": 2},
                ],
                "pairings": [{"system_a": "model_a", "system_b": "model_b"}],
                "case_meta": {},
            },
            {
                "case_id": "disagreement_case",
                "context": "Disagreement case context.",
                "tags": [],
                "responses": [
                    {"system_id": "model_a", "response_id": "disagreement::model_a", "text": "disagreement_model_a", "token_count": 8, "word_count": 2},
                    {"system_id": "model_b", "response_id": "disagreement::model_b", "text": "disagreement_model_b", "token_count": 8, "word_count": 2},
                ],
                "pairings": [{"system_a": "model_a", "system_b": "model_b"}],
                "case_meta": {},
            },
            {
                "case_id": "high_risk_case",
                "context": "High-risk case context.",
                "tags": ["high_risk"],
                "responses": [
                    {"system_id": "model_a", "response_id": "highrisk::model_a", "text": "highrisk_model_a", "token_count": 8, "word_count": 2},
                    {"system_id": "model_b", "response_id": "highrisk::model_b", "text": "highrisk_model_b", "token_count": 8, "word_count": 2},
                ],
                "pairings": [{"system_a": "model_a", "system_b": "model_b"}],
                "case_meta": {},
            },
            {
                "case_id": "persistent_case",
                "context": "Persistent disagreement case context.",
                "tags": [],
                "responses": [
                    {"system_id": "model_a", "response_id": "persistent::model_a", "text": "persistent_model_a", "token_count": 8, "word_count": 2},
                    {"system_id": "model_b", "response_id": "persistent::model_b", "text": "persistent_model_b", "token_count": 8, "word_count": 2},
                ],
                "pairings": [{"system_a": "model_a", "system_b": "model_b"}],
                "case_meta": {},
            },
        ],
    }
    judge_manifest = {
        "manifest_version": "pairwise.judges.v2",
        "judges": [
            {
                "judge_id": "primary_judge",
                "display_name": "Primary Judge",
                "hf_source": "Jackrong/Qwopus3.5-27B-v3.5-GGUF",
                "local_model_id": "Jackrong/Qwopus3.5-27B-v3.5-GGUF",
                "role": "primary",
                "generation_params": {"temperature": 0.1, "max_tokens": 64, "top_p": 0.9},
            },
            {
                "judge_id": "audit_judge",
                "display_name": "Audit Judge",
                "hf_source": "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill",
                "local_model_id": "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill",
                "role": "audit",
                "generation_params": {"temperature": 0.1, "max_tokens": 64, "top_p": 0.9},
            },
            {
                "judge_id": "escalation_one",
                "display_name": "Escalation One",
                "hf_source": "TeichAI/Qwen3-14B-GPT-5.2-High-Reasoning-Distill-GGUF",
                "local_model_id": "TeichAI/Qwen3-14B-GPT-5.2-High-Reasoning-Distill-GGUF",
                "role": "escalation",
                "escalation_rank": 1,
                "generation_params": {"temperature": 0.1, "max_tokens": 64, "top_p": 0.9},
            },
            {
                "judge_id": "escalation_two",
                "display_name": "Escalation Two",
                "hf_source": "google/gemma-4-31B-it",
                "local_model_id": "google/gemma-4-31B-it",
                "role": "escalation",
                "escalation_rank": 2,
                "generation_params": {"temperature": 0.1, "max_tokens": 64, "top_p": 0.9},
            },
        ],
    }

    case_manifest_path = _write_json(tmp_path / "study_a_case_manifest.json", case_manifest)
    judge_manifest_path = _write_json(tmp_path / "judge_panel.v2.json", judge_manifest)
    config_path = _write_json(
        tmp_path / "study_a.run_config.json",
        {
            "run_id": "pairwise_study_a_v2",
            "layer": "core",
            "slice_id": "study_a",
            "case_manifest_path": str(case_manifest_path),
            "judge_manifest_path": str(judge_manifest_path),
            "rubric_family": "core_communication",
            "run_mode": "stacked",
            "orders": ["AB", "BA"],
            "output_root": str(tmp_path / "metric-results" / "pairwise"),
        },
    )

    def fake_call(_self, *, model_string: str, temperature: float, max_tokens: int, top_p: float, system_prompt: str, prompt: str) -> str:
        del temperature, max_tokens, top_p, system_prompt
        if "Routine case context." in prompt:
            return f"Clear winner.\n{_winner_marker_for(prompt, 'routine_model_a', 'routine_model_b')}"
        if "Disagreement case context." in prompt:
            winner_text = "disagreement_model_a"
            if model_string == "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill":
                winner_text = "disagreement_model_b"
            return f"Clear winner.\n{_winner_marker_for(prompt, winner_text, 'disagreement_model_b' if winner_text.endswith('_a') else 'disagreement_model_a')}"
        if "High-risk case context." in prompt:
            return f"Escalated but unanimous.\n{_winner_marker_for(prompt, 'highrisk_model_a', 'highrisk_model_b')}"
        if "Persistent disagreement case context." in prompt:
            winner_text = "persistent_model_a"
            if model_string in {
                "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill",
                "google/gemma-4-31B-it",
            }:
                winner_text = "persistent_model_b"
            return f"Still split.\n{_winner_marker_for(prompt, winner_text, 'persistent_model_b' if winner_text.endswith('_a') else 'persistent_model_a')}"
        raise AssertionError("Unexpected prompt")

    monkeypatch.setattr(PairwiseRunner, "_call_judge", fake_call)

    run_spec = load_pairwise_run_spec(config_path)
    runner = PairwiseRunner(run_spec=run_spec, case_manifest=case_manifest)
    raw_records, parsed_records = runner.run_all()

    assert len(raw_records) == (1 * 6 * 2 * 2) + (3 * 6 * 2 * 4)
    assert len(parsed_records) == len(raw_records)
    assert {record["order"] for record in parsed_records} == {"AB", "BA"}
    assert {"routine", "escalated"} == {record["judge_stage"] for record in parsed_records}

    aggregate_blob = PairwiseAggregator(run_spec=run_spec, case_manifest=case_manifest).aggregate(parsed_records)
    written = PairwiseReportBuilder(run_spec=run_spec).build(aggregate_blob)

    assert written["aggregate_path"].exists()
    assert written["report_path"].exists()
    assert written["win_rates_csv_path"].exists()
    assert written["execution_summary_csv_path"].exists()
    assert written["persistent_disagreement_csv_path"].exists()
    assert not aggregate_blob["pooled_complete"]
    assert aggregate_blob["pooled"]["status"] == "incomplete"
    assert aggregate_blob["execution_summary"]["routine_two_judge_results"]["count"] == 6
    assert aggregate_blob["execution_summary"]["escalated_four_judge_results"]["count"] == 18
    assert aggregate_blob["execution_summary"]["escalated_four_judge_results"]["resolved_count"] == 6
    assert aggregate_blob["execution_summary"]["uncertain_case_count"] == 12
