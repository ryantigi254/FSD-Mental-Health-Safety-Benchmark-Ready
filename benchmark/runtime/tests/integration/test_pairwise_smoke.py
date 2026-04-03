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


@pytest.mark.integration
def test_pairwise_smoke_run_writes_outputs(tmp_path: Path, monkeypatch):
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
                "case_id": "case_1",
                "context": "User asks for support.",
                "tags": [],
                "responses": [
                    {"system_id": "model_a", "response_id": "case_1::model_a", "text": "A short validating reply.", "token_count": 8, "word_count": 4},
                    {"system_id": "model_b", "response_id": "case_1::model_b", "text": "A longer grounding reply with one suggestion.", "token_count": 12, "word_count": 7},
                ],
                "pairings": [{"system_a": "model_a", "system_b": "model_b"}],
                "case_meta": {},
            }
        ],
    }
    judge_manifest = {
        "manifest_version": "pairwise.judges.v1",
        "judges": [
            {
                "judge_id": f"judge_{index}",
                "display_name": f"Judge {index}",
                "hf_source": f"org/model-{index}",
                "local_model_id": f"local-model-{index}",
                "generation_params": {"temperature": 0.1, "max_tokens": 64, "top_p": 0.9},
            }
            for index in range(4)
        ],
    }

    case_manifest_path = _write_json(tmp_path / "study_a_case_manifest.json", case_manifest)
    judge_manifest_path = _write_json(tmp_path / "judge_panel.v1.json", judge_manifest)
    config_path = _write_json(
        tmp_path / "study_a.run_config.json",
        {
            "run_id": "pairwise_study_a_v1",
            "layer": "core",
            "slice_id": "study_a",
            "case_manifest_path": str(case_manifest_path),
            "judge_manifest_path": str(judge_manifest_path),
            "rubric_family": "core_communication",
            "orders": ["AB", "BA"],
            "output_root": str(tmp_path / "metric-results" / "pairwise"),
        },
    )

    def fake_chat_completion(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        return "More concise.\n[[A]]" if "Response A" in prompt else "More concise.\n[[B]]"

    monkeypatch.setattr(
        "reliable_clinical_benchmark.pairwise.runner.chat_completion",
        fake_chat_completion,
    )

    run_spec = load_pairwise_run_spec(config_path)
    runner = PairwiseRunner(run_spec=run_spec, case_manifest=case_manifest)
    raw_records, parsed_records = runner.run_all()

    assert len(raw_records) == 4 * 6 * 2
    assert len(parsed_records) == 4 * 6 * 2
    assert {record["order"] for record in parsed_records} == {"AB", "BA"}

    aggregate_blob = PairwiseAggregator(run_spec=run_spec, case_manifest=case_manifest).aggregate(parsed_records)
    written = PairwiseReportBuilder(run_spec=run_spec).build(aggregate_blob)

    assert written["aggregate_path"].exists()
    assert written["report_path"].exists()
    assert written["win_rates_csv_path"].exists()
    assert aggregate_blob["pooled_complete"]
