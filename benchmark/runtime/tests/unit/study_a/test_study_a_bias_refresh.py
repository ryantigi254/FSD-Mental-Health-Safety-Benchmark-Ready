from __future__ import annotations

from pathlib import Path

from reliable_clinical_benchmark.study_a_bias_refresh import (
    find_stale_case_ids,
    replace_rows_by_id,
    write_jsonl_rows_atomic,
)


def _case(
    case_id: str,
    prompt: str,
    bias_feature: str = "homeless",
    bias_label: str = "lifestyle related",
    source_openr1_id: int = 6559,
    source_openr1_split: str = "train",
) -> dict:
    return {
        "id": case_id,
        "prompt": prompt,
        "bias_feature": bias_feature,
        "bias_label": bias_label,
        "metadata": {
            "dimension": "substance_use",
            "persona_id": "agnes",
            "source_openr1_id": source_openr1_id,
            "source_openr1_split": source_openr1_split,
        },
    }


def _row(case_id: str, prompt: str, output_text: str = "old") -> dict:
    return {
        "id": case_id,
        "prompt": prompt,
        "bias_feature": "homeless",
        "bias_label": "lifestyle related",
        "metadata": {
            "dimension": "substance_use",
            "persona_id": "agnes",
            "source_openr1_id": 6559,
            "source_openr1_split": "train",
        },
        "output_text": output_text,
        "status": "ok",
        "model_name": "qwen3-lmstudio",
    }


def test_find_stale_case_ids_uses_source_linkage_not_prompt_text() -> None:
    case_map = {
        "abias_0001": _case("abias_0001", "Fresh v5 prompt one.", source_openr1_id=7001),
        "abias_0002": _case("abias_0002", "Stable v5 prompt two."),
        "abias_0003": _case("abias_0003", "Stable v5 prompt three.", source_openr1_split="test"),
    }
    existing_rows = [
        _row("abias_0001", "Old prompt one without <END>."),
        _row("abias_0002", "Different prompt text but same source link."),
        _row("abias_0003", "Prompt text unchanged."),
    ]
    existing_rows[2]["metadata"]["source_openr1_split"] = "train"

    stale_ids = find_stale_case_ids(existing_rows, case_map)

    assert stale_ids == ["abias_0001", "abias_0003"]


def test_replace_rows_by_id_preserves_order_and_rewrites_file(tmp_path: Path) -> None:
    original_rows = [
        _row("abias_0001", "old prompt 1", output_text="old-1"),
        _row("abias_0002", "old prompt 2", output_text="old-2"),
        _row("abias_0003", "old prompt 3", output_text="old-3"),
    ]
    replacements = {
        "abias_0002": {
            **_row("abias_0002", "new prompt 2", output_text="new-2"),
            "timestamp": "2026-03-10T00:00:00.000000Z",
        }
    }

    updated_rows, replaced_count = replace_rows_by_id(original_rows, replacements)

    assert replaced_count == 1
    assert [row["id"] for row in updated_rows] == ["abias_0001", "abias_0002", "abias_0003"]
    assert updated_rows[1]["prompt"] == "new prompt 2"
    assert updated_rows[1]["output_text"] == "new-2"
    assert updated_rows[0]["output_text"] == "old-1"
    assert updated_rows[2]["output_text"] == "old-3"

    out_path = tmp_path / "study_a_bias_generations.jsonl"
    write_jsonl_rows_atomic(out_path, updated_rows)
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert '"id": "abias_0002"' in lines[1]
    assert '"output_text": "new-2"' in lines[1]
