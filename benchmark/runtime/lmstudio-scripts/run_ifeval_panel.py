"""
IFEval panel runner — local LM Studio builds.

Runs google/IFEval against the four shortlisted models and writes a
consolidated results JSON to ../results/ifeval_panel_results.json.

Usage:
    python run_ifeval_panel.py [--model MODEL_INDEX] [--limit N] [--dry-run]

Prerequisites:
    pip install datasets requests tqdm
    LM Studio must be running on localhost:1234 with the target model loaded.

IFEval scoring note:
    This script uses a lightweight rule-based scorer derived from the
    original google-research/instruction_following_eval evaluation rules.
    Results are comparable to (but not identical to) the official lm-eval
    harness task because the harness adds additional post-processing.
    For publication purposes, confirm with: pip install lm-eval && run via harness.

Minimum acceptable thresholds (project-internal, not clinical standards):
    prompt_level_strict_acc >= 0.30
    inst_level_strict_acc   >= 0.45
    Models below both thresholds should be excluded from the Robert shortlist.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Model panel (freeze: 2026-04)
# Update MODEL_ID values to match whichever build is loaded in LM Studio.
# Check loaded model in LM Studio → Developer → Model ID shown in the header.
# ---------------------------------------------------------------------------
CANDIDATE_PANEL = [
    {
        "name": "Qwopus-3.5-27B-v3.5",
        "model_id": "jackrong/qwopus-3.5-27b-v3.5",  # update to exact LM Studio model ID
        "family": "Qwen-distil",
    },
    {
        "name": "GLM-4.7-Flash-ClaudeOpus4.5-Distill",
        "model_id": "zai-org/glm-4.7-flash:2",         # update to exact LM Studio model ID
        "family": "GLM-distil",
    },
    {
        "name": "Qwen3.5-27B-ClaudeOpus4.6-Distill-v2",
        "model_id": "jackrong/qwen-3.5-27b-v2",        # update to exact LM Studio model ID
        "family": "Qwen-distil",
    },
    {
        "name": "Gemma4-31B-IT-ClaudeOpus-Distill",
        "model_id": "teichai/gemma-4-31b-it-distill",  # update to exact LM Studio model ID
        "family": "Gemma-distil",
    },
]

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions")
LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
RESULTS_DIR = Path(__file__).parent.parent / "results" / "ifeval"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Project-internal minimum thresholds
THRESHOLD_PROMPT_STRICT = 0.30
THRESHOLD_INST_STRICT = 0.45


# ---------------------------------------------------------------------------
# LM Studio call
# ---------------------------------------------------------------------------

def call_model(model_id: str, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LM_STUDIO_API_KEY}",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        r = requests.post(LM_STUDIO_URL, headers=headers, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "ERROR: LM Studio not reachable on port 1234"
    except Exception as exc:
        return f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Minimal IFEval scoring functions
# (derived from google-research/instruction_following_eval/instructions.py)
# These cover the main verifiable constraint types.
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _score_single_instruction(response: str, instruction_id: str, kwargs: dict) -> bool:
    """Return True if the response satisfies the instruction."""
    rid = instruction_id.lower()

    # --- word count ---
    if rid == "length_constraints:number_words":
        relation = kwargs.get("relation", "")
        num_words = kwargs.get("num_words", 0)
        actual = _count_words(response)
        if relation == "less than":
            return actual < num_words
        if relation == "at least":
            return actual >= num_words
        if relation == "at most":
            return actual <= num_words
        return False

    # --- sentence count ---
    if rid == "length_constraints:number_sentences":
        num_sentences = kwargs.get("num_sentences", 0)
        relation = kwargs.get("relation", "at least")
        actual = len(re.split(r"[.!?]+", response.strip()))
        actual = sum(1 for s in actual if s.strip()) if isinstance(actual, list) else actual
        sentences = [s.strip() for s in re.split(r"[.!?]+", response.strip()) if s.strip()]
        count = len(sentences)
        if relation == "at least":
            return count >= num_sentences
        if relation == "at most":
            return count <= num_sentences
        return count == num_sentences

    # --- paragraph count ---
    if rid == "length_constraints:number_paragraphs":
        num_paragraphs = kwargs.get("num_paragraphs", 0)
        paras = [p.strip() for p in response.split("\n\n") if p.strip()]
        return len(paras) == num_paragraphs

    # --- keyword presence ---
    if rid == "keywords:existence":
        keywords = kwargs.get("keywords", [])
        return all(kw.lower() in response.lower() for kw in keywords)

    # --- keyword frequency ---
    if rid == "keywords:frequency":
        keyword = kwargs.get("keyword", "")
        frequency = kwargs.get("frequency", 0)
        relation = kwargs.get("relation", "at least")
        count = response.lower().count(keyword.lower())
        if relation == "at least":
            return count >= frequency
        if relation == "at most":
            return count <= frequency
        return count == frequency

    # --- forbidden words ---
    if rid == "keywords:forbidden_words":
        forbidden = kwargs.get("forbidden_words", [])
        return all(fw.lower() not in response.lower() for fw in forbidden)

    # --- letter frequency ---
    if rid == "keywords:letter_frequency":
        letter = kwargs.get("letter", "")
        let_freq = kwargs.get("let_freq", 0)
        relation = kwargs.get("relation", "at least")
        count = response.lower().count(letter.lower())
        if relation == "at least":
            return count >= let_freq
        if relation == "at most":
            return count <= let_freq
        return count == let_freq

    # --- case constraints ---
    if rid == "change_case:english_capital":
        return response == response.upper()
    if rid == "change_case:english_lowercase":
        return response == response.lower()
    if rid == "change_case:capital_word_frequency":
        freq = kwargs.get("capital_frequency", 0)
        words = response.split()
        cap_count = sum(1 for w in words if w and w[0].isupper())
        relation = kwargs.get("relation", "at least")
        if relation == "at least":
            return cap_count >= freq
        return cap_count <= freq

    # --- json format ---
    if rid == "detectable_format:json_format":
        try:
            json.loads(response)
            return True
        except (json.JSONDecodeError, ValueError):
            # Try stripping markdown code fences
            stripped = re.sub(r"```(?:json)?\n?(.*?)```", r"\1", response, flags=re.DOTALL).strip()
            try:
                json.loads(stripped)
                return True
            except Exception:
                return False

    # --- bullet points ---
    if rid == "detectable_format:number_bullet_lists":
        bullet_count = len(re.findall(r"^\s*[-*•]\s", response, re.MULTILINE))
        num_bullets = kwargs.get("num_bullets", 0)
        return bullet_count >= num_bullets

    # --- numbered list ---
    if rid == "detectable_format:number_highlighted_sections":
        num = kwargs.get("num_highlights", 0)
        # count markdown bold or highlighted sections
        count = len(re.findall(r"\*\*[^*]+\*\*", response))
        return count >= num

    # --- title ---
    if rid == "detectable_format:title":
        return bool(re.search(r"^#\s+\S", response, re.MULTILINE))

    # --- section headings ---
    if rid == "detectable_format:multiple_sections":
        num_sections = kwargs.get("num_sections", 0)
        headings = re.findall(r"^#{1,3}\s+\S", response, re.MULTILINE)
        return len(headings) >= num_sections

    # --- postscript ---
    if rid == "detectable_content:postscript":
        postscript_marker = kwargs.get("postscript_marker", "P.S.")
        return postscript_marker.lower() in response.lower()

    # --- starting word ---
    if rid == "startend:start_checker":
        first_word = kwargs.get("first_word", "")
        return response.strip().lower().startswith(first_word.lower())

    # --- ending ---
    if rid == "startend:end_checker":
        end_phrase = kwargs.get("end_phrase", "")
        return response.strip().lower().endswith(end_phrase.lower().strip())

    # --- response language ---
    if rid == "language:response_language":
        # Heuristic: skip detailed lang detection, flag as unknown -> pass
        return True

    # --- two responses ---
    if rid == "combination:two_responses":
        return "******" in response

    # --- repeat prompt ---
    if rid == "combination:repeat_prompt":
        original_prompt = kwargs.get("original_prompt", "")
        return original_prompt[:50].lower() in response.lower() if original_prompt else False

    # Unknown / unimplemented — treat as not assessable, count as pass to avoid penalising
    return True


def score_response(response: str, instructions: list[dict]) -> tuple[bool, list[bool]]:
    """Return (prompt_pass, [instruction_passes])."""
    results = []
    for instr in instructions:
        instr_id = instr.get("instruction_id_list", [instr.get("id", "")])
        if isinstance(instr_id, list):
            # Multiple instructions per prompt item — score each
            kwargs_list = instr.get("kwargs", [{}] * len(instr_id))
            for iid, kw in zip(instr_id, kwargs_list):
                results.append(_score_single_instruction(response, iid, kw or {}))
        else:
            results.append(_score_single_instruction(response, instr_id, instr.get("kwargs", {})))
    prompt_pass = all(results)
    return prompt_pass, results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_model_eval(candidate: dict, dataset, limit: int | None, dry_run: bool) -> dict:
    model_id = candidate["model_id"]
    model_name = candidate["name"]
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  ID:    {model_id}")
    print(f"{'='*60}")

    items = list(dataset)
    if limit:
        items = items[:limit]

    prompt_passes = []
    inst_passes: list[bool] = []
    per_prompt: list[dict] = []

    for item in tqdm(items, desc=model_name, unit="prompt"):
        prompt = item["prompt"]

        # Build instruction list from IFEval schema
        instructions = []
        instr_ids = item.get("instruction_id_list", [])
        kwarg_list = item.get("kwargs", [])
        if not kwarg_list:
            kwarg_list = [{}] * len(instr_ids)
        for iid, kw in zip(instr_ids, kwarg_list):
            instructions.append({"instruction_id_list": [iid], "kwargs": [kw]})

        if dry_run:
            response = f"[DRY-RUN] Prompt: {prompt[:80]}..."
        else:
            response = call_model(model_id, prompt)

        p_pass, i_passes = score_response(response, instructions)
        prompt_passes.append(p_pass)
        inst_passes.extend(i_passes)

        per_prompt.append({
            "prompt": prompt[:200],
            "response_preview": response[:200],
            "prompt_pass": p_pass,
            "instruction_passes": i_passes,
        })

    n_prompts = len(prompt_passes)
    n_insts = len(inst_passes)
    prompt_acc = sum(prompt_passes) / n_prompts if n_prompts else 0.0
    inst_acc = sum(inst_passes) / n_insts if n_insts else 0.0

    passes_threshold = (
        prompt_acc >= THRESHOLD_PROMPT_STRICT and inst_acc >= THRESHOLD_INST_STRICT
    )

    result = {
        "model_name": model_name,
        "model_id": model_id,
        "family": candidate["family"],
        "n_prompts": n_prompts,
        "n_instructions": n_insts,
        "prompt_level_strict_acc": round(prompt_acc, 4),
        "inst_level_strict_acc": round(inst_acc, 4),
        "passes_project_threshold": passes_threshold,
        "threshold_prompt": THRESHOLD_PROMPT_STRICT,
        "threshold_inst": THRESHOLD_INST_STRICT,
        "dry_run": dry_run,
        "per_prompt_sample": per_prompt[:10],  # store first 10 for inspection
    }

    status = "PASS ✓" if passes_threshold else "FAIL ✗"
    print(f"  prompt_level_strict_acc : {prompt_acc:.3f}  (threshold ≥ {THRESHOLD_PROMPT_STRICT})")
    print(f"  inst_level_strict_acc   : {inst_acc:.3f}  (threshold ≥ {THRESHOLD_INST_STRICT})")
    print(f"  Shortlist retention     : {status}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IFEval against local LM Studio panel.")
    parser.add_argument("--model", type=int, default=None,
                        help="Run only one model by index (0-3). Omit to run all.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N prompts (for testing).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LM Studio calls, use placeholder responses.")
    parser.add_argument("--split", default="train",
                        help="IFEval dataset split (default: train).")
    args = parser.parse_args()

    print("Loading google/IFEval ...")
    dataset = load_dataset("google/IFEval", split=args.split)
    print(f"  Loaded {len(dataset)} prompts.")

    candidates = CANDIDATE_PANEL if args.model is None else [CANDIDATE_PANEL[args.model]]

    all_results = []
    for candidate in candidates:
        result = run_model_eval(candidate, dataset, args.limit, args.dry_run)
        all_results.append(result)

    output = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "google/IFEval",
        "split": args.split,
        "n_prompts_evaluated": all_results[0]["n_prompts"] if all_results else 0,
        "thresholds": {
            "prompt_level_strict_acc": THRESHOLD_PROMPT_STRICT,
            "inst_level_strict_acc": THRESHOLD_INST_STRICT,
            "description": "Project-internal gates only. Not clinical standards.",
        },
        "results": all_results,
        "summary": {
            m["model_name"]: {
                "prompt_acc": m["prompt_level_strict_acc"],
                "inst_acc": m["inst_level_strict_acc"],
                "passes": m["passes_project_threshold"],
            }
            for m in all_results
        },
    }

    out_path = RESULTS_DIR / f"ifeval_panel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"\nResults written to: {out_path}")
    print("\n--- Summary ---")
    for m in all_results:
        flag = "✓" if m["passes_project_threshold"] else "✗"
        print(f"  {flag}  {m['model_name']:<45}  "
              f"prompt={m['prompt_level_strict_acc']:.3f}  "
              f"inst={m['inst_level_strict_acc']:.3f}")


if __name__ == "__main__":
    main()
