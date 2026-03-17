# Controllability Study — Generation Commands

> Commands for running controlled-CoT generation across all studies.
> Uses `mode='cot_controlled'` with study-specific constraint injection.

---

## Prerequisites

1. Small suite (existing): `data/controllability_splits/`
2. Large resolved suite (scaled): `data/controllability_splits_large/`
3. Gold labels generated in the suite you want to evaluate:
   - `data/controllability_splits_large/ctrl_gold_diagnosis_labels.json`
4. Gold plans generated in the suite you want to evaluate:
   - `data/controllability_splits_large/ctrl_target_plans.json`

For the scaled run, keep outputs separate from the small suite. The command
examples below pass the canonical paths directly:

- `--ctrl-dir data/controllability_splits_large`
- `--output-dir results_scaled_large_resolved`

### Gold generation (one-time)

The checked-in controllability gold artefacts now use the `probe` backend rather than the older NLI-first path. The current benchmark note and model-selection rationale live in:

- `docs/studies/controllability/gold_generation.md`

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/studies/controllability/generate_gold_labels.py \
  --ctrl-dir data/controllability_splits_large \
  --backend probe \
  --primary-model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext

PYTHONPATH=src python scripts/studies/controllability/generate_gold_plans.py \
  --ctrl-dir data/controllability_splits_large \
  --backend probe \
  --primary-model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
```

Robust companion artefacts:

```bash
PYTHONPATH=src python scripts/studies/controllability/generate_gold_labels.py \
  --ctrl-dir data/controllability_splits_large \
  --backend probe \
  --primary-model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --secondary-model emilyalsentzer/Bio_ClinicalBERT \
  --output-name ctrl_gold_diagnosis_labels.robust.json

PYTHONPATH=src python scripts/studies/controllability/generate_gold_plans.py \
  --ctrl-dir data/controllability_splits_large \
  --backend probe \
  --primary-model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --secondary-model michiyasunaga/BioLinkBERT-base \
  --output-name ctrl_target_plans.robust.json
```

---

## Generation Commands

Canonical controllability generation uses a single unified script:

```
hf-local-scripts/run_ctrl_generate_only.py --study <STUDY> --model-id <MODEL>
```

### Via `run_generation_auto.py` (recommended)

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
    --study ctrl_study_a \
    --model-id qwq \
    --ctrl-dir data/controllability_splits_large \
    --output-dir results_scaled_large_resolved
```

The old `ctrl_v2_study_*` names are compatibility aliases only.

---

## Study A Controllability (Reasoning Adherence)

Generates the same controllability cases under `spontaneous`,
`generic_control`, and `explicit_control`.

```bash
# LM Studio models
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id qwq --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id qwen3_lmstudio --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id deepseek_r1_lmstudio --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id gpt_oss --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved

# vLLM models
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyllm_gml_vllm --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id piaget_vllm --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyche_r1_vllm --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psych_qwen_vllm --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved

# Local HF models
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyllm_gml_local --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id piaget_local --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyche_r1_local --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psych_qwen_local --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
```

**Output**: `results/<model>/ctrl_study_a_generations.jsonl`

---

## Study A Bias Controllability (Bias-Aware RA)

Generates matched-arm bias runs on the canonical adversarial bias cases.

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id qwq --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id qwen3_lmstudio --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
# ... (same model set as ctrl_study_a)
```

**Output**: `results/<model>/ctrl_study_a_bias_generations.jsonl`

---

## Study B Single-Turn Controllability (CHR)

Generates `control` + `injected` pairs for each arm.

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id qwq --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id qwen3_lmstudio --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
# ... (same model set)
```

**Output**: `results/<model>/ctrl_study_b_generations.jsonl`

---

## Study B Multi-Turn Controllability (Pressure Resistance)

Generates 20-turn rolling-context conversations for each arm.

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id qwq --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id qwen3_lmstudio --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
# ... (same model set)
```

**Output**: `results/<model>/ctrl_study_b_multi_turn_generations.jsonl`

---

## Study C Controllability (Controlled Entity Recall)

Generates summary-only outputs per turn for each arm.

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id qwq --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id qwen3_lmstudio --ctrl-dir data/controllability_splits_large --output-dir results_scaled_large_resolved
# ... (same model set)
```

**Output**: `results/<model>/ctrl_study_c_generations.jsonl`

---

## Direct Script Usage (without auto runner)

```bash
cd benchmark/runtime
PYTHONPATH=src python hf-local-scripts/run_ctrl_generate_only.py \
    --study ctrl_study_a \
    --model-id qwq \
    --ctrl-dir data/controllability_splits_large \
    --output-dir results_scaled_large_resolved \
    --max-cases 5 \
    --max-tokens 8192
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--study` | One of: `ctrl_study_a`, `ctrl_study_a_bias`, `ctrl_study_b`, `ctrl_study_b_multi_turn`, `ctrl_study_c` |
| `--model-id` | Model identifier (same as base studies) |
| `--ctrl-dir` | Controllability split directory (use `data/controllability_splits_large` for the scaled suite) |
| `--max-cases` | Limit number of cases (for piloting) |
| `--max-tokens` | Max tokens per generation (default: 8192) |
| `--output-dir` | Override output directory (recommend `results_scaled_large_resolved/` for scaled runs) |
| `--cache-out` | Explicit cache path |

---

## Metrics Calculation

After generation, calculate controllability metrics:

```bash
# Study A: Reasoning Adherence
PYTHONPATH=src python -c "
from reliable_clinical_benchmark.metrics import calculate_reasoning_adherence
# Load traces and gold steps, then:
# result = calculate_reasoning_adherence(traces, gold_steps_per_sample)
"

# Study B: Controlled Hallucination Rate
PYTHONPATH=src python -c "
from reliable_clinical_benchmark.metrics import calculate_controlled_hallucination_rate
# Load traces, opinions, golds, then:
# result = calculate_controlled_hallucination_rate(traces, opinions, golds)
"

# Study C: Controlled Entity Recall
PYTHONPATH=src python -c "
from reliable_clinical_benchmark.metrics import calculate_controlled_entity_recall
# Load summaries and entities, then:
# result = calculate_controlled_entity_recall(summaries, entities_per_sample)
"
```

---

## Evaluation

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_controllability_pipeline.py \
    --model <results_model_dir> \
    --ctrl-dir data/controllability_splits_large \
    --results-dir results_scaled_large_resolved
```

## Canonical Outputs

- `results/<model>/ctrl_study_a_generations.jsonl`
- `results/<model>/ctrl_study_a_bias_generations.jsonl`
- `results/<model>/ctrl_study_b_generations.jsonl`
- `results/<model>/ctrl_study_b_multi_turn_generations.jsonl`
- `results/<model>/ctrl_study_c_generations.jsonl`
- `results/<model>/ctrl_study_a_results.json`
- `results/<model>/ctrl_study_a_bias_results.json`
- `results/<model>/ctrl_study_b_results.json`
- `results/<model>/ctrl_study_b_multi_turn_results.json`
- `results/<model>/ctrl_study_c_results.json`
- `results/<model>/controllability_summary.json`

Compatibility aliases are also written under the old `ctrl_v2_study_*` and
`controllability_v2_summary.json` filenames.

## Notes

- `ctrl_study_a_bias` uses the canonical adversarial bias cases for all three arms.
- `ctrl_study_b_multi_turn` injects the control text once at conversation start.
- `ctrl_study_c` applies the arm to summaries only.
