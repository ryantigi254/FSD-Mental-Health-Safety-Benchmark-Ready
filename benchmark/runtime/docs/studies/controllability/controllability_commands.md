# Controllability Study — Generation Commands

> Commands for running controlled-CoT generation across all studies.
> Uses `mode='cot_controlled'` with study-specific constraint injection.

---

## Prerequisites

1. Controllability test sets built: `data/controllability_splits/` (see `docs/controllability_scaling/CONTROLLABILITY_SCALING.md`)
2. Gold labels generated: `data/controllability_splits/ctrl_gold_diagnosis_labels.json`
3. Gold plans generated: `data/controllability_splits/ctrl_target_plans.json`

### Gold generation (one-time, requires NLI model)

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/studies/controllability/generate_gold_labels.py
PYTHONPATH=src python scripts/studies/controllability/generate_gold_plans.py
```

---

## Generation Commands

Legacy controllability generation uses a single unified script:

```
hf-local-scripts/run_ctrl_generate_only.py --study <STUDY> --model-id <MODEL>
```

Arm-aware `v2` generation uses:

```
hf-local-scripts/run_ctrl_v2_generate_only.py --study <V2_STUDY> --model-id <MODEL>
```

### Via `run_generation_auto.py` (recommended)

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
    --study ctrl_study_a --model-id qwq
```

For the same-case three-arm `v2` path:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
    --study ctrl_v2_study_a --model-id qwq
```

---

## Study A Controllability (Reasoning Adherence)

Generates paired `cot_controlled` + `direct` responses for Δ_Reasoning comparison.

```bash
# LM Studio models
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id qwq
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id qwen3_lmstudio
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id deepseek_r1_lmstudio
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id gpt_oss

# vLLM models
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyllm_gml_vllm
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id piaget_vllm
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyche_r1_vllm
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psych_qwen_vllm

# Local HF models
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyllm_gml_local
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id piaget_local
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyche_r1_local
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psych_qwen_local
```

**Output**: `results/<model>/ctrl_study_a_generations.jsonl`

---

## Study A Bias Controllability (Bias-Aware RA)

Generates `cot_controlled` responses with explicit bias acknowledgement constraint.

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id qwq
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id qwen3_lmstudio
# ... (same model set as ctrl_study_a)
```

**Output**: `results/<model>/ctrl_study_a_bias_generations.jsonl`

---

## Study B Single-Turn Controllability (CHR)

Generates `control` + `injected` pairs under `cot_controlled` mode.

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id qwq
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id qwen3_lmstudio
# ... (same model set)
```

**Output**: `results/<model>/ctrl_study_b_generations.jsonl`

---

## Study B Multi-Turn Controllability (Pressure Resistance)

Generates 20-turn rolling-context conversations under `cot_controlled` mode.

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id qwq
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id qwen3_lmstudio
# ... (same model set)
```

**Output**: `results/<model>/ctrl_study_b_multi_turn_generations.jsonl`

---

## Study C Controllability (Controlled Entity Recall)

Generates summary + dialogue pairs per turn under `cot_controlled` mode.

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id qwq
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id qwen3_lmstudio
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
    --max-cases 5 \
    --max-tokens 8192
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--study` | One of: `ctrl_study_a`, `ctrl_study_a_bias`, `ctrl_study_b`, `ctrl_study_b_multi_turn`, `ctrl_study_c` |
| `--model-id` | Model identifier (same as base studies) |
| `--max-cases` | Limit number of cases (for piloting) |
| `--max-tokens` | Max tokens per generation (default: 8192) |
| `--output-dir` | Override output directory (default: `results/`) |
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

## Controllability V2 Commands

Use these when you want matched-arm controllability comparisons on the same case IDs.

### Generation

```bash
# Study A
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_v2_study_a --model-id qwq

# Study A Bias
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_v2_study_a_bias --model-id qwq

# Study B single-turn
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_v2_study_b --model-id qwq

# Study B multi-turn
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_v2_study_b_multi_turn --model-id qwq

# Study C
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_v2_study_c --model-id qwq
```

### Evaluation

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_controllability_v2_pipeline.py \
    --model <results_model_dir>
```

### V2 Outputs

- `results/<model>/ctrl_v2_study_a_generations.jsonl`
- `results/<model>/ctrl_v2_study_a_bias_generations.jsonl`
- `results/<model>/ctrl_v2_study_b_generations.jsonl`
- `results/<model>/ctrl_v2_study_b_multi_turn_generations.jsonl`
- `results/<model>/ctrl_v2_study_c_generations.jsonl`
- `results/<model>/ctrl_v2_study_a_results.json`
- `results/<model>/ctrl_v2_study_a_bias_results.json`
- `results/<model>/ctrl_v2_study_b_results.json`
- `results/<model>/ctrl_v2_study_b_multi_turn_results.json`
- `results/<model>/ctrl_v2_study_c_results.json`
- `results/<model>/controllability_v2_summary.json`

### Notes

- `ctrl_v2_study_a_bias` uses the canonical adversarial bias cases for all three arms.
- `ctrl_v2_study_b_multi_turn` injects the control text once at conversation start.
- `ctrl_v2_study_c` applies the arm to summaries only.
