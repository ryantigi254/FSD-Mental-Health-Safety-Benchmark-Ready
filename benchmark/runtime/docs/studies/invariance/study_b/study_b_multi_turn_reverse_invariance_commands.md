# Study B Multi-Turn - Reverse Invariance Commands

> **Direction:** Controllability -> Invariance.
> Measures which perturbation families change Study B Multi-Turn controllability
> metrics most.

Use `--output-dir results_ctrl_invariance` for all generation commands below.

Generation caches for each reverse study should be saved under `results_ctrl_invariance/<MODEL>/...`, mirroring the per-model layout used in `results` and `results_invariance`.

---

## Ctrl Base Runs

### LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 2
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id qwen3_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id piaget_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 4
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id qwq --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id deepseek_r1_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psyllm_gml_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id piaget_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psyche_r1_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psych_qwen_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psyllm_gml_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id piaget_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psyche_r1_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psych_qwen_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
```

---

## Ctrl Variant-Family Runs

Use `data/invariance/ctrl/variants/v2_1/study_b_multi_turn` to run every
Study B multi-turn ctrl variant-family child in one go.

### LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id gpt_oss --ctrl-dir data/invariance/ctrl/variants/v2_1/study_b_multi_turn --output-dir results_ctrl_invariance --workers 2
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id qwen3_lmstudio --ctrl-dir data/invariance/ctrl/variants/v2_1/study_b_multi_turn --output-dir results_ctrl_invariance --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id piaget_lmstudio --ctrl-dir data/invariance/ctrl/variants/v2_1/study_b_multi_turn --output-dir results_ctrl_invariance --workers 4
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id qwq --ctrl-dir data/invariance/ctrl/variants/v2_1/study_b_multi_turn --output-dir results_ctrl_invariance --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id deepseek_r1_lmstudio --ctrl-dir data/invariance/ctrl/variants/v2_1/study_b_multi_turn --output-dir results_ctrl_invariance --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psyllm_gml_local --ctrl-dir data/invariance/ctrl/variants/v2_1/study_b_multi_turn --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id piaget_local --ctrl-dir data/invariance/ctrl/variants/v2_1/study_b_multi_turn --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psyche_r1_local --ctrl-dir data/invariance/ctrl/variants/v2_1/study_b_multi_turn --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psych_qwen_local --ctrl-dir data/invariance/ctrl/variants/v2_1/study_b_multi_turn --output-dir results_ctrl_invariance --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psyllm_gml_vllm --ctrl-dir data/invariance/ctrl/variants/v2_1/study_b_multi_turn --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id piaget_vllm --ctrl-dir data/invariance/ctrl/variants/v2_1/study_b_multi_turn --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psyche_r1_vllm --ctrl-dir data/invariance/ctrl/variants/v2_1/study_b_multi_turn --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psych_qwen_vllm --ctrl-dir data/invariance/ctrl/variants/v2_1/study_b_multi_turn --output-dir results_ctrl_invariance
```

---

## Direct Runner

```powershell
python hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_b_multi_turn --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --max-cases 3 --workers 2
```

## Checks

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --check-only
python hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_b_multi_turn --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 2 --max-cases 3
```

---

## Variant Families

| Tag | Type | Data Root |
|-----|------|-----------|
| `schedule_earlier` | schedule | `data/invariance/ctrl/variants/v2_1/study_b_multi_turn/schedule_earlier` |
| `schedule_later` | schedule | `data/invariance/ctrl/variants/v2_1/study_b_multi_turn/schedule_later` |
| `tone_gentle` | tone | `data/invariance/ctrl/variants/v2_1/study_b_multi_turn/tone_gentle` |
| `tone_direct` | tone | `data/invariance/ctrl/variants/v2_1/study_b_multi_turn/tone_direct` |
| `tone_confrontational` | tone | `data/invariance/ctrl/variants/v2_1/study_b_multi_turn/tone_confrontational` |
| `pressure_milder` | pressure | `data/invariance/ctrl/variants/v2_1/study_b_multi_turn/pressure_milder` |
| `pressure_stronger` | pressure | `data/invariance/ctrl/variants/v2_1/study_b_multi_turn/pressure_stronger` |

## Comparison (All Variants)

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache schedule_earlier=results_ctrl_invariance/<MODEL>/study_b_multi_turn/schedule_earlier/study_b_multi_turn_generations.jsonl \
  --variant-cache schedule_later=results_ctrl_invariance/<MODEL>/study_b_multi_turn/schedule_later/study_b_multi_turn_generations.jsonl \
  --variant-cache tone_gentle=results_ctrl_invariance/<MODEL>/study_b_multi_turn/tone_gentle/study_b_multi_turn_generations.jsonl \
  --variant-cache tone_direct=results_ctrl_invariance/<MODEL>/study_b_multi_turn/tone_direct/study_b_multi_turn_generations.jsonl \
  --variant-cache tone_confrontational=results_ctrl_invariance/<MODEL>/study_b_multi_turn/tone_confrontational/study_b_multi_turn_generations.jsonl \
  --variant-cache pressure_milder=results_ctrl_invariance/<MODEL>/study_b_multi_turn/pressure_milder/study_b_multi_turn_generations.jsonl \
  --variant-cache pressure_stronger=results_ctrl_invariance/<MODEL>/study_b_multi_turn/pressure_stronger/study_b_multi_turn_generations.jsonl \
  --variant-type schedule_earlier=schedule \
  --variant-type schedule_later=schedule \
  --variant-type tone_gentle=tone \
  --variant-type tone_direct=tone \
  --variant-type tone_confrontational=tone \
  --variant-type pressure_milder=pressure \
  --variant-type pressure_stronger=pressure \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance.json
```

## Per-Variant (Individual)

```bash
# Schedule Earlier
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache schedule_earlier=results_ctrl_invariance/<MODEL>/study_b_multi_turn/schedule_earlier/study_b_multi_turn_generations.jsonl \
  --variant-type schedule_earlier=schedule \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_schedule_earlier.json

# Schedule Later
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache schedule_later=results_ctrl_invariance/<MODEL>/study_b_multi_turn/schedule_later/study_b_multi_turn_generations.jsonl \
  --variant-type schedule_later=schedule \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_schedule_later.json

# Tone Gentle
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache tone_gentle=results_ctrl_invariance/<MODEL>/study_b_multi_turn/tone_gentle/study_b_multi_turn_generations.jsonl \
  --variant-type tone_gentle=tone \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_tone_gentle.json

# Tone Direct
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache tone_direct=results_ctrl_invariance/<MODEL>/study_b_multi_turn/tone_direct/study_b_multi_turn_generations.jsonl \
  --variant-type tone_direct=tone \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_tone_direct.json

# Tone Confrontational
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache tone_confrontational=results_ctrl_invariance/<MODEL>/study_b_multi_turn/tone_confrontational/study_b_multi_turn_generations.jsonl \
  --variant-type tone_confrontational=tone \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_tone_confrontational.json

# Pressure Milder
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache pressure_milder=results_ctrl_invariance/<MODEL>/study_b_multi_turn/pressure_milder/study_b_multi_turn_generations.jsonl \
  --variant-type pressure_milder=pressure \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_pressure_milder.json

# Pressure Stronger
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache pressure_stronger=results_ctrl_invariance/<MODEL>/study_b_multi_turn/pressure_stronger/study_b_multi_turn_generations.jsonl \
  --variant-type pressure_stronger=pressure \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_pressure_stronger.json
```

## Tone Gradient

Run gentle ? direct ? confrontational together:

```bash
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache tone_gentle=results_ctrl_invariance/<MODEL>/study_b_multi_turn/tone_gentle/study_b_multi_turn_generations.jsonl \
  --variant-cache tone_direct=results_ctrl_invariance/<MODEL>/study_b_multi_turn/tone_direct/study_b_multi_turn_generations.jsonl \
  --variant-cache tone_confrontational=results_ctrl_invariance/<MODEL>/study_b_multi_turn/tone_confrontational/study_b_multi_turn_generations.jsonl \
  --variant-type tone_gentle=tone \
  --variant-type tone_direct=tone \
  --variant-type tone_confrontational=tone \
  --variant-intensity tone_gentle=1 \
  --variant-intensity tone_direct=2 \
  --variant-intensity tone_confrontational=3 \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_tone_gradient.json
```

## GPT-OSS-20B

LM Studio model. Recommended max 2 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_b_multi_turn \
  --model-id gpt_oss \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_ctrl_invariance \
  --workers 2
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 2
```

## Qwen3-8B

LM Studio model. Recommended max 6 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_b_multi_turn \
  --model-id qwen3_lmstudio \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_ctrl_invariance \
  --workers 6
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id qwen3_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 6
```

## QwQ-32B

LM Studio model. Recommended max 6 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_b_multi_turn \
  --model-id qwq \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_ctrl_invariance \
  --workers 6
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id qwq --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 6
```

## DeepSeek-R1 Distill Qwen 14B

LM Studio model. Recommended max 4 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_b_multi_turn \
  --model-id deepseek_r1_lmstudio \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_ctrl_invariance \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id deepseek_r1_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 4
```

## Qwen 3.5 27B Distilled (mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2)

LM Studio model via Apple Silicon MLX. Recommended max 4 workers on 48 GB.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_b_multi_turn \
  --model-id "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0" \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_ctrl_invariance \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0" --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 8
python scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance --model-id "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0" --env mh-llm-benchmark-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance_reverse --workers 8
```

## MedGemma 27B Text (`google.medgemma-27b-text-it`)

In LM Studio, load the model as `google.medgemma-27b-text-it`. The benchmark runner uses `--model-id medgemma_lmstudio` (aliases include that identifier). Recommended max 4 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_b_multi_turn \
  --model-id medgemma_lmstudio \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_ctrl_invariance \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id medgemma_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 4
```



