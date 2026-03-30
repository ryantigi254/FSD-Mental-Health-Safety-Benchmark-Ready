# Study A — Reverse Invariance Commands

> **Direction:** Controllability → Invariance.
> Measures which perturbation families change Study A controllability metrics most.

Use `--output-dir results_ctrl_invariance` for all generation commands below.

---

## Ctrl Base Runs

### LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 2
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id qwen3_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id qwq --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id deepseek_r1_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyllm_gml_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id piaget_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyche_r1_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psych_qwen_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyllm_gml_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id piaget_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyche_r1_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psych_qwen_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
```

---

## Ctrl Variant-Family Runs

Use `data/invariance/ctrl/variants/v2_1/study_a` to run every Study A
ctrl variant-family child in one go.

### LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id gpt_oss --ctrl-dir data/invariance/ctrl/variants/v2_1/study_a --output-dir results_ctrl_invariance --workers 2
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id qwen3_lmstudio --ctrl-dir data/invariance/ctrl/variants/v2_1/study_a --output-dir results_ctrl_invariance --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id qwq --ctrl-dir data/invariance/ctrl/variants/v2_1/study_a --output-dir results_ctrl_invariance --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id deepseek_r1_lmstudio --ctrl-dir data/invariance/ctrl/variants/v2_1/study_a --output-dir results_ctrl_invariance --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyllm_gml_local --ctrl-dir data/invariance/ctrl/variants/v2_1/study_a --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id piaget_local --ctrl-dir data/invariance/ctrl/variants/v2_1/study_a --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyche_r1_local --ctrl-dir data/invariance/ctrl/variants/v2_1/study_a --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psych_qwen_local --ctrl-dir data/invariance/ctrl/variants/v2_1/study_a --output-dir results_ctrl_invariance --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyllm_gml_vllm --ctrl-dir data/invariance/ctrl/variants/v2_1/study_a --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id piaget_vllm --ctrl-dir data/invariance/ctrl/variants/v2_1/study_a --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psyche_r1_vllm --ctrl-dir data/invariance/ctrl/variants/v2_1/study_a --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id psych_qwen_vllm --ctrl-dir data/invariance/ctrl/variants/v2_1/study_a --output-dir results_ctrl_invariance
```

---

## Direct Runner

```powershell
python hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_a --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --max-cases 5 --workers 2
```

## Checks

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --check-only
python hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_a --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 2 --max-cases 5
```

---

## Variant Families

| Tag | Type | Data Root |
|-----|------|-----------|
| `lexical` | paraphrase | `data/invariance/ctrl/variants/v2_1/study_a/lexical` |
| `surface` | paraphrase | `data/invariance/ctrl/variants/v2_1/study_a/surface` |
| `syntax` | paraphrase | `data/invariance/ctrl/variants/v2_1/study_a/syntax` |
| `instruction` | paraphrase | `data/invariance/ctrl/variants/v2_1/study_a/instruction` |

## Comparison

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_a \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_a_generations.jsonl \
  --variant-cache lexical=results_ctrl_invariance/<MODEL>/study_a/lexical/study_a_generations.jsonl \
  --variant-cache surface=results_ctrl_invariance/<MODEL>/study_a/surface/study_a_generations.jsonl \
  --variant-cache syntax=results_ctrl_invariance/<MODEL>/study_a/syntax/study_a_generations.jsonl \
  --variant-cache instruction=results_ctrl_invariance/<MODEL>/study_a/instruction/study_a_generations.jsonl \
  --variant-type lexical=paraphrase \
  --variant-type surface=paraphrase \
  --variant-type syntax=paraphrase \
  --variant-type instruction=paraphrase \
  --out metric-results/<MODEL>/study_a_reverse_invariance.json
```

## Per-Variant (Individual)

Run each variant family separately to isolate its effect:

```bash
# Lexical
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_a \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_a_generations.jsonl \
  --variant-cache lexical=results_ctrl_invariance/<MODEL>/study_a/lexical/study_a_generations.jsonl \
  --variant-type lexical=paraphrase \
  --out metric-results/<MODEL>/study_a_reverse_invariance_lexical.json

# Surface
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_a \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_a_generations.jsonl \
  --variant-cache surface=results_ctrl_invariance/<MODEL>/study_a/surface/study_a_generations.jsonl \
  --variant-type surface=paraphrase \
  --out metric-results/<MODEL>/study_a_reverse_invariance_surface.json

# Syntax
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_a \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_a_generations.jsonl \
  --variant-cache syntax=results_ctrl_invariance/<MODEL>/study_a/syntax/study_a_generations.jsonl \
  --variant-type syntax=paraphrase \
  --out metric-results/<MODEL>/study_a_reverse_invariance_syntax.json

# Instruction
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_a \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_a_generations.jsonl \
  --variant-cache instruction=results_ctrl_invariance/<MODEL>/study_a/instruction/study_a_generations.jsonl \
  --variant-type instruction=paraphrase \
  --out metric-results/<MODEL>/study_a_reverse_invariance_instruction.json
```

## GPT-OSS-20B

LM Studio model. Recommended max 2 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_a \
  --model-id gpt_oss \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_ctrl_invariance \
  --workers 2
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 2
```

## Qwen3-8B

LM Studio model. Recommended max 6 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_a \
  --model-id qwen3_lmstudio \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_ctrl_invariance \
  --workers 6
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id qwen3_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 6
```

## QwQ-32B

LM Studio model. Recommended max 6 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_a \
  --model-id qwq \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_ctrl_invariance \
  --workers 6
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id qwq --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 6
```

## DeepSeek-R1 Distill Qwen 14B

LM Studio model. Recommended max 4 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_a \
  --model-id deepseek_r1_lmstudio \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_ctrl_invariance \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id deepseek_r1_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 4
```

## Qwen 3.5 27B Distilled (mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2)

LM Studio model via Apple Silicon MLX. Recommended max 4 workers on 48 GB.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_a \
  --model-id "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0" \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_ctrl_invariance \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0" --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 8
```

## MedGemma 27B Text (`google.medgemma-27b-text-it`)

In LM Studio, load the model as `google.medgemma-27b-text-it`. The benchmark runner uses `--model-id medgemma_lmstudio` (aliases include that identifier). Recommended max 4 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_a \
  --model-id medgemma_lmstudio \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_ctrl_invariance \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a --model-id medgemma_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 4
```
