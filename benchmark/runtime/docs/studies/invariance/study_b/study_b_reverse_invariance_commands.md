# Study B — Reverse Invariance Commands

> **Direction:** Controllability → Invariance.
> Measures which perturbation families change Study B controllability metrics most.

## Variant Families

| Tag | Type | Data Root |
|-----|------|-----------|
| `paraphrase` | paraphrase | `data/invariance/ctrl/variants/v2_1/study_b/paraphrase` |
| `mild` | intensity | `data/invariance/ctrl/variants/v2_1/study_b/mild` |
| `moderate` | intensity | `data/invariance/ctrl/variants/v2_1/study_b/moderate` |
| `strong` | intensity | `data/invariance/ctrl/variants/v2_1/study_b/strong` |
| `question` | framing | `data/invariance/ctrl/variants/v2_1/study_b/question` |
| `cultural` | framing | `data/invariance/ctrl/variants/v2_1/study_b/cultural` |

## Comparison (All Variants)

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_generations.jsonl \
  --variant-cache paraphrase=results_ctrl_invariance/<MODEL>/study_b/paraphrase/study_b_generations.jsonl \
  --variant-cache mild=results_ctrl_invariance/<MODEL>/study_b/mild/study_b_generations.jsonl \
  --variant-cache moderate=results_ctrl_invariance/<MODEL>/study_b/moderate/study_b_generations.jsonl \
  --variant-cache strong=results_ctrl_invariance/<MODEL>/study_b/strong/study_b_generations.jsonl \
  --variant-cache question=results_ctrl_invariance/<MODEL>/study_b/question/study_b_generations.jsonl \
  --variant-cache cultural=results_ctrl_invariance/<MODEL>/study_b/cultural/study_b_generations.jsonl \
  --variant-type paraphrase=paraphrase \
  --variant-type mild=intensity \
  --variant-type moderate=intensity \
  --variant-type strong=intensity \
  --variant-type question=framing \
  --variant-type cultural=framing \
  --out metric-results/<MODEL>/study_b_reverse_invariance.json
```

## Per-Variant (Individual)

```bash
# Paraphrase
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_generations.jsonl \
  --variant-cache paraphrase=results_ctrl_invariance/<MODEL>/study_b/paraphrase/study_b_generations.jsonl \
  --variant-type paraphrase=paraphrase \
  --out metric-results/<MODEL>/study_b_reverse_invariance_paraphrase.json

# Mild
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_generations.jsonl \
  --variant-cache mild=results_ctrl_invariance/<MODEL>/study_b/mild/study_b_generations.jsonl \
  --variant-type mild=intensity \
  --out metric-results/<MODEL>/study_b_reverse_invariance_mild.json

# Moderate
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_generations.jsonl \
  --variant-cache moderate=results_ctrl_invariance/<MODEL>/study_b/moderate/study_b_generations.jsonl \
  --variant-type moderate=intensity \
  --out metric-results/<MODEL>/study_b_reverse_invariance_moderate.json

# Strong
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_generations.jsonl \
  --variant-cache strong=results_ctrl_invariance/<MODEL>/study_b/strong/study_b_generations.jsonl \
  --variant-type strong=intensity \
  --out metric-results/<MODEL>/study_b_reverse_invariance_strong.json

# Question
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_generations.jsonl \
  --variant-cache question=results_ctrl_invariance/<MODEL>/study_b/question/study_b_generations.jsonl \
  --variant-type question=framing \
  --out metric-results/<MODEL>/study_b_reverse_invariance_question.json

# Cultural
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_generations.jsonl \
  --variant-cache cultural=results_ctrl_invariance/<MODEL>/study_b/cultural/study_b_generations.jsonl \
  --variant-type cultural=framing \
  --out metric-results/<MODEL>/study_b_reverse_invariance_cultural.json
```

## Intensity Dose-Response

Run mild → moderate → strong together to check for monotonic degradation:

```bash
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_generations.jsonl \
  --variant-cache mild=results_ctrl_invariance/<MODEL>/study_b/mild/study_b_generations.jsonl \
  --variant-cache moderate=results_ctrl_invariance/<MODEL>/study_b/moderate/study_b_generations.jsonl \
  --variant-cache strong=results_ctrl_invariance/<MODEL>/study_b/strong/study_b_generations.jsonl \
  --variant-type mild=intensity \
  --variant-type moderate=intensity \
  --variant-type strong=intensity \
  --variant-intensity mild=1 \
  --variant-intensity moderate=2 \
  --variant-intensity strong=3 \
  --out metric-results/<MODEL>/study_b_reverse_invariance_dose_response.json
```

## Qwen 3.5 27B Distilled (mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2)

LM Studio model via Apple Silicon MLX. Recommended max 4 workers on 48 GB.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_invariance_generate_only.py \
  --study study_b_invariance \
  --model-id qwen3.5-distilled \
  --data-dir data/invariance/v5/base/v2_1 \
  --output-dir results_invariance_reverse \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id qwen3.5-distilled --env mh-llm-benchmark-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance_reverse --workers 4
```
