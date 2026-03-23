# Study C — Reverse Invariance Commands

> **Direction:** Controllability → Invariance.
> Measures which perturbation families change Study C controllability metrics most.

## Variant Families

| Tag | Type | Data Root |
|-----|------|-----------|
| `summary_short` | summary | `data/invariance/ctrl/variants_v2_1/study_c/summary_short` |
| `summary_long` | summary | `data/invariance/ctrl/variants_v2_1/study_c/summary_long` |
| `patient_turn_rephrase` | paraphrase | `data/invariance/ctrl/variants_v2_1/study_c/patient_turn_rephrase` |
| `noncritical_reorder` | reorder | `data/invariance/ctrl/variants_v2_1/study_c/noncritical_reorder` |

## Comparison (All Variants)

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_c \
  --data-root data/invariance/ctrl/base_v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_c_generations.jsonl \
  --variant-cache summary_short=results_ctrl_invariance/<MODEL>/study_c/summary_short/study_c_generations.jsonl \
  --variant-cache summary_long=results_ctrl_invariance/<MODEL>/study_c/summary_long/study_c_generations.jsonl \
  --variant-cache patient_turn_rephrase=results_ctrl_invariance/<MODEL>/study_c/patient_turn_rephrase/study_c_generations.jsonl \
  --variant-cache noncritical_reorder=results_ctrl_invariance/<MODEL>/study_c/noncritical_reorder/study_c_generations.jsonl \
  --variant-type summary_short=summary \
  --variant-type summary_long=summary \
  --variant-type patient_turn_rephrase=paraphrase \
  --variant-type noncritical_reorder=reorder \
  --out metric-results/<MODEL>/study_c_reverse_invariance.json
```

## Per-Variant (Individual)

```bash
# Summary Short
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_c \
  --data-root data/invariance/ctrl/base_v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_c_generations.jsonl \
  --variant-cache summary_short=results_ctrl_invariance/<MODEL>/study_c/summary_short/study_c_generations.jsonl \
  --variant-type summary_short=summary \
  --out metric-results/<MODEL>/study_c_reverse_invariance_summary_short.json

# Summary Long
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_c \
  --data-root data/invariance/ctrl/base_v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_c_generations.jsonl \
  --variant-cache summary_long=results_ctrl_invariance/<MODEL>/study_c/summary_long/study_c_generations.jsonl \
  --variant-type summary_long=summary \
  --out metric-results/<MODEL>/study_c_reverse_invariance_summary_long.json

# Patient Turn Rephrase
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_c \
  --data-root data/invariance/ctrl/base_v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_c_generations.jsonl \
  --variant-cache patient_turn_rephrase=results_ctrl_invariance/<MODEL>/study_c/patient_turn_rephrase/study_c_generations.jsonl \
  --variant-type patient_turn_rephrase=paraphrase \
  --out metric-results/<MODEL>/study_c_reverse_invariance_patient_turn_rephrase.json

# Noncritical Reorder
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_c \
  --data-root data/invariance/ctrl/base_v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_c_generations.jsonl \
  --variant-cache noncritical_reorder=results_ctrl_invariance/<MODEL>/study_c/noncritical_reorder/study_c_generations.jsonl \
  --variant-type noncritical_reorder=reorder \
  --out metric-results/<MODEL>/study_c_reverse_invariance_noncritical_reorder.json
```
