# Study B Multi-Turn — Reverse Invariance Commands

> **Direction:** Controllability → Invariance.
> Measures which perturbation families change Study B Multi-Turn controllability
> metrics most.

## Variant Families

| Tag | Type | Data Root |
|-----|------|-----------|
| `schedule_earlier` | schedule | `data/invariance/ctrl_variants_v2_1/study_b_multi_turn/schedule_earlier` |
| `schedule_later` | schedule | `data/invariance/ctrl_variants_v2_1/study_b_multi_turn/schedule_later` |
| `tone_gentle` | tone | `data/invariance/ctrl_variants_v2_1/study_b_multi_turn/tone_gentle` |
| `tone_direct` | tone | `data/invariance/ctrl_variants_v2_1/study_b_multi_turn/tone_direct` |
| `tone_confrontational` | tone | `data/invariance/ctrl_variants_v2_1/study_b_multi_turn/tone_confrontational` |
| `pressure_milder` | pressure | `data/invariance/ctrl_variants_v2_1/study_b_multi_turn/pressure_milder` |
| `pressure_stronger` | pressure | `data/invariance/ctrl_variants_v2_1/study_b_multi_turn/pressure_stronger` |

## Comparison (All Variants)

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl_base_v2_1 \
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
  --data-root data/invariance/ctrl_base_v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache schedule_earlier=results_ctrl_invariance/<MODEL>/study_b_multi_turn/schedule_earlier/study_b_multi_turn_generations.jsonl \
  --variant-type schedule_earlier=schedule \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_schedule_earlier.json

# Schedule Later
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl_base_v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache schedule_later=results_ctrl_invariance/<MODEL>/study_b_multi_turn/schedule_later/study_b_multi_turn_generations.jsonl \
  --variant-type schedule_later=schedule \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_schedule_later.json

# Tone Gentle
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl_base_v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache tone_gentle=results_ctrl_invariance/<MODEL>/study_b_multi_turn/tone_gentle/study_b_multi_turn_generations.jsonl \
  --variant-type tone_gentle=tone \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_tone_gentle.json

# Tone Direct
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl_base_v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache tone_direct=results_ctrl_invariance/<MODEL>/study_b_multi_turn/tone_direct/study_b_multi_turn_generations.jsonl \
  --variant-type tone_direct=tone \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_tone_direct.json

# Tone Confrontational
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl_base_v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache tone_confrontational=results_ctrl_invariance/<MODEL>/study_b_multi_turn/tone_confrontational/study_b_multi_turn_generations.jsonl \
  --variant-type tone_confrontational=tone \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_tone_confrontational.json

# Pressure Milder
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl_base_v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache pressure_milder=results_ctrl_invariance/<MODEL>/study_b_multi_turn/pressure_milder/study_b_multi_turn_generations.jsonl \
  --variant-type pressure_milder=pressure \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_pressure_milder.json

# Pressure Stronger
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl_base_v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_b_multi_turn_generations.jsonl \
  --variant-cache pressure_stronger=results_ctrl_invariance/<MODEL>/study_b_multi_turn/pressure_stronger/study_b_multi_turn_generations.jsonl \
  --variant-type pressure_stronger=pressure \
  --out metric-results/<MODEL>/study_b_multi_turn_reverse_invariance_pressure_stronger.json
```

## Tone Gradient

Run gentle → direct → confrontational together:

```bash
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_b_multi_turn \
  --data-root data/invariance/ctrl_base_v2_1 \
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
