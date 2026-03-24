# Reverse Evaluation: Controllability → Invariance

> **Direction:** Which perturbation families change controllability metrics most?
>
> Uses `run_controllability_comparison.py` with variant-family generation caches
> as `--variant-cache` inputs. The ctrl suite has one control arm
> (`explicit_control`), so this measures perturbation × control interaction
> rather than arm-vs-arm comparison.

---

## Prerequisites

1. Generate base results from `data/invariance/ctrl/base/v2_1` (see `invariance_ctrl_commands.md`).
2. Generate variant results for each family under `data/invariance/ctrl/variants/v2_1/study_*/`.
3. Both base and variant caches must exist before running comparison.

---

## Study A — Reverse Invariance

Variant families: `lexical`, `surface`, `syntax`, `instruction`

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

---

## Study B — Reverse Invariance

Variant families: `paraphrase`, `mild`, `moderate`, `strong`, `question`, `cultural`

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

---

## Study B Multi-Turn — Reverse Invariance

Variant families: `schedule_earlier`, `schedule_later`, `tone_gentle`, `tone_direct`, `tone_confrontational`, `pressure_milder`, `pressure_stronger`

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

---

## Study C — Reverse Invariance

Variant families: `summary_short`, `summary_long`, `patient_turn_rephrase`, `noncritical_reorder`

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_c \
  --data-root data/invariance/ctrl/base/v2_1 \
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

---

## Per-Study Command Files

- [`study_a/study_a_reverse_invariance_commands.md`](study_a/study_a_reverse_invariance_commands.md)
- [`study_a/study_a_bias_reverse_invariance_commands.md`](study_a/study_a_bias_reverse_invariance_commands.md)
- [`study_b/study_b_reverse_invariance_commands.md`](study_b/study_b_reverse_invariance_commands.md)
- [`study_b/study_b_multi_turn_reverse_invariance_commands.md`](study_b/study_b_multi_turn_reverse_invariance_commands.md)
- [`study_c/study_c_reverse_invariance_commands.md`](study_c/study_c_reverse_invariance_commands.md)

---

## Interpreting Results

Each output JSON contains per-variant controllability deltas. Key questions:

- **Which variant families cause the largest ctrl metric shift?** — signals that
  the perturbation interacts with the control condition.
- **Do intensity-graded variants (mild → strong) show monotonic degradation?** —
  confirms a dose-response relationship between perturbation strength and ctrl
  robustness.
- **Is the shift consistent across studies?** — a variant family that destabilises
  ctrl in Study A but not Study B suggests study-specific sensitivity.

## Qwen 3.5 27B Distilled (mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2)

LM Studio model via Apple Silicon MLX. Recommended max 4 workers on 48 GB.
See per-study invariance command files for specific commands.
