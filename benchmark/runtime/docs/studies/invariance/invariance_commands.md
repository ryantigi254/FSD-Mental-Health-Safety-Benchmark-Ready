# Invariance Study — Generation Commands

> Commands for building sampled invariance roots and running the dedicated
> invariance generation targets.

---

## Profiles

### Profile A: frozen `v5`

- Source root: `data/frozen_splits/v5`
- Sampled root: `data/frozen_splits/v5_invariance_samples`
- Recommended output dir: `results_invariance_v5`

### Profile B: controllability-backed invariance

- Source root: `data/controllability_splits_large_resolved`
- Sampled root: `data/controllability_splits_large_resolved_invariance_samples`
- Recommended output dir: `results_invariance_controllability`

The controllability-backed sample uses a slightly smaller default budget than
the main `v5` invariance profile:

- Study A: `140`
- Study B: `150`
- Study B multi-turn: `10`
- Study C: `12`

---

## Build the sampled root

### Main `v5` invariance sample

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/studies/v5_review/build_invariance_splits.py \
  --sample-profile v5 \
  --data-root data/frozen_splits/v5 \
  --output-root data/frozen_splits/v5_invariance_samples
```

### Controllability-backed invariance sample

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/studies/v5_review/build_invariance_splits.py \
  --sample-profile controllability \
  --data-root data/controllability_splits_large_resolved \
  --output-root data/controllability_splits_large_resolved_invariance_samples
```

### Single-study manifest only

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/studies/v5_review/generate_invariance_manifest.py \
  --study study_b \
  --sample-profile controllability \
  --data-root data/controllability_splits_large_resolved
```

---

## Build variant roots from the fixed sample

Apply variant families to the same sampled root rather than resampling per
family.

### Build the full controllability-backed variant matrix

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/invariance/build_variant_family_matrix.py \
  --base-root data/controllability_splits_large_resolved_invariance_samples \
  --output-root data/invariance_variants/controllability
```

### Build a selected study or variant only

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/invariance/build_variant_family_matrix.py \
  --base-root data/controllability_splits_large_resolved_invariance_samples \
  --output-root data/invariance_variants/controllability \
  --study study_c \
  --variant patient_turn_rephrase
```

Concrete variant menu:

- Study A: `lexical`, `surface`, `syntax`, `instruction`
- Study B single-turn: `paraphrase`, `mild`, `moderate`, `strong`, `question`, `cultural`
- Study B multi-turn: `schedule_earlier`, `schedule_later`, `tone_gentle`, `tone_direct`, `tone_confrontational`, `pressure_milder`, `pressure_stronger`
- Study C: `summary_short`, `summary_long`, `patient_turn_rephrase`, `noncritical_reorder`

---

## Generation commands

Set the sampled root and output directory first.

### `v5`

```bash
cd benchmark/runtime
export INVARIANCE_DATA_DIR=data/frozen_splits/v5_invariance_samples
export INVARIANCE_RESULTS_DIR=results_invariance_v5
```

### Controllability-backed invariance

```bash
cd benchmark/runtime
export INVARIANCE_DATA_DIR=data/controllability_splits_large_resolved_invariance_samples
export INVARIANCE_RESULTS_DIR=results_invariance_controllability
```

### Study A invariance

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_a_invariance \
  --model-id qwq \
  --data-dir "$INVARIANCE_DATA_DIR" \
  --output-dir "$INVARIANCE_RESULTS_DIR"
```

### Study B invariance

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_b_invariance \
  --model-id qwq \
  --data-dir "$INVARIANCE_DATA_DIR" \
  --output-dir "$INVARIANCE_RESULTS_DIR"
```

### Study B multi-turn invariance

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_b_multi_turn_invariance \
  --model-id qwq \
  --data-dir "$INVARIANCE_DATA_DIR" \
  --output-dir "$INVARIANCE_RESULTS_DIR"
```

### Study C invariance

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_c_invariance \
  --model-id qwq \
  --data-dir "$INVARIANCE_DATA_DIR" \
  --output-dir "$INVARIANCE_RESULTS_DIR"
```

The same pattern works for the other supported model IDs.

### Variant-root runs

Point `--data-dir` at a specific variant root and pass `--variant-tag` so the
default cache filename stays unique:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_b_invariance \
  --model-id qwq \
  --data-dir data/invariance_variants/controllability/study_b/cultural \
  --output-dir results_invariance_controllability \
  --variant-tag cultural
```

That produces a cache like:

- `results_invariance_controllability/<model>/study_b_invariance_cultural_generations.jsonl`

Per-study command notes:

- `docs/studies/invariance/study_a/study_a_invariance_commands.md`
- `docs/studies/invariance/study_b/study_b_invariance_commands.md`
- `docs/studies/invariance/study_b/study_b_multi_turn_invariance_commands.md`
- `docs/studies/invariance/study_c/study_c_invariance_commands.md`

---

## Direct runner usage

```bash
cd benchmark/runtime
PYTHONPATH=src python hf-local-scripts/run_study_a_invariance_generate_only.py \
  --model-id qwq \
  --data-dir "$INVARIANCE_DATA_DIR" \
  --output-dir "$INVARIANCE_RESULTS_DIR" \
  --max-samples 5
```

The other runners follow the same shape:

- `hf-local-scripts/run_study_b_invariance_generate_only.py`
- `hf-local-scripts/run_study_b_multi_turn_invariance_generate_only.py`
- `hf-local-scripts/run_study_c_invariance_generate_only.py`

---

## Paired comparison commands

After generation, compare a base cache with a variant cache:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_invariance_comparison.py \
  --study study_b \
  --data-root "$INVARIANCE_DATA_DIR" \
  --base-cache "$INVARIANCE_RESULTS_DIR/qwq/study_b_generations.jsonl" \
  --variant-cache "$INVARIANCE_RESULTS_DIR/qwq/study_b_invariance_generations.jsonl" \
  --variant-name controllability_sample \
  --out metric-results/qwq/study_b_invariance_from_controllability.json
```

Export per-case deltas if needed:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/export_invariance_case_deltas.py \
  --study study_b \
  --data-root "$INVARIANCE_DATA_DIR" \
  --base-cache "$INVARIANCE_RESULTS_DIR/qwq/study_b_generations.jsonl" \
  --variant-cache "$INVARIANCE_RESULTS_DIR/qwq/study_b_invariance_generations.jsonl" \
  --out metric-results/qwq/study_b_invariance_case_deltas.jsonl
```

---

## Notes

- The materialised sampled roots always use the canonical invariance layout,
  even when the source root is controllability-specific.
- Variant roots keep that same layout and only change the prompt or turn
  surface for the selected family.
- Cache filenames stay the same across profiles, so keep the output dirs
  separate if you want to compare `v5` and controllability-backed runs side by
  side. Use `--variant-tag` when multiple families share one output dir.
- The `controllability` profile is meant to reuse the invariance machinery on
  the control-conditioned suite, not to replace the main frozen `v5`
  invariance diagnostic.
