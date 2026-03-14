# Study A Invariance Commands

## Scope

Study A invariance generation writes to:

- `results.../<model>/study_a_invariance_generations.jsonl`

## `v5` sampled root

```bash
cd benchmark/runtime
export INVARIANCE_DATA_DIR=data/frozen_splits/v5_invariance_samples
export INVARIANCE_RESULTS_DIR=results_invariance_v5
```

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_a_invariance \
  --model-id qwq \
  --data-dir "$INVARIANCE_DATA_DIR" \
  --output-dir "$INVARIANCE_RESULTS_DIR"
```

## Variant matrix

Study A concrete variants on the same sampled root:

- `lexical`
- `surface`
- `syntax`
- `instruction`

Build them with:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/invariance/build_variant_family_matrix.py \
  --base-root "$INVARIANCE_DATA_DIR" \
  --output-root data/invariance_variants \
  --study study_a
```

Run one family without resampling:

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_a_invariance \
  --model-id qwq \
  --data-dir data/invariance_variants/study_a/lexical \
  --output-dir "$INVARIANCE_RESULTS_DIR" \
  --variant-tag lexical
```

## Controllability-backed sampled root

```bash
cd benchmark/runtime
export INVARIANCE_DATA_DIR=data/controllability_splits_large_resolved_invariance_samples
export INVARIANCE_RESULTS_DIR=results_invariance_controllability
```

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_a_invariance \
  --model-id qwq \
  --data-dir "$INVARIANCE_DATA_DIR" \
  --output-dir "$INVARIANCE_RESULTS_DIR"
```

## Direct runner

```bash
cd benchmark/runtime
PYTHONPATH=src python hf-local-scripts/run_study_a_invariance_generate_only.py \
  --model-id qwq \
  --data-dir "$INVARIANCE_DATA_DIR" \
  --output-dir "$INVARIANCE_RESULTS_DIR" \
  --max-samples 5
```
