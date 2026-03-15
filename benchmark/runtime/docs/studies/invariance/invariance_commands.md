# Invariance Study Generation Commands

> Commands for building sampled invariance roots and running the dedicated
> invariance generation targets.

---

## Profiles

### Profile A: frozen `v5` variants

- Build source root: `data/frozen_splits/v5`
- Prepared variant root: `data/invariance_variants/v5`
- Recommended output dir: `results_invariance_v5`

### Profile B: controllability-backed invariance

- Source root: `data/controllability_splits_large_resolved`
- Sampled root: `data/invariance_variants/controllability/base`
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

If you are running the prepared variant pack instead, use
`data/invariance_variants/v5`. The invariance runner now resolves the default
child automatically for each study.

### Controllability-backed invariance sample

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/studies/v5_review/build_invariance_splits.py \
  --sample-profile controllability \
  --data-root data/controllability_splits_large_resolved \
  --output-root data/invariance_variants/controllability/base
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

## Generation commands

Set the study-specific root and output directory first.

### `v5` variant pack

Set the bundle root once. The invariance runner now resolves the right child
folder automatically for each study.

```bash
cd benchmark/runtime
export INVARIANCE_VARIANTS_ROOT=data/invariance_variants/v5
export INVARIANCE_DATA_DIR="$INVARIANCE_VARIANTS_ROOT"
export INVARIANCE_RESULTS_DIR=results_invariance_v5
```

```powershell
cd benchmark/runtime
$env:INVARIANCE_VARIANTS_ROOT = 'data/invariance_variants/v5'
$env:INVARIANCE_DATA_DIR = $env:INVARIANCE_VARIANTS_ROOT
$env:INVARIANCE_RESULTS_DIR = 'results_invariance_v5'
```

Default child mapping for that root:

- Study A invariance: `study_a/lexical`
- Study B invariance: `study_b/mild`
- Study B multi-turn invariance: `study_b_multi_turn/schedule_earlier`
- Study C invariance: `study_c/noncritical_reorder`

### Controllability-backed invariance

```bash
cd benchmark/runtime
export INVARIANCE_DATA_DIR=data/invariance_variants/controllability/base
export INVARIANCE_RESULTS_DIR=results_invariance_controllability
```

```powershell
cd benchmark/runtime
$env:INVARIANCE_DATA_DIR = 'data/invariance_variants/controllability/base'
$env:INVARIANCE_RESULTS_DIR = 'results_invariance_controllability'
```

### Study A invariance

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --workers 2
```

### Study A bias invariance

```powershell
$env:BIAS_INVARIANCE_DATA_PATH = 'data/frozen_splits/v5/adversarial_bias/biased_vignettes.json'
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --data-path $env:BIAS_INVARIANCE_DATA_PATH --output-dir $env:INVARIANCE_RESULTS_DIR --workers 2
```

### Study B invariance

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --workers 2
```

### Study B multi-turn invariance

```powershell
python scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --workers 2
```

### Study C invariance

```powershell
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --workers 2
```

The same pattern works for other supported model IDs; set `$env:INVARIANCE_DATA_DIR` to the top-level bundle root and `$env:INVARIANCE_RESULTS_DIR` to the output root.

To switch those same commands to the controllability-backed invariance sample,
replace the env block with the controllability-backed one above and keep the
same study commands.

Per-study command notes:

- `docs/studies/invariance/study_a/study_a_invariance_commands.md`
- `docs/studies/invariance/study_a/study_a_bias_invariance_commands.md`
- `docs/studies/invariance/study_b/study_b_invariance_commands.md`
- `docs/studies/invariance/study_b/study_b_multi_turn_invariance_commands.md`
- `docs/studies/invariance/study_c/study_c_invariance_commands.md`

---

## Direct runner usage

```powershell
python hf-local-scripts/run_invariance_generate_only.py --study study_a_invariance --model-id gpt_oss_lmstudio --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --max-samples 5 --workers 2
```

The same runner also supports:

- `--study study_b_invariance`
- `--study study_b_multi_turn_invariance`
- `--study study_c_invariance`

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
- Cache filenames stay the same across profiles, so keep the output dirs
  separate if you want to compare `v5` and controllability-backed runs side by
  side.
- The `controllability` profile is meant to reuse the invariance machinery on
  the control-conditioned suite, not to replace the main frozen `v5`
  invariance diagnostic.
