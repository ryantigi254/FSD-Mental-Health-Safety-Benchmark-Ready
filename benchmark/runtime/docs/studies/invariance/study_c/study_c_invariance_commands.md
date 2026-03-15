# Study C Invariance Commands

## Scope

Study C invariance generation writes to:

- `results.../<model>/study_c_invariance_generations.jsonl`

## Data Root Overrides

Use these overrides so the same commands can target either invariance root.

### `v5` variant root

```powershell
cd benchmark/runtime
$env:INVARIANCE_DATA_DIR = 'data/frozen_splits/v5_invariance_variants'
$env:INVARIANCE_RESULTS_DIR = 'results_invariance_v5'
```

The runner auto-selects the Study C child variant from that bundle root.

### Controllability-backed sampled root

```powershell
cd benchmark/runtime
$env:INVARIANCE_DATA_DIR = 'data/controllability_splits_large_resolved_invariance_samples'
$env:INVARIANCE_RESULTS_DIR = 'results_invariance_controllability'
```

## Generation (automatic runner)

```powershell
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --workers 2
```

Other models (same pattern; adjust `--env` and `--workers` as needed):

```powershell
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id qwen3_lmstudio --env mh-llm-benchmark-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --workers 6
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id qwq --env mh-llm-benchmark-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --workers 6
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --workers 4
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id piaget_local --env mh-llm-local-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --quantization 4bit
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR
```

## Variant matrix

Build the Study C family menu from the same sampled root:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/invariance/build_variant_family_matrix.py \
  --base-root data/controllability_splits_large_resolved_invariance_samples \
  --output-root data/invariance_variants/controllability \
  --study study_c
```

Available variants:

- `summary_short`
- `summary_long`
- `patient_turn_rephrase`
- `noncritical_reorder`

Run one family without resampling:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_c_invariance \
  --model-id qwq \
  --data-dir data/invariance_variants/controllability/study_c/patient_turn_rephrase \
  --output-dir results_invariance_controllability \
  --variant-tag patient_turn_rephrase
```

## Direct runner

```powershell
python hf-local-scripts/run_invariance_generate_only.py --study study_c_invariance --model-id gpt_oss_lmstudio --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --max-cases 3 --workers 2
```

## Workers

`study_c_invariance` supports `--workers`. If omitted, default is auto (e.g. 4 for LM Studio, 1 for vLLM/local).

## Useful checks

```powershell
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --check-only
python hf-local-scripts/run_invariance_generate_only.py --study study_c_invariance --model-id gpt_oss_lmstudio --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --workers 2 --max-cases 3
```
