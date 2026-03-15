# Study B Invariance Commands

## Scope

Study B single-turn invariance generation writes to:

- `results.../<model>/study_b_invariance_generations.jsonl`

## Data Root Overrides

Use these overrides so the same commands can target either invariance root.

### `v5` variant root

```powershell
cd benchmark/runtime
$env:INVARIANCE_DATA_DIR = 'data/frozen_splits/v5_invariance_variants'
$env:INVARIANCE_RESULTS_DIR = 'results_invariance_v5'
```

The runner auto-selects the Study B child variant from that bundle root.

### Controllability-backed sampled root

```powershell
cd benchmark/runtime
$env:INVARIANCE_DATA_DIR = 'data/controllability_splits_large_resolved_invariance_samples'
$env:INVARIANCE_RESULTS_DIR = 'results_invariance_controllability'
```

## Generation (automatic runner)

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --workers 2
```

Other models (same pattern; adjust `--env` and `--workers` as needed):

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id qwen3_lmstudio --env mh-llm-benchmark-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --workers 6
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id qwq --env mh-llm-benchmark-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --workers 6
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --workers 4
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_local --env mh-llm-local-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --quantization 4bit
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR
```

## Variant matrix

Study B single-turn concrete variants on the same sampled root:

- `paraphrase`
- `mild`
- `moderate`
- `strong`
- `question`
- `cultural`

Build them with:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/invariance/build_variant_family_matrix.py \
  --base-root data/controllability_splits_large_resolved_invariance_samples \
  --output-root data/invariance_variants/controllability \
  --study study_b
```

Run one family without resampling:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_b_invariance \
  --model-id qwq \
  --data-dir data/invariance_variants/controllability/study_b/cultural \
  --output-dir results_invariance_controllability \
  --variant-tag cultural
```

## Direct runner

```powershell
python hf-local-scripts/run_invariance_generate_only.py --study study_b_invariance --model-id gpt_oss_lmstudio --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --max-samples 5 --workers 2
```

## Workers

`study_b_invariance` supports `--workers`. If omitted, default is auto (e.g. 4 for LM Studio, 1 for vLLM/local).

## Useful checks

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --check-only
python hf-local-scripts/run_invariance_generate_only.py --study study_b_invariance --model-id gpt_oss_lmstudio --data-dir $env:INVARIANCE_DATA_DIR --output-dir $env:INVARIANCE_RESULTS_DIR --workers 2 --max-samples 5
```
