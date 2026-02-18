# Study A Bias Commands

## Scope
Study A bias generation writes to `results/<model-folder>/study_a_bias_generations.jsonl` and metrics write to `metric-results/.../study_a`.

**Data source**: `data/releases/clinician_readiness_v0.3_2026-02-16/adversarial_bias/biased_vignettes.json` (2000 vignettes, SHA256-verified in release manifest).

## One-Time Setup

### Mac/Linux
```bash
cd "<path-to-repo>/benchmark/runtime"
```

### Windows (PC)
```powershell
cd "E:\22837352\NLP\FSD-Mental-Health-Safety-Benchmark-Ready\benchmark\runtime"
```

## Generation Commands (Automatic Cross-Platform Runner)

### Mac/Linux
```bash
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id qwen3_lmstudio --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id qwq --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psyllm_gml_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id piaget_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psyche_r1_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psych_qwen_local --env mh-llm-local-env --quantization 4bit
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id qwen3_lmstudio --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id qwq --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psyllm_gml_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id piaget_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psyche_r1_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psych_qwen_local --env mh-llm-local-env --quantization 4bit
```

### Direct Script (Alternative)
If you prefer to call the underlying script directly instead of `run_generation_auto.py`:
```bash
# Set PYTHONPATH first (from benchmark/runtime/)
export PYTHONPATH=src   # Mac/Linux
$env:PYTHONPATH="src"   # Windows PowerShell

# LM Studio models
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id qwq
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id deepseek_r1_lmstudio
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id qwen3_lmstudio

# Local HF models
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id psyllm_gml_local
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id piaget_local
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id psyche_r1_local
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id psych_qwen_local --quantization 4bit
```

## Metrics Commands

### Mac/Linux
```bash
RUN_TAG="$(date +%Y%m%d_%H%M)"; OUT_ROOT="metric-results/misc/${RUN_TAG}"; mkdir -p "${OUT_ROOT}/study_a"; conda run -n mh-llm-benchmark-env env PYTHONPATH=src python scripts/studies/study_a/metrics/calculate_bias.py --use-cleaned --output-dir "${OUT_ROOT}/study_a"
RUN_TAG="$(date +%Y%m%d_%H%M)"; OUT_ROOT="metric-results/misc/${RUN_TAG}"; mkdir -p "${OUT_ROOT}/study_a"; conda run -n mh-llm-benchmark-env env PYTHONPATH=src python scripts/studies/study_a/metrics/calculate_metrics.py --use-cleaned --output-dir "${OUT_ROOT}/study_a"
```

### Windows (PC)
```powershell
$RUN_TAG=Get-Date -Format "yyyyMMdd_HHmm"; $OUT_ROOT="metric-results/misc/$RUN_TAG"; New-Item -ItemType Directory -Force -Path "$OUT_ROOT/study_a" | Out-Null; conda run -n mh-llm-benchmark-env python scripts/studies/study_a/metrics/calculate_bias.py --use-cleaned --output-dir "$OUT_ROOT/study_a"
$RUN_TAG=Get-Date -Format "yyyyMMdd_HHmm"; $OUT_ROOT="metric-results/misc/$RUN_TAG"; New-Item -ItemType Directory -Force -Path "$OUT_ROOT/study_a" | Out-Null; conda run -n mh-llm-benchmark-env python scripts/studies/study_a/metrics/calculate_metrics.py --use-cleaned --output-dir "$OUT_ROOT/study_a"
```

## Workers
`study_a_bias` generation supports `--workers`. If not passed, default is auto:
- `4` for LM Studio models (`qwen3_lmstudio`, `qwq`, `deepseek_r1_lmstudio`, `gpt_oss_lmstudio`)
- `1` for non-LM Studio/local HF models

## Useful Checks
```bash
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id gpt_oss_lmstudio --check-only
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio --workers 8 --max-cases 5
```

## Key Paths

| Item | Path (relative to `benchmark/runtime/`) |
|---|---|
| Bias data (default) | `data/releases/clinician_readiness_v0.3_2026-02-16/adversarial_bias/biased_vignettes.json` |
| Data manifest | `data/releases/clinician_readiness_v0.3_2026-02-16/manifest.json` |
| Generation script | `hf-local-scripts/run_study_a_bias_generate_only.py` |
| Auto-runner wrapper | `scripts/dev/run_generation_auto.py` |
| Generation output | `results/<model-folder>/study_a_bias_generations.jsonl` |
| Processed pipeline | `processed/study_a_bias_pipeline/<model-folder>/study_a_bias_processed.jsonl` |
| Metrics script | `scripts/studies/study_a/metrics/calculate_bias.py` |
| Metrics output | `metric-results/study_a_bias/study_a_bias_metrics.json` |
