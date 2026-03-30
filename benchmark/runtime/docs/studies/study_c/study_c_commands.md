# Study C Commands

## Scope
Study C generation writes to `results/<model-folder>/study_c_generations.jsonl` and metrics write to `metric-results/.../study_c`.

## One-Time Setup

### Windows (PC)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
```

## Optional Gold Plans (for continuity score stability)

### Windows (PC)
```powershell
conda run -n mh-llm-benchmark-env env PYTHONPATH=src python scripts/studies/study_c/gold_plans/generate_nli_plans.py --force
```

## Generation Commands (Automatic Cross-Platform Runner)

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_c --model-id qwen3_lmstudio --env mh-llm-benchmark-env --workers 6
python scripts/dev/run_generation_auto.py --study study_c --model-id qwq --env mh-llm-benchmark-env --workers 6
python scripts/dev/run_generation_auto.py --study study_c --model-id medgemma_lmstudio --env mh-llm-benchmark-env --workers 4
python scripts/dev/run_generation_auto.py --study study_c --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --workers 4
python scripts/dev/run_generation_auto.py --study study_c --model-id gpt_oss --env mh-llm-benchmark-env --workers 2
```

## Local HF Models (mh-llm-local-env)

Use these for the four HF-local models:

```powershell
python scripts/dev/run_generation_auto.py --study study_c --model-id psyllm_gml_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_c --model-id piaget_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_c --model-id psyche_r1_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_c --model-id psych_qwen_local --env mh-llm-local-env --quantization 4bit
```

## vLLM (Local HF Models Only)

This section is for the four HF-local models when served via vLLM's OpenAI-compatible server.

- Client `--workers` should remain `1`.
- Concurrency is controlled server-side via `--max-num-seqs` (recommended sweep: `2 → 4 → 8 → 12`).

### vLLM Server (per model)

#### PsyLLM-8B (`psyllm_gml_vllm`, default port 8101)

Windows (PC) / WSL:
```bash
python -m vllm.entrypoints.openai.api_server --model "GMLHUHE/PsyLLM-8B" --download-dir "./models/vllm" --host 0.0.0.0 --port 8101 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager --max-model-len 24576
```

#### Piaget-8B (`piaget_vllm`, default port 8102)

Windows (PC) / WSL:
```bash
python -m vllm.entrypoints.openai.api_server --model "gustavecortal/Piaget-8B" --download-dir "./models/vllm" --host 0.0.0.0 --port 8102 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager --max-model-len 24576
```

#### Psyche-R1 (`psyche_r1_vllm`, default port 8103)

Windows (PC) / WSL:
```bash
python -m vllm.entrypoints.openai.api_server --model "MindIntLab/Psyche-R1" --download-dir "./models/vllm" --host 0.0.0.0 --port 8103 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager --max-model-len 24576
```

#### Psych_Qwen_32B (`psych_qwen_vllm`, default port 8104)

Windows (PC) / WSL:
```bash
python -m vllm.entrypoints.openai.api_server --model "Compumacy/Psych_Qwen_32B" --download-dir "./models/vllm" --host 0.0.0.0 --port 8104 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager --max-model-len 24576 --quantization bitsandbytes --load-format bitsandbytes
```

### vLLM Generation Commands

#### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_c --model-id psyllm_gml_vllm --env mh-llm-vllm-env
python scripts/dev/run_generation_auto.py --study study_c --model-id piaget_vllm --env mh-llm-vllm-env
python scripts/dev/run_generation_auto.py --study study_c --model-id psyche_r1_vllm --env mh-llm-vllm-env
python scripts/dev/run_generation_auto.py --study study_c --model-id psych_qwen_vllm --env mh-llm-vllm-env
```

## Metrics Commands

### Windows (PC)
```powershell
conda run -n mh-llm-benchmark-env env PYTHONPATH=src python scripts/studies/study_c/metrics/calculate_metrics.py --use-cleaned --output-dir metric-results/study_c
```

## Workers
Study C generation supports `--workers`. If not passed, default is auto:
- `6` for `qwen3_lmstudio` and `qwq` in the Windows (PC) examples above
- `4` for `medgemma_lmstudio` and for auto-default LM Studio runs when `--workers` is omitted
- `4` / `2` for `deepseek_r1_lmstudio` / `gpt_oss` as in the examples
- `1` for non-LM Studio/local HF models

## Useful Checks
```bash
python scripts/dev/run_generation_auto.py --study study_c --model-id gpt_oss --check-only
python hf-local-scripts/run_study_c_generate_only.py --model-id gpt_oss --workers 8 --max-cases 5
```

## Temporary Option (Ollama Minimax M2.5 Cloud)
Use this only as a temporary model path, not as part of the main benchmark model set.

```powershell
python scripts/dev/run_generation_auto.py --study study_c --model-id ollama_minimax_m2_5_cloud --env mh-llm-benchmark-env --workers 4
```
