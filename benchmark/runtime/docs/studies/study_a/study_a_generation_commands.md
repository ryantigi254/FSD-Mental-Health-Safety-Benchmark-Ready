# Study A Generation Commands

## Scope
Study A generation writes to `results/<model-folder>/study_a_generations.jsonl`.

## One-Time Setup

### Windows (PC)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
```

## Generation Commands (Automatic Cross-Platform Runner)

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_a --model-id qwen3_lmstudio --env mh-llm-benchmark-env --workers 6
python scripts/dev/run_generation_auto.py --study study_a --model-id qwq --env mh-llm-benchmark-env --workers 6
python scripts/dev/run_generation_auto.py --study study_a --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --workers 4
python scripts/dev/run_generation_auto.py --study study_a --model-id gpt_oss --env mh-llm-benchmark-env --workers 2
```

## Local HF Models (mh-llm-local-env)

Use these for the four HF-local models:

```powershell
python scripts/dev/run_generation_auto.py --study study_a --model-id psyllm_gml_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a --model-id piaget_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a --model-id psyche_r1_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a --model-id psych_qwen_local --env mh-llm-local-env --quantization 4bit
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
python scripts/dev/run_generation_auto.py --study study_a --model-id psyllm_gml_vllm --env mh-llm-vllm-env
python scripts/dev/run_generation_auto.py --study study_a --model-id piaget_vllm --env mh-llm-vllm-env
python scripts/dev/run_generation_auto.py --study study_a --model-id psyche_r1_vllm --env mh-llm-vllm-env
python scripts/dev/run_generation_auto.py --study study_a --model-id psych_qwen_vllm --env mh-llm-vllm-env
```

## Metrics Commands

### Windows (PC)
```powershell
conda run -n mh-llm-benchmark-env env PYTHONPATH=src python scripts/studies/study_a/metrics/calculate_metrics.py --use-cleaned --output-dir metric-results/study_a
conda run -n mh-llm-benchmark-env env PYTHONPATH=src python scripts/studies/study_a/metrics/calculate_bias.py --use-cleaned --output-dir metric-results/study_a
```

## Workers
`study_a` generation supports `--workers`. If not passed, default is auto:
- `4` for LM Studio models (`qwen3_lmstudio`, `qwq`, `deepseek_r1_lmstudio`, `gpt_oss`)
- `1` for non-LM Studio/local HF models

## Useful Checks
```bash
python scripts/dev/run_generation_auto.py --study study_a --model-id gpt_oss --check-only
python hf-local-scripts/run_study_a_generate_only.py --model-id gpt_oss --workers 8 --max-cases 5
```

## Temporary Option (Ollama Minimax M2.5 Cloud)
Use this only as a temporary model path, not as part of the main benchmark model set.

```powershell
python scripts/dev/run_generation_auto.py --study study_a --model-id ollama_minimax_m2_5_cloud --env mh-llm-benchmark-env --workers 4
```

## Qwen 3.5 27B Distilled (mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2)

LM Studio model via Apple Silicon MLX. Recommended max 4 workers on 48 GB.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_study_a_generate_only.py --model-id qwen3.5-distilled --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_a --model-id qwen3.5-distilled --env mh-llm-benchmark-env --workers 4
```
