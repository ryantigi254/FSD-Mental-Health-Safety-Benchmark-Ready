# Study B Controllability Commands

## Scope
Study B single-turn controllability generation writes to `results/<model-folder>/ctrl_study_b_generations.jsonl`.

## Canonical Scaled Paths

These commands use the large resolved suite directly:

- `--ctrl-dir data/controllability_splits_large_resolved`
- `--output-dir results_scaled_large_resolved`

## One-Time Setup

### Windows (PC)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
```

## Generation Commands (Automatic Cross-Platform Runner)

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id qwen3_lmstudio --env mh-llm-benchmark-env --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id qwq --env mh-llm-benchmark-env --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved --workers 4
python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id gpt_oss --env mh-llm-benchmark-env --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved --workers 2
```

## Local HF Models (mh-llm-local-env)

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id psyllm_gml_local --env mh-llm-local-env --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved
python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id piaget_local --env mh-llm-local-env --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved
python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id psyche_r1_local --env mh-llm-local-env --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved
python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id psych_qwen_local --env mh-llm-local-env --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved --quantization 4bit
```

## vLLM (Local HF Models Only)

- Client `--workers` should remain `1`.
- Concurrency is controlled server-side via `--max-num-seqs` (recommended sweep: `2 -> 4 -> 8 -> 12`).

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

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id psyllm_gml_vllm --env mh-llm-vllm-env --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved
python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id piaget_vllm --env mh-llm-vllm-env --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved
python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id psyche_r1_vllm --env mh-llm-vllm-env --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved
python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id psych_qwen_vllm --env mh-llm-vllm-env --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved
```

## Direct Runner

```powershell
python hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_b --model-id qwq --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved --max-cases 5 --max-tokens 8192 --workers 6
```

## Workers

`ctrl_study_b` generation supports `--workers`. If not passed, default is auto:
- `4` for LM Studio models (`qwen3_lmstudio`, `qwq`, `deepseek_r1_lmstudio`, `gpt_oss`)
- `1` for vLLM and local HF models

## Useful Checks

```bash
python scripts/dev/run_generation_auto.py --study ctrl_study_b --model-id gpt_oss --env mh-llm-benchmark-env --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved --check-only
python hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_b --model-id gpt_oss --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved --workers 8 --max-cases 5
```
