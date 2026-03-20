# Study C Controllability Commands

## Scope
Study C controllability generation writes to `results/<model-folder>/ctrl_study_c_generations.jsonl`.
Canonical Study C generation is arm-aware. Each case is evaluated across
`spontaneous`, `generic_control`, and `explicit_control`.

## Scaled Suite Overrides

Use these overrides for the large resolved suite so you do not mix caches with the
small-scale run:

```bash
cd benchmark/runtime
export CTRL_DIR=data/controllability_splits_large_resolved
export CTRL_RESULTS_DIR=results_scaled_large_resolved
```

Example:

```powershell
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id qwq `
  --study ctrl_study_c `
  --ctrl-dir "$CTRL_DIR" `
  --output-dir "$CTRL_RESULTS_DIR" `
  --workers 6
```

## One-Time Setup

### Windows (PC)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
```

## Generation Commands (Direct Runner)

### Windows (PC)
```powershell
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id qwen3_lmstudio `
  --study ctrl_study_c `
  --workers 6
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id qwq `
  --study ctrl_study_c `
  --workers 6
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id deepseek_r1_lmstudio `
  --study ctrl_study_c `
  --workers 4
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id gpt_oss `
  --study ctrl_study_c `
  --workers 2
```

## Local HF Models (mh-llm-local-env)

```powershell
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id psyllm_gml_local `
  --study ctrl_study_c
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id piaget_local `
  --study ctrl_study_c
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id psyche_r1_local `
  --study ctrl_study_c
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id psych_qwen_local `
  --study ctrl_study_c `
  --quantization 4bit
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
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id psyllm_gml_vllm `
  --study ctrl_study_c
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id piaget_vllm `
  --study ctrl_study_c
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id psyche_r1_vllm `
  --study ctrl_study_c
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id psych_qwen_vllm `
  --study ctrl_study_c
```

## Direct Runner

```powershell
python hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_c --model-id qwq --ctrl-dir data/controllability_splits_large_resolved --output-dir results_scaled_large_resolved --max-cases 3 --workers 6
```

`--max-tokens` is optional. For LM Studio and vLLM models, omitting it lets the
serving stack control the effective completion limit.

## Workers

`ctrl_study_c` generation supports `--workers`. If not passed, default is auto:
- `4` for LM Studio models (`qwen3_lmstudio`, `qwq`, `deepseek_r1_lmstudio`, `gpt_oss`)
- `1` for vLLM and local HF models

## Useful Checks

```bash
python hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_c --model-id gpt_oss --workers 8 --max-cases 3
```
