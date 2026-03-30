# Study A Bias Controllability Commands

## Scope
Study A bias controllability generation writes to `results/<model-folder>/ctrl_study_a_bias_generations.jsonl`.
Canonical Study A bias generation is arm-aware and runs each case under
`spontaneous`, `generic_control`, and `explicit_control`.
By default, Study A bias loads data from
`data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json`.

## Data Directory

Use `--data-path` to point Study A bias at the merged suite.

```bash
cd benchmark/runtime
export CTRL_RESULTS_DIR=results_ctrl_v2_1
export BIAS_DATA_PATH=data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json
```

Example:

```powershell
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id qwq `
  --study ctrl_study_a_bias `
  --data-path "$BIAS_DATA_PATH" `
  --output-dir "$CTRL_RESULTS_DIR" `
  --workers 6
```

Effective direct runner command shape:

```powershell
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id gpt_oss `
  --study ctrl_study_a_bias `
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json `
  --workers 2
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
  --study ctrl_study_a_bias `
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json `
  --workers 6
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id qwq `
  --study ctrl_study_a_bias `
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json `
  --workers 6
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id deepseek_r1_lmstudio `
  --study ctrl_study_a_bias `
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json `
  --workers 4
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id gpt_oss `
  --study ctrl_study_a_bias `
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json `
  --workers 2
```

## Local HF Models (mh-llm-local-env)

```powershell
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id psyllm_gml_local `
  --study ctrl_study_a_bias `
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id piaget_local `
  --study ctrl_study_a_bias `
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id psyche_r1_local `
  --study ctrl_study_a_bias `
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id psych_qwen_local `
  --study ctrl_study_a_bias `
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json `
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
  --study ctrl_study_a_bias `
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id piaget_vllm `
  --study ctrl_study_a_bias `
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id psyche_r1_vllm `
  --study ctrl_study_a_bias `
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id psych_qwen_vllm `
  --study ctrl_study_a_bias `
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json
```

## Direct Runner

```powershell
python hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_a_bias --model-id qwen3_lmstudio --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json --output-dir results_ctrl_v2_1 --max-cases 5 --workers 6
```

`--max-tokens` is optional. For LM Studio and vLLM models, omitting it lets the
serving stack control the effective completion limit.

## Workers

`ctrl_study_a_bias` generation supports `--workers`. If not passed, default is auto:
- `6` for `qwen3_lmstudio` and `qwq` in the Windows (PC) examples above
- `4` for `medgemma_lmstudio` and for auto-default LM Studio runs when `--workers` is omitted
- `4` / `2` for `deepseek_r1_lmstudio` / `gpt_oss` as in the examples
- `1` for vLLM and local HF models

## Useful Checks

```bash
python hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_a_bias --model-id gpt_oss --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json --workers 8 --max-cases 5
```

## Qwen 3.5 27B Distilled (mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2)

LM Studio model via Apple Silicon MLX. Recommended max 4 workers on 48 GB.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --model-id qwen3.5-distilled \
  --study ctrl_study_a_bias \
  --workers 4
```

### Windows (PC)
```powershell
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id qwen3.5-distilled `
  --study ctrl_study_a_bias `
  --workers 4
```

## MedGemma 27B Text (`google.medgemma-27b-text-it`)

In LM Studio, load the model as `google.medgemma-27b-text-it`. The benchmark runner uses `--model-id medgemma_lmstudio` (aliases include that identifier). Recommended max 4 workers.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --model-id medgemma_lmstudio \
  --study ctrl_study_a_bias \
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json \
  --workers 4
```

### Windows (PC)
```powershell
python hf-local-scripts/run_ctrl_generate_only.py `
  --model-id medgemma_lmstudio `
  --study ctrl_study_a_bias `
  --data-path data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json `
  --workers 4
```
