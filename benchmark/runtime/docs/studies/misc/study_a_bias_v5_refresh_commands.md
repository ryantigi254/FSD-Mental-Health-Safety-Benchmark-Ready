# Study A Bias v5 Refresh Commands

## Scope
Use these commands to refresh existing `study_a_bias_generations.jsonl` files when the v5 adversarial bias source changed some prompts.

The refresh script:
- reads an existing `results/<model-folder>/study_a_bias_generations.jsonl`
- compares each row to `data/frozen_splits/v5/adversarial_bias/biased_vignettes.json` by `id`
- regenerates only stale ids whose v5 source row changed
- rewrites the JSONL in place

## One-Time Setup

### Windows (PC)
```powershell
cd "E:\22837352\NLP\FSD-Mental-Health-Safety-Benchmark-Ready\benchmark\runtime"
```

## Dry Run Checks

Use `--dry-run` first to see which ids will be regenerated.

### LM Studio models
```powershell
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id qwen3_lmstudio --cache-path results/qwen3-lmstudio/study_a_bias_generations.jsonl --dry-run
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id qwq --cache-path results/qwq/study_a_bias_generations.jsonl --dry-run
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id deepseek_r1_lmstudio --cache-path results/deepseek-r1-lmstudio/study_a_bias_generations.jsonl --dry-run
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id gpt_oss_lmstudio --cache-path results/gpt-oss-20b/study_a_bias_generations.jsonl --dry-run
```

### Local HF models
```powershell
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id psyllm_gml_local --cache-path results/psyllm-gml-local/study_a_bias_generations.jsonl --dry-run
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id piaget_local --cache-path results/piaget-8b-local/study_a_bias_generations.jsonl --dry-run
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id psyche_r1_local --cache-path results/psyche-r1-local/study_a_bias_generations.jsonl --dry-run
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id psych_qwen_local --cache-path results/psych-qwen-32b-local/study_a_bias_generations.jsonl --dry-run --quantization 4bit
```

## Refresh Commands

Use `--backup` so the original cache is kept as `study_a_bias_generations.jsonl.bak`.

### LM Studio models with worker settings
```powershell
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id qwen3_lmstudio --cache-path results/qwen3-lmstudio/study_a_bias_generations.jsonl --workers 6 --backup
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id qwq --cache-path results/qwq/study_a_bias_generations.jsonl --workers 6 --backup
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id deepseek_r1_lmstudio --cache-path results/deepseek-r1-lmstudio/study_a_bias_generations.jsonl --workers 4 --backup
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id gpt_oss_lmstudio --cache-path results/gpt-oss-20b/study_a_bias_generations.jsonl --workers 2 --backup
```

### Local HF models
```powershell
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id psyllm_gml_local --cache-path results/psyllm-gml-local/study_a_bias_generations.jsonl --backup
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id piaget_local --cache-path results/piaget-8b-local/study_a_bias_generations.jsonl --backup
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id psyche_r1_local --cache-path results/psyche-r1-local/study_a_bias_generations.jsonl --backup
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id psych_qwen_local --cache-path results/psych-qwen-32b-local/study_a_bias_generations.jsonl --backup --quantization 4bit
```

## vLLM (Local HF Models Only)

This section is for the four HF-local models when served via vLLM's OpenAI-compatible server.

- Client `--workers` should remain `1`.
- Concurrency is controlled server-side via `--max-num-seqs` (recommended sweep: `2 -> 4 -> 8 -> 12`).

### vLLM Server (per model)

#### PsyLLM-8B (`psyllm_gml_vllm`, default port 8101)

Mac/Linux:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model "GMLHUHE/PsyLLM-8B" \
  --download-dir "models/vllm" \
  --host 0.0.0.0 \
  --port 8101 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 4 \
  --enforce-eager \
  --max-model-len 24576
```

Windows (PC) / WSL:
```bash
python -m vllm.entrypoints.openai.api_server --model "GMLHUHE/PsyLLM-8B" --download-dir "models/vllm" --host 0.0.0.0 --port 8101 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager --max-model-len 24576
```

#### Piaget-8B (`piaget_vllm`, default port 8102)

Mac/Linux:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model "gustavecortal/Piaget-8B" \
  --download-dir "models/vllm" \
  --host 0.0.0.0 \
  --port 8102 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 4 \
  --enforce-eager \
  --max-model-len 24576
```

Windows (PC) / WSL:
```bash
python -m vllm.entrypoints.openai.api_server --model "gustavecortal/Piaget-8B" --download-dir "models/vllm" --host 0.0.0.0 --port 8102 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager --max-model-len 24576
```

#### Psyche-R1 (`psyche_r1_vllm`, default port 8103)

Mac/Linux:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model "MindIntLab/Psyche-R1" \
  --download-dir "models/vllm" \
  --host 0.0.0.0 \
  --port 8103 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 4 \
  --enforce-eager \
  --max-model-len 24576
```

Windows (PC) / WSL:
```bash
python -m vllm.entrypoints.openai.api_server --model "MindIntLab/Psyche-R1" --download-dir "models/vllm" --host 0.0.0.0 --port 8103 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager --max-model-len 24576
```

#### Psych_Qwen_32B (`psych_qwen_vllm`, default port 8104)

Mac/Linux:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model "Compumacy/Psych_Qwen_32B" \
  --download-dir "./models/vllm" \
  --host 0.0.0.0 \
  --port 8104 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 4 \
  --enforce-eager \
  --quantization bitsandbytes \
  --load-format bitsandbytes
```

Windows (PC) / WSL:
```bash
python -m vllm.entrypoints.openai.api_server --model "Compumacy/Psych_Qwen_32B" --download-dir "models/vllm" --host 0.0.0.0 --port 8104 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager --max-model-len 24576 --quantization bitsandbytes --load-format bitsandbytes
```

### vLLM Refresh Commands

Mac/Linux:
```bash
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id psyllm_gml_vllm --cache-path results/psyllm-gml-local/study_a_bias_generations.jsonl --backup
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id piaget_vllm --cache-path results/piaget-8b-local/study_a_bias_generations.jsonl --backup
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id psyche_r1_vllm --cache-path results/psyche-r1-local/study_a_bias_generations.jsonl --backup
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id psych_qwen_vllm --cache-path results/psych-qwen-32b-local/study_a_bias_generations.jsonl --backup
```

Windows (PC):
```powershell
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id psyllm_gml_vllm --cache-path results/psyllm-gml-local/study_a_bias_generations.jsonl --backup
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id piaget_vllm --cache-path results/piaget-8b-local/study_a_bias_generations.jsonl --backup
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id psyche_r1_vllm --cache-path results/psyche-r1-local/study_a_bias_generations.jsonl --backup
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id psych_qwen_vllm --cache-path results/psych-qwen-32b-local/study_a_bias_generations.jsonl --backup
```

## Partial Refresh

Use `--max-cases` to refresh only the first few stale ids after diffing.

```powershell
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id qwen3_lmstudio --cache-path results/qwen3-lmstudio/study_a_bias_generations.jsonl --dry-run --max-cases 10
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id qwen3_lmstudio --cache-path results/qwen3-lmstudio/study_a_bias_generations.jsonl --max-cases 10 --backup
```

## Explicit v5 Source

The script already defaults to v5, but you can pass the file explicitly:

```powershell
python hf-local-scripts/refresh_study_a_bias_from_v5.py --model-id qwen3_lmstudio --cache-path results/qwen3-lmstudio/study_a_bias_generations.jsonl --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --dry-run
```

## Notes

- Matching is by Study A bias case id such as `abias_0007`.
- A row is treated as stale when its cached prompt, `bias_feature`, `bias_label`, or `metadata` no longer matches the current v5 source row for that id.
- The script only refreshes ids already present in the target JSONL file.
- `psych-qwen-space-api` is not included here because it is a results folder, not a supported runnable model alias in the current runner stack.
