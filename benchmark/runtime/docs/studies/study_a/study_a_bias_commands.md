# Study A Bias Commands

## Scope
Study A bias generation writes to `results/<model-folder>/study_a_bias_generations.jsonl` and metrics write to `metric-results/.../study_a`.

## One-Time Setup

### Windows (PC)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
```

## Generation Commands (Automatic Cross-Platform Runner)

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id qwen3_lmstudio --env mh-llm-benchmark-env --workers 6
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id qwq --env mh-llm-benchmark-env --workers 6
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id medgemma_lmstudio --env mh-llm-benchmark-env --workers 4
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --workers 4
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --workers 2
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psych_qwen_32b --env mh-llm-benchmark-env --workers 4
```

## Local HF Models (mh-llm-local-env)

Use these for the four HF-local models:

```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psyllm_gml_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id piaget_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psyche_r1_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psych_qwen_local --env mh-llm-local-env --quantization 4bit
```

## Study A Bias (Mac Apple Silicon, MLX)
This is Mac-only and separate from the standard cross-machine commands above.

### Generation command
```bash
PYTHONPATH=src python hf-local-scripts/run_study_a_bias_generate_only.py --model-id psych_qwen_local --model models/Psych_Qwen_32B --quantization 4bit --data-path data/adversarial_bias/biased_vignettes.json --output-dir results
```

### Batch cutoff benchmark (MLX)
```bash
python scripts/studies/study_a/run_study_a_bias_mlx_batch_cutoff.py --model-id psych_qwen_local --model models/Psych_Qwen_32B_MLX_4bit --data-path data/adversarial_bias/biased_vignettes.json --output-dir results --max-cases 24 --max-tokens 64 --batch-sizes 1 2 4 8 --trials 1
```

## vLLM (Local HF Models Only)

This section is for the four HF-local models when served via vLLM's OpenAI-compatible server.

- Client `--workers` should remain `1`.
- Concurrency is controlled server-side via `--max-num-seqs` (recommended sweep: `2 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 4 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 8 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 12`).

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
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psyllm_gml_vllm --env mh-llm-vllm-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id piaget_vllm --env mh-llm-vllm-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psyche_r1_vllm --env mh-llm-vllm-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psych_qwen_vllm --env mh-llm-vllm-env
```

## Metrics Commands

### Windows (PC)
```powershell
$RUN_TAG=Get-Date -Format "yyyyMMdd_HHmm"; $OUT_ROOT="metric-results/misc/$RUN_TAG"; New-Item -ItemType Directory -Force -Path "$OUT_ROOT/study_a" | Out-Null; conda run -n mh-llm-benchmark-env python scripts/studies/study_a/metrics/calculate_bias.py --use-cleaned --output-dir "$OUT_ROOT/study_a"
$RUN_TAG=Get-Date -Format "yyyyMMdd_HHmm"; $OUT_ROOT="metric-results/misc/$RUN_TAG"; New-Item -ItemType Directory -Force -Path "$OUT_ROOT/study_a" | Out-Null; conda run -n mh-llm-benchmark-env python scripts/studies/study_a/metrics/calculate_metrics.py --use-cleaned --output-dir "$OUT_ROOT/study_a"
```

## Workers
`study_a_bias` generation supports `--workers`. If not passed, default is auto:
- `6` for `qwen3_lmstudio` and `qwq` in the Windows (PC) examples above
- `4` for `medgemma_lmstudio`, `psych_qwen_32b` (LM Studio), and for auto-default LM Studio runs when `--workers` is omitted
- `4` / `2` for `deepseek_r1_lmstudio` / `gpt_oss_lmstudio` as in the examples
- `1` for non-LM Studio/local HF models

## Useful Checks
```bash
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id gpt_oss_lmstudio --check-only
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio --workers 8 --max-cases 5
```
