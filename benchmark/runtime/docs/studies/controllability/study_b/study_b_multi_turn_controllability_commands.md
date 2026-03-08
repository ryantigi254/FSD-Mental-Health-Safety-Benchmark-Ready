# Study B Multi-Turn Controllability Commands

## Generation

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id qwq
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id qwen3_lmstudio
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id deepseek_r1_lmstudio
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id gpt_oss
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psyllm_gml_vllm
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id piaget_vllm
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psyche_r1_vllm
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psych_qwen_vllm
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psyllm_gml_local
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id piaget_local
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psyche_r1_local
PYTHONPATH=src python scripts/dev/run_generation_auto.py --study ctrl_study_b_multi_turn --model-id psych_qwen_local
```

## Direct Runner

```bash
cd benchmark/runtime
PYTHONPATH=src python hf-local-scripts/run_ctrl_generate_only.py \
    --study ctrl_study_b_multi_turn \
    --model-id qwq \
    --max-cases 3 \
    --max-tokens 8192
```

## Output

```text
results/<model>/ctrl_study_b_multi_turn_generations.jsonl
```
