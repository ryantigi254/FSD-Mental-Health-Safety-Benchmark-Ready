# Run All Core Pairwise Studies

This note shows the canonical way to run all core pairwise slices when only one LM Studio judge model can be loaded at a time.

## Core slices covered

- `study_a`
- `study_a_bias`
- `study_b`
- `study_b_multiturn`
- `study_c`

## Role order

Run the judges in this order:

1. `primary`
2. `audit`
3. `escalation_1`
4. `escalation_2`

The runner uses the role metadata from `judge_panel.v3.json`, so the judges do not all do the same work:

- `primary` runs from a cold state across every pair
- `audit` runs only after primary coverage exists
- `escalation_1` runs only on comparisons that actually escalate
- `escalation_2` runs only on comparisons that still need the second escalation pass

## Before starting

Regenerate the canonical manifests and configs if needed:

```bash
python3 benchmark/runtime/scripts/pairwise/write_default_pairwise_manifests.py
```

## 1. Primary pass across all core studies

Load this LM Studio model first:

- `qwopus3.5-27b-v3.5`

Then run:

```bash
for slice in study_a study_a_bias study_b study_b_multiturn study_c; do
  python3 benchmark/runtime/scripts/pairwise/run_pairwise.py \
    --config "benchmark/runtime/metric-results/pairwise/manifests/configs/${slice}.run_config.json" \
    --judge-id jackrong_qwopus35_27b_v35
done
```

## 2. Audit pass across all core studies

Load this LM Studio model next:

- `glm-4.7-flash-claude-opus-4.5-high-reasoning-distill`

Then run:

```bash
for slice in study_a study_a_bias study_b study_b_multiturn study_c; do
  python3 benchmark/runtime/scripts/pairwise/run_pairwise.py \
    --config "benchmark/runtime/metric-results/pairwise/manifests/configs/${slice}.run_config.json" \
    --judge-id teichai_glm47_flash_opus45
done
```

## 3. Escalation 1 across all core studies

Load this LM Studio model next:

- `qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2`

Then run:

```bash
for slice in study_a study_a_bias study_b study_b_multiturn study_c; do
  python3 benchmark/runtime/scripts/pairwise/run_pairwise.py \
    --config "benchmark/runtime/metric-results/pairwise/manifests/configs/${slice}.run_config.json" \
    --judge-id jackrong_qwen35_27b_claude46_opus_v2
done
```

## 4. Escalation 2 across all core studies

Load this LM Studio model next:

- `gemma-4-31b-it-claude-opus-distill-v2`

Then run:

```bash
for slice in study_a study_a_bias study_b study_b_multiturn study_c; do
  python3 benchmark/runtime/scripts/pairwise/run_pairwise.py \
    --config "benchmark/runtime/metric-results/pairwise/manifests/configs/${slice}.run_config.json" \
    --judge-id teichai_gemma4_31b_it_opus
done
```

## What this does

- each command runs one judge role across every core slice
- aggregate and report files are rebuilt from the parsed outputs already on disk after each pass
- the escalation commands usually do much less work than the primary command because they only touch the escalated comparisons

## Optional dry-run check

If you want to see planned work before running a slice:

```bash
python3 benchmark/runtime/scripts/pairwise/run_pairwise.py \
  --config benchmark/runtime/metric-results/pairwise/manifests/configs/study_a.run_config.json \
  --judge-id jackrong_qwopus35_27b_v35 \
  --dry-run
```
