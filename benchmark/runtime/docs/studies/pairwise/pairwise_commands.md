# Pairwise Commands

## Write the canonical judge panel and default slice configs

```bash
python3 scripts/pairwise/write_default_pairwise_manifests.py
```

## Build one frozen case manifest

```bash
python3 scripts/pairwise/build_case_manifest.py --slice-id study_a
```

## Build all currently supported core manifests

```bash
for slice in study_a study_a_bias study_b study_b_multiturn study_c; do
  python3 scripts/pairwise/build_case_manifest.py --slice-id "$slice"
done
```

## Dry-run one slice

```bash
python3 scripts/pairwise/run_pairwise.py \
  --config metric-results/pairwise/manifests/configs/study_a.run_config.json \
  --dry-run
```

## Run one slice against LM Studio

```bash
python3 scripts/pairwise/run_pairwise.py \
  --config metric-results/pairwise/manifests/configs/study_a.run_config.json
```

## Write the local LM Studio panel for this machine

```powershell
.\.venv-pairwise-22837352-20260430\Scripts\python.exe scripts/pairwise/write_local_lmstudio_pairwise_manifests.py
```

This writes `judge_panel.local.json` and `*.local.run_config.json` files for:

- `qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2`
- `gemma-4-31b-it-claude-opus-distill-v2`

The local manifests omit `max_tokens`; set the output budget in LM Studio.

## Run one stacked judge pass at a time in LM Studio

```bash
python3 scripts/pairwise/run_pairwise.py \
  --config metric-results/pairwise/manifests/configs/study_a.run_config.json \
  --judge-id jackrong_qwopus35_27b_v35
```

Repeat with one loaded LM Studio model at a time. The runner rebuilds the aggregate report from all parsed judge-pass files already on disk after each pass.

## Run one local LM Studio judge pass

```powershell
.\.venv-pairwise-22837352-20260430\Scripts\python.exe scripts/pairwise/run_pairwise.py `
  --config metric-results/pairwise/manifests/configs/study_a.local.run_config.json `
  --judge-id local_qwen35_primary `
  --workers 4
```

Repeat with:

- `local_qwen35_primary`
- `local_gemma4_audit`
- `local_qwen35_escalation`
- `local_gemma4_escalation`

Full run guides:

- [Local LM Studio commands](local_lmstudio_commands.md)
- [Run all core studies](core/run_all_core_studies.md)
- [Run all controllability studies](controllability/run_all_controllability_studies.md)
- [Run all invariance studies](invariance/run_all_invariance_studies.md)

## Run one slice in exhaustive audit mode

```bash
python3 scripts/pairwise/run_pairwise.py \
  --config metric-results/pairwise/manifests/configs/study_a.run_config.json \
  --dry-run
```

Then copy the config, change `"run_mode": "all_judges"`, and rerun for a full four-judge audit pass.

## Run the judge-audit summary

```bash
python3 scripts/pairwise/run_judge_audit.py \
  --manifest metric-results/pairwise/judge_audit/manifests/gold_audit_slice.v1.json \
  --parsed-dir metric-results/pairwise/parsed/pairwise_study_a_v3
```

## Generate the dedicated notebook family

```bash
python3 scripts/pairwise/generate_pairwise_notebooks.py
```
