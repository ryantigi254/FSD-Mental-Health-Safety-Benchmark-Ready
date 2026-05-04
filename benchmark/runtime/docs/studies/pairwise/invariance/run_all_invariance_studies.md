# Run All Invariance Pairwise Studies

This note shows the commands for every invariance pairwise family.

## Slices Covered

- `invariance`
- `invariance_under_control`
- `control_under_invariance`

## Canonical Four-Judge Panel

Run each judge role with the matching LM Studio model loaded.

### Primary

Load `qwopus3.5-27b-v3.5`, then run:

```powershell
cd benchmark/runtime
$slices = @("invariance", "invariance_under_control", "control_under_invariance")
foreach ($slice in $slices) {
  python scripts/pairwise/run_pairwise.py `
    --config "metric-results/pairwise/manifests/configs/$slice.run_config.json" `
    --judge-id jackrong_qwopus35_27b_v35
}
```

### Audit

Load `glm-4.7-flash-claude-opus-4.5-high-reasoning-distill`, then run:

```powershell
cd benchmark/runtime
$slices = @("invariance", "invariance_under_control", "control_under_invariance")
foreach ($slice in $slices) {
  python scripts/pairwise/run_pairwise.py `
    --config "metric-results/pairwise/manifests/configs/$slice.run_config.json" `
    --judge-id teichai_glm47_flash_opus45
}
```

### Escalation 1

Load `qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2`, then run:

```powershell
cd benchmark/runtime
$slices = @("invariance", "invariance_under_control", "control_under_invariance")
foreach ($slice in $slices) {
  python scripts/pairwise/run_pairwise.py `
    --config "metric-results/pairwise/manifests/configs/$slice.run_config.json" `
    --judge-id jackrong_qwen35_27b_claude46_opus_v2
}
```

### Escalation 2

Load `gemma-4-31b-it-claude-opus-distill-v2`, then run:

```powershell
cd benchmark/runtime
$slices = @("invariance", "invariance_under_control", "control_under_invariance")
foreach ($slice in $slices) {
  python scripts/pairwise/run_pairwise.py `
    --config "metric-results/pairwise/manifests/configs/$slice.run_config.json" `
    --judge-id teichai_gemma4_31b_it_opus
}
```

## Local Two-Model Panel

Create local configs first:

```powershell
cd benchmark/runtime
.\.venv-pairwise-22837352-20260430\Scripts\python.exe scripts/pairwise/write_local_lmstudio_pairwise_manifests.py
```

Then run the local judge IDs:

```powershell
cd benchmark/runtime
$slices = @("invariance", "invariance_under_control", "control_under_invariance")
$judges = @(
  "local_qwen35_primary",
  "local_gemma4_audit",
  "local_qwen35_escalation",
  "local_gemma4_escalation"
)
foreach ($judge in $judges) {
  foreach ($slice in $slices) {
    .\.venv-pairwise-22837352-20260430\Scripts\python.exe scripts/pairwise/run_pairwise.py `
      --config "metric-results/pairwise/manifests/configs/$slice.local.run_config.json" `
      --judge-id $judge
  }
}
```

