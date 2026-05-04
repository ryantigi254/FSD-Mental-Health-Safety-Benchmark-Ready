# Local LM Studio Pairwise Commands

These commands use the local two-model LM Studio panel written by:

```powershell
cd benchmark/runtime
.\.venv-pairwise-22837352-20260430\Scripts\python.exe scripts/pairwise/write_local_lmstudio_pairwise_manifests.py
```

The local panel uses:

- `local_qwen35_primary` -> `qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2`
- `local_gemma4_audit` -> `gemma-4-31b-it-claude-opus-distill-v2`
- `local_qwen35_escalation` -> `qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2`
- `local_gemma4_escalation` -> `gemma-4-31b-it-claude-opus-distill-v2`

The local judge manifests intentionally omit `max_tokens`; set the high output budget in LM Studio.

## Dry-Run One Slice

```powershell
cd benchmark/runtime
.\.venv-pairwise-22837352-20260430\Scripts\python.exe scripts/pairwise/run_pairwise.py `
  --config metric-results/pairwise/manifests/configs/study_a.local.run_config.json `
  --workers 1 `
  --dry-run
```

## Worker Tuning

Use `--workers` to run multiple comparison groups in parallel for the currently selected judge. Start low and watch LM Studio throughput, VRAM, context pressure, and error rate.

Use `--max-groups` for a small pilot before a full slice. Those pilot records are valid partial progress and the runner will skip completed records when you continue.

```powershell
cd benchmark/runtime
$workerCounts = @(1, 2, 3, 4)
foreach ($workers in $workerCounts) {
  Measure-Command {
    .\.venv-pairwise-22837352-20260430\Scripts\python.exe scripts/pairwise/run_pairwise.py `
      --config metric-results/pairwise/manifests/configs/study_a.local.run_config.json `
      --judge-id local_qwen35_primary `
      --workers $workers `
      --max-groups 10
  }
}
```

On 2026-05-01, a local Qwen primary pilot on `study_a` found `--workers 4` fastest for three comparison groups / six real judge calls. Recheck after LM Studio server settings or loaded model mix changes.

## Core Slices

Slices:

- `study_a`
- `study_a_bias`
- `study_b`
- `study_b_multiturn`
- `study_c`

### Qwen Primary

Load `qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2`, then run:

```powershell
cd benchmark/runtime
$slices = @("study_a", "study_a_bias", "study_b", "study_b_multiturn", "study_c")
foreach ($slice in $slices) {
  .\.venv-pairwise-22837352-20260430\Scripts\python.exe scripts/pairwise/run_pairwise.py `
    --config "metric-results/pairwise/manifests/configs/$slice.local.run_config.json" `
    --judge-id local_qwen35_primary `
    --workers 4
}
```

### Gemma Audit

Load `gemma-4-31b-it-claude-opus-distill-v2`, then run:

```powershell
cd benchmark/runtime
$slices = @("study_a", "study_a_bias", "study_b", "study_b_multiturn", "study_c")
foreach ($slice in $slices) {
  .\.venv-pairwise-22837352-20260430\Scripts\python.exe scripts/pairwise/run_pairwise.py `
    --config "metric-results/pairwise/manifests/configs/$slice.local.run_config.json" `
    --judge-id local_gemma4_audit `
    --workers 4
}
```

### Qwen Escalation

Load `qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2`, then run:

```powershell
cd benchmark/runtime
$slices = @("study_a", "study_a_bias", "study_b", "study_b_multiturn", "study_c")
foreach ($slice in $slices) {
  .\.venv-pairwise-22837352-20260430\Scripts\python.exe scripts/pairwise/run_pairwise.py `
    --config "metric-results/pairwise/manifests/configs/$slice.local.run_config.json" `
    --judge-id local_qwen35_escalation `
    --workers 4
}
```

### Gemma Escalation

Load `gemma-4-31b-it-claude-opus-distill-v2`, then run:

```powershell
cd benchmark/runtime
$slices = @("study_a", "study_a_bias", "study_b", "study_b_multiturn", "study_c")
foreach ($slice in $slices) {
  .\.venv-pairwise-22837352-20260430\Scripts\python.exe scripts/pairwise/run_pairwise.py `
    --config "metric-results/pairwise/manifests/configs/$slice.local.run_config.json" `
    --judge-id local_gemma4_escalation `
    --workers 4
}
```

## Controllability Slices

Slices:

- `study_a_controllability`
- `study_a_bias_controllability`
- `study_b_controllability`
- `study_b_multiturn_controllability`
- `study_c_controllability`

```powershell
cd benchmark/runtime
$slices = @(
  "study_a_controllability",
  "study_a_bias_controllability",
  "study_b_controllability",
  "study_b_multiturn_controllability",
  "study_c_controllability"
)
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
      --judge-id $judge `
      --workers 4
  }
}
```

## Invariance Slices

Slices:

- `invariance`
- `invariance_under_control`
- `control_under_invariance`

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
      --judge-id $judge `
      --workers 4
  }
}
```
