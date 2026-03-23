# Controllability Gold Generation

This note records the current gold-generation policy for controllability and why the benchmark no longer treats the NLI-first path as the canonical source of truth.

## Current Policy

- Study A gold diagnosis labels now use the `probe` backend by default.
- Study C gold plans now use the `probe` backend by default.
- The NLI backend is still kept in the scripts for audit and ablation work, but it is no longer the preferred path for checked-in controllability gold artefacts.

Canonical checked-in outputs:

- `data/controllability/controllability_splits_v2_1/ctrl_gold_diagnosis_labels.json`
- `data/controllability/controllability_splits_v2_1/ctrl_gold_diagnosis_labels.robust.json`
- `data/controllability/controllability_splits_v2_1/ctrl_target_plans.json`
- `data/controllability/controllability_splits_v2_1/ctrl_target_plans.robust.json`

## Why The NLI-First Path Was Replaced

The original controllability gold-label path treated diagnosis selection as an entailment-ranking problem over candidate hypotheses. That was useful for experimentation, but it was not robust enough to be the canonical labelling path for the frozen controllability artefacts.

The practical failure mode was simple:

- most cases did not clear the NLI confidence thresholds cleanly
- many labels had to come from ranked rescue rather than threshold-clearing entailment
- the output taxonomy could drift away from the benchmark label set

That meant the old path was reproducible, but not strong enough to justify as the benchmark-default label generator.

## Benchmark Method Used To Choose The Probe Models

Model selection was done locally on the controllability artefacts rather than by intuition alone.

### Study A label benchmark

Benchmark script:

- `scripts/dev/benchmark_diagnosis_backbones.py`

Saved benchmark summary:

- `docs/misc/diagnosis_backbone_benchmark_summary.json`

Protocol:

- frozen encoder embeddings
- one-vs-rest logistic probe
- 3-fold stratified cross-validation
- prompt text only
- honest reporting on the 16-label subset with at least 3 examples

Metrics checked:

- macro-F1
- per-label recall
- top-2 recall
- Brier score
- expected calibration error
- abstain coverage versus accuracy
- pairwise confusion on overlapping labels

Single-model results from the saved benchmark summary:

| Model | Accuracy | Macro-F1 | Top-2 Recall | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| `BiomedBERT` | 0.434 | 0.227 | 0.583 | 0.814 | 0.219 |
| `BioClinicalBERT` | 0.386 | 0.195 | 0.532 | 0.797 | 0.074 |
| `BioLinkBERT` | 0.369 | 0.185 | 0.563 | 0.800 | 0.080 |
| `DeBERTa-v3-base` | 0.315 | 0.175 | 0.447 | 0.851 | 0.076 |
| `BlueBERT` | 0.346 | 0.155 | 0.539 | 0.802 | 0.066 |

Decision:

- canonical single-model Study A golds use `BiomedBERT`
- robust Study A golds use `BiomedBERT + BioClinicalBERT`

Reason:

- `BiomedBERT` was the best raw labeler on the local taxonomy
- `BioClinicalBERT` improved the agreement-gated robustness story without replacing the stronger primary model

Agreement rates in the checked-in robust artefacts:

- merged suite: `1431 / 2543` = `56.3%`

### Study C plan benchmark

Study C plan golds are not free-form plan extraction. The current generator first predicts the underlying condition from the summary context, then renders a deterministic target plan from that condition.

That means the practical question is:

- which backbone best recovers the condition from `patient_summary + critical_entities`
- and which pair gives a useful robustness check without turning the plan path into an abstain-heavy research branch

Decision:

- canonical single-model Study C golds use `BiomedBERT`
- robust Study C golds use `BiomedBERT + BioLinkBERT`

Reason:

- `BiomedBERT` stays the best all-round single model across the controllability stack
- `BioLinkBERT` was the better secondary partner for the current Study C condition-recovery proxy

Agreement rates in the checked-in robust artefacts:

- merged suite: `124 / 142` = `87.3%`

## Current Backend Semantics

The checked-in `probe` outputs are explicitly weakly supervised.

Each probe-backed file records:

- `backend = "probe"`
- `primary_model`
- `secondary_model` where used
- `probe_meta.weak_supervision_source = "split_metadata_inferred_condition"`

The current robust outputs are not strict abstention files. On disagreement:

- agreement is recorded in `probe_meta`
- the generator currently falls back to the primary model prediction

So the robust files should be read as:

- primary prediction
- plus a recorded secondary-model agreement signal

not as pure agreement-only gold sets.

## How To Re-Test The Choice

If the label taxonomy changes, the weak labels change materially, or a new backbone is proposed, rerun the benchmark before changing the canonical models.

Minimum re-test contract:

1. Run `scripts/dev/benchmark_diagnosis_backbones.py`
2. Compare `BiomedBERT`, `BioClinicalBERT`, `BioLinkBERT`, `DeBERTa-v3-base`, and any proposed new model
3. Re-check:
   - macro-F1
   - top-2 recall
   - ECE or Brier
   - abstain curves
   - confusion on overlapping labels
4. Update the saved benchmark summary and this note if the ranking changes

## Current Commands

Run from `benchmark/runtime`.

```bash
export KMP_USE_SHM=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src

python3 scripts/studies/controllability/generate_gold_labels.py \
  --ctrl-dir data/controllability/controllability_splits_v2_1 \
  --backend probe \
  --primary-model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext

python3 scripts/studies/controllability/generate_gold_labels.py \
  --ctrl-dir data/controllability/controllability_splits_v2_1 \
  --backend probe \
  --primary-model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --secondary-model emilyalsentzer/Bio_ClinicalBERT \
  --output-name ctrl_gold_diagnosis_labels.robust.json

python3 scripts/studies/controllability/generate_gold_plans.py \
  --ctrl-dir data/controllability/controllability_splits_v2_1 \
  --backend probe \
  --primary-model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext

python3 scripts/studies/controllability/generate_gold_plans.py \
  --ctrl-dir data/controllability/controllability_splits_v2_1 \
  --backend probe \
  --primary-model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --secondary-model michiyasunaga/BioLinkBERT-base \
  --output-name ctrl_target_plans.robust.json
```
