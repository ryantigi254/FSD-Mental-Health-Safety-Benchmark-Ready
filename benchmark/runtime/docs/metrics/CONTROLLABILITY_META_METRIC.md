# Controllability as a meta-metric

Controllability is the next layer on top of the clinician-ready study metrics and the newly added invariance workflow.

It should be treated as a **meta-metric**, not as a replacement for the canonical Study A / B / C metrics.

## Why controllability sits above the base studies

The benchmark's base metrics already measure study-specific behaviour:

- Study A: faithfulness / instruction-sensitive reasoning
- Study B: sycophancy under explicit pressure
- Study C: longitudinal memory / drift

Controllability asks a second-order question:

> how strongly does the model's behaviour move when we apply an explicit control signal, and how stable is that movement under harmless prompt variation?

That makes it a meta-metric over the existing observables rather than a new standalone study.

## Operational definition

For a study-specific behavioural score `B` on case `s`:

- `B_s(ctrl_off)`: baseline behaviour without the control signal
- `B_s(ctrl_on)`: behaviour with the control signal present

Define the per-case controllability effect:

- `delta_C_s = B_s(ctrl_on) - B_s(ctrl_off)`

Then aggregate at study level with a robust summary such as the median:

- `C_study = median_s(delta_C_s)`

Bootstrap over evaluation units (IDs / conversations / cases) with at least `1000` resamples, following the same paired bootstrap semantics used by the invariance comparison code.

## Mapping onto the existing studies

### Study B single-turn

This is the strongest first anchor for controllability.

- Behavioural score `B`: `sycophancy_probability` or injected agreement rate
- Control signal: incorrect opinion wording / intensity / ordering
- Recommended first comparison: base injected prompt vs mild / moderate / strong injected variants

### Study B multi-turn

- Behavioural score `B`: `turn_of_flip` or flip probability by a fixed turn
- Control signal: pressure schedule, pressure tone, clinician pushback style
- Recommended first comparison: earlier vs later pressure spike, with paired ToF deltas

### Study A

- Behavioural score `B`: `faithfulness_gap` or `step_f1`
- Control signal: instruction phrasing, direct-vs-reasoned framing, explicit answer-format directives
- Recommended first comparison: instruction rephrasing variants over the sampled invariance split

### Study C

- Behavioural score `B`: `entity_recall_t10` or `knowledge_conflict_rate`
- Control signal: what to preserve in summaries, recall emphasis, summary framing
- Recommended first comparison: summary wording variants over the sampled invariance split

## Why invariance is the right substrate

The new invariance flow already gives the required machinery:

- deterministic sampled split manifests
- paired cache comparison
- bootstrap confidence intervals
- metric-aligned evaluation units
- persona / risk / age coverage summaries

Controllability should reuse that machinery rather than creating a separate sampling stack.

In practice, the next implementation step is:

1. keep the current invariance split roots as the base evaluation subset
2. add explicit **control-signal variants** (pressure intensity, instruction wording, ordering)
3. compute per-case `delta_C`
4. report study-level controllability plus robustness-of-controllability under paraphrase variants

## Recommended next implementation order

1. **Study B single-turn controllability first**
   - highest signal
   - lowest implementation ambiguity
   - directly aligned with the benchmark's pressure-resistance framing

2. **Study B multi-turn controllability second**
   - use `pressure_schedule` and `pressure_style`
   - measure ToF sensitivity to schedule/tone changes

3. **Study A instruction controllability third**
   - use instruction phrasing changes
   - track `faithfulness_gap` and `step_f1`

4. **Study C memory controllability fourth**
   - use summary framing variants
   - track recall preservation and contradiction stability

## Proposed implementation artefacts

Recommended next additions:

- `scripts/evaluation/run_controllability_comparison.py`
- `scripts/evaluation/summarize_invariance_results.py`
- `notebooks/invariance/controllability_analysis.ipynb`
- `notebooks/invariance/invariance_analysis.ipynb`
- `metric-results/{model-id}/controllability/...`

The comparison script should mirror `run_invariance_comparison.py`, but compute:

- per-case `delta_C`
- study-level median / mean controllability
- paired bootstrap CI
- controllability robustness under semantic paraphrase

## Interpretation guidance

Controllability has two distinct axes:

1. **Responsiveness**
   - does the model move when the control signal changes?

2. **Robustness**
   - does that movement stay consistent under semantically harmless rewording?

That separation matches the reasoning-failure framing in:

- *Large-Language-Model Reasoning Failures*. arXiv:2602.06176, 2026. Available at `https://arxiv.org/pdf/2602.06176`.

Use that paper's distinction between **application-specific** and **robustness** failures when writing model failure cards:

- failure to resist clinician/user pressure -> application-specific controllability failure
- large controllability swings under harmless paraphrase -> robustness failure

## Current repo status

Implemented now:

- frozen `v5` invariance sampled split root
- dedicated invariance generation runners
- paired invariance comparison with fail-closed pairing

Not yet implemented:

- explicit controllability comparison runner
- controllability aggregate JSON outputs
- controllability notebook / reporting cards

So the next engineering task is not new sampling; it is building the controllability layer **on top of** the invariance stack that now exists.
