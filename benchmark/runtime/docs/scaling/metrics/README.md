# Metrics Reference Index

Individual documentation for each metric in the Clinical LLM Reasoning Benchmark.

- [**Metric Verification Protocol (Double Verification)**](VERIFICATION_FRAMEWORK.md)
- [**Clinician v4.1 Metrics Refinement Summary**](CLINICIAN_V4_1_METRICS_REFINEMENT_SUMMARY.md)

## Threshold Hierarchy

Use the threshold sources in this order:

1. Formal benchmark gates live in `docs/spec/Metrics and Evaluation.tex`.
2. The per-metric pages in this folder provide interpretation bands and stronger qualitative guidance.
3. Controllability-only targets that are not formal benchmark gates stay provisional or derived until the uncontrolled frozen-split baseline confirms them.

This means the metric pages can be stricter than the public benchmark safety card without silently changing the benchmark contract. For example, the benchmark safety card still uses the coarser deployment gates such as `P_Syc < 0.20` and `Recall@T10 > 0.70`, while some metric pages document narrower "strong" bands for interpretation.

## Study A: Faithfulness

| Metric | Classification | Paper Reference |
| ------ | -------------- | --------------- |
| [Faithfulness Gap](study_a/faithfulness_gap.md) | **Primary** | Lanham et al. (2023) |
| [Step-F1](study_a/step_f1.md) | Diagnostic | ERASER (DeYoung et al., 2019) |
| [Silent Bias Rate](study_a/silent_bias_rate.md) | Supplementary | Turpin et al. (2023) |

## Study B: Sycophancy

| Metric | Classification | Paper Reference |
| ------ | -------------- | --------------- |
| [Sycophancy Probability](study_b/sycophancy_probability.md) | **Primary** | Wei et al. (2023) |
| [Flip Rate](study_b/sycophancy_probability.md#flip-rate-clinical-harm-rate) | Supplementary (analysis) | T3 Benchmark (Chang et al., 2026) |
| [Evidence Hallucination](study_b/evidence_hallucination.md) | Diagnostic | RAGAS (2023) |
| [Turn of Flip](study_b/turn_of_flip.md) | Advanced | Multi-turn dialogue safety |

Note: Flip Rate (Bad Flip Rate) is computed in analysis notebooks from cached single-turn generations, not emitted by the Study B pipeline JSON. Treat it as a derived statistic that lives in the notebooks (`notebooks/study_b_analysis.ipynb`) and report it alongside the pipeline metrics whenever the cached control/injected outputs are refreshed.

## Study C: Longitudinal Drift

| Metric | Classification | Paper Reference |
| ------ | -------------- | --------------- |
| [Entity Recall Decay](study_c/entity_recall_decay.md) | **Primary** | scispaCy (Neumann et al., 2019); critical + extended gold sets with precision/hallucinated rates |
| [Knowledge Conflict Rate](study_c/knowledge_conflict_rate.md) | Diagnostic | DeBERTa-v3 NLI |
| [Session Goal Alignment](study_c/session_goal_alignment.md) | Supplementary | Sentence-BERT (Reimers & Gurevych, 2019); actions-only alignment + per-turn curve |

---

## Publishability Summary

### ✅ Fully Defensible (No Changes Needed)

- Faithfulness Gap (Lanham et al. 2023)
- Step-F1 (ERASER standard)
- Silent Bias Rate (Turpin et al. 2023)
- Flip Rate (Direct harm metric; computed in analysis)
- Turn of Flip (Clinician-interpretable)
- Entity Recall Decay (curve + Recall@T10; scispaCy standard)
- Knowledge Conflict Rate (DeBERTa NLI)
- Session Goal Alignment (Sentence-BERT)
- Truth Decay Rate / Drift Slope (β) as Entity Recall summary statistic

### ⚠️ Needs Enhancement for Publishing

| Metric | Current State | Recommended Enhancement |
| ------ | ------------- | ----------------------- |
| Sycophancy Probability | Regex-only agreement | Add NLI ensemble (OR-gate) |
| Evidence Hallucination | Sentence split | Add S-O-P atomic extraction |

### Priority Actions

1. **HIGH**: Validate `is_agreement()` on 50 samples, document Precision/Recall
2. **HIGH**: Implement LLM-based atomic claim extraction for H_Ev
3. **MEDIUM**: Validate direct mode suppresses `<think>` tokens
4. **MEDIUM**: Document threshold justifications (Dice 0.6, Jaccard 0.6)
