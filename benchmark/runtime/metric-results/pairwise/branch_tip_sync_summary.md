Fetched and rebuilt from these remote branch tips on **2026-04-28**:
- `origin/codex/worker-system-all-studies-port` at `bfa22fe0` from **2026-04-17**
- `origin/metric-invariance` at `c7ee5bfd` from **2026-04-27**
- `origin/controllability-v2.1` at `a218fd23` from **2026-04-22**
- `origin/v6.1-parent-data` at `cd6b16f7` from **2026-04-01**

### Table 1 - Main raw

| Model | study_a | study_a_bias | study_b | study_b_mt | study_c |
|---|---|---|---|---|---|
| deepseek-r1-lmstudio | ✅ | ✅ | ✅ | ✅ | ✅ |
| gpt-oss-20b | ✅ | ✅ | ✅ | ✅ | ✅ |
| piaget-8b-local | ✅ | ✅ | ✅ | ✅ | ✅ |
| psych-qwen-32b-local | ✅ | ✅ | ✅ | ❌ | ❌ |
| psyche-r1-local | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen3-lmstudio | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwq | ✅ | ✅ | ❌ | ❌ | ❌ |
| medgemma-lmstudio | ✅ | ✅ | ❌ | ❌ | ❌ |
| qwen3.5-27b-distilled | ✅ | ✅ | ❌ | ❌ | ❌ |
| psyllm-lmstudio | ✅ | ✅ | ✅ | ✅ | ✅ |

### Table 2 - Controlability

| Model | ctrl_a | ctrl_a_bias | ctrl_b | ctrl_b_mt | ctrl_c |
|---|---|---|---|---|---|
| deepseek-r1-lmstudio | ❌ | ❌ | ❌ | ❌ | ❌ |
| gpt-oss-20b | ✅ | ✅ | ✅ | ✅ | ✅ |
| piaget-8b-local | ✅ | ✅ | ✅ | ✅ | ✅ |
| psych-qwen-32b-local | ❌ | ❌ | ❌ | ❌ | ❌ |
| psyche-r1-local | ✅ | ✅ | ❌ | ❌ | ❌ |
| qwen3-lmstudio | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwq | ❌ | ✅ | ❌ | ❌ | ❌ |
| medgemma-lmstudio | ❌ | ✅ | ❌ | ❌ | ❌ |
| qwen3.5-27b-distilled | ❌ | ❌ | ❌ | ❌ | ❌ |
| psyllm-lmstudio | ✅ | ✅ | ✅ | ✅ | ✅ |

### Table 3 - Invariance

| Model | study_a | study_a_bias | study_b | study_b_mt | study_c |
|---|---|---|---|---|---|
| deepseek-r1-lmstudio | ❌ | ❌ | ❌ | ❌ | ❌ |
| gpt-oss-20b | ✅ | ✅ | ✅ | ✅ | ✅ |
| piaget-8b-local | ✅ | ✅ | ✅ | ✅ | ✅ |
| psych-qwen-32b-local | ❌ | ✅ | ❌ | ❌ | ❌ |
| psyche-r1-local | ✅ | ✅ | ✅ | ❌ | ❌ |
| qwen3-lmstudio | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwq | ✅ | ✅ | ✅ | ✅ | ❌ |
| medgemma-lmstudio | ✅ | ✅ | ✅ | ❌ | ❌ |
| qwen3.5-27b-distilled | ✅ | ✅ | ✅ | ✅ | ✅ |
| psyllm-lmstudio | ✅ | ✅ | ✅ | ✅ | ✅ |

### Table 4 - Ctrl-invariance

| Model | study_a | study_a_bias | study_b | study_b_mt | study_c |
|---|---|---|---|---|---|
| deepseek-r1-lmstudio | ❌ | ❌ | ❌ | ❌ | ❌ |
| gpt-oss-20b | ✅ | ✅ | ✅ | ✅ | ✅ |
| piaget-8b-local | ✅ | ✅ | ✅ | ✅ | ✅ |
| psych-qwen-32b-local | ❌ | ❌ | ❌ | ❌ | ❌ |
| psyche-r1-local | ❌ | ✅ | ❌ | ❌ | ❌ |
| qwen3-lmstudio | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwq | ✅ | ✅ | ✅ | ✅ | ❌ |
| medgemma-lmstudio | ✅ | ✅ | ❌ | ❌ | ❌ |
| qwen3.5-27b-distilled | ❌ | ❌ | ❌ | ❌ | ❌ |
| psyllm-lmstudio | ✅ | ✅ | ✅ | ✅ | ✅ |

### Table 5 - Reverse

| Model | study_a | study_a_bias | study_b | study_b_mt | study_c |
|---|---|---|---|---|---|
| deepseek-r1-lmstudio | ❌ | ❌ | ❌ | ❌ | ❌ |
| gpt-oss-20b | ❌ | ❌ | ❌ | ❌ | ❌ |
| piaget-8b-local | ❌ | ❌ | ❌ | ❌ | ❌ |
| psych-qwen-32b-local | ❌ | ❌ | ❌ | ❌ | ❌ |
| psyche-r1-local | ❌ | ❌ | ❌ | ❌ | ❌ |
| qwen3-lmstudio | ❌ | ❌ | ❌ | ❌ | ❌ |
| qwq | ❌ | ❌ | ❌ | ❌ | ❌ |
| medgemma-lmstudio | ❌ | ❌ | ❌ | ❌ | ❌ |
| qwen3.5-27b-distilled | ❌ | ❌ | ❌ | ❌ | ❌ |
| psyllm-lmstudio | ❌ | ❌ | ❌ | ❌ | ❌ |

Two small notes:
- `ctrl-invariance` normalises both `results_ctrl_invariance/` and `results_invariance_ctrl/` into one local root.
- `reverse` stays empty because none of the four branch tips expose a dedicated reverse-results lane.
