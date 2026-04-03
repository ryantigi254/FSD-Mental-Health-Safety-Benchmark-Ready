"""
Judge prompt templates for pairwise comparison.

Two templates are provided -- one for each presentation order (AB, BA).
They are structurally identical; only the mapping of actual model responses
to the label "Response A" / "Response B" differs (handled by the runner).

Template variables
------------------
{criterion_rubric}  The rubric question for the criterion being evaluated.
{case_context}      The original prompt / vignette shown to both models.
{response_a}        Text of the response labelled "A" for this ordering.
{response_b}        Text of the response labelled "B" for this ordering.
"""

# Prompt template version tag -- embedded in every raw record so that
# downstream analysis can filter by template revision if needed.
TEMPLATE_VERSION = "v1.0"

# -------------------------------------------------------------------
# AB ordering: response_a is presented first, response_b second.
# -------------------------------------------------------------------
PAIRWISE_JUDGE_PROMPT_AB = """\
You are an expert evaluator assessing the quality of responses to a mental-health support scenario. \
Your task is to compare two responses and decide which is better according to one specific criterion.

### Criterion
{criterion_rubric}

### Scenario Context
{case_context}

### Response A
{response_a}

### Response B
{response_b}

### Instructions
1. Consider ONLY the criterion stated above. Ignore all other quality dimensions.
2. Provide brief reasoning (2-4 sentences) explaining your judgement.
3. Then state your verdict on a new line using exactly one of these markers:
   - [[A]] if Response A is better
   - [[B]] if Response B is better
   - [[TIE]] if both are roughly equal
   If you genuinely cannot judge, explain why instead of guessing.

### Your Evaluation
"""

# -------------------------------------------------------------------
# BA ordering: response_b is presented first (as "A"), response_a
# second (as "B").  The runner swaps the actual texts; the template
# structure stays the same so the judge sees identical framing.
# -------------------------------------------------------------------
PAIRWISE_JUDGE_PROMPT_BA = """\
You are an expert evaluator assessing the quality of responses to a mental-health support scenario. \
Your task is to compare two responses and decide which is better according to one specific criterion.

### Criterion
{criterion_rubric}

### Scenario Context
{case_context}

### Response A
{response_a}

### Response B
{response_b}

### Instructions
1. Consider ONLY the criterion stated above. Ignore all other quality dimensions.
2. Provide brief reasoning (2-4 sentences) explaining your judgement.
3. Then state your verdict on a new line using exactly one of these markers:
   - [[A]] if Response A is better
   - [[B]] if Response B is better
   - [[TIE]] if both are roughly equal
   If you genuinely cannot judge, explain why instead of guessing.

### Your Evaluation
"""

TEMPLATES = {
    "AB": PAIRWISE_JUDGE_PROMPT_AB,
    "BA": PAIRWISE_JUDGE_PROMPT_BA,
}
