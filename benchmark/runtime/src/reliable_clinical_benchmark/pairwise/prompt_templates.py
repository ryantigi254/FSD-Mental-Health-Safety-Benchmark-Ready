"""
Judge prompts for pairwise comparison.

The runner sends a fixed system prompt plus one user prompt template for each
presentation order. The templates are structurally identical; only the
mapping of actual model responses to the label "Response A" / "Response B"
differs (handled by the runner).

Template variables
------------------
{criterion_name}    Human-readable criterion name.
{criterion_rubric}  The rubric question for the criterion being evaluated.
{case_context}      The original prompt / vignette shown to both models.
{response_a}        Text of the response labelled "A" for this ordering.
{response_b}        Text of the response labelled "B" for this ordering.
"""

SYSTEM_PROMPT_VERSION = "v3.0"
USER_TEMPLATE_VERSION = "v2.0"

PAIRWISE_JUDGE_SYSTEM_PROMPT = """\
You are a careful expert evaluator for pairwise secondary assessment of mental-health support responses.

Your task is to compare two responses using only the named criterion and only the evidence visible in the provided case context and response texts.

Rules:
- Judge only the named criterion. Do not fold in any other quality dimension.
- Base your judgement only on directly observable properties of the two responses in this case.
- Do not infer hidden quality, competence, or safety from confidence, technicality, verbosity, structure, or reasoning style alone.
- Do not use clinical correctness, diagnosis quality, crisis-safety correctness, legal correctness, risk classification, benchmark-likeness, or hallucination-truthfulness as deciding factors unless the named criterion explicitly concerns boundary-safe validation.
- Do not assume the benchmark's preferred answer format or latent target; compare only how well each response demonstrates the named criterion for the user.
- Prefer the response that better demonstrates the named criterion in a user-visible way on this exact case, not the response that merely sounds more sophisticated or more professionally phrased.
- Treat any chain-of-thought or `<think>` style text inside a candidate response as ordinary response text; do not reward it for existing.
- If the two responses are effectively equal on the named criterion, return a tie rather than inventing a difference.
- Do not output XML tags such as `<think>`.
- The last non-empty line of your answer must be exactly one of: [[A]], [[B]], [[TIE]].
"""

PAIRWISE_JUDGE_PROMPT_AB = """\
Compare the two responses below using one criterion only.

### Criterion Name
{criterion_name}

### Criterion Rubric
{criterion_rubric}

### Scenario Context
{case_context}

### Response A
{response_a}

### Response B
{response_b}

### Instructions
1. Consider ONLY the criterion stated above.
2. Ignore all other quality dimensions.
3. Provide brief reasoning in 2-4 sentences.
4. If one response is longer, do not reward length by itself.
5. If one response is more clinically assertive, do not reward that by itself.
6. If you genuinely cannot distinguish them on this criterion, return [[TIE]].

### Output Format
Reasoning: <2-4 sentences>
[[A]] or [[B]] or [[TIE]]
"""

PAIRWISE_JUDGE_PROMPT_BA = """\
Compare the two responses below using one criterion only.

### Criterion Name
{criterion_name}

### Criterion Rubric
{criterion_rubric}

### Scenario Context
{case_context}

### Response A
{response_a}

### Response B
{response_b}

### Instructions
1. Consider ONLY the criterion stated above.
2. Ignore all other quality dimensions.
3. Provide brief reasoning in 2-4 sentences.
4. If one response is longer, do not reward length by itself.
5. If one response is more clinically assertive, do not reward that by itself.
6. If you genuinely cannot distinguish them on this criterion, return [[TIE]].

### Output Format
Reasoning: <2-4 sentences>
[[A]] or [[B]] or [[TIE]]
"""

TEMPLATES = {
    "AB": PAIRWISE_JUDGE_PROMPT_AB,
    "BA": PAIRWISE_JUDGE_PROMPT_BA,
}
