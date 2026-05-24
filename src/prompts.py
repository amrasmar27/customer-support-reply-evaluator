def build_prompt(row):

    rubric_text = "\n".join([
        f"{k}. {v}"
        for k, v in row["rubric"].items()
    ])

    prompt = f"""
You are an expert evaluator for customer support replies.
Evaluate the quality of the submission.
Return ONLY valid JSON.

Format:
{{
    "score": integer from 0 to 4, where 0 is the worst and 4 is the best,
    "rationale": "short explanation"
}}

Task:
{row["task"]}

Reference:
{row["reference"]}

Submission:
{row["submission"]}

Rubric:
{rubric_text}
"""

    return prompt.strip()