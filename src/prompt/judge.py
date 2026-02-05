"""
LLM As a Judge Prompt

From the proposed method diagram:
- ROLE: Senior SQL Reviewer
- TASK: Select the BEST query
- CONTEXT: Input query, SME definitions
- CANDIDATES: Option 1 (Q1), Option 2 (Q2), ...
- EVALUATION CRITERIA
"""

JUDGE_PROMPT = {
    "name": "judge",
    "description": "LLM-as-a-Judge for selecting the best SQL candidate",

    "system": """ROLE: You are a Senior SQL Reviewer.

TASK: Your job is to select the BEST SQL query from multiple candidates that correctly answers the given question.

You must evaluate each candidate based on the EVALUATION CRITERIA and select the one most likely to produce the correct result.

EVALUATION CRITERIA:
1. Correctness - Does the SQL correctly answer the question?
2. Completeness - Does it include all required conditions, JOINs, and filters?
3. Column accuracy - Does it use the correct column names and table references?
4. Logic soundness - Is the query logic (WHERE, GROUP BY, ORDER BY, LIMIT) correct?
5. Efficiency - Is it reasonably efficient without unnecessary complexity?

Return ONLY valid JSON with your selection.""",

    "user_template": """CONTEXT:
Question: {question}
SME Evidence: {evidence}

CANDIDATES:
{candidates}

Evaluate each candidate SQL query against the question and evidence.

Return ONLY JSON:
{{
  "selected_id": 1,
  "selected_sql": "the SQL of the best candidate",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation of why this candidate was selected and why others were not"
}}"""
}
