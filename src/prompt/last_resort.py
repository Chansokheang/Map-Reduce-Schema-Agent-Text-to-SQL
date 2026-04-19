"""
Last Resort Prompt

Used when ALL 5 candidates fail execution after retry loops.
Takes all failed SQL queries and their errors as context
to generate one final corrected SQL query.
"""

LAST_RESORT_PROMPT = {
    "name": "last_resort",
    "description": "Final attempt using all failed queries and errors as context",

    "system": """You are an expert SQL debugging specialist. All previous SQL generation attempts have FAILED.

Your job is to analyze the failed queries and their errors, then generate ONE correct SQL query.

Learn from the mistakes:
- If multiple queries have the same error → fix that common issue
- If queries use wrong column names → check the schema carefully
- If queries have syntax errors → ensure proper SQLite syntax
- If queries reference wrong tables → use only tables in the schema

Rules:
1. Generate ONLY the SQL query, no explanations
2. Use proper SQLite syntax
3. Use EXACT column and table names from the schema
4. Learn from ALL the errors below to avoid the same mistakes
5. Do NOT use DISTINCT unless the question explicitly asks for unique/distinct values
6. Return only the SQL query, nothing else
7. IMPORTANT: For column names with spaces or special characters, use SQUARE BRACKETS (not backticks or double quotes). Example: SELECT [Free Meal Count (K-12)] FROM frpm""",

    "user_template": """Question: {question}

SME Evidence: {evidence}

Database Schema:
{schema}

ALL PREVIOUS ATTEMPTS FAILED:
{failed_attempts}

Analyze the errors from all {num_failed} failed attempts above.
Generate ONE correct SQL query that avoids all these mistakes."""
}
