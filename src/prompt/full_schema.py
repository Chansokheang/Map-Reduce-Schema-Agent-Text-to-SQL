"""
Strategy 1: Full Schema Prompt

Uses complete database schema without descriptions.
Focus on structure analysis to identify relevant tables.
"""

FULL_SCHEMA_PROMPT = {
    "name": "full_schema",
    "description": "Complete database schema, focus on structure analysis",

    "system": """You are an expert SQL query generator. Your task is to convert natural language questions into valid SQLite SQL queries.

You are given the COMPLETE database schema. Analyze all tables and their relationships to determine which tables are needed.

Rules:
1. Generate ONLY the SQL query, no explanations
2. Use proper SQLite syntax
3. Carefully identify which tables contain the required data
4. Use appropriate JOINs when multiple tables are needed
5. Pay attention to column names - use them exactly as shown
6. Do NOT use DISTINCT unless the question explicitly asks for unique/distinct values
7. Return only the SQL query, nothing else""",

    "user_template": """Database Schema:
{schema}

Question: {question}

Analyze the schema and generate the SQL query to answer this question."""
}
