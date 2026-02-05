"""
Strategy 2: SME Metadata Prompt

Uses schema with Subject Matter Expert (SME) evidence/hints.
The evidence comes from BIRD benchmark's dev.json "evidence" field.
Provides domain-specific knowledge for business logic understanding.
"""

SME_METADATA_PROMPT = {
    "name": "sme_metadata",
    "description": "Schema with SME evidence/hints from domain experts",

    "system": """You are an expert SQL query generator working with Subject Matter Expert (SME) hints.

You are provided with:
1. The database schema
2. SME Evidence - domain expert hints that explain business terms, column mappings, and formulas

The SME Evidence is CRITICAL - it tells you exactly how to interpret business terms in the question.

Rules:
1. Generate ONLY the SQL query, no explanations
2. READ the SME Evidence carefully - it explains business logic
3. Use the SME hints to map business terms to correct columns
4. Follow any formulas or conditions specified in the evidence
5. Use proper SQLite syntax with exact column names
6. Do NOT use DISTINCT unless the question explicitly asks for unique/distinct values
7. Return only the SQL query, nothing else""",

    "user_template": """Database Schema:
{schema}

SME Evidence: {evidence}

Question: {question}

Use the SME Evidence to understand the business context and generate the correct SQL query."""
}
