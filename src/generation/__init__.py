"""
Candidate Generation Module

Generates SQL candidates using different schema/profile combinations:
- Focused Schema: Schema filtered by relevance to query
- Full Schema: Complete database schema
- Minimal Profile: Basic column descriptions
- SME Metadata: Subject Matter Expert annotations
- Full Profile: Minimal + SME combined
"""

from .candidate_generator import (
    CandidateGenerator,
    SQLCandidate,
    GenerationStrategy,
)
from .prompt_builder import PromptBuilder

__all__ = [
    "CandidateGenerator",
    "SQLCandidate",
    "GenerationStrategy",
    "PromptBuilder",
]
