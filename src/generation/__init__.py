"""
Candidate Generation Module

Generates SQL candidates using 5 context-aware strategies:
1. Full Schema - Complete database schema
2. SME Metadata - Schema with evidence/hints from domain experts
3. Minimal Profile - Schema with basic column descriptions
4. Focused Schema - Relevant tables only (from Map-Reduce Agent)
5. Full Profile - Focused schema + descriptions + evidence
"""

from .candidate_generator import (
    CandidateGenerator,
    SQLCandidate,
)
from .prompt_builder import PromptBuilder, ContextStrategy

__all__ = [
    "CandidateGenerator",
    "SQLCandidate",
    "ContextStrategy",
    "PromptBuilder",
]
