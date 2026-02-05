"""
SQL Generation Prompts Module

Contains 5 context-aware prompts for SQL generation:
1. Full Schema - Complete database schema
2. SME Metadata - Subject Matter Expert annotations
3. Minimal Profile - Basic column descriptions
4. Focused Schema - Relevant tables only (from Map-Reduce Agent)
5. Full Profile - Combined Minimal + SME
"""

from .full_schema import FULL_SCHEMA_PROMPT
from .sme_metadata import SME_METADATA_PROMPT
from .minimal_profile import MINIMAL_PROFILE_PROMPT
from .focused_schema import FOCUSED_SCHEMA_PROMPT
from .full_profile import FULL_PROFILE_PROMPT
from .judge import JUDGE_PROMPT
from .last_resort import LAST_RESORT_PROMPT

# All prompts indexed by strategy name
STRATEGY_PROMPTS = {
    "full_schema": FULL_SCHEMA_PROMPT,
    "sme_metadata": SME_METADATA_PROMPT,
    "minimal_profile": MINIMAL_PROFILE_PROMPT,
    "focused_schema": FOCUSED_SCHEMA_PROMPT,
    "full_profile": FULL_PROFILE_PROMPT,
}

__all__ = [
    "FULL_SCHEMA_PROMPT",
    "SME_METADATA_PROMPT",
    "MINIMAL_PROFILE_PROMPT",
    "FOCUSED_SCHEMA_PROMPT",
    "FULL_PROFILE_PROMPT",
    "JUDGE_PROMPT",
    "LAST_RESORT_PROMPT",
    "STRATEGY_PROMPTS",
]
