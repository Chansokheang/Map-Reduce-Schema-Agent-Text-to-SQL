"""
QA-SQL: Query Augmentation to SQL

A pipeline for converting natural language queries to SQL using:
- Map-Reduce Schema Agent for query decomposition
- Multiple candidate generation strategies
- LLM-based candidate selection
"""

from .pipeline import QASQLPipeline, PipelineResult
from .utils import Config, LLMClient

__version__ = "0.1.0"

__all__ = [
    "QASQLPipeline",
    "PipelineResult",
    "Config",
    "LLMClient",
]
