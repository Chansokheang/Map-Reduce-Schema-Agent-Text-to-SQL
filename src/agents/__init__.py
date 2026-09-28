"""
Map-Reduce Schema Agent Module

Implements the "Map-Reduce" Schema Agent pattern:
- Manager: Agentic Decomposition - breaks down NL query into sub-tasks
- Workers: Parallel Verification - verify schema relevance for each sub-task
"""

from .manager import SchemaManager, DecomposedQuery
from .worker import SchemaWorker, VerificationResult

__all__ = [
    "SchemaManager",
    "DecomposedQuery",
    "SchemaWorker",
    "VerificationResult",
]
