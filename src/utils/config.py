"""
Configuration Module

Handles pipeline configuration and settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class Config:
    """
    Pipeline configuration.

    Contains all settings for the QA-SQL pipeline.
    """

    # Paths
    data_dir: Path = Path("data/bird_data")
    schema_dir: Path = None
    profile_dir: Path = None
    output_dir: Path = Path("output")

    # LLM Settings
    llm_model: str = "claude-sonnet-4-5-20250929"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.0

    # Schema Agent Settings
    max_workers: int = 4
    relevance_threshold: float = 0.5

    # Candidate Generation Settings
    num_candidates: int = 5
    generation_strategies: list[str] = field(default_factory=lambda: [
        "focused_schema",
        "full_schema",
        "minimal_profile",
        "sme_metadata",
        "full_profile"
    ])

    # Execution Settings
    query_timeout: float = 30.0
    max_refinement_attempts: int = 2

    def __post_init__(self):
        """Set derived paths after initialization."""
        if self.schema_dir is None:
            self.schema_dir = self.data_dir / "schemas"
        if self.profile_dir is None:
            self.profile_dir = self.data_dir / "descriptions"

    @classmethod
    def from_json(cls, path: Path) -> 'Config':
        """
        Load configuration from JSON file.

        Args:
            path: Path to JSON config file

        Returns:
            Config instance
        """
        # TODO: Implement JSON loading
        pass

    def to_json(self, path: Path):
        """
        Save configuration to JSON file.

        Args:
            path: Path to save JSON config
        """
        # TODO: Implement JSON saving
        pass

    def to_dict(self) -> dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Configuration dictionary
        """
        # TODO: Implement dict conversion
        pass
