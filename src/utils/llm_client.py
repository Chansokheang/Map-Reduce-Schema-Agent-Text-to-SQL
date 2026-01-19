"""
LLM Client Wrapper

Provides unified interface for LLM interactions across all modules.
"""

import os
from pathlib import Path
from anthropic import Anthropic


def load_api_key() -> str:
    """Load API key from environment or .env file."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key

    # Try to load from .env file
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent / ".env",
        Path.home() / ".env",
    ]

    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ANTHROPIC_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"\'')

    return None


class LLMClient:
    """
    Unified LLM client wrapper using Anthropic API.
    """

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        api_key: str = None
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model identifier
            api_key: Optional API key (uses env var if not provided)
        """
        self.model = model
        self.api_key = api_key or load_api_key()

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it via:\n"
                "  1. Environment variable: export ANTHROPIC_API_KEY='your-key'\n"
                "  2. .env file with: ANTHROPIC_API_KEY=your-key"
            )

        self.client = Anthropic(api_key=self.api_key)

    def complete(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.0
    ) -> str:
        """
        Get a completion from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum response tokens
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if temperature > 0:
            kwargs["temperature"] = temperature

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text
