"""
LLM Client Wrapper

Provides unified interface for LLM interactions across all modules.
Supports Anthropic (Claude), OpenAI, and Ollama providers.
"""

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import requests


def load_api_key(key_name: str = "ANTHROPIC_API_KEY") -> str:
    """Load API key from environment or .env file.

    Args:
        key_name: Name of the environment variable (default: ANTHROPIC_API_KEY)

    Returns:
        API key string or None if not found
    """
    api_key = os.environ.get(key_name)
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
                    if line.startswith(f"{key_name}="):
                        return line.split("=", 1)[1].strip().strip('"\'')

    return None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
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
        pass


class AnthropicClient(BaseLLMClient):
    """LLM client using Anthropic API."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        api_key: str = None
    ):
        """
        Initialize the Anthropic client.

        Args:
            model: Model identifier
            api_key: Optional API key (uses env var if not provided)
        """
        from anthropic import Anthropic

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
        """Get a completion from Claude."""
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


class OpenAIClient(BaseLLMClient):
    """LLM client using OpenAI API with rate limit handling."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = None,
        max_retries: int = 5,
        base_delay: float = 1.0
    ):
        """
        Initialize the OpenAI client.

        Args:
            model: Model identifier (e.g., "gpt-4o", "gpt-4o-mini", "gpt-4-turbo")
            api_key: Optional API key (uses env var if not provided)
            max_retries: Maximum number of retries for rate limit errors
            base_delay: Base delay in seconds for exponential backoff
        """
        from openai import OpenAI

        self.model = model
        self.api_key = api_key or load_api_key("OPENAI_API_KEY")
        self.max_retries = max_retries
        self.base_delay = base_delay

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it via:\n"
                "  1. Environment variable: export OPENAI_API_KEY='your-key'\n"
                "  2. .env file with: OPENAI_API_KEY=your-key"
            )

        self.client = OpenAI(api_key=self.api_key)

    def complete(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.0
    ) -> str:
        """Get a completion from OpenAI with retry logic for rate limits."""
        from openai import RateLimitError, APIError

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content

            except RateLimitError as e:
                last_error = e
                # Extract wait time from error message if available
                wait_time = self.base_delay * (2 ** attempt)  # Exponential backoff

                # Try to parse suggested wait time from error message
                error_msg = str(e)
                if "Please try again in" in error_msg:
                    try:
                        import re
                        match = re.search(r'try again in (\d+(?:\.\d+)?)(ms|s)', error_msg)
                        if match:
                            value = float(match.group(1))
                            unit = match.group(2)
                            if unit == 'ms':
                                wait_time = max(value / 1000 + 0.5, wait_time)
                            else:
                                wait_time = max(value + 0.5, wait_time)
                    except:
                        pass

                if attempt < self.max_retries - 1:
                    print(f"  Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 2}/{self.max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"OpenAI rate limit exceeded after {self.max_retries} retries: {e}"
                    )

            except APIError as e:
                # For other API errors, retry with backoff
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.base_delay * (2 ** attempt)
                    print(f"  API error, waiting {wait_time:.1f}s before retry {attempt + 2}/{self.max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise

        raise last_error


class OllamaClient(BaseLLMClient):
    """LLM client using Ollama local server."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the Ollama client.

        Args:
            model: Model name (e.g., "llama3.2", "mistral", "codellama")
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"

        # Verify connection
        self._verify_connection()

    def _verify_connection(self):
        """Verify Ollama server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running:\n"
                "  1. Install Ollama: https://ollama.ai\n"
                "  2. Start the server: ollama serve\n"
                "  3. Pull your model: ollama pull llama3.2"
            )
        except requests.exceptions.Timeout:
            raise ConnectionError(
                f"Timeout connecting to Ollama at {self.base_url}"
            )

    def complete(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.0
    ) -> str:
        """Get a completion from Ollama."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }

        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=120  # 2 minute timeout for generation
            )
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")


# Keep LLMClient as an alias for backwards compatibility
class LLMClient(AnthropicClient):
    """
    Unified LLM client wrapper using Anthropic API.

    This is maintained for backwards compatibility.
    For new code, use create_llm_client() factory function.
    """
    pass


def create_llm_client(
    provider: Literal["anthropic", "openai", "ollama"] = "anthropic",
    model: str = None,
    api_key: str = None,
    ollama_base_url: str = "http://localhost:11434"
) -> BaseLLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider: LLM provider ("anthropic", "openai", or "ollama")
        model: Model name (provider-specific)
        api_key: API key (for Anthropic or OpenAI)
        ollama_base_url: Ollama server URL (only for Ollama)

    Returns:
        Configured LLM client instance
    """
    if provider == "anthropic":
        default_model = "claude-3-5-haiku-20241022"
        return AnthropicClient(
            model=model or default_model,
            api_key=api_key
        )
    elif provider == "openai":
        default_model = "gpt-4o-mini"
        return OpenAIClient(
            model=model or default_model,
            api_key=api_key
        )
    elif provider == "ollama":
        default_model = "llama3.2"
        return OllamaClient(
            model=model or default_model,
            base_url=ollama_base_url
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            "Supported providers: 'anthropic', 'openai', 'ollama'"
        )
