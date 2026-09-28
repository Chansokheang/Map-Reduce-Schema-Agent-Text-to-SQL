"""
LLM Client Wrapper

Provides unified interface for LLM interactions across all modules.
Supports Anthropic (Claude), OpenAI, Ollama, Claude Code Headless, Gemma,
and vLLM (OpenAI-compatible self-hosted endpoints) providers.
"""

import os
import random
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import requests


# Strips reasoning-model "thinking" blocks (Qwen3, DeepSeek-R1 etc) that
# would otherwise break downstream JSON parsers.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class AnthropicExhaustedError(Exception):
    """
    Raised when Anthropic API calls fail after all retry attempts.

    This is a *terminal* exception meant to bubble past the generic
    `except Exception` handlers in manager.py / candidate_generator.py so
    the current batch halts cleanly. Caller (run_batch) should print a
    resume hint and exit non-zero.
    """
    pass


class AnthropicRefusalError(AnthropicExhaustedError):
    """
    Raised when Anthropic persistently refuses a prompt
    (`stop_reason=refusal`). Unlike generic exhaustion, this is
    *deterministic per-prompt* — retries won't help. Callers may catch
    this specifically and fall back to a non-LLM path (heuristic
    decomposition, default scoring, etc.) so the batch can continue.
    """
    pass


# HTTP status codes considered transient (retryable).
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


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
    """LLM client using Anthropic API with fail-fast retry logic.

    Every ``complete()`` call is attempted up to ``max_attempts`` times
    (default 3 = 1 original + 2 retries). Transient failures (429 rate
    limit, 529 overloaded, 5xx server errors, timeouts) back off
    exponentially with jitter — if the server returns a ``retry-after``
    header we honor it instead. On exhaustion we raise
    ``AnthropicExhaustedError`` so the batch halts rather than silently
    producing bad SQL.
    """

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        api_key: str = None,
        max_attempts: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 30.0,
        jitter: float = 0.25,
    ):
        """
        Args:
            model: Model identifier
            api_key: Optional API key (uses env var if not provided)
            max_attempts: Total attempts per call, including the first.
            base_delay: First-retry sleep in seconds; doubled each retry.
            max_delay: Cap for any single backoff interval.
            jitter: Symmetric jitter factor (0.25 → ±25%).
        """
        from anthropic import Anthropic

        self.model = model
        self.api_key = api_key or load_api_key()
        self.max_attempts = max(1, max_attempts)
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it via:\n"
                "  1. Environment variable: export ANTHROPIC_API_KEY='your-key'\n"
                "  2. .env file with: ANTHROPIC_API_KEY=your-key"
            )

        self.client = Anthropic(api_key=self.api_key)

    def _compute_backoff(self, attempt_idx: int, retry_after_header) -> float:
        """Choose sleep duration before the next retry.

        attempt_idx is 0-indexed (0 = between attempt 1 and attempt 2).
        Respects retry-after if the server suggested one; otherwise
        exponential backoff with symmetric jitter, capped at max_delay.
        """
        if retry_after_header is not None:
            try:
                return max(0.0, min(float(retry_after_header), self.max_delay))
            except (TypeError, ValueError):
                pass

        delay = min(self.base_delay * (2 ** attempt_idx), self.max_delay)
        jitter_span = delay * self.jitter
        return max(0.0, delay + random.uniform(-jitter_span, jitter_span))

    @staticmethod
    def _classify(exc) -> tuple[bool, int | None, object]:
        """Return (is_retryable, status_code, retry_after_header)."""
        status = getattr(exc, "status_code", None)
        if status is None:
            response = getattr(exc, "response", None)
            if response is not None:
                status = getattr(response, "status_code", None)

        retry_after = None
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", None)
            if headers is not None:
                try:
                    retry_after = headers.get("retry-after")
                except AttributeError:
                    retry_after = None

        # Retryable if HTTP status is transient OR it's a connection/timeout error
        # without a status attached (e.g. anthropic.APIConnectionError).
        from anthropic import APIConnectionError, APITimeoutError

        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True, status, retry_after
        if status in _RETRYABLE_STATUS:
            return True, status, retry_after
        return False, status, retry_after

    def complete(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.0
    ) -> str:
        """Get a completion from Claude with fail-fast retry on transient errors."""
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

        last_exc = None
        last_empty_reason = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.messages.create(**kwargs)
                print(f"  [AnthropicClient] raw response (attempt {attempt}):")
                # try:
                #     print(f"    {response.model_dump_json()}")
                # except Exception:
                #     try:
                #         print(f"    {response.model_dump()}")
                #     except Exception:
                #         print(f"    {response!r}")
            except Exception as exc:
                last_exc = exc
                retryable, status, retry_after = self._classify(exc)

                if not retryable:
                    raise AnthropicExhaustedError(
                        f"Non-retryable Anthropic error "
                        f"(status={status}): {exc}"
                    ) from exc

                if attempt >= self.max_attempts:
                    break

                delay = self._compute_backoff(attempt - 1, retry_after)
                print(
                    f"  [AnthropicClient] attempt {attempt}/{self.max_attempts} "
                    f"failed (status={status}); retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                continue

            # Defensively extract text. response.content is a list of blocks
            # (text / thinking / tool_use); it can be empty, or contain only
            # non-text blocks. Naive `response.content[0].text` raises
            # IndexError / AttributeError in those cases, which previously
            # halted the whole batch.
            text_parts = []
            for block in getattr(response, "content", None) or []:
                if getattr(block, "type", None) == "text":
                    text_parts.append(getattr(block, "text", "") or "")
            text = "".join(text_parts)
            if text:
                return text

            stop_reason = getattr(response, "stop_reason", None)
            last_empty_reason = stop_reason
            if attempt >= self.max_attempts:
                break

            # If we ran out of output tokens before producing text, double
            # max_tokens for the next attempt (capped at 8192). Otherwise
            # retry with the same params — likely a transient model glitch.
            if stop_reason == "max_tokens":
                new_max = min(kwargs["max_tokens"] * 2, 8192)
                if new_max > kwargs["max_tokens"]:
                    print(
                        f"  [AnthropicClient] hit max_tokens "
                        f"({kwargs['max_tokens']}); bumping to {new_max}"
                    )
                    kwargs["max_tokens"] = new_max

            delay = self._compute_backoff(attempt - 1, None)
            print(
                f"  [AnthropicClient] empty response on attempt "
                f"{attempt}/{self.max_attempts} (stop_reason={stop_reason}); "
                f"retrying in {delay:.1f}s"
            )
            time.sleep(delay)

        # All attempts failed.
        if last_exc is None:
            # Persistent refusal → raise the specific subclass so callers
            # can fall back without halting the batch.
            if last_empty_reason == "refusal":
                raise AnthropicRefusalError(
                    f"Anthropic refused the prompt after "
                    f"{self.max_attempts} attempts (stop_reason=refusal)"
                )
            raise AnthropicExhaustedError(
                f"Anthropic returned empty text after "
                f"{self.max_attempts} attempts (stop_reason={last_empty_reason})"
            )
        raise AnthropicExhaustedError(
            f"Anthropic API exhausted after {self.max_attempts} attempts: "
            f"{last_exc}"
        ) from last_exc


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


class ClaudeCodeHeadlessClient(BaseLLMClient):
    """LLM client using Claude Code CLI via claude-code-headless package.

    Uses Claude Max subscription instead of API credits.
    Requires: pip install claude-code-headless
    Prerequisites: Claude Code CLI installed and authenticated (claude login)
    """

    def __init__(self, model: str = None):
        """
        Initialize the Claude Code Headless client.

        Args:
            model: Model identifier passed through to claude-code-headless.
                   Accepts shortcuts ("sonnet", "haiku", "opus") or full IDs
                   (e.g. "claude-sonnet-4-6", "claude-opus-4-7").
                   Defaults to "claude-sonnet-4-6" (Sonnet 4.6).
        """
        try:
            from claude_code_headless import call_claude_with_system
            self._call_claude = call_claude_with_system
        except ImportError:
            raise ImportError(
                "claude-code-headless not installed. Install it via:\n"
                "  pip install claude-code-headless\n\n"
                "Prerequisites:\n"
                "  1. Install Claude Code CLI: npm install -g @anthropic-ai/claude-code\n"
                "  2. Authenticate: claude login"
            )

        self.model = model or "claude-sonnet-4-6"

    def complete(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.0
    ) -> str:
        """Get a completion from Claude via Claude Code CLI.

        Note: max_tokens and temperature are not directly supported by
        claude-code-headless. The CLI uses its own defaults.
        """
        try:
            import random
            rate_limit = random.uniform(5, 10)
            if system_prompt:
                response = self._call_claude(
                    prompt=prompt,
                    system=system_prompt,
                    model=self.model,
                    rate_limit=rate_limit
                )
            else:
                from claude_code_headless import call_claude
                response = call_claude(prompt, model=self.model, rate_limit=rate_limit)

            return response
        except Exception as e:
            raise RuntimeError(f"Claude Code Headless error: {e}")


class GemmaClient(BaseLLMClient):
    """LLM client for the GPU-local Gemma chat-completions endpoint.

    Endpoint: {base_url}/api/v1/{model}/chat/completions
    Auth:     header `x-api-key: <api_key>`
    Payload:  OpenAI-style {"stream": bool, "messages": [...]}

    Response shape assumed to follow OpenAI chat completions:
    {"choices": [{"message": {"content": "..."}}]}
    """

    # Fallback default for the GPU-local Gemma endpoint. Resolution order:
    # explicit api_key arg > GEMMA_API_KEY env var > .env file > this default.
    DEFAULT_API_KEY = "sk-d7a20eb034c847e8994e192b40c69a61"

    def __init__(
        self,
        model: str = "gemma-4-E4B-8b-instruct",
        api_key: str = None,
        base_url: str = "http://gpu-local.sovanreach.com:9020",
        timeout: float = 120.0,
    ):
        self.model = model
        self.api_key = api_key or load_api_key("GEMMA_API_KEY") or self.DEFAULT_API_KEY
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/v1/{self.model}/chat/completions"
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "Gemma API key not found. Set it via:\n"
                "  1. Pass api_key= directly to GemmaClient()\n"
                "  2. Environment variable: export GEMMA_API_KEY='your-key'\n"
                "  3. .env file with: GEMMA_API_KEY=your-key"
            )

    def complete(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.0
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "stream": False,
            "messages": messages,
        }
        # Pass optional sampling knobs only if they look set; server may ignore unknown keys.
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.api_url, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Gemma API request error: {e}")
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(
                f"Gemma API returned unexpected response shape: {e}. "
                f"Raw: {response.text[:500] if 'response' in locals() else 'n/a'}"
            )


class VLLMClient(BaseLLMClient):
    """LLM client for an OpenAI-compatible self-hosted vLLM endpoint.

    Tested against https://llm.aiclab.dev (Qwen/Qwen3-14B) but works with
    any vLLM server exposing /v1/chat/completions. Uses the OpenAI Python
    SDK with a custom ``base_url`` so retry/streaming/etc. come for free.

    Reasoning-model handling:
      - Qwen3 emits ``<think>...</think>`` blocks before the answer. The
        pipeline's downstream JSON parsers can't tolerate that, so we
        request ``enable_thinking: False`` via ``chat_template_kwargs``
        (Qwen3 convention exposed by vLLM) AND strip any ``<think>`` block
        that slips through defensively.
    """

    def __init__(
        self,
        model: str = "qwen3-14b",
        api_key: str = None,
        base_url: str = "https://llm.aiclab.dev",
        max_retries: int = 5,
        base_delay: float = 1.0,
        timeout: float = 120.0,
        enable_thinking: bool = False,
    ):
        from openai import OpenAI

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or load_api_key("VLLM_API_KEY")
        self.max_retries = max(1, max_retries)
        self.base_delay = base_delay
        self.timeout = timeout
        self.enable_thinking = enable_thinking

        if not self.api_key:
            raise ValueError(
                "VLLM_API_KEY not found. Set it via:\n"
                "  1. Environment variable: export VLLM_API_KEY='your-key'\n"
                "  2. .env file with: VLLM_API_KEY=your-key\n"
                "  3. Pass --vllm-api-key on the CLI"
            )

        # vLLM exposes a standard /v1 prefix; OpenAI SDK appends the rest.
        # Override the User-Agent: the default `OpenAI/Python ...` UA is
        # blocked by the aiclab.dev WAF (verified empirically — 403 "Your
        # request was blocked"). Any non-OpenAI UA passes through.
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=f"{self.base_url}/v1",
            timeout=self.timeout,
            default_headers={"User-Agent": "qa-sql-pipeline/1.0"},
        )

    def complete(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        from openai import RateLimitError, APIError

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Disable Qwen3-style reasoning by default so JSON-parsing callers
        # don't choke on <think>...</think> prefixes. vLLM passes
        # `chat_template_kwargs` straight to the tokenizer's chat template.
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=extra_body,
                )
                content = response.choices[0].message.content or ""
                # Defensive strip — if `enable_thinking=False` is ignored
                # by the server, kill the <think> block locally so
                # downstream JSON parsers still work.
                return _THINK_BLOCK_RE.sub("", content).lstrip()

            except RateLimitError as e:
                last_error = e
                wait_time = self.base_delay * (2 ** attempt)
                if attempt < self.max_retries - 1:
                    print(
                        f"  [VLLMClient] rate-limited; "
                        f"waiting {wait_time:.1f}s before retry "
                        f"{attempt + 2}/{self.max_retries}"
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"vLLM rate limit exceeded after "
                        f"{self.max_retries} retries: {e}"
                    )

            except APIError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.base_delay * (2 ** attempt)
                    print(
                        f"  [VLLMClient] API error; "
                        f"waiting {wait_time:.1f}s before retry "
                        f"{attempt + 2}/{self.max_retries}"
                    )
                    time.sleep(wait_time)
                else:
                    raise

        raise last_error


# Keep LLMClient as an alias for backwards compatibility
class LLMClient(AnthropicClient):
    """
    Unified LLM client wrapper using Anthropic API.

    This is maintained for backwards compatibility.
    For new code, use create_llm_client() factory function.
    """
    pass


def create_llm_client(
    provider: Literal["anthropic", "openai", "ollama", "headless", "gemma", "vllm"] = "anthropic",
    model: str = None,
    api_key: str = None,
    ollama_base_url: str = "http://localhost:11434",
    gemma_base_url: str = "http://gpu-local.sovanreach.com:9020",
    vllm_base_url: str = "https://llm.aiclab.dev",
    vllm_enable_thinking: bool = False,
) -> BaseLLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider: LLM provider ("anthropic", "openai", "ollama", "headless", "gemma", or "vllm")
        model: Model name (provider-specific)
        api_key: API key (for Anthropic, OpenAI, Gemma, or vLLM)
        ollama_base_url: Ollama server URL (only for Ollama)
        gemma_base_url: Gemma endpoint base URL (only for Gemma)
        vllm_base_url: vLLM OpenAI-compatible endpoint (only for vLLM)
        vllm_enable_thinking: leave Qwen3 "thinking" output on (default False)

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
    elif provider == "headless":
        return ClaudeCodeHeadlessClient(model=model)
    elif provider == "gemma":
        default_model = "gemma-4-E4B-8b-instruct"
        return GemmaClient(
            model=model or default_model,
            api_key=api_key,
            base_url=gemma_base_url,
        )
    elif provider == "vllm":
        default_model = "qwen3-14b"
        return VLLMClient(
            model=model or default_model,
            api_key=api_key,
            base_url=vllm_base_url,
            enable_thinking=vllm_enable_thinking,
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            "Supported providers: 'anthropic', 'openai', 'ollama', 'headless', 'gemma', 'vllm'"
        )
