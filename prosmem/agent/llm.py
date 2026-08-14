"""LLM wrapper for any OpenAI-compatible chat-completions endpoint.

The endpoint and key come from ProsMemConfig (env: OPENAI_BASE_URL /
OPENAI_API_KEY); an optional HTTP(S) proxy can be set via config.http_proxy.
"""

from __future__ import annotations

import time

import httpx
from openai import OpenAI

from prosmem.core.config import ProsMemConfig


class LLMClient:
    """Thin chat-completions wrapper with retry and explicit close()."""

    MAX_RETRIES = 5
    RETRY_DELAY = 5  # seconds (exponential: 5, 10, 15, 20, 25)

    def __init__(self, config: ProsMemConfig) -> None:
        self.config = config

        http_client = None
        if config.http_proxy:
            http_client = httpx.Client(proxy=config.http_proxy, timeout=120.0)
        self.client = OpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url or None,
            http_client=http_client,
        )

        self.total_tokens_used = 0

    def close(self) -> None:
        """Close the underlying HTTP connection pool to release sockets/file
        descriptors. The OpenAI() instance owns an httpx.Client whose pooled
        sockets are NOT reclaimed promptly by GC; without an explicit close,
        per-sample LLMClient instances accumulate FDs and exhaust RLIMIT_NOFILE
        on long runs. Idempotent and safe to call multiple times."""
        client = getattr(self, "client", None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        response_format: dict | None = None,
    ) -> str:
        """Send chat completion request with retry on transient errors.

        `response_format` (e.g. {"type": "json_object"}) is forwarded when set —
        used by the intention encoder to force strict JSON output. Optional so
        every existing caller is unaffected."""
        resolved_model = model or self.config.main_model
        extra = {"response_format": response_format} if response_format else {}
        last_err = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=resolved_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                )
                usage = response.usage
                if usage:
                    self.total_tokens_used += usage.total_tokens
                return response.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
        raise last_err  # type: ignore[misc]
