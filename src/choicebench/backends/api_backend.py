# src/choicebench/backends/api_backend.py

from __future__ import annotations

import asyncio
import concurrent.futures
from pathlib import Path

from choicebench.backends.base import BaseBackend
from choicebench.clients.base import BaseClient
from choicebench.infra.cache import CachingClientWrapper, ResponseCache
from choicebench.clients.types import ModelRequest, ModelResponse


class APIBackend(BaseBackend):
    """Wraps one of the provider clients in src/choicebench/clients/
    (OpenAIClient, GeminiClient, GroqClient, TogetherAIClient) behind the
    BaseBackend interface, so runners depend only on BaseBackend and never
    import a provider client directly.

    All generation goes through the async generate_batch() path.
    generate() raises RuntimeError — it must not be called inside a running
    event loop (i.e. from within async_main or any coroutine it drives).
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        client: BaseClient,
        cache_dir: Path,
        temperature: float,
        max_tokens: int,
        seed: int,
        concurrency_limit: int = 10,
    ) -> None:
        self._provider = provider
        self._model_name = model_name
        self._raw_client = client  # unwrapped; used for isinstance checks and score_options
        self._client = CachingClientWrapper(client, ResponseCache(cache_dir=cache_dir))
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed
        self._concurrency_limit = concurrency_limit
        # Semaphore created lazily inside a running event loop — not in __init__.
        self._semaphore: asyncio.Semaphore | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return the shared semaphore, creating it lazily on first use."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._concurrency_limit)
        return self._semaphore

    def _make_request(self, prompt: str) -> ModelRequest:
        return ModelRequest(
            provider=self._provider,
            model_name=self._model_name,
            payload=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            seed=self._seed,
            request_logprobs=False,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """Not supported in the async pipeline.

        APIBackend requires a running event loop — call generate_batch()
        from an async context instead. This method exists only to satisfy
        the BaseBackend interface; it must never be called from within a
        coroutine driven by asyncio.run().
        """
        raise RuntimeError(
            "APIBackend.generate() cannot be called inside a running event loop. "
            "Use await backend.generate_batch([prompt]) from an async runner method."
        )

    async def generate_single_async(self, prompt: str) -> ModelResponse:
        """Execute one request, respecting this backend's concurrency limit."""
        request = self._make_request(prompt)
        async with self._get_semaphore():
            return await self._client.generate(request)

    async def generate_batch(self, prompts: list[str]) -> list[ModelResponse]:
        """Execute all prompts concurrently, bounded by concurrency_limit.

        The semaphore in _get_semaphore() (and in BaseClient) controls how
        many requests are in-flight simultaneously. All prompts are submitted
        to asyncio.gather() so they race for semaphore slots — no prompt
        waits for the previous one to finish before being submitted.
        """
        return list(await asyncio.gather(
            *[self.generate_single_async(p) for p in prompts]
        ))

    def is_async_capable(self) -> bool:
        return True

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model_name

    def supports_score_options(self) -> bool:
        """Return True when the underlying client is a VLLMClient.

        Only VLLMClient exposes score_options_async(); all other provider
        clients (OpenAI, Anthropic, Gemini, …) are opaque and cannot return
        per-token log probabilities.
        """
        from choicebench.clients.vllm_client import VLLMClient
        return isinstance(self._raw_client, VLLMClient)

    @property
    def supports_logprobs(self) -> bool:
        return self.supports_score_options()

    def score_options(self, prompt: str, options: list[str], **kwargs) -> list[float]:
        """Return log probabilities for each option via the vLLM scoring API.

        Bridges sync→async by running score_options_async() in a dedicated
        thread with its own event loop. This allows synchronous callers (e.g.
        PriDeRunner) to use the vLLM logprob API without needing to be async.

        Raises NotImplementedError if the underlying client is not a VLLMClient.
        Check supports_score_options() before calling.
        """
        if not self.supports_score_options():
            raise NotImplementedError(
                "score_options() requires a VLLMClient backend. "
                "Check backend.supports_score_options() before calling."
            )
        from choicebench.clients.vllm_client import VLLMClient
        raw: VLLMClient = self._raw_client  # type: ignore[assignment]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run, raw.score_options_async(prompt, options)
            )
            logprob_dict: dict[str, float] = future.result()

        return [logprob_dict.get(opt, -100.0) for opt in options]
