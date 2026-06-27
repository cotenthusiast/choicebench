# src/mcq_eval/backends/api_backend.py

import asyncio
from pathlib import Path

from mcq_eval.backends.base import BaseBackend
from mcq_eval.clients.base import BaseClient
from mcq_eval.infra.cache import CachingClientWrapper, ResponseCache
from mcq_eval.clients.types import ModelRequest

class APIBackend(BaseBackend):
    """Wraps one of the provider clients in src/mcq_eval/clients/
    (OpenAIClient, GeminiClient, GroqClient, TogetherAIClient) behind the
    BaseBackend interface, so runners depend only on BaseBackend and never
    import a provider client directly.
    """

    def __init__(self, provider: str, model_name: str, client: BaseClient, cache_dir: Path, temperature: float, max_tokens: int, seed: int) -> None:
        self._provider = provider
        self._model_name = model_name
        self._client = CachingClientWrapper(client, ResponseCache(cache_dir = cache_dir))  # Wrap the client with caching
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed

    def generate(self, prompt: str) -> str:
        """Generate text from the model using the provided prompt."""
        request = ModelRequest(
            provider=self.provider,
            model_name=self.model_name,
            payload=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            seed=self._seed,
            request_logprobs=False,
        )
        response = asyncio.run(self._client.generate(request))
        return response.raw_text
    
    @property
    def provider(self) -> str:
        return self._provider
    
    @property
    def model_name(self) -> str:
        return self._model_name