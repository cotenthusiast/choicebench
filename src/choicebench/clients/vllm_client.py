# src/choicebench/clients/vllm_client.py

import os

import openai
from openai import AsyncOpenAI

from choicebench.clients.base import BaseClient, map_api_status_error
from choicebench.config.providers import MAX_RETRIES, TIMEOUT
from choicebench.clients.types import (
    ModelRequest,
    ModelResponse,
    UsageInfo,
    SUCCESS_STATUS,
    ProviderResponseError,
    ProviderCallError,
    ProviderTimeoutError,
)

_DEFAULT_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_API_KEY = "vllm"


class VLLMClient(BaseClient):
    """Async client for a locally-running vLLM server (OpenAI-compatible API).

    vLLM exposes an OpenAI-compatible REST API at a user-specified base URL.
    This client uses the Chat Completions endpoint (/v1/chat/completions) rather
    than the newer Responses API, which vLLM does not implement.

    Tested against: vLLM 0.6.x, openai SDK 1.x.

    API key: reads VLLM_API_KEY from the environment if set; otherwise defaults
    to the string "vllm" (vLLM accepts any non-empty value).

    All retry / backoff / concurrency-limit behaviour is inherited from BaseClient
    unchanged — same as OpenAIClient/TogetherAIClient/AnthropicClient. This client
    is generate-only (no logprob/score_options support), mirroring TogetherAIClient.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: int = TIMEOUT,
        concurrency_limit: int = 10,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        super().__init__(
            provider="vllm",
            model_name=model_name,
            timeout=timeout,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries,
        )
        self._api_key = os.getenv("VLLM_API_KEY", _DEFAULT_API_KEY)
        self._base_url = base_url
        self.client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    async def _call_vllm(self, create_kwargs: dict[str, object]):
        """Make the vLLM chat-completions call, remapping SDK exceptions to our taxonomy."""
        try:
            return await self.client.chat.completions.create(**create_kwargs)
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except openai.APIConnectionError as exc:
            raise ProviderCallError(str(exc)) from exc
        except openai.APIStatusError as exc:
            raise map_api_status_error(exc) from exc

    @staticmethod
    def _extract_response(response) -> tuple[str, str | None, UsageInfo | None]:
        """Parse a vLLM chat-completions response into (raw_text, finish_reason, usage)."""
        if not response.choices:
            raise ProviderResponseError("vLLM returned no choices")

        raw_text = response.choices[0].message.content
        if raw_text is None or raw_text.strip() == "":
            raise ProviderResponseError("client response is empty")

        usage = None
        if response.usage is not None:
            usage = UsageInfo(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return raw_text, response.choices[0].finish_reason, usage

    async def _generate_provider_response(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        create_kwargs: dict[str, object] = {
            "model": request.model_name,
            "messages": [{"role": "user", "content": request.payload}],
        }
        if request.temperature is not None:
            create_kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            create_kwargs["max_tokens"] = request.max_tokens

        response = await self._call_vllm(create_kwargs)
        raw_text, finish_reason, usage = self._extract_response(response)

        return ModelResponse(
            provider=request.provider,
            model_name=request.model_name,
            status=SUCCESS_STATUS,
            latency_seconds=0.0,
            raw_text=raw_text,
            finish_reason=finish_reason,
            usage=usage,
            error=None,
            timestamp_utc=None,
        )
