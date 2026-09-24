# src/choicebench/clients/openrouter_client.py

"""Async client for OpenRouter's OpenAI-compatible endpoint.

Generic provider/model gateway: any OpenRouter-routed model works through
this one client. Which upstream provider serves a given model, and whether
OpenRouter is allowed to silently fall back to a different one, are pure
config choices (ModelConfig.upstream_provider / .allow_fallbacks) -- no
model- or provider-specific branching lives here. A caller that never sets
upstream_provider gets OpenRouter's normal automatic routing/fallback
behavior unchanged.
"""

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

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(BaseClient):
    """Async client for the OpenRouter API (OpenAI-compatible endpoint)."""

    def __init__(
        self,
        model_name: str,
        timeout: int = TIMEOUT,
        concurrency_limit: int = 10,
        max_retries: int = MAX_RETRIES,
        api_key: str | None = None,
        base_url: str | None = None,
        upstream_provider: str | None = None,
        allow_fallbacks: bool = True,
    ) -> None:
        super().__init__(
            provider="openrouter",
            model_name=model_name,
            timeout=timeout,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries,
        )
        from choicebench.config.providers import OPENROUTER_API_KEY
        self.client = AsyncOpenAI(
            api_key=api_key or OPENROUTER_API_KEY,
            base_url=base_url or _OPENROUTER_BASE_URL,
            timeout=timeout,
            max_retries=0,
        )
        self._upstream_provider = upstream_provider
        self._allow_fallbacks = allow_fallbacks

    def _provider_routing_extra_body(self) -> dict[str, object] | None:
        """Build OpenRouter's `provider` routing block, if pinning is configured.

        Returns None (no extra_body sent at all) when upstream_provider is
        unset, preserving OpenRouter's default automatic routing for callers
        that don't care which upstream serves the request.
        """
        if self._upstream_provider is None:
            return None
        return {
            "provider": {
                "order": [self._upstream_provider],
                "allow_fallbacks": self._allow_fallbacks,
            }
        }

    async def _call_openrouter(self, create_kwargs: dict[str, object]):
        """Make the OpenRouter chat-completions call, remapping SDK exceptions to our taxonomy."""
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
        """Parse an OpenRouter chat-completions response into (raw_text, finish_reason, usage)."""
        raw_text = None
        finish_reason = None

        choices = getattr(response, "choices", None)
        if choices:
            first_choice = choices[0]
            message_obj = getattr(first_choice, "message", None)
            raw_text = getattr(message_obj, "content", None)
            finish_reason = getattr(first_choice, "finish_reason", None)

        if raw_text is None or raw_text.strip() == "":
            raise ProviderResponseError("client response is empty")

        usage = None
        usage_raw = getattr(response, "usage", None)
        if usage_raw is not None:
            prompt_tokens = getattr(usage_raw, "prompt_tokens", None)
            completion_tokens = getattr(usage_raw, "completion_tokens", None)
            total_tokens = getattr(usage_raw, "total_tokens", None)

            usage = UsageInfo(
                prompt_tokens=prompt_tokens or 0,
                completion_tokens=completion_tokens or 0,
                total_tokens=(
                    total_tokens
                    if total_tokens is not None
                    else (prompt_tokens or 0) + (completion_tokens or 0)
                ),
            )

        return raw_text, finish_reason, usage

    async def _generate_provider_response(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        messages: list[dict] = [{"role": "user", "content": request.payload}]

        create_kwargs: dict[str, object] = {
            "model": request.model_name,
            "messages": messages,
        }

        if request.temperature is not None:
            create_kwargs["temperature"] = request.temperature

        if request.max_tokens is not None:
            create_kwargs["max_tokens"] = request.max_tokens

        if request.seed is not None:
            create_kwargs["seed"] = request.seed

        extra_body = self._provider_routing_extra_body()
        if extra_body is not None:
            create_kwargs["extra_body"] = extra_body

        response = await self._call_openrouter(create_kwargs)
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
            logprobs=None,
        )
