# src/choicebench/clients/anthropic_client.py

import anthropic
from anthropic import AsyncAnthropic

from choicebench.clients.base import BaseClient
from choicebench.config.providers import MAX_RETRIES, MAX_TOKENS, TIMEOUT
from choicebench.clients.types import (
    ModelRequest,
    ModelResponse,
    UsageInfo,
    SUCCESS_STATUS,
    ProviderResponseError,
    ProviderCallError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderConfigurationError,
)


class AnthropicClient(BaseClient):
    """Async client for the Anthropic API."""

    def __init__(
        self,
        model_name: str,
        timeout: int = TIMEOUT,
        concurrency_limit: int = 10,
        max_retries: int = MAX_RETRIES,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            provider="anthropic",
            model_name=model_name,
            timeout=timeout,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries,
        )
        from choicebench.config.providers import ANTHROPIC_API_KEY
        self.client = AsyncAnthropic(
            api_key=api_key or ANTHROPIC_API_KEY,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    async def _generate_provider_response(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        create_kwargs: dict[str, object] = {
            "model": request.model_name,
            "messages": [{"role": "user", "content": request.payload}],
            # max_tokens is required by the Anthropic API; fall back to MAX_TOKENS
            # if the request doesn't specify one.
            "max_tokens": request.max_tokens if request.max_tokens is not None else MAX_TOKENS,
        }
        if request.temperature is not None:
            create_kwargs["temperature"] = request.temperature

        try:
            response = await self.client.messages.create(**create_kwargs)
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderCallError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            message = str(exc)

            if exc.status_code == 429:
                raise ProviderRateLimitError(message) from exc

            if exc.status_code in {400, 401, 403, 404, 422}:
                raise ProviderConfigurationError(message) from exc

            raise ProviderCallError(message) from exc

        content = getattr(response, "content", None)
        raw_text = None
        if content:
            raw_text = getattr(content[0], "text", None)

        if raw_text is None or raw_text.strip() == "":
            raise ProviderResponseError("client response is empty")

        finish_reason = getattr(response, "stop_reason", None)

        usage = None
        usage_raw = getattr(response, "usage", None)
        if usage_raw is not None:
            prompt_tokens = getattr(usage_raw, "input_tokens", None)
            completion_tokens = getattr(usage_raw, "output_tokens", None)

            usage = UsageInfo(
                prompt_tokens=prompt_tokens or 0,
                completion_tokens=completion_tokens or 0,
                total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
            )

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
