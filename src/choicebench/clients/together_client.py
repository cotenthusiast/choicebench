# src/choicebench/clients/together_client.py

import openai
from openai import AsyncOpenAI

from choicebench.clients.base import BaseClient
from choicebench.config.providers import MAX_RETRIES, TIMEOUT
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

_TOGETHER_BASE_URL = "https://api.together.xyz/v1"


class TogetherAIClient(BaseClient):
    """Async client for the Together AI API (OpenAI-compatible endpoint)."""

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
            provider="together",
            model_name=model_name,
            timeout=timeout,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries,
        )
        from choicebench.config.providers import TOGETHER_API_KEY
        self.client = AsyncOpenAI(
            api_key=api_key or TOGETHER_API_KEY,
            base_url=base_url or _TOGETHER_BASE_URL,
            timeout=timeout,
            max_retries=0,
        )

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

        try:
            response = await self.client.chat.completions.create(**create_kwargs)
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except openai.APIConnectionError as exc:
            raise ProviderCallError(str(exc)) from exc
        except openai.APIStatusError as exc:
            message = str(exc)

            if exc.status_code == 429:
                raise ProviderRateLimitError(message) from exc

            if exc.status_code in {400, 401, 403, 404, 422}:
                raise ProviderConfigurationError(message) from exc

            raise ProviderCallError(message) from exc

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
