# src/choicebench/clients/gemini_client.py

from google import genai
from google.genai import types, errors

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


class GeminiClient(BaseClient):
    """Async client for the Google Gemini API."""

    def __init__(
        self,
        model_name: str,
        timeout: int = TIMEOUT,
        concurrency_limit: int = 10,
        max_retries: int = MAX_RETRIES,
        min_delay_seconds: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            provider="gemini",
            model_name=model_name,
            timeout=timeout,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries,
            min_delay_seconds=min_delay_seconds,
        )
        from choicebench.config.providers import GEMINI_API_KEY
        self.client = genai.Client(api_key=api_key or GEMINI_API_KEY)

    async def _generate_provider_response(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        try:
            response = await self.client.aio.models.generate_content(
                model=request.model_name,
                contents=request.payload,
                config=types.GenerateContentConfig(
                    temperature=request.temperature,
                    max_output_tokens=request.max_tokens,
                ),
            )
        except errors.APIError as exc:
            message = exc.message or str(exc)

            if exc.code == 429:
                raise ProviderRateLimitError(message) from exc

            if exc.code in {400, 401, 403, 404}:
                raise ProviderConfigurationError(message) from exc

            if exc.code in {408, 504}:
                raise ProviderTimeoutError(message) from exc

            raise ProviderCallError(message) from exc

        raw_text = getattr(response, "text", None)
        if raw_text is None or raw_text.strip() == "":
            raise ProviderResponseError("client response is empty")

        finish_reason = None
        candidates = getattr(response, "candidates", None)
        if candidates:
            candidate_finish_reason = getattr(candidates[0], "finish_reason", None)
            if candidate_finish_reason is not None:
                finish_reason = (
                    getattr(candidate_finish_reason, "name", None)
                    or getattr(candidate_finish_reason, "value", None)
                    or str(candidate_finish_reason)
                )

        usage = None
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is not None:
            prompt_tokens = getattr(usage_metadata, "prompt_token_count", None)
            total_tokens = getattr(usage_metadata, "total_token_count", None)

            completion_tokens = getattr(
                usage_metadata, "candidates_token_count", None
            )
            if completion_tokens is None:
                completion_tokens = getattr(
                    usage_metadata, "response_token_count", None
                )

            if completion_tokens is None and (
                prompt_tokens is not None and total_tokens is not None
            ):
                completion_tokens = max(total_tokens - prompt_tokens, 0)

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
            latency_seconds=0.0,  # handled in generate()
            raw_text=raw_text,
            finish_reason=finish_reason,
            usage=usage,
            error=None,
            timestamp_utc=None,  # handled in generate()
        )
