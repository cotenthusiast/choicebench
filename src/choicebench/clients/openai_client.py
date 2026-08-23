# src/choicebench/clients/openai_client.py

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


def _openai_finish_reason(response) -> str | None:
    """Provenance-only mapping of Responses-API lifecycle to finish_reason.

    Preserves exact provider truth; the single normalized case is
    max-output-token exhaustion -> 'length' (semantically identical to the
    Chat Completions 'length' used by the other clients).
    """
    status = getattr(response, "status", None)
    if status is None:
        return None
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        if reason == "max_output_tokens":
            return "length"
        if reason:
            return str(reason)
        return "incomplete"
    return str(status)


class OpenAIClient(BaseClient):
    """Async client for the OpenAI API."""

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
            provider="openai",
            model_name=model_name,
            timeout=timeout,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries,
        )
        from choicebench.config.providers import OPENAI_API_KEY
        self.client = AsyncOpenAI(
            api_key=api_key or OPENAI_API_KEY,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    async def _call_openai(self, create_kwargs: dict[str, object]):
        """Make the OpenAI Responses API call, remapping SDK exceptions to our taxonomy."""
        try:
            return await self.client.responses.create(**create_kwargs)
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

    @staticmethod
    def _extract_response(response) -> tuple[str, str | None, UsageInfo | None]:
        """Parse an OpenAI Responses API response into (raw_text, finish_reason, usage)."""
        raw_text = getattr(response, "output_text", None)
        if raw_text is None or raw_text.strip() == "":
            raise ProviderResponseError("client response is empty")

        finish_reason = _openai_finish_reason(response)

        usage = None
        usage_raw = getattr(response, "usage", None)
        if usage_raw is not None:
            prompt_tokens = getattr(usage_raw, "input_tokens", None)
            completion_tokens = getattr(usage_raw, "output_tokens", None)
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
        create_kwargs: dict[str, object] = {
            "model": request.model_name,
            "input": request.payload,
        }
        if request.temperature is not None:
            create_kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            create_kwargs["max_output_tokens"] = request.max_tokens

        response = await self._call_openai(create_kwargs)
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
