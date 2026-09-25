# src/choicebench/clients/anthropic_client.py

import anthropic
from anthropic import AsyncAnthropic

from choicebench.clients.base import batch_line_failure, BaseClient
from choicebench.config.providers import MAX_RETRIES, MAX_TOKENS, TIMEOUT
from choicebench.clients.types import (
    BATCH_COMPLETED,
    BATCH_IN_PROGRESS,
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

    async def _call_anthropic(self, create_kwargs: dict[str, object]):
        """Make the Anthropic SDK call, remapping SDK exceptions to our taxonomy."""
        try:
            return await self.client.messages.create(**create_kwargs)
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

    @staticmethod
    def _extract_response(response) -> tuple[str, str | None, UsageInfo | None]:
        """Parse an Anthropic Messages response into (raw_text, finish_reason, usage)."""
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

        return raw_text, finish_reason, usage

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

        response = await self._call_anthropic(create_kwargs)
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

    async def submit_batch(self, requests: list[ModelRequest]) -> str:
        """Submit all requests as one Anthropic Message Batch job.

        custom_id is the request's index (stringified) -- fetch_batch_results
        reassembles by this; results are explicitly not guaranteed to stream
        back in request order.
        """
        batch_requests = []
        for i, request in enumerate(requests):
            params: dict[str, object] = {
                "model": request.model_name,
                "messages": [{"role": "user", "content": request.payload}],
                "max_tokens": request.max_tokens if request.max_tokens is not None else MAX_TOKENS,
            }
            if request.temperature is not None:
                params["temperature"] = request.temperature
            batch_requests.append({"custom_id": str(i), "params": params})

        batch = await self.client.messages.batches.create(requests=batch_requests)
        return batch.id

    async def poll_batch(self, batch_id: str) -> str:
        """Return the normalized BATCH_* status for an in-flight batch job.

        Anthropic's job-level processing_status only ever reaches
        "in_progress" or "ended" (or "canceling", which this pipeline never
        triggers) -- there is no job-level "failed" state. Per-request
        outcomes (succeeded/errored/canceled/expired) are only known once
        the job has ended, and are handled individually in
        fetch_batch_results -- never surfaced here.
        """
        batch = await self.client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return BATCH_COMPLETED
        return BATCH_IN_PROGRESS

    async def fetch_batch_results(
            self, batch_id: str, requests: list[ModelRequest],
    ) -> list[ModelResponse]:
        """Stream a completed batch's results and reassemble into `requests`
        order via custom_id (results are not guaranteed to stream in request
        order)."""
        response_by_custom_id: dict[str, ModelResponse] = {}

        decoder = await self.client.messages.batches.results(batch_id)
        async for item in decoder:
            custom_id = item.custom_id
            result = item.result
            if result.type == "succeeded":
                try:
                    raw_text, finish_reason, usage = self._extract_response(result.message)
                    request = requests[int(custom_id)]
                    response_by_custom_id[custom_id] = ModelResponse(
                        provider=request.provider, model_name=request.model_name,
                        status=SUCCESS_STATUS, latency_seconds=0.0, raw_text=raw_text,
                        finish_reason=finish_reason, usage=usage, error=None,
                        timestamp_utc=None,
                    )
                except Exception as exc:
                    response_by_custom_id[custom_id] = batch_line_failure(
                        requests, custom_id, str(exc), default_provider="anthropic",
                    )
            else:
                response_by_custom_id[custom_id] = batch_line_failure(
                    requests, custom_id, self._batch_result_error_message(result),
                    default_provider="anthropic",
                )

        results = []
        for i, request in enumerate(requests):
            custom_id = str(i)
            response = response_by_custom_id.get(custom_id)
            if response is None:
                response = batch_line_failure(
                    requests, custom_id, f"No result for custom_id={custom_id!r}.",
                    default_provider="anthropic",
                )
            results.append(response)
        return results

    @staticmethod
    def _batch_result_error_message(result) -> str:
        """Human-readable message for a non-succeeded batch result
        (errored/canceled/expired -- see MessageBatchResult's discriminated
        union in the anthropic SDK)."""
        if result.type == "errored":
            error = getattr(result, "error", None)
            inner = getattr(error, "error", None)
            message = getattr(inner, "message", None)
            return message or str(error)
        return f"Batch request {result.type}."
