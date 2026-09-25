# src/choicebench/clients/openai_client.py

import json

import openai
from openai import AsyncOpenAI
from openai.types.responses import Response as OpenAIResponsesResponse

from choicebench.clients.base import batch_line_failure, BaseClient
from choicebench.config.providers import MAX_RETRIES, TIMEOUT
from choicebench.clients.types import (
    BATCH_COMPLETED,
    BATCH_FAILED,
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

# Batch job statuses that mean "still processing" vs. "done, retrieve
# outputs" vs. "terminal failure" -- see clients/types.py's normalized
# BATCH_* vocabulary this collapses down to.
_BATCH_STATUS_IN_PROGRESS = {"validating", "in_progress", "finalizing", "cancelling"}
_BATCH_STATUS_FAILED = {"failed", "expired", "cancelled"}


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

    async def submit_batch(self, requests: list[ModelRequest]) -> str:
        """Submit all requests as one OpenAI Batch API job against the
        /v1/responses endpoint (matching _generate_provider_response's own
        Responses-API call), via a single uploaded JSONL input file.

        custom_id is the request's index (stringified) -- fetch_batch_results
        reassembles by this, not by output file line order (the API does
        not guarantee it matches input order).
        """
        lines = []
        for i, request in enumerate(requests):
            body: dict[str, object] = {"model": request.model_name, "input": request.payload}
            if request.temperature is not None:
                body["temperature"] = request.temperature
            if request.max_tokens is not None:
                body["max_output_tokens"] = request.max_tokens
            lines.append(json.dumps({
                "custom_id": str(i), "method": "POST", "url": "/v1/responses", "body": body,
            }))
        jsonl_bytes = ("\n".join(lines) + "\n").encode("utf-8")

        uploaded = await self.client.files.create(
            file=("batch_input.jsonl", jsonl_bytes), purpose="batch",
        )
        batch = await self.client.batches.create(
            input_file_id=uploaded.id, endpoint="/v1/responses", completion_window="24h",
        )
        return batch.id

    async def poll_batch(self, batch_id: str) -> str:
        """Return the normalized BATCH_* status for an in-flight batch job."""
        batch = await self.client.batches.retrieve(batch_id)
        if batch.status == "completed":
            return BATCH_COMPLETED
        if batch.status in _BATCH_STATUS_FAILED:
            return BATCH_FAILED
        return BATCH_IN_PROGRESS

    async def fetch_batch_results(
            self, batch_id: str, requests: list[ModelRequest],
    ) -> list[ModelResponse]:
        """Retrieve a completed batch's output (and error) files, parse each
        line, and reassemble into `requests` order via custom_id.

        Successful lines live in output_file_id, per-request failures in the
        separate error_file_id -- a line's "response"/"error" is never both
        populated (see fixtures in the test file for the exact shapes).
        """
        batch = await self.client.batches.retrieve(batch_id)
        response_by_custom_id: dict[str, ModelResponse] = {}

        if batch.output_file_id:
            content = await self.client.files.content(batch.output_file_id)
            for line in content.text.splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                custom_id = record["custom_id"]
                body = (record.get("response") or {}).get("body")
                if body is None:
                    continue
                try:
                    request = requests[int(custom_id)]
                    parsed = OpenAIResponsesResponse.model_validate(body)
                    raw_text, finish_reason, usage = self._extract_response(parsed)
                    response_by_custom_id[custom_id] = ModelResponse(
                        provider=request.provider, model_name=request.model_name,
                        status=SUCCESS_STATUS, latency_seconds=0.0, raw_text=raw_text,
                        finish_reason=finish_reason, usage=usage, error=None,
                        timestamp_utc=None,
                    )
                except Exception as exc:
                    response_by_custom_id[custom_id] = batch_line_failure(
                        requests, custom_id, str(exc), default_provider="openai",
                    )

        if batch.error_file_id:
            content = await self.client.files.content(batch.error_file_id)
            for line in content.text.splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                custom_id = record["custom_id"]
                error = record.get("error") or {}
                response_by_custom_id[custom_id] = batch_line_failure(
                    requests, custom_id, str(error.get("message", error)), default_provider="openai",
                )

        results = []
        for i, request in enumerate(requests):
            custom_id = str(i)
            response = response_by_custom_id.get(custom_id)
            if response is None:
                response = batch_line_failure(
                    requests, custom_id,
                    f"No output or error line for custom_id={custom_id!r}.",
                    default_provider="openai",
                )
            results.append(response)
        return results
