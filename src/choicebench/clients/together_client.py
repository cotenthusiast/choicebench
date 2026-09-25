# src/choicebench/clients/together_client.py

import json
import tempfile
from pathlib import Path

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from together import AsyncTogether

from choicebench.clients.base import batch_line_failure, BaseClient, map_api_status_error
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
    ProviderTimeoutError,
)

_TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# Together's native SDK's batch job status vocabulary (distinct from the
# OpenAI-compatible chat-completions surface this client's synchronous path
# uses -- Together's Batch API has no OpenAI-compatible surface at all, so
# it goes through the separate `together` package instead).
_BATCH_STATUS_FAILED = {"FAILED", "EXPIRED", "CANCELLED"}


class TogetherAIClient(BaseClient):
    """Async client for the Together AI API (OpenAI-compatible endpoint).

    Batch methods (submit_batch/poll_batch/fetch_batch_results) use a
    separate AsyncTogether client instead -- Together's native Batch API
    (client.batches / client.files.upload) is not exposed through the
    OpenAI-compatible endpoint the synchronous path above uses.
    """

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
        resolved_api_key = api_key or TOGETHER_API_KEY
        self.client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=base_url or _TOGETHER_BASE_URL,
            timeout=timeout,
            max_retries=0,
        )
        self._batch_client = AsyncTogether(api_key=resolved_api_key)

    async def _call_together(self, create_kwargs: dict[str, object]):
        """Make the Together chat-completions call, remapping SDK exceptions to our taxonomy."""
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
        """Parse a Together chat-completions response into (raw_text, finish_reason, usage)."""
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

        response = await self._call_together(create_kwargs)
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

    async def submit_batch(self, requests: list[ModelRequest]) -> str:
        """Submit all requests as one Together Batch API job against the
        /v1/chat/completions endpoint (matching _generate_provider_response's
        own chat-completions call).

        Together's files.upload() takes a file PATH, not raw bytes -- the
        JSONL is written to a temp file, uploaded, then removed regardless
        of outcome. custom_id is the request's index (stringified) --
        fetch_batch_results reassembles by this (batch results are
        explicitly documented as arbitrary-order).
        """
        lines = []
        for i, request in enumerate(requests):
            body: dict[str, object] = {
                "model": request.model_name,
                "messages": [{"role": "user", "content": request.payload}],
            }
            if request.temperature is not None:
                body["temperature"] = request.temperature
            if request.max_tokens is not None:
                body["max_tokens"] = request.max_tokens
            if request.seed is not None:
                body["seed"] = request.seed
            lines.append(json.dumps({
                "custom_id": str(i), "method": "POST", "url": "/v1/chat/completions", "body": body,
            }))
        jsonl_text = "\n".join(lines) + "\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
        ) as f:
            f.write(jsonl_text)
            temp_path = f.name
        try:
            uploaded = await self._batch_client.files.upload(file=temp_path, purpose="batch-api")
            created = await self._batch_client.batches.create(
                input_file_id=uploaded.id, endpoint="/v1/chat/completions",
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)
        return created.job.id

    async def poll_batch(self, batch_id: str) -> str:
        """Return the normalized BATCH_* status for an in-flight batch job."""
        job = await self._batch_client.batches.retrieve(batch_id)
        if job.status == "COMPLETED":
            return BATCH_COMPLETED
        if job.status in _BATCH_STATUS_FAILED:
            return BATCH_FAILED
        return BATCH_IN_PROGRESS

    async def fetch_batch_results(
            self, batch_id: str, requests: list[ModelRequest],
    ) -> list[ModelResponse]:
        """Retrieve a completed batch's output (and error) files, parse each
        line, and reassemble into `requests` order via custom_id.

        Successful lines live in output_file_id, per-request failures in
        the separate error_file_id (mirroring OpenAIClient's own batch
        format -- Together's BatchJob carries the identical two-file
        fields, and its Batch API is explicitly modeled on OpenAI's).
        """
        job = await self._batch_client.batches.retrieve(batch_id)
        response_by_custom_id: dict[str, ModelResponse] = {}

        if job.output_file_id:
            content = await self._batch_client.files.content(job.output_file_id)
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
                    parsed = ChatCompletion.model_validate(body)
                    raw_text, finish_reason, usage = self._extract_response(parsed)
                    response_by_custom_id[custom_id] = ModelResponse(
                        provider=request.provider, model_name=request.model_name,
                        status=SUCCESS_STATUS, latency_seconds=0.0, raw_text=raw_text,
                        finish_reason=finish_reason, usage=usage, error=None,
                        timestamp_utc=None,
                    )
                except Exception as exc:
                    response_by_custom_id[custom_id] = batch_line_failure(
                        requests, custom_id, str(exc), default_provider="together",
                    )

        if job.error_file_id:
            content = await self._batch_client.files.content(job.error_file_id)
            for line in content.text.splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                custom_id = record["custom_id"]
                error = record.get("error") or {}
                response_by_custom_id[custom_id] = batch_line_failure(
                    requests, custom_id, str(error.get("message", error)), default_provider="together",
                )

        results = []
        for i, request in enumerate(requests):
            custom_id = str(i)
            response = response_by_custom_id.get(custom_id)
            if response is None:
                response = batch_line_failure(
                    requests, custom_id,
                    f"No output or error line for custom_id={custom_id!r}.",
                    default_provider="together",
                )
            results.append(response)
        return results
