# src/choicebench/clients/deepinfra_client.py

import json

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

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

_DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

# DeepInfra's Batch API is fully OpenAI-SDK-compatible -- both the
# synchronous chat-completions path above and the batch methods below use
# the plain `openai` package pointed at DeepInfra's base_url, no separate
# provider SDK needed (unlike Together, whose Batch API is not exposed
# through its OpenAI-compatible endpoint).
_BATCH_STATUS_FAILED = {"failed", "expired", "cancelled"}


class DeepInfraClient(BaseClient):
    """Async client for the DeepInfra API (OpenAI-compatible endpoint).

    Used for Batch API execution of the Llama-3.1-8B-Instruct model
    (synchronous execution for this model goes through OpenRouter with
    upstream_provider=deepinfra pinning instead -- see openrouter_client.py
    -- this client exists specifically because DeepInfra's Batch API is
    not reachable through OpenRouter's proxy).
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
            provider="deepinfra",
            model_name=model_name,
            timeout=timeout,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries,
        )
        from choicebench.config.providers import DEEPINFRA_API_KEY
        self.client = AsyncOpenAI(
            api_key=api_key or DEEPINFRA_API_KEY,
            base_url=base_url or _DEEPINFRA_BASE_URL,
            timeout=timeout,
            max_retries=0,
        )

    async def _call_deepinfra(self, create_kwargs: dict[str, object]):
        """Make the DeepInfra chat-completions call, remapping SDK exceptions to our taxonomy."""
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
        """Parse a DeepInfra chat-completions response into (raw_text, finish_reason, usage)."""
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

        response = await self._call_deepinfra(create_kwargs)
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
        """Submit all requests as one DeepInfra Batch API job against the
        /v1/chat/completions endpoint (matching _generate_provider_response's
        own chat-completions call). Fully OpenAI-SDK-compatible: same
        files.create(purpose="batch") + batches.create() shape OpenAIClient
        uses, just pointed at DeepInfra's base_url and chat-completions
        instead of responses.

        custom_id is the request's index (stringified) -- fetch_batch_results
        reassembles by this, not by output file line order.
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
            lines.append(json.dumps({
                "custom_id": str(i), "method": "POST", "url": "/v1/chat/completions", "body": body,
            }))
        jsonl_bytes = ("\n".join(lines) + "\n").encode("utf-8")

        uploaded = await self.client.files.create(
            file=("batch_input.jsonl", jsonl_bytes), purpose="batch",
        )
        batch = await self.client.batches.create(
            input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h",
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
        """Retrieve a completed batch's output file, parse each line, and
        reassemble into `requests` order via custom_id.

        DeepInfra's docs show a single output file where a failed line
        carries a populated "error" and null "response" (rather than
        OpenAI's separate error_file_id) -- both are handled here regardless
        of which file (output_file_id and/or error_file_id, if DeepInfra
        does provide one) a given line came from.
        """
        batch = await self.client.batches.retrieve(batch_id)
        response_by_custom_id: dict[str, ModelResponse] = {}

        for file_id in filter(None, [batch.output_file_id, getattr(batch, "error_file_id", None)]):
            content = await self.client.files.content(file_id)
            for line in content.text.splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                custom_id = record["custom_id"]
                error = record.get("error")
                if error is not None:
                    response_by_custom_id[custom_id] = batch_line_failure(
                        requests, custom_id, str(error.get("message", error)),
                        default_provider="deepinfra",
                    )
                    continue
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
                        requests, custom_id, str(exc), default_provider="deepinfra",
                    )

        results = []
        for i, request in enumerate(requests):
            custom_id = str(i)
            response = response_by_custom_id.get(custom_id)
            if response is None:
                response = batch_line_failure(
                    requests, custom_id,
                    f"No output or error line for custom_id={custom_id!r}.",
                    default_provider="deepinfra",
                )
            results.append(response)
        return results
