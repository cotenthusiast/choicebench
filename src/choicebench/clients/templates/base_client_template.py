# src/choicebench/clients/templates/base_client_template.py
#
# Template for adding a new API provider client.
#
# WHEN TO USE THIS vs THE BACKEND TEMPLATE
#   Use this template for hosted API providers (OpenAI, Anthropic, Cohere, etc.)
#   where you make HTTP calls to a remote endpoint.
#   Use backends/templates/base_backend_template.py for local models where you
#   control the inference process (HuggingFace, vLLM, llama.cpp).
#
# WHAT YOU IMPLEMENT
#   Only _generate_provider_response(). Everything else — retry logic,
#   exponential backoff, semaphore-based concurrency limiting, min-delay
#   spacing, request/response validation — is handled by BaseClient.
#
# REGISTRATION
#   1. Copy this file to src/choicebench/clients/my_provider_client.py
#   2. Add to CLIENT_REGISTRY in src/choicebench/registry.py:
#        from choicebench.clients.my_provider_client import MyProviderClient
#        CLIENT_REGISTRY["my_provider"] = MyProviderClient
#   3. Add the API key to .env:
#        MY_PROVIDER_API_KEY=sk-...
#   4. Load it in src/choicebench/config/providers.py:
#        MY_PROVIDER_API_KEY = os.getenv("MY_PROVIDER_API_KEY")
#   5. Use in your YAML config:
#        models:
#          - model_name_or_path: my-provider/my-model
#            backend: api
#            provider: my_provider

from __future__ import annotations

from choicebench.clients.base import BaseClient
from choicebench.clients.types import (
    ModelRequest,
    ModelResponse,
    ProviderCallError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    SUCCESS_STATUS,
    UsageInfo,
)
from choicebench.config.providers import MAX_RETRIES, TIMEOUT


class YourProviderClient(BaseClient):
    # Rename this class: e.g. AnthropicClient, CohereClient.

    def __init__(
        self,
        model_name: str,
        timeout: int = TIMEOUT,
        concurrency_limit: int = 10,
        max_retries: int = MAX_RETRIES,
        min_delay_seconds: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        # Call super().__init__() with your provider name string and the model name.
        # The retry loop, semaphore, and delay logic live entirely in BaseClient.
        super().__init__(
            provider="your_provider",   # lowercase string, matches YAML and CLIENT_REGISTRY key
            model_name=model_name,
            timeout=timeout,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries,
            min_delay_seconds=min_delay_seconds,
        )
        # Initialize the provider's SDK client here.
        # Example:
        #   import your_sdk
        #   self.client = your_sdk.AsyncClient(api_key=api_key, timeout=timeout)
        self.api_key = api_key

    async def _generate_provider_response(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        """Execute a single provider API call.

        BaseClient calls this inside its retry loop with semaphore and
        min-delay enforcement already applied. You do NOT need to handle
        retry, backoff, concurrency, or timing here.

        How request fields map to typical API parameters:
            request.payload      → the prompt / messages content
            request.model_name   → model identifier string
            request.temperature  → temperature / sampling parameter
            request.max_tokens   → max_tokens / max_completion_tokens
            request.seed         → seed (if the provider supports it)

        Raise typed provider exceptions so BaseClient can categorize them:
            ProviderTimeoutError      → retryable; wraps timeout errors
            ProviderRateLimitError    → retryable; wraps 429 / quota errors
            ProviderCallError         → retryable; wraps generic API errors
            ProviderConfigurationError → NOT retried; wraps 400/401/404/422
            ProviderResponseError     → retryable in v0.1; empty/malformed response
                                      (it subclasses ProviderCallError)

        Args:
            request: Validated ModelRequest for this provider/model pair.

        Returns:
            ModelResponse with status=SUCCESS_STATUS, raw_text set,
            and latency_seconds=0.0 (BaseClient fills the real latency).
        """
        # ── 1. Build the API call arguments ─────────────────────────────────
        # call_kwargs: dict[str, object] = {
        #     "model": request.model_name,
        #     "messages": [{"role": "user", "content": request.payload}],
        #     "temperature": request.temperature,
        #     "max_tokens": request.max_tokens,
        # }
        # if request.seed is not None:
        #     call_kwargs["seed"] = request.seed

        # ── 2. Call the provider SDK ─────────────────────────────────────────
        # try:
        #     response = await self.client.messages.create(**call_kwargs)
        # except YourSDK.TimeoutError as exc:
        #     raise ProviderTimeoutError(str(exc)) from exc
        # except YourSDK.RateLimitError as exc:
        #     raise ProviderRateLimitError(str(exc)) from exc
        # except YourSDK.APIStatusError as exc:
        #     if exc.status_code in {400, 401, 403, 404}:
        #         raise ProviderConfigurationError(str(exc)) from exc
        #     raise ProviderCallError(str(exc)) from exc

        # ── 3. Extract raw_text from the response ────────────────────────────
        # raw_text = response.content[0].text   # Anthropic style
        # raw_text = response.choices[0].message.content  # OpenAI style
        # if not raw_text or not raw_text.strip():
        #     raise ProviderResponseError("Provider returned an empty response.")

        # ── 4. Build and return a ModelResponse ──────────────────────────────
        # Always pass latency_seconds=0.0 — BaseClient overwrites it.
        # return ModelResponse(
        #     provider=request.provider,
        #     model_name=request.model_name,
        #     status=SUCCESS_STATUS,
        #     latency_seconds=0.0,
        #     raw_text=raw_text,
        #     finish_reason=None,   # optional; fill if the API exposes it
        #     usage=UsageInfo(      # optional; fill if the API exposes token counts
        #         prompt_tokens=response.usage.input_tokens,
        #         completion_tokens=response.usage.output_tokens,
        #         total_tokens=response.usage.total_tokens,
        #     ),
        #     error=None,
        #     timestamp_utc=None,
        # )

        raise NotImplementedError("TODO: implement _generate_provider_response()")
