# src/choicebench/clients/vllm_client.py

import os

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

_DEFAULT_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_API_KEY = "vllm"
# Cover A–D plus common space-prefixed token variants (" A", " B", …)
_TOP_LOGPROBS = 20


class VLLMClient(BaseClient):
    """Async client for a locally-running vLLM server (OpenAI-compatible API).

    vLLM exposes an OpenAI-compatible REST API at a user-specified base URL.
    This client uses the Chat Completions endpoint (/v1/chat/completions) rather
    than the newer Responses API, which vLLM does not implement.

    score_options_async() is implemented via logprobs=True / top_logprobs=20 on
    the Chat Completions endpoint — this returns the next-token distribution at
    the first generated position, which is what MCQ scoring methods need.
    vLLM also supports prompt_logprobs via extra_body, but that approach requires
    one call per option token and more complex response parsing; see
    audits/implementation-notes.md for a comparison.

    Tested against: vLLM 0.6.x, openai SDK 1.x.

    API key: reads VLLM_API_KEY from the environment if set; otherwise defaults
    to the string "vllm" (vLLM accepts any non-empty value).

    All retry / backoff / concurrency-limit behaviour is inherited from BaseClient
    unchanged.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: int = TIMEOUT,
        concurrency_limit: int = 10,
        max_retries: int = MAX_RETRIES,
        min_delay_seconds: float = 0.0,
    ) -> None:
        super().__init__(
            provider="vllm",
            model_name=model_name,
            timeout=timeout,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries,
            min_delay_seconds=min_delay_seconds,
        )
        self._api_key = os.getenv("VLLM_API_KEY", _DEFAULT_API_KEY)
        self._base_url = base_url
        self.client = AsyncOpenAI(
            api_key=self._api_key,
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
        }
        if request.temperature is not None:
            create_kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            create_kwargs["max_tokens"] = request.max_tokens

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

        if not response.choices:
            raise ProviderResponseError("vLLM returned no choices")

        raw_text = response.choices[0].message.content
        if raw_text is None or raw_text.strip() == "":
            raise ProviderResponseError("client response is empty")

        usage = None
        if response.usage is not None:
            usage = UsageInfo(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return ModelResponse(
            provider=request.provider,
            model_name=request.model_name,
            status=SUCCESS_STATUS,
            latency_seconds=0.0,
            raw_text=raw_text,
            finish_reason=response.choices[0].finish_reason,
            usage=usage,
            error=None,
            timestamp_utc=None,
        )

    async def score_options_async(
        self,
        prompt: str,
        options: list[str],
    ) -> dict[str, float]:
        """Return log probabilities for each option token.

        Issues a single chat completions call with max_tokens=1,
        logprobs=True, and top_logprobs=20 to get the next-token
        distribution at the first generated position. Options not
        present in the top-20 receive a floor value of -100.0.

        Each option should be a single token as seen by the model's
        tokenizer (e.g. "A", "B", "C", "D"). Multi-token options will
        silently receive the floor value.

        A fresh AsyncOpenAI client is created for each call to avoid
        event-loop conflicts when this method is invoked from a separate
        thread (as APIBackend.score_options() does).
        """
        # Fresh client per call — avoids httpx connection-pool event-loop
        # affinity conflicts when called from APIBackend.score_options() via
        # a ThreadPoolExecutor thread.
        scoring_client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self.timeout,
            max_retries=0,
        )
        try:
            response = await scoring_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                logprobs=True,
                top_logprobs=_TOP_LOGPROBS,
                temperature=0.0,
            )
        finally:
            await scoring_client.aclose()

        if not response.choices or not response.choices[0].logprobs:
            raise ProviderResponseError("vLLM returned no logprob data")

        content_logprobs = response.choices[0].logprobs.content
        if not content_logprobs:
            raise ProviderResponseError("vLLM returned empty logprob content")

        top_lp = content_logprobs[0].top_logprobs

        # Build a lookup that covers both exact tokens ("A") and space-prefixed
        # variants (" A") that some tokenizers emit. Exact token takes priority.
        logprob_map: dict[str, float] = {}
        for entry in top_lp:
            if entry.token not in logprob_map:
                logprob_map[entry.token] = entry.logprob
            stripped = entry.token.strip()
            if stripped and stripped not in logprob_map:
                logprob_map[stripped] = entry.logprob

        return {opt: logprob_map.get(opt, -100.0) for opt in options}
