# src/mcq_eval/clients/together_client.py

import openai
from openai import AsyncOpenAI

from mcq_eval.clients.base import BaseClient
from mcq_eval.clients.types import (
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
_TOP_LOGPROBS = 5
# Assistant prefill injected when request_logprobs=True.  By continuing from
# this prefix the model's first generated token is almost always a bare letter
# (A/B/C/D), so letter logprobs appear in top_logprobs at position 0.
_LOGPROB_PREFILL = "The answer is "


class TogetherAIClient(BaseClient):
    """Async client for the Together AI API (OpenAI-compatible endpoint).

    Supports optional per-token log-probability output via the
    ``request_logprobs`` field on ``ModelRequest``.  When enabled, an
    assistant prefill is added so the first generated token is a choice letter,
    the top-20 token log-probabilities for that position are stored in
    ``ModelResponse.logprobs``, and ``raw_text`` begins with the chosen letter.
    """

    def __init__(
        self,
        model_name: str,
        timeout: int = 30,
        concurrency_limit: int = 10,
        max_retries: int = 3,
        min_delay_seconds: float = 0.0,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            provider="together",
            model_name=model_name,
            timeout=timeout,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries,
            min_delay_seconds=min_delay_seconds,
        )
        from mcq_eval.config.models import TOGETHER_API_KEY
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
        use_logprobs = getattr(request, "request_logprobs", False)
        messages: list[dict] = [{"role": "user", "content": request.payload}]
        if use_logprobs:
            # Assistant prefill: model generates starting after this prefix so
            # the first token is a letter whose logprob we can directly read.
            messages.append({"role": "assistant", "content": _LOGPROB_PREFILL})

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

        if use_logprobs:
            create_kwargs["logprobs"] = True
            create_kwargs["top_logprobs"] = _TOP_LOGPROBS

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
        logprobs = None

        choices = getattr(response, "choices", None)
        if choices:
            first_choice = choices[0]
            message_obj = getattr(first_choice, "message", None)
            raw_text = getattr(message_obj, "content", None)
            finish_reason = getattr(first_choice, "finish_reason", None)

            if use_logprobs:
                lp_obj = getattr(first_choice, "logprobs", None)
                if lp_obj is not None:
                    content_lp = getattr(lp_obj, "content", None) or []
                    if content_lp:
                        # Standard OpenAI-format: list of token objects
                        logprobs = [
                            {
                                "token": entry.token,
                                "logprob": entry.logprob,
                                "top_logprobs": [
                                    {"token": tl.token, "logprob": tl.logprob}
                                    for tl in (getattr(entry, "top_logprobs", None) or [])
                                ],
                            }
                            for entry in content_lp
                        ]
                    else:
                        # Together AI non-standard format: parallel arrays where
                        # top_logprobs is a list of {token: logprob} dicts per position
                        tokens = getattr(lp_obj, "tokens", None) or []
                        token_logprobs = getattr(lp_obj, "token_logprobs", None) or []
                        top_logprobs_list = getattr(lp_obj, "top_logprobs", None) or []
                        logprobs = [
                            {
                                "token": tok,
                                "logprob": lp,
                                "top_logprobs": [
                                    {"token": k, "logprob": v}
                                    for k, v in (tlp.items() if isinstance(tlp, dict) else [])
                                ],
                            }
                            for tok, lp, tlp in zip(tokens, token_logprobs, top_logprobs_list)
                        ]

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
            metadata=request.metadata,
            raw_text=raw_text,
            finish_reason=finish_reason,
            usage=usage,
            error=None,
            timestamp_utc=None,
            logprobs=logprobs,
        )
