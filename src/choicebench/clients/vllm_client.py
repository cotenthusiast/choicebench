# src/choicebench/clients/vllm_client.py

import os

import openai
from openai import AsyncOpenAI

from choicebench.clients.base import BaseClient, map_api_status_error
from choicebench.config.providers import MAX_RETRIES, TIMEOUT
from choicebench.clients.types import (
    ModelRequest,
    ModelResponse,
    UsageInfo,
    SUCCESS_STATUS,
    ProviderResponseError,
    ProviderCallError,
    ProviderTimeoutError,
)

_DEFAULT_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
_DEFAULT_API_KEY = "vllm"


class VLLMClient(BaseClient):
    """Async client for a locally-running vLLM server (OpenAI-compatible API).

    vLLM exposes an OpenAI-compatible REST API at a user-specified base URL.
    ChoiceBench renders prompts before calling the backend, so this client
    tokenizes that exact string with the checkpoint tokenizer and sends the
    resulting IDs to /v1/completions. This avoids vLLM applying a chat template.

    The model's HF tokenizer and generation config are loaded lazily from
    model_name_or_path (normally a local snapshot on Kelvin2).

    API key: reads VLLM_API_KEY from the environment if set; otherwise defaults
    to the string "vllm" (vLLM accepts any non-empty value).

    All retry / backoff / concurrency-limit behaviour is inherited from BaseClient
    unchanged — same as OpenAIClient/TogetherAIClient/AnthropicClient. This client
    is generate-only (no logprob/score_options support), mirroring TogetherAIClient.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: int = TIMEOUT,
        concurrency_limit: int = 10,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        super().__init__(
            provider="vllm",
            model_name=model_name,
            timeout=timeout,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries,
        )
        self._api_key = os.getenv("VLLM_API_KEY", _DEFAULT_API_KEY)
        self._base_url = base_url
        self._tokenizer = None
        self._stop_token_ids = None
        self.client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    async def _call_vllm(self, create_kwargs: dict[str, object]):
        """Make a vLLM completion call, remapping SDK exceptions."""
        try:
            return await self.client.completions.create(**create_kwargs)
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except openai.APIConnectionError as exc:
            raise ProviderCallError(str(exc)) from exc
        except openai.APIStatusError as exc:
            raise map_api_status_error(exc) from exc

    @staticmethod
    def _extract_response(response) -> tuple[str, str | None, UsageInfo | None]:
        """Parse a vLLM completion into (raw_text, finish_reason, usage)."""
        if not response.choices:
            raise ProviderResponseError("vLLM returned no choices")

        raw_text = response.choices[0].text
        if raw_text is None or raw_text.strip() == "":
            raise ProviderResponseError("client response is empty")

        usage = None
        if response.usage is not None:
            usage = UsageInfo(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return raw_text, response.choices[0].finish_reason, usage

    def _load_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer, GenerationConfig
        except ImportError as exc:
            raise ImportError(
                "transformers is required to tokenize prompts for the vLLM backend."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        generation_config = GenerationConfig.from_pretrained(self.model_name)
        eos_token_id = generation_config.eos_token_id
        self._stop_token_ids = (
            list(eos_token_id) if isinstance(eos_token_id, (list, tuple))
            else [eos_token_id] if eos_token_id is not None else []
        )

    async def _generate_provider_response(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self._load_tokenizer()
        prompt_token_ids = self._tokenizer(
            request.payload, add_special_tokens=True
        )["input_ids"]
        create_kwargs: dict[str, object] = {
            "model": request.model_name,
            "prompt": prompt_token_ids,
            "extra_body": {
                "stop_token_ids": self._stop_token_ids,
                "skip_special_tokens": True,
            },
        }
        if request.temperature is not None:
            create_kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            create_kwargs["max_tokens"] = request.max_tokens

        response = await self._call_vllm(create_kwargs)
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
