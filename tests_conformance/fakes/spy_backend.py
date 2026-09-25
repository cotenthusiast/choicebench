"""Spy/fake backend for the conformance suite.

Implements the real ``choicebench.backends.base.BaseBackend`` interface so
production runner code (``ExperimentRunner`` and subclasses) can be driven
through it unmodified. Records every ``generate()``/``score_options()``
call so tests can assert exact call counts and inspect exactly what was
sent, without ever touching a real provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from choicebench.backends.base import BaseBackend
from choicebench.clients.types import ErrorInfo, ModelResponse, FAILURE_STATUS, SUCCESS_STATUS

Responder = Callable[[str], str]


@dataclass
class SpyCall:
    call_index: int
    prompt: str
    response_text: str | None
    raised: str | None = None


@dataclass
class SpyScoreCall:
    call_index: int
    prompt: str
    options: list[str]
    scores: list[float]


class SpyBackend(BaseBackend):
    """Records every call; returns deterministic markers or scripted answers.

    Args:
        default_text: text returned by generate() when no responder/queue
            entry applies to that call. Defaults to a distinctive
            "CALL_%03d" marker per call, so call N's output is identifiable
            in a downstream field without any other configuration.
        responder: optional callable(prompt) -> str, consulted for every
            generate() call. Takes priority over default_text; still
            recorded like any other call.
        raise_on_call: optional set of 1-based call indices (matching
            call_index below) on which generate()/score_options() raises
            RuntimeError instead of returning -- for exercising a runner's
            existing try/except in _call_backend_generate without a test
            ever calling that private method directly.
        fixed_scores: default score_options() return value (dict of
            option -> logprob), used unless score_responder overrides.
        score_responder: optional callable(prompt, options) -> list[float].
        supports_logprobs: whether this backend advertises score_options()
            support (True by default, like DummyBackend, so it can serve
            PriDe/cyclic_logprob tests without extra configuration).
    """

    def __init__(
        self,
        *,
        default_text: str | None = None,
        responder: Responder | None = None,
        raise_on_call: set[int] | None = None,
        fixed_scores: dict[str, float] | None = None,
        score_responder: Callable[[str, list[str]], list[float]] | None = None,
        supports_logprobs: bool = True,
        model_name: str = "spy-model",
        provider: str = "spy-provider",
    ) -> None:
        self._default_text = default_text
        self._responder = responder
        self._raise_on_call = raise_on_call or set()
        self._fixed_scores = fixed_scores or {}
        self._score_responder = score_responder
        self._supports_logprobs = supports_logprobs
        self._model_name = model_name
        self._provider = provider
        self.calls: list[SpyCall] = []
        self.score_calls: list[SpyScoreCall] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def supports_logprobs(self) -> bool:
        return self._supports_logprobs

    def is_async_capable(self) -> bool:
        # Deliberately False: ExperimentRunner._call_backend_generate() only
        # routes through the sync generate() path when this is False. A spy
        # that claimed async capability would need generate_single_async()
        # implemented too, and every call-count assertion in this suite is
        # written against the sync path.
        return False

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def generate(self, prompt: str, **kwargs) -> str:
        call_index = len(self.calls) + 1
        if call_index in self._raise_on_call:
            self.calls.append(SpyCall(call_index, prompt, None, raised="RuntimeError"))
            raise RuntimeError(f"SpyBackend: scripted failure on call {call_index}")
        if self._responder is not None:
            text = self._responder(prompt)
        elif self._default_text is not None:
            text = self._default_text
        else:
            text = f"CALL_{call_index:03d}"
        self.calls.append(SpyCall(call_index, prompt, text))
        return text

    async def generate_batch(self, prompts: list[str]) -> list["ModelResponse"]:
        """Async batch path (ExperimentRunner.run_many_async /
        TwoStageRunner.run_many_async both call backend.generate_batch()
        directly, unconditionally -- unlike the sync run_one path, which
        gates on is_async_capable()). Shares the SAME self.calls list/index
        counter as generate(), so CALL_NNN markers stay globally sequential
        regardless of which path a test exercises.

        Scripted failures (raise_on_call) surface as a FAILURE_STATUS
        ModelResponse here (never a raised exception) -- matching how a real
        APIBackend never lets a provider-level failure propagate as a Python
        exception out of generate_batch().
        """
        responses: list[ModelResponse] = []
        for prompt in prompts:
            call_index = len(self.calls) + 1
            if call_index in self._raise_on_call:
                self.calls.append(SpyCall(call_index, prompt, None, raised="RuntimeError"))
                responses.append(ModelResponse(
                    provider=self._provider, model_name=self._model_name,
                    status=FAILURE_STATUS, latency_seconds=0.0,
                    error=ErrorInfo("SpyBackendScriptedFailure", f"scripted failure on call {call_index}", False, "spy"),
                ))
                continue
            if self._responder is not None:
                text = self._responder(prompt)
            elif self._default_text is not None:
                text = self._default_text
            else:
                text = f"CALL_{call_index:03d}"
            self.calls.append(SpyCall(call_index, prompt, text))
            responses.append(ModelResponse(
                provider=self._provider, model_name=self._model_name,
                status=SUCCESS_STATUS, latency_seconds=0.0, raw_text=text,
            ))
        return responses

    def score_options(self, prompt: str, options: list[str], **kwargs) -> list[float]:
        if not options:
            raise ValueError("options must not be empty.")
        call_index = len(self.score_calls) + 1
        if self._score_responder is not None:
            scores = list(self._score_responder(prompt, options))
        else:
            scores = [self._fixed_scores.get(opt, -5.0) for opt in options]
        self.score_calls.append(SpyScoreCall(call_index, prompt, list(options), scores))
        return scores
