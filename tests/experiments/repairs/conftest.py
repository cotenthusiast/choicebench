# tests/experiments/repairs/conftest.py

from __future__ import annotations

import pytest

from choicebench.clients.types import ModelResponse
from choicebench.clients.types import SUCCESS_STATUS, FAILURE_STATUS


class MockAsyncBackend:
    """Test double implementing the subset of APIBackend's interface the
    repair harness actually uses: generate_single_async(). Queues fixed
    raw-text responses (or exceptions), each returned with a positive,
    distinguishable latency_seconds so assert_fresh_call's sanity check
    passes for every queued success.
    """

    def __init__(self, responses: list[str | Exception], provider: str = "openai", model_name: str = "gpt-4.1-mini"):
        self._responses = list(responses)
        self._call_count = 0
        self._provider = provider
        self._model_name = model_name
        self.prompts_received: list[str] = []

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate_single_async(self, prompt: str) -> ModelResponse:
        self.prompts_received.append(prompt)
        if self._call_count >= len(self._responses):
            raise RuntimeError("MockAsyncBackend has no more queued responses.")
        entry = self._responses[self._call_count]
        self._call_count += 1
        if isinstance(entry, Exception):
            return ModelResponse(
                provider=self._provider,
                model_name=self._model_name,
                status=FAILURE_STATUS,
                latency_seconds=0.01,
                raw_text=None,
            )
        return ModelResponse(
            provider=self._provider,
            model_name=self._model_name,
            status=SUCCESS_STATUS,
            latency_seconds=0.42,
            raw_text=entry,
        )


@pytest.fixture
def three_option_row() -> dict:
    """A realistic repaired 3-option row (mirrors the real frozen content
    for question_id 79e8c959bbeb74a0)."""
    return {
        "question_id": "79e8c959bbeb74a0",
        "subject": "arc_challenge",
        "question_text": (
            "A toy truck rolls over a smooth surface. If the surface is "
            "covered with sand, the truck will most likely roll"
        ),
        "choice_a": "slower",
        "choice_b": "faster",
        "choice_c": "at the same speed",
        "correct_option": "A",
        "model_name": "gpt-4.1-mini",
    }


@pytest.fixture
def four_option_row() -> dict:
    """An ordinary 4-option row, NOT in the repair set — used to prove the
    harness never touches ordinary questions."""
    return {
        "question_id": "some_ordinary_4_option_question",
        "subject": "arc_challenge",
        "question_text": "An ordinary question with four real options?",
        "choice_a": "alpha",
        "choice_b": "beta",
        "choice_c": "gamma",
        "choice_d": "delta",
        "correct_option": "B",
        "model_name": "gpt-4.1-mini",
    }
