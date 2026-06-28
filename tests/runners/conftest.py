# tests/runners/conftest.py

import pytest

from choicebench.backends.base import BaseBackend


class MockBackend(BaseBackend):
    """Test double: queues fixed responses for backend.generate() and score_options().

    Each queued entry is either a value (returned) or an Exception instance
    (raised). For generate(), entries are str | Exception. For score_options(),
    entries are list[float] | Exception.

    ExperimentRunner._call_backend_generate catches exceptions from generate()
    and turns them into a failure ModelResponse (result["error_type"] becomes
    the exception's class name). score_options() exceptions propagate as-is.
    """

    def __init__(
            self,
            responses: list[str | Exception] | None = None,
            score_responses: list[list[float] | Exception] | None = None,
            provider: str = "openai",
            model_name: str = "gpt-4.1-mini",
            supports_logprobs: bool = False,
    ) -> None:
        self._provider = provider
        self._model_name = model_name
        self._supports_logprobs = supports_logprobs
        self._responses = list(responses) if responses else []
        self._call_count = 0
        self._score_responses = list(score_responses) if score_responses else []
        self._score_call_count = 0
        self.requests_received: list[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def supports_logprobs(self) -> bool:
        return self._supports_logprobs

    def generate(self, prompt: str, **kwargs) -> str:
        self.requests_received.append(prompt)
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
            self._call_count += 1
            if isinstance(response, Exception):
                raise response
            return response
        raise RuntimeError("MockBackend has no more queued responses.")

    def score_options(self, prompt: str, options: list[str], **kwargs) -> list[float]:
        if self._score_call_count < len(self._score_responses):
            entry = self._score_responses[self._score_call_count]
            self._score_call_count += 1
            if isinstance(entry, Exception):
                raise entry
            return entry
        raise NotImplementedError(
            f"MockBackend: no queued score_responses for call "
            f"{self._score_call_count + 1}."
        )


@pytest.fixture
def runner_question_row() -> dict[str, object]:
    """A normalized question row with all fields runners expect."""
    return {
        "question_id": "4865890d7f0efae8",
        "subject": "computer_security",
        "question_text": "Which protocol is primarily used to securely browse websites?",
        "choice_a": "FTP",
        "choice_b": "HTTP",
        "choice_c": "HTTPS",
        "choice_d": "SMTP",
        "correct_option": "C",
        "correct_answer_text": "HTTPS",
    }


@pytest.fixture
def canonical_options() -> dict[str, str]:
    """The canonical option mapping for the sample question."""
    return {
        "A": "FTP",
        "B": "HTTP",
        "C": "HTTPS",
        "D": "SMTP",
    }
