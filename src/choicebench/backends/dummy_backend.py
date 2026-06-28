# src/choicebench/backends/dummy_backend.py

from choicebench.backends.base import BaseBackend

_DEFAULT_TEXT = "The answer is A."
_DEFAULT_SCORES: dict[str, float] = {"A": -0.1, "B": -1.5, "C": -2.0, "D": -2.5}


class DummyBackend(BaseBackend):
    """Always returns a fixed answer without calling any model or client.

    For exercising runner/method logic in tests without API calls or loaded
    weights. supports_logprobs is True so PriDe-style logprob consumers can
    be exercised in tests too — score_options() returns fixed scores rather
    than raising NotImplementedError like a real API backend would.
    """

    def __init__(
        self,
        fixed_text: str = _DEFAULT_TEXT,
        fixed_scores: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            fixed_text: Text returned by every generate() call.
            fixed_scores: Scores returned by score_options(). Keys are option
                labels; any requested option not in the dict gets -5.0.
        """
        self._fixed_text = fixed_text
        self._fixed_scores = fixed_scores if fixed_scores is not None else dict(_DEFAULT_SCORES)

    @property
    def model_name(self) -> str:
        return "dummy"

    @property
    def provider(self) -> str:
        return "dummy"

    @property
    def supports_logprobs(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs) -> str:
        return self._fixed_text

    def score_options(self, prompt: str, options: list[str], **kwargs) -> list[float]:
        if not options:
            raise ValueError("options must not be empty.")
        return [self._fixed_scores.get(opt, -5.0) for opt in options]
