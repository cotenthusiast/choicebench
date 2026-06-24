# src/twoprompt/methods/templates/base_method.py
#
# Template for implementing a new MCQ method. Copy this file into
# src/twoprompt/methods/<your_method_name>.py and fill in the TODOs.
#
# NOTE: methods must not import clients, runners, checkpoint logic, or
# config. The only framework dependency a method needs is BaseBackend —
# everything else (provider auth, retries, caching, checkpointing, job
# orchestration) happens beneath or above this class, never inside it.
#
# NOTE: use self.backend.generate(prompt, **kwargs) -> str for plain text
# generation. Use self.backend.score_options(prompt, options) -> list[float]
# only for logprob-based methods (e.g. PriDe), and always check
# self.backend.supports_logprobs first — most backends (all API backends in
# this project) do not implement score_options() and will raise
# NotImplementedError if you call it unconditionally.

from __future__ import annotations

from typing import Any

from twoprompt.backends.base import BaseBackend

"""
Method: <Your Method Name>
---------------------------
Description: <one paragraph — what does this method do, and why?>
Reference: <paper title and citation, or "n/a" if this is not from a paper>
Backend requirements: generate only | generate + score_options
Logprob support required: yes | no
"""


class YourMethodRunner:
    """TODO: rename this class to describe your method, e.g. MyDebiasRunner."""

    def __init__(self, backend: BaseBackend, **kwargs: Any) -> None:
        """
        Args:
            backend: Any object implementing BaseBackend — APIBackend,
                HuggingFaceBackend, DummyBackend, etc. Your method must work
                with any backend that satisfies the contract it needs (check
                backend.supports_logprobs if you need score_options()).
            **kwargs: TODO — whatever per-method configuration you need
                (e.g. number of samples, a calibration set, fallback flags).
        """
        self.backend = backend
        # TODO: store any other config your method needs.

    def run(self, prompt: str, options: list[str]) -> dict:
        """TODO: implement your method's logic here.

        This is a minimal synchronous stub — adapt the signature to whatever
        shape your method needs (e.g. async def run_one(question_row, ...)
        if you want to match the existing runners' batch-processing shape).

        Returns:
            TODO — return whatever your method produces. At minimum this
            should include enough information for a caller to determine the
            model's final answer letter, e.g.:
                {"final_choice": "B", "raw_text": "...", ...}
        """
        # --- Example: plain text generation ---
        # raw_text = self.backend.generate(prompt, max_new_tokens=50)
        # final_choice = parse_letter_from(raw_text)  # your own parsing logic

        # --- Example: logprob-based scoring (only if your method needs it) ---
        # if not self.backend.supports_logprobs:
        #     raise NotImplementedError(
        #         f"{self.__class__.__name__} requires a backend with "
        #         "score_options() support (e.g. HuggingFaceBackend)."
        #     )
        # scores = self.backend.score_options(prompt, options)
        # final_choice = options[scores.index(max(scores))]

        raise NotImplementedError("TODO: implement run().")
