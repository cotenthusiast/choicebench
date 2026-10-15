# src/choicebench/backends/base.py
#
# Abstract interface that all inference backends must implement.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # clients.types imports config.schema, which imports the backends
    from choicebench.clients.types import ModelResponse


class BaseBackend(ABC):

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name, e.g. 'openai', 'huggingface'."""
        ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a text completion for the given prompt."""
        ...

    def score_options(self, prompt: str, options: list[str], **kwargs) -> list[float]:
        """
        Return log-probabilities for each option, in the same order as the input list.

        options must be single tokens as seen by the model's tokenizer — pass option
        labels (e.g. "A", "B", "C", "D"), not option text. Passing multi-token strings
        will raise ValueError in backends that enforce this (e.g. HuggingFaceBackend).

        Raises NotImplementedError if this backend does not support logprob scoring.
        API backends intentionally do not implement this method — it is only available
        via HuggingFaceBackend or other backends with direct model access.

        Args:
            prompt: the full prompt string, including the question and all options
            options: list of single-token option label strings, e.g. ["A", "B", "C", "D"]
            **kwargs: passed through to the backend implementation

        Returns:
            list of floats (log-probabilities), same length and order as options
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support option scoring / logprobs. "
            "Only HuggingFaceBackend (or backends with direct model access) implement "
            "score_options(). Check backend.supports_logprobs before calling."
        )

    @property
    def supports_logprobs(self) -> bool:
        """Return True if this backend supports score_options."""
        return False

    def is_async_capable(self) -> bool:
        """Return True if this backend supports the async generate_batch() path.

        Runners and run_method() branch on this: an async-capable backend is driven
        through generate_batch()/generate_single_async(), any other through the
        blocking generate().
        """
        return False

    async def generate_single_async(self, prompt: str) -> "ModelResponse":
        """Run one prompt through the async path and return a ModelResponse.

        Part of the async contract: implemented by every backend whose
        is_async_capable() is True (APIBackend, BatchAPIBackend, a batched
        HuggingFaceBackend). Failures are returned as a failure ModelResponse
        rather than raised, where the backend can classify them.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is not async-capable (is_async_capable() is False); "
            "use generate() instead."
        )

    async def generate_batch(self, prompts: list[str]) -> "list[ModelResponse]":
        """Run many prompts and return one ModelResponse per prompt, in input order.

        Part of the async contract (see generate_single_async). Backends are free
        to bound concurrency or chunk internally; callers rely only on the
        one-response-per-prompt, same-order guarantee.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is not async-capable (is_async_capable() is False); "
            "use generate() instead."
        )
