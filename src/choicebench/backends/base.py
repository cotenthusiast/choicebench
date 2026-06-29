# src/choicebench/backends/base.py
#
# Abstract interface that all inference backends must implement.

from abc import ABC, abstractmethod

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

    def supports_score_options(self) -> bool:
        """Return True if this backend supports score_options_async() for async logprob inference.

        Overridden by APIBackend when the underlying client is VLLMClient.
        All other backends return False.
        """
        return False

    @property
    def supports_logprobs(self) -> bool:
        """Return True if this backend supports score_options."""
        return False

    def is_async_capable(self) -> bool:
        """Return True if this backend supports the async generate_batch() path."""
        return False
