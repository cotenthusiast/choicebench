# src/choicebench/backends/templates/base_backend_template.py
#
# Template for implementing a new inference backend.
#
# BACKEND vs CLIENT — WHEN TO USE WHICH
#   Backend  — direct model access. Use when you control the model weights
#              or the inference process. Examples: HuggingFace Transformers,
#              vLLM, llama.cpp, ONNX Runtime.
#   Client   — API provider. Use when you call a hosted endpoint via HTTP.
#              Examples: OpenAI, Gemini, Groq, Together AI.
#   If you're adding support for a new hosted provider, use the client template
#   instead (see clients/templates/base_client_template.py).
#
# REGISTRATION
#   Add a branch to build_backend() in scripts/run_experiment.py:
#     elif backend_type == "my_backend":
#         return MyBackend(model_config.model_name_or_path, ...)
#   Add "backend: my_backend" to the model entry in your YAML config.
#
# WHAT THE FRAMEWORK EXPECTS
#   - generate(prompt) → str: plain text output. Must be synchronous.
#   - score_options(prompt, options) → list[float]: log-probs per option.
#       Only implement this if your backend has direct token-level access.
#       Leave as-is (raises NotImplementedError) if not applicable.
#   - supports_logprobs → bool: return True only if score_options works.

from __future__ import annotations

from choicebench.backends.base import BaseBackend


class YourBackend(BaseBackend):
    # Rename this class: e.g. VLLMBackend, OnnxBackend.

    def __init__(self, model_name_or_path: str, device: str = "cuda") -> None:
        # Load the model and tokenizer here.
        # Store anything you need across calls as instance attributes.
        self._model_name = model_name_or_path
        self._device = device

        # Example: load a HuggingFace model
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # self._model = AutoModelForCausalLM.from_pretrained(
        #     model_name_or_path, device_map=device,
        # )

    @property
    def model_name(self) -> str:
        # Return the canonical model identifier. Used in output filenames.
        return self._model_name

    @property
    def provider(self) -> str:
        # Short provider string, e.g. "huggingface", "vllm", "onnx".
        # Appears in result rows but has no functional effect for backends.
        return "your_provider"

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a text completion for the given prompt.

        This is the only method required for methods that use text generation
        (direct_mcq, two_stage, cyclic_permutation). The return value is
        the raw model output string — do not parse or score it here.

        Args:
            prompt: The full prompt string to send to the model.
            **kwargs: Passed through from the runner; usually unused.

        Returns:
            Raw generated text as a single string.
        """
        raise NotImplementedError("TODO: implement generate()")

    # ── Optional: logprob scoring ────────────────────────────────────────────
    # Only implement the two methods below if your backend has access to
    # per-token log-probabilities (e.g. HuggingFace forward pass).
    # Methods like PriDe require this. API backends leave these as-is.

    @property
    def supports_logprobs(self) -> bool:
        # Return True here AND implement score_options() below.
        return False

    def score_options(self, prompt: str, options: list[str], **kwargs) -> list[float]:
        """Return log-probabilities for each option token.

        Args:
            prompt:  Full prompt string including the question and options.
            options: Single-token option labels, e.g. ["A", "B", "C", "D"].
                     Multi-token strings should raise ValueError.

        Returns:
            list[float] of log-probabilities, same length and order as options.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support logprob scoring. "
            "Set supports_logprobs = True and implement score_options() to enable it."
        )
