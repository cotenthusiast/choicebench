# src/choicebench/backends/hf_backend.py

from __future__ import annotations

import logging

from choicebench.backends.base import BaseBackend

logger = logging.getLogger(__name__)

_DEFAULT_MAX_NEW_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_DO_SAMPLE = False


class HuggingFaceBackend(BaseBackend):
    """Local inference backend using HuggingFace transformers AutoModelForCausalLM.

    Works with any causal LM loadable via from_pretrained (Qwen, Llama, etc.).
    torch and transformers are imported lazily inside load() so importing this
    module does not require them to be installed.

    This is the only backend that implements score_options() with real
    logprobs (single forward pass over the full vocabulary) — PriDe requires
    full-vocabulary logits, which API providers do not expose cleanly. API
    backends raise NotImplementedError on score_options() by design.

    Usage:
        backend = HuggingFaceBackend("Qwen/Qwen2.5-7B-Instruct")
        backend.load()
        text = backend.generate(prompt)
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        **generation_kwargs,
    ) -> None:
        """
        Args:
            model_name_or_path: HuggingFace hub ID or local model directory path.
            device: "cuda", "cpu", "mps", or "auto" (device_map=auto for
                multi-GPU). Defaults to "cuda".
            **generation_kwargs: Default overrides for generate(), e.g.
                max_new_tokens=256, temperature=0.7, do_sample=True. Any of
                these can also be passed per-call to generate().
        """
        self._model_path = model_name_or_path
        self._device = device
        self._default_generation_kwargs = generation_kwargs
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._F = None
        self._loaded = False

    @property
    def model_name(self) -> str:
        return self._model_path

    @property
    def provider(self) -> str:
        return "huggingface"

    @property
    def supports_logprobs(self) -> bool:
        return True

    def load(self) -> None:
        """Load tokenizer and model weights into memory. Idempotent."""
        if self._loaded:
            return

        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for HuggingFaceBackend. "
                "Install them with: pip install -e '.[hf]'"
            ) from exc

        self._torch = torch
        self._F = F

        logger.info("Loading tokenizer: %s", self._model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            trust_remote_code=True,
        )

        logger.info("Loading model: %s  device=%s", self._model_path, self._device)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.float16,
            device_map=self._device,
            trust_remote_code=True,
        )
        self._model.eval()
        self._loaded = True
        logger.info("Model ready: %s", self._model_path)

    def _get_input_device(self):
        """Device input tensors should be moved to before a forward pass.

        With device_map="auto" the model may span multiple GPUs; inputs go to
        whichever device holds the embedding layer (the first parameter).
        """
        if self._device == "auto":
            return next(self._model.parameters()).device
        return self._device

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a text completion for the given prompt.

        Args:
            prompt: Full input prompt string.
            **kwargs: max_new_tokens, temperature, do_sample, seed — override
                the constructor's generation_kwargs for this call only.

        Returns:
            Decoded completion text (prompt tokens excluded).
        """
        if not self._loaded:
            raise RuntimeError("Call load() before generate().")

        merged = {**self._default_generation_kwargs, **kwargs}
        max_new_tokens = merged.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS)
        temperature = merged.get("temperature", _DEFAULT_TEMPERATURE)
        do_sample = merged.get("do_sample", _DEFAULT_DO_SAMPLE)
        seed = merged.get("seed")

        if seed is not None:
            self._torch.manual_seed(seed)

        inputs = self._tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[-1]
        input_device = self._get_input_device()
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        generate_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        # temperature is only meaningful when sampling; passing it with greedy
        # decoding triggers a HuggingFace warning.
        if do_sample and temperature > 0.0:
            generate_kwargs["temperature"] = temperature

        with self._torch.no_grad():
            outputs = self._model.generate(**inputs, **generate_kwargs)

        generated_ids = outputs[0][prompt_len:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    def score_options(self, prompt: str, options: list[str], **kwargs) -> list[float]:
        """
        Return log-probabilities for each option label token.

        options must be single tokens as seen by this model's tokenizer. Pass option
        labels ("A", "B", "C", "D"), not full option text. This is by design — MCQ
        methods in this framework score labels, not text. Raises ValueError if any
        option encodes to more than one token.

        Must call load() before invoking this method.

        Args:
            prompt: the full prompt string
            options: list of single-token label strings, e.g. ["A", "B", "C", "D"]
            **kwargs: unused, accepted for interface compatibility

        Returns:
            list of floats (log-probabilities), same length and order as options

        Raises:
            ValueError: if load() has not been called, options is empty, or any
                option encodes to != 1 token
        """
        if not self._loaded:
            raise ValueError("Call load() before score_options().")
        if not options:
            raise ValueError("options must not be empty.")

        # Design decision: single-token constraint is intentional.
        # score_options() scores option labels (A, B, C, D), not option text.
        # Multi-token options would require summing log-probs over variable-length
        # sequences, making scores incomparable across options of different lengths.
        # If you need to score full option text, implement a separate method in your
        # method class — do not extend score_options() to handle it.
        option_token_ids: dict[str, int] = {}
        for opt in options:
            ids = self._tokenizer.encode(opt, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"score_options() cannot score option label {opt!r}: it encodes "
                    f"to {len(ids)} tokens in {self._model_path}'s tokenizer, but "
                    f"scoring requires each option label to be exactly one token "
                    f"(so per-letter log-probs are comparable). Most tokenizers emit "
                    f"a single token for a bare 'A'–'D' only after a leading space; "
                    f"check whether this tokenizer needs the space-prefixed form "
                    f"(e.g. ' {opt}' instead of {opt!r}). This is a hard constraint of "
                    f"label scoring, not a transient error — pride / cyclic_logprob "
                    f"cannot run on this model until option labels are single-token."
                )
            option_token_ids[opt] = ids[0]

        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_device = self._get_input_device()
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with self._torch.no_grad():
            # Single forward pass; logits shape is (1, seq_len, vocab_size).
            outputs = self._model(**inputs)

        # Position -1 predicts the token that follows the entire prompt.
        last_logits = outputs.logits[0, -1, :]
        log_probs = self._F.log_softmax(last_logits, dim=-1)

        scores = {opt: log_probs[tid].item() for opt, tid in option_token_ids.items()}
        return [scores[opt] for opt in options]
