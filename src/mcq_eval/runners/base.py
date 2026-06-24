# src/mcq_eval/runners/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from pathlib import Path

from mcq_eval.backends.base import BaseBackend
from mcq_eval.clients.types import (
    ErrorInfo,
    FAILURE_STATUS,
    ModelRequest,
    ModelResponse,
    RequestMetadata,
    SUCCESS_STATUS,
)
from mcq_eval.config.models import MAX_TOKENS, SEED, TEMPERATURE
from mcq_eval.parsing.parser import parse_model_answer
from mcq_eval.parsing.types import ParseResult
from mcq_eval.pipeline.prompt_builder import load_prompt_templates
from mcq_eval.scoring.scorer import score_prediction
from mcq_eval.scoring.types import ScoreResult


class ExperimentRunner(ABC):
    """Abstract base runner for all experimental conditions.

    This class owns the shared infrastructure used by every condition:
    model request construction, result-row assembly, parsing, and scoring.
    Subclasses implement the condition-specific execution shape by defining
    ``run_one`` and ``run_many``.
    """

    def __init__(
            self,
            backend: BaseBackend,
            method_name: str,
            split_name: str,
            prompt_version: str,
            prompts_dir: Path,
            run_id: str,
            temperature: float = TEMPERATURE,
            max_tokens: int = MAX_TOKENS,
            seed: int | None = SEED,
            perturbation_name: str | None = None,
    ) -> None:
        self.backend = backend
        self.method_name = method_name
        self.split_name = split_name
        self.prompt_version = prompt_version
        self.run_id = run_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.perturbation_name = perturbation_name
        self._prompts = load_prompt_templates(prompt_version, prompts_dir)

    @abstractmethod
    def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Execute a single question through this experimental condition.

        Args:
            question_row: Normalized question record.
            sample_index: Repetition index for this question within the run.

        Returns:
            Flat result dictionary ready for serialization.
        """

    def run_many(self, question_rows: Sequence[Any]) -> list[dict]:
        """Execute multiple questions through this experimental condition.

        Synchronous: backend calls (generate/score_options) are compute-bound
        local forward passes or already-blocking API calls, not awaitable
        I/O, so there is no concurrency to gain from asyncio here.

        Args:
            question_rows: Sequence of normalized question records.

        Returns:
            List of flat result dictionaries, one per question.
        """
        return [self.run_one(row, i) for i, row in enumerate(question_rows)]

    def _call_backend_generate(
            self,
            model_request: ModelRequest,
            prompt: str,
    ) -> ModelResponse:
        """Call backend.generate() and wrap the result in a ModelResponse.

        BaseBackend.generate() takes a bare prompt string and returns a bare
        string (or raises) — it carries no latency/usage/finish_reason
        metadata, unlike the old client-level ModelResponse contract. This
        wraps that minimal contract back into the existing ModelResponse
        shape so _build_result_row keeps working unchanged. latency_seconds,
        finish_reason, usage, and timestamp_utc are not available at this
        layer and are left at their empty defaults.

        Args:
            model_request: The request metadata object for this call (used
                only for its .metadata field here — the prompt itself is
                passed separately since BaseBackend.generate() takes a plain
                string, not a ModelRequest).
            prompt: The prompt string to send to the backend.

        Returns:
            A ModelResponse reflecting success (raw_text set) or failure
            (error set), matching the shape _build_result_row expects.
        """
        try:
            raw_text = self.backend.generate(prompt)
        except Exception as exc:
            return ModelResponse(
                provider=self.backend.provider,
                model_name=self.backend.model_name,
                status=FAILURE_STATUS,
                latency_seconds=0.0,
                metadata=model_request.metadata,
                error=ErrorInfo(type(exc).__name__, str(exc), False, "backend_generate"),
            )
        return ModelResponse(
            provider=self.backend.provider,
            model_name=self.backend.model_name,
            status=SUCCESS_STATUS,
            latency_seconds=0.0,
            metadata=model_request.metadata,
            raw_text=raw_text,
        )

    def _build_model_request(
            self,
            question_row: Any,
            prompt: str,
            sample_index: int,
            request_logprobs: bool = False,
    ) -> ModelRequest:
        """Construct a standardized model request from a question and prompt.

        Args:
            question_row: Normalized question record.
            prompt: Fully formatted prompt string.
            sample_index: Repetition index for this question within the run.
            request_logprobs: Whether to ask the provider for token log-probabilities
                (Together only in this codebase).

        Returns:
            A validated ModelRequest ready for backend execution.
        """
        metadata = RequestMetadata(
            question_id=question_row["question_id"],
            split_name=self.split_name,
            method_name=self.method_name,
            subject=question_row["subject"],
            run_id=self.run_id,
            prompt_version=self.prompt_version,
            perturbation_name=self.perturbation_name,
            sample_index=sample_index,
        )

        return ModelRequest(
            provider=self.backend.provider,
            model_name=self.backend.model_name,
            payload=prompt,
            metadata=metadata,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            request_logprobs=request_logprobs,
        )

    @staticmethod
    def _build_result_row(
            question_row: Any,
            prompt: str,
            model_request: ModelRequest,
            model_response: ModelResponse | None,
            parsed_result: ParseResult | None,
            score_result: ScoreResult | None,
            error: str | None = None,
    ) -> dict:
        """Assemble a flat result dictionary from all pipeline outputs.

        Args:
            question_row: Normalized question record.
            prompt: Prompt string that was sent to the model.
            model_request: The request object sent to the backend.
            model_response: The response object returned by the backend,
                or None if the call was never made.
            parsed_result: Structured parse output, or None.
            score_result: Structured score output, or None.
            error: Optional error message for failures that occur
                outside the backend layer.

        Returns:
            Flat dictionary containing all trace, model, parse, and
            score fields for one experimental observation.
        """
        metadata = model_request.metadata

        return {
            # --- trace metadata ---
            "run_id": metadata.run_id,
            "question_id": metadata.question_id,
            "split_name": metadata.split_name,
            "subject": metadata.subject,
            "method_name": metadata.method_name,
            "prompt_version": metadata.prompt_version,
            "perturbation_name": metadata.perturbation_name,
            "sample_index": metadata.sample_index,
            # --- model config ---
            "provider": model_request.provider,
            "model_name": model_request.model_name,
            "temperature": model_request.temperature,
            "max_tokens": model_request.max_tokens,
            "seed": model_request.seed,
            # --- question content ---
            "question_text": question_row["question_text"],
            "choice_a": question_row["choice_a"],
            "choice_b": question_row["choice_b"],
            "choice_c": question_row["choice_c"],
            "choice_d": question_row["choice_d"],
            "correct_option": question_row["correct_option"],
            # --- prompt ---
            "prompt": prompt,
            # --- model output ---
            "model_status": model_response.status if model_response else None,
            "raw_text": model_response.raw_text if model_response else None,
            "finish_reason": model_response.finish_reason if model_response else None,
            "latency_seconds": model_response.latency_seconds if model_response else None,
            "timestamp_utc": model_response.timestamp_utc if model_response else None,
            # --- error info ---
            "error_type": (
                model_response.error.error_type
                if model_response and model_response.error
                else None
            ),
            "error_message": (
                model_response.error.message
                if model_response and model_response.error
                else error
            ),
            "error_stage": (
                model_response.error.stage
                if model_response and model_response.error
                else None
            ),
            "error_retryable": (
                model_response.error.retryable
                if model_response and model_response.error
                else None
            ),
            # --- parse output ---
            "parsed_choice": parsed_result.final_choice if parsed_result else None,
            "parse_status": parsed_result.status if parsed_result else None,
            "normalized_text": parsed_result.normalized_text if parsed_result else None,
            "parse_reason": parsed_result.reason if parsed_result else None,
            # --- score output ---
            "is_correct": score_result.is_correct if score_result else None,
            "score_status": score_result.status if score_result else None,
        }

    def _parse_and_score(
            self,
            raw_text: str,
            correct_option: str,
            options: dict[str, str],
    ) -> tuple[ParseResult, ScoreResult]:
        """Parse a model response and score it against the gold answer.

        Args:
            raw_text: Raw model output text.
            correct_option: Ground-truth answer letter.
            options: Mapping from answer letter to answer text.

        Returns:
            Tuple of (ParseResult, ScoreResult).
        """
        parse_result = self._parse(raw_text, options)
        score_result = self._score(parse_result, correct_option)
        return parse_result, score_result

    @staticmethod
    def _parse(
            raw_text: str,
            options: dict[str, str],
    ) -> ParseResult:
        """Parse a model response."""
        return parse_model_answer(raw_text, options)

    @staticmethod
    def _score(
            parse_result: ParseResult,
            correct_option: str,
        ) -> ScoreResult:
        """Score a parsed model response."""
        return score_prediction(parse_result, correct_option)


    @staticmethod
    def _build_options(question_row: Any) -> dict[str, str]:
        """Extract the option letter-to-text mapping from a question row.

        Args:
            question_row: Normalized question record.

        Returns:
            Mapping from canonical answer letters to their text.
        """
        return {
            "A": question_row["choice_a"],
            "B": question_row["choice_b"],
            "C": question_row["choice_c"],
            "D": question_row["choice_d"],
        }