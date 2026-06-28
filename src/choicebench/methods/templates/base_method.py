# src/choicebench/methods/templates/base_method.py
#
# Template for implementing a new MCQ evaluation method.
#
# NAMING CONVENTION
#   File name:  my_method_name.py  (snake_case)
#   Class name: MyMethodNameRunner  (CamelCase + "Runner" suffix)
#   YAML key:   "my_method_name"  (same as file name, without .py)
#
# REGISTRATION
#   Built-in (lives inside this repo):
#     1. Copy this file to src/choicebench/methods/library/my_method_name.py
#     2. Import and add to METHOD_REGISTRY in src/choicebench/registry.py
#     3. Add re-export to src/choicebench/methods/library/__init__.py
#
#   External (lives in your own package):
#     Use "module.path:ClassName" in the YAML config — no framework files needed.
#     Example:  methods:
#                 - name: my_package.methods.my_method:MyMethodRunner
#                   params:
#                     num_rollouts: 4
#
# METHOD-SPECIFIC PARAMETERS
#   YAML params are passed to your runner constructor as keyword arguments.
#   Example:
#     def __init__(self, num_rollouts: int = 4, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.num_rollouts = num_rollouts
#
# BACKEND METHODS AVAILABLE
#   self.backend.generate(prompt) -> str
#       Plain text generation. Works with all backends.
#   self.backend.score_options(prompt, options) -> list[float]
#       Log-probabilities for each option label. Only works when
#       self.backend.supports_logprobs is True (HuggingFaceBackend and
#       compatible backends). API backends raise NotImplementedError.
#       Always check supports_logprobs before calling.

from __future__ import annotations

from typing import Any

from choicebench.methods.base import ExperimentRunner
from choicebench.pipeline.prompt_builder import build_direct_mcq_prompt

"""
Method: <Your Method Name>
---------------------------
Description: <one paragraph — what does this method do, and why?>
Reference: <paper title and citation, or "n/a" if not from a paper>
Backend requirements: generate only | generate + score_options
Logprob support required: yes | no
Calls per question: <how many backend calls does run_one() make per question?>
"""


class YourMethodRunner(ExperimentRunner):
    # Rename this class: e.g. TwoPassDebiasingRunner.

    def __init__(self, num_rollouts: int = 4, **kwargs) -> None:
        # All framework args (backend, method_name, split_name, prompt_version,
        # prompts_dir, run_id, temperature, max_tokens, seed,
        # perturbation_name, model_label) are forwarded automatically by
        # instantiate_runner() via **kwargs.
        # YAML params are forwarded as constructor kwargs too:
        #   methods:
        #     - name: my_method_name
        #       params:
        #         num_rollouts: 8
        super().__init__(**kwargs)

        # Add any per-run state your method needs here.
        # Examples:
        self.num_rollouts = num_rollouts
        #   self._calibration = None      # populated lazily in run_one()

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Execute one question through this experimental condition.

        This is the only method you MUST implement. The framework calls
        run_many() → run_one() for every question in the benchmark.

        Args:
            question_row: Normalized question record. Guaranteed keys:
                question_id, subject, question_text,
                choice_a, choice_b, choice_c, choice_d, correct_option,
                correct_answer_text.
                v0.1 uses legacy A-D columns. Use self._build_options(row)
                instead of hardcoding A-D if your method can support rows with
                missing trailing choices.
            sample_index: Repetition index for this question within the run.
                The default run_many() passes a zero-based row index.

        Returns:
            Flat result dictionary. ALWAYS build this via _build_result_row() —
            do not construct it by hand. The schema is load-bearing for the
            evaluate_run.py pipeline.
        """
        # ── Step 1: build the prompt ────────────────────────────────────────
        # self._prompts is a dict loaded from the prompts/{prompt_version}/
        # directory. Keys match template file names (without .txt).
        # Example:  template = self._prompts["direct_mcq"]
        #           prompt = build_direct_mcq_prompt(
        #               template=template,
        #               question=question_row["question_text"],
        #               options=self._build_options(question_row),
        #           )
        prompt = self._build_prompt(question_row)

        # ── Step 2: call the backend ────────────────────────────────────────
        # _call_backend_generate wraps generate() and returns a ModelResponse
        # for both success and failure cases (never raises).
        model_response = self._call_backend_generate(prompt)

        parsed_result = None
        score_result = None

        if model_response.is_success():
            # ── Step 3: parse + score ────────────────────────────────────────
            # _parse_and_score extracts the final letter choice from raw_text,
            # restricted to the letters present in options, then checks it
            # against correct_option.
            # Returns (ParseResult, ScoreResult) — both are dataclasses.
            options = self._build_options(question_row)
            parsed_result, score_result = self._parse_and_score(
                raw_text=model_response.raw_text,
                correct_option=question_row["correct_option"],
                options=options,
            )

        # ── (Optional) logprob-based scoring ───────────────────────────────
        # Uncomment this block if your method needs per-option log-probs
        # (e.g. PriDe-style debiasing). Delete if not needed.
        #
        # if not self.backend.supports_logprobs:
        #     raise NotImplementedError(
        #         f"{self.__class__.__name__} requires score_options() support. "
        #         "Use HuggingFaceBackend or another logprob-capable backend."
        #     )
        # options = self._build_options(question_row)
        # letters = list(options.keys())
        # logprobs = self.backend.score_options(prompt, letters)
        # best_letter = letters[logprobs.index(max(logprobs))]
        # Then create a ParseResult/ScoreResult or attach method-specific
        # columns before returning _build_result_row().

        # ── Step 4: assemble result row ─────────────────────────────────────
        # _build_result_row fills in all trace/model/parse/score fields.
        # Pass None for parsed_result/score_result on backend failure.
        return self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            model_response=model_response,
            parsed_result=parsed_result,
            score_result=score_result,
        )

    def _build_prompt(self, question_row: Any) -> str:
        """Construct the prompt string for one question.

        self._prompts["<template_name>"] is a string template loaded from
        prompts/{prompt_version}/<template_name>.txt. Built-in MCQ templates
        use {question} and {options}; build_direct_mcq_prompt() renders the
        dynamic option block from the row's available choices. If your method
        has no prompt template, build the string directly here.
        """
        # Example using a template:
        # return build_direct_mcq_prompt(
        #     template=self._prompts["direct_mcq"],
        #     question=question_row["question_text"],
        #     options=self._build_options(question_row),
        # )
        raise NotImplementedError("TODO: implement _build_prompt()")
