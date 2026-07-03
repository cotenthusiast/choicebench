# tests/runners/test_cyclic_logprob.py

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from choicebench.methods.library.cyclic_logprob import CyclicLogprobRunner
from choicebench.methods.library.pride_math import equation1_cyclic_debiased_content_probs
from choicebench.methods.library.permutation import PermutationRunner
from choicebench.backends.base import BaseBackend

from tests.runners.conftest import MockBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(backend, **kw) -> CyclicLogprobRunner:
    return CyclicLogprobRunner(
        backend=backend,
        method_name="cyclic_logprob",
        split_name="test",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run",
        **kw,
    )


class _MockVLLMRawClient:
    """Simulates VLLMClient.score_options_async() with a queue of fixed responses."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.calls_received: list[tuple[str, list[str]]] = []

    async def score_options_async(
        self, prompt: str, options: list[str]
    ) -> dict[str, float]:
        self.calls_received.append((prompt, options))
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            if isinstance(resp, Exception):
                raise resp
            return resp
        raise RuntimeError("_MockVLLMRawClient: no more queued responses")


class _MockVLLMBackend(BaseBackend):
    """Mock backend shaped like APIBackend(VLLMClient)."""

    def __init__(
        self,
        score_async_responses: list | None = None,
        supports_score_options_val: bool = True,
    ) -> None:
        self._raw_client = _MockVLLMRawClient(score_async_responses or [])
        self._supports_val = supports_score_options_val

    @property
    def model_name(self) -> str:
        return "mock-vllm-model"

    @property
    def provider(self) -> str:
        return "vllm"

    @property
    def supports_logprobs(self) -> bool:
        return True

    def supports_score_options(self) -> bool:
        return self._supports_val

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def score_options(self, prompt: str, options: list[str], **kwargs) -> list[float]:
        raise NotImplementedError("use score_options_async")


# ---------------------------------------------------------------------------
# Tests: class attributes
# ---------------------------------------------------------------------------

class TestCyclicLogprobRunnerClassAttributes:
    def test_requires_score_options_is_true(self):
        assert CyclicLogprobRunner.requires_score_options is True


# ---------------------------------------------------------------------------
# Tests: run_one schema
# ---------------------------------------------------------------------------

class TestCyclicLogprobRunnerRunOne:
    def test_run_one_produces_extra_fields(self, runner_question_row):
        """run_one() result must include cyclic_logprob_inference_mode and option_distributions_json."""
        # 4 score_options calls — one per cyclic permutation
        backend = MockBackend(
            score_responses=[
                [-0.1, -2.0, -3.0, -4.0],
                [-0.1, -2.0, -3.0, -4.0],
                [-0.1, -2.0, -3.0, -4.0],
                [-0.1, -2.0, -3.0, -4.0],
            ],
            supports_logprobs=True,
        )
        runner = _make_runner(backend)
        row = runner.run_one(runner_question_row, sample_index=0)

        assert row["cyclic_logprob_inference_mode"] == "eq1_averaging"
        assert row["option_distributions_json"] is not None

    def test_option_distributions_json_is_valid_matrix(self, runner_question_row):
        """option_distributions_json must parse to a 4x4 list of lists."""
        backend = MockBackend(
            score_responses=[[-1.0, -2.0, -3.0, -4.0]] * 4,
            supports_logprobs=True,
        )
        runner = _make_runner(backend)
        row = runner.run_one(runner_question_row, sample_index=0)

        mat = json.loads(row["option_distributions_json"])
        assert len(mat) == 4
        for dist_row in mat:
            assert len(dist_row) == 4
            assert abs(sum(dist_row) - 1.0) < 1e-6, "each row must be a probability distribution"

    def test_run_one_scores_correct_answer(self, runner_question_row):
        """When the correct option (C=HTTPS) has the highest logprob, is_correct=True."""
        # correct_option is "C". Give C the highest logprob in every permutation.
        canon = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        perms = PermutationRunner._generate_permutations(canon)

        score_responses = []
        for perm in perms:
            letters = list(perm.keys())
            # Find which letter carries HTTPS in this permutation, give it -0.1
            scores = []
            for letter in letters:
                scores.append(-0.1 if perm[letter] == "HTTPS" else -5.0)
            score_responses.append(scores)

        backend = MockBackend(score_responses=score_responses, supports_logprobs=True)
        runner = _make_runner(backend)
        row = runner.run_one(runner_question_row, sample_index=0)

        assert row["parsed_choice"] == "C"
        assert row["is_correct"] is True
        # PF-3: success rows carry a non-None answer_status despite never calling generate().
        assert row["answer_status"] == "success"

    @pytest.mark.parametrize(
        "n_failed",
        [0, 2, 3],  # none, some, all-but-one (of 4 permutations)
    )
    def test_run_one_reports_n_permutations_failed(self, runner_question_row, n_failed):
        """PF-4: n_permutations_failed/_total must count uniform-fallback permutations."""
        good = [-0.1, -2.0, -3.0, -4.0]
        # First (4 - n_failed) succeed, the rest raise → uniform fallback.
        score_responses = [good] * (4 - n_failed) + [RuntimeError("down")] * n_failed
        backend = MockBackend(score_responses=score_responses, supports_logprobs=True)
        runner = _make_runner(backend)
        row = runner.run_one(runner_question_row, sample_index=0)

        assert row["n_permutations_total"] == 4
        assert row["n_permutations_failed"] == n_failed
        # As long as ≥1 permutation succeeded, an answer is still produced.
        assert row["parsed_choice"] is not None

    def test_run_one_base_schema_fields_present(self, runner_question_row):
        """Result row must carry all standard base schema fields."""
        backend = MockBackend(
            score_responses=[[-1.0, -2.0, -3.0, -4.0]] * 4,
            supports_logprobs=True,
        )
        runner = _make_runner(backend)
        row = runner.run_one(runner_question_row, sample_index=0)

        for field in ("run_id", "question_id", "method_name", "split_name",
                      "provider", "model_name", "prompt"):
            assert field in row, f"missing field: {field}"

        assert row["method_name"] == "cyclic_logprob"
        assert row["run_id"] == "test_run"

    def test_run_one_all_score_options_fail_leaves_no_answer(self, runner_question_row):
        """When all score_options calls raise, parsed_choice and is_correct are None."""
        backend = MockBackend(
            score_responses=[RuntimeError("server down")] * 4,
            supports_logprobs=True,
        )
        runner = _make_runner(backend)
        row = runner.run_one(runner_question_row, sample_index=0)

        assert row["parsed_choice"] is None
        assert row["is_correct"] is None
        assert row["answer_status"] == "failure"

    def test_run_one_makes_n_score_options_calls(self, runner_question_row):
        """run_one() must call score_options() exactly N times (once per permutation)."""
        backend = MockBackend(
            score_responses=[[-1.0, -2.0, -3.0, -4.0]] * 4,
            supports_logprobs=True,
        )
        runner = _make_runner(backend)
        runner.run_one(runner_question_row, sample_index=0)

        assert backend._score_call_count == 4

    def test_run_one_never_calls_generate(self, runner_question_row):
        """CyclicLogprobRunner must never call backend.generate()."""
        backend = MockBackend(
            score_responses=[[-1.0, -2.0, -3.0, -4.0]] * 4,
            supports_logprobs=True,
        )
        runner = _make_runner(backend)
        runner.run_one(runner_question_row, sample_index=0)

        assert len(backend.requests_received) == 0


# ---------------------------------------------------------------------------
# Tests: Eq. (1) averaging vs majority-vote divergence
# ---------------------------------------------------------------------------

class TestEquation1VsMajorityVote:
    def test_eq1_and_majority_vote_can_disagree(self):
        """Construct a logprob matrix where Eq.(1) and majority-vote argmax diverge.

        Setup: every permutation has a strong positional bias toward letter A,
        so majority-vote (argmax per row, then unpermute) produces a four-way
        tie resolved by the first vote (canonical A), while Eq.(1) averages the
        FULL distribution and correctly identifies content slot 1 (canonical B)
        as the winner because the non-A mass is concentrated there.
        """
        # Each row: A has the highest probability (positional bias)
        # but B's content signal accumulates more across permutations.
        mat = np.array([
            [0.40, 0.35, 0.15, 0.10],  # perm 0
            [0.40, 0.30, 0.20, 0.10],  # perm 1
            [0.40, 0.25, 0.25, 0.10],  # perm 2
            [0.40, 0.25, 0.25, 0.10],  # perm 3
        ], dtype=np.float64)

        # Majority vote: argmax of each row is A (index 0) in all permutations.
        # After un-permuting: perm k → A maps to canonical slot k.
        # So votes are A(0), B(1), C(2), D(3) — four-way tie, first = A.
        majority_vote_canonical_idx = 0  # A

        content_probs = equation1_cyclic_debiased_content_probs(mat)
        eq1_canonical_idx = int(np.argmax(content_probs))

        # Verify the divergence: Eq.(1) should pick B (index 1).
        assert eq1_canonical_idx == 1, (
            f"Expected Eq.(1) to pick content slot 1 (B); got {eq1_canonical_idx}. "
            f"content_probs={content_probs}"
        )
        assert eq1_canonical_idx != majority_vote_canonical_idx, (
            "Eq.(1) and majority vote must disagree on this example."
        )

    def test_eq1_output_is_valid_probability_distribution(self):
        """equation1_cyclic_debiased_content_probs must return a normalized distribution."""
        mat = np.array([
            [0.50, 0.30, 0.20],
            [0.50, 0.30, 0.20],
            [0.50, 0.30, 0.20],
        ], dtype=np.float64)
        out = equation1_cyclic_debiased_content_probs(mat)
        assert abs(out.sum() - 1.0) < 1e-9
        assert all(v >= 0 for v in out)


# ---------------------------------------------------------------------------
# Tests: run_many_async
# ---------------------------------------------------------------------------

class TestCyclicLogprobRunManyAsync:
    pytestmark = pytest.mark.asyncio

    async def test_raises_not_implemented_when_no_score_options(
        self, runner_question_row
    ):
        """run_many_async() must raise NotImplementedError when backend lacks score_options."""
        backend = MockBackend(supports_logprobs=False)
        runner = _make_runner(backend)
        df = pd.DataFrame([runner_question_row])

        with pytest.raises(NotImplementedError, match="score_options"):
            await runner.run_many_async(df)

    async def test_run_many_async_returns_one_result_per_question(
        self, runner_question_row
    ):
        """run_many_async() must return exactly one result row per question."""
        n_questions = 3
        # 4 permutations × 3 questions = 12 async calls
        resp = {"A": -0.1, "B": -2.0, "C": -3.0, "D": -4.0}
        backend = _MockVLLMBackend(score_async_responses=[resp] * 12)
        runner = _make_runner(backend)
        df = pd.DataFrame([runner_question_row] * n_questions)

        results = await runner.run_many_async(df)

        assert len(results) == n_questions
        assert backend._raw_client._call_count == n_questions * 4

    async def test_run_many_async_result_has_extra_fields(self, runner_question_row):
        """Each result row from run_many_async() must include the cyclic-specific fields."""
        resp = {"A": -0.1, "B": -2.0, "C": -3.0, "D": -4.0}
        backend = _MockVLLMBackend(score_async_responses=[resp] * 4)
        runner = _make_runner(backend)
        df = pd.DataFrame([runner_question_row])

        results = await runner.run_many_async(df)
        row = results[0]

        assert row["cyclic_logprob_inference_mode"] == "eq1_averaging"
        assert row["option_distributions_json"] is not None
        mat = json.loads(row["option_distributions_json"])
        assert len(mat) == 4 and len(mat[0]) == 4

    async def test_run_many_async_scores_correct_answer(self, runner_question_row):
        """With C having the highest logprob and uniform positional signal, is_correct=True."""
        # correct_option is C; give C the highest logprob in the canonical permutation
        # and distribute evenly in others so Eq.(1) still selects C.
        canon = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        perms = PermutationRunner._generate_permutations(canon)

        async_responses = []
        for perm in perms:
            letters = list(perm.keys())
            resp = {}
            for letter in letters:
                resp[letter] = -0.1 if perm[letter] == "HTTPS" else -5.0
            async_responses.append(resp)

        backend = _MockVLLMBackend(score_async_responses=async_responses)
        runner = _make_runner(backend)
        df = pd.DataFrame([runner_question_row])

        results = await runner.run_many_async(df)

        assert results[0]["is_correct"] is True
        assert results[0]["parsed_choice"] == "C"

    async def test_run_many_async_partial_failure_handled_gracefully(
        self, runner_question_row
    ):
        """When some (not all) score_options_async calls fail, result is still produced."""
        good = {"A": -0.1, "B": -2.0, "C": -3.0, "D": -4.0}
        responses = [good, RuntimeError("timeout"), good, good]
        backend = _MockVLLMBackend(score_async_responses=responses)
        runner = _make_runner(backend)
        df = pd.DataFrame([runner_question_row])

        results = await runner.run_many_async(df)

        assert len(results) == 1
        assert results[0]["option_distributions_json"] is not None

    async def test_run_many_async_cyclic_logprob_inference_mode(
        self, runner_question_row
    ):
        """cyclic_logprob_inference_mode must be 'eq1_averaging' in async path."""
        resp = {"A": -0.1, "B": -2.0, "C": -3.0, "D": -4.0}
        backend = _MockVLLMBackend(score_async_responses=[resp] * 4)
        runner = _make_runner(backend)
        df = pd.DataFrame([runner_question_row])

        results = await runner.run_many_async(df)

        assert results[0]["cyclic_logprob_inference_mode"] == "eq1_averaging"
