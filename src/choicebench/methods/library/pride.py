# src/choicebench/methods/library/pride.py

"""
Method: PriDe (Permutation Debiasing)
--------------------------------------------------------
Description: Estimates the model's positional bias prior on a calibration set
disjoint from the evaluation split, by running cyclic permutation rollouts and
reading each rotation's per-letter log-probabilities via the backend (Eq. 7),
then debiases each evaluation question's observed logprob distribution
against that prior (Eq. 8) and picks the argmax. Calls backend.score_options()
directly — the backend is responsible for returning a clean per-letter
logprob list.
Reference: Zheng et al., ICLR 2024, "Large Language Models Are Not Robust
Multiple Choice Selectors" (arXiv:2309.03882), §3, Eq. 1/7/8.
Backend requirements: generate + score_options
Logprob support required: yes
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from choicebench.parsing.types import PARSE_OK, ParseResult
from choicebench.pipeline.prompt_builder import build_direct_mcq_prompt
from choicebench.methods.base import ExperimentRunner
from choicebench.methods.library.permutation import PermutationRunner
from choicebench.methods.library.pride_math import (
    OPTION_LETTERS,
    CalibrationState,
    apply_debiased_choice_from_defaults,
    average_prior_probability_vectors,
    calibration_state_from_sidecar,
    calibration_state_uniform,
    equation7_prior_from_rollouts,
    logprob_map_to_label_distribution,
)

logger = logging.getLogger(__name__)

_SIDE_SCHEMA_VERSION = 3


def _pick_calibration_rows(
        full: list[dict],
        k: int,
        seed: int,
) -> tuple[list[str], list[dict]]:
    """Random subset of size *k* from *full* using a seeded shuffle."""
    import random
    if not full:
        return [], []
    kk = max(0, min(int(k), len(full)))
    if kk == 0:
        return [], []
    if kk == len(full):
        chosen_idx = list(range(len(full)))
    else:
        rng = random.Random(int(seed))
        idx = list(range(len(full)))
        rng.shuffle(idx)
        chosen_idx = sorted(idx[:kk])
    rows = [full[i] for i in chosen_idx]
    qids = [r["question_id"] for r in rows]
    return qids, rows


class PriDeRunner(ExperimentRunner):
    """Cyclic permutation prior estimation (Paper §3) then Eq.(8) transfer inference.

    Calibration questions must be disjoint from evaluation questions so that
    the estimated prior is not contaminated by in-distribution label leakage.
    All evaluation rows use ``eq8_transfer`` mode.

    Logprobs come from backend.score_options() — no provider/client logic.
    """

    requires_score_options: bool = True

    def __init__(
            self,
            backend: Any,
            method_name: str,
            split_name: str,
            prompt_version: str,
            prompts_dir: Path,
            run_id: str,
            temperature: float | None = None,
            max_tokens: int | None = None,
            seed: int | None = None,
            perturbation_name: str | None = None,
            *,
            calibration_n: int = 50,
            calibration_seed: int = 42,
            calibration_benchmark: str = "",
            calibration_runs_dir: Path | None = None,
            calibration_questions: list[dict] | None = None,
            preflight_questions: list[dict] | None = None,
    ) -> None:
        # Precedence: calibration_questions > preflight_questions > uniform prior.
        # calibration_questions is the explicit API; preflight_questions is the
        # generic key the orchestrator passes when a preflight block is configured.
        if calibration_questions is None and preflight_questions is not None:
            calibration_questions = preflight_questions
        kw: dict[str, Any] = dict(
            backend=backend,
            method_name=method_name,
            split_name=split_name,
            prompt_version=prompt_version,
            prompts_dir=prompts_dir,
            run_id=run_id,
        )
        if temperature is not None:
            kw["temperature"] = temperature
        if max_tokens is not None:
            kw["max_tokens"] = max_tokens
        if seed is not None:
            kw["seed"] = seed
        if perturbation_name is not None:
            kw["perturbation_name"] = perturbation_name
        super().__init__(**kw)

        # PriDe needs per-letter logprobs, so it requires a backend that
        # implements score_options(). We gate on the capability flag rather
        # than a provider name, so any logprob-capable backend works.
        if not backend.supports_logprobs:
            raise ValueError(
                f"PriDe requires a backend with score_options() support; "
                f"{backend.__class__.__name__} does not "
                f"(supports_logprobs=False)."
            )

        self._calibration_n = max(0, int(calibration_n))
        self._calibration_seed = int(calibration_seed)
        self._calibration_benchmark = calibration_benchmark or split_name
        self._calibration_runs_dir = Path(calibration_runs_dir or Path("."))
        self._calibration_questions: list[dict] = list(calibration_questions or [])

        self._calibration_ready: bool = False
        self._calibration_state: CalibrationState = calibration_state_uniform()

    def _sidecar_path(self) -> Path:
        slug = self.backend.model_name.replace("/", "_").replace(" ", "_")
        return (
            self._calibration_runs_dir
            / self.run_id
            / f"pride_calibration__{slug}__{self._calibration_benchmark}.json"
        )

    def run_many(self, question_rows: Sequence[Any]) -> list[dict]:
        # Fully synchronous, matching ExperimentRunner.run_many: backend
        # generate()/score_options() calls are blocking (local forward passes
        # or already-blocking API calls), so there is no concurrency to gain
        # from asyncio here. Calibration is fit once, before the eval loop.
        self._ensure_calibration()
        return [self.run_one(row, i) for i, row in enumerate(question_rows)]

    def _ensure_calibration(self) -> None:
        if self._calibration_ready:
            return

        cal_qids, cal_rows = _pick_calibration_rows(
            self._calibration_questions,
            self._calibration_n,
            self._calibration_seed,
        )
        sorted_ids = tuple(sorted(cal_qids))

        # Try to reuse a matching sidecar from a previous run.
        path = self._sidecar_path()
        if sorted_ids and path.exists():
            try:
                blob = json.loads(path.read_text())
                if (
                    blob.get("schema_version") == _SIDE_SCHEMA_VERSION
                    and tuple(sorted(blob.get("calibration_question_ids") or [])) == sorted_ids
                    and int(blob.get("calibration_seed", -1)) == self._calibration_seed
                ):
                    self._calibration_state = calibration_state_from_sidecar(blob)
                    self._calibration_ready = True
                    logger.info(
                        "PriDe loaded sidecar (K=%d) → %s",
                        len(sorted_ids),
                        path,
                    )
                    return
            except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
                logger.warning("PriDe sidecar unreadable (%s); refitting.", exc)

        if not cal_rows:
            logger.warning(
                "PriDe: no calibration questions available — using uniform prior."
            )
            self._calibration_state = calibration_state_uniform()
        else:
            for row in cal_rows:
                self._require_four_options(row)
            prior_vectors: list[np.ndarray] = []
            for row in cal_rows:
                roll_mat = self._cyclic_rollout_prob_matrix(row)
                prior_vectors.append(equation7_prior_from_rollouts(roll_mat))

            pep_global = average_prior_probability_vectors(prior_vectors)
            self._calibration_state = CalibrationState(
                peprior_probs={
                    OPTION_LETTERS[i]: float(pep_global[i])
                    for i in range(len(OPTION_LETTERS))
                },
                epsilon=1e-12,
                estimation_question_ids=tuple(sorted_ids),
            )

        self._calibration_ready = True

        sidecar_payload = {
            "schema_version": _SIDE_SCHEMA_VERSION,
            "version": self._calibration_state.version,
            "calibration_seed": self._calibration_seed,
            "n_options": len(OPTION_LETTERS),
            "calibration_question_ids": list(sorted_ids),
            "peprior_probs": {
                L: float(self._calibration_state.peprior_probs.get(L, 0.0))
                for L in OPTION_LETTERS
            },
            "epsilon": self._calibration_state.epsilon,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sidecar_payload, indent=2))

    def _cyclic_rollout_prob_matrix(self, question_row: Any) -> np.ndarray:
        """Run all cyclic permutations through score_options() → shape-(n,n) matrix.

        Per-question math is option-count-agnostic: ``letters`` is this
        question's real option keys (whatever length), and
        logprob_map_to_label_distribution(..., letters=letters) handles any
        length >= 2.

        Known limitation: _ensure_calibration averages per-question prior
        vectors with average_prior_probability_vectors(), which requires every
        vector to be the same length — i.e. it assumes all calibration
        questions share one option count. Mixing, say, 3-option and 4-option
        calibration questions in the same run would need _ensure_calibration
        switched to average_prior_probability_dicts() (masked per-letter
        averaging) and this method returning per-question (vector, letters)
        pairs instead of a bare array.
        """
        canon = self._build_options(question_row)
        letters = list(canon.keys())
        permutations = PermutationRunner._generate_permutations(canon)
        prompts = [
            PermutationRunner._build_permuted_prompt(
                question_row,
                perm,
                self._prompts["direct_mcq"],
            )
            for perm in permutations
        ]

        uni = np.ones(len(letters), dtype=np.float64) / len(letters)
        rows: list[np.ndarray] = []
        n_success = 0
        last_exc: Exception | None = None
        for prompt in prompts:
            try:
                scores = self.backend.score_options(prompt, letters)
                lp_map = dict(zip(letters, scores))
                rows.append(logprob_map_to_label_distribution(lp_map, letters=letters))
                n_success += 1
            except Exception as exc:
                logger.warning("PriDe calibration: score_options failed — %s", exc)
                rows.append(uni.copy())
                last_exc = exc

        if n_success == 0:
            qid = question_row.get("question_id", "<unknown>")
            raise RuntimeError(
                f"PriDe calibration: score_options() failed for all {len(prompts)} "
                f"permutations of calibration question {qid!r}. "
                "Ensure the backend implements score_options() when providing "
                "calibration_questions."
            ) from last_exc

        return np.stack(rows, axis=0).astype(np.float64)

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        self._ensure_calibration()

        options = self._build_options(question_row)
        self._require_four_options(question_row)
        letters = list(options.keys())
        prompt = self._build_prompt(question_row)

        adjusted_letter: str | None = None
        score_adjusted = None
        lp_map: dict[str, float] = {}
        scoring_error: str | None = None

        try:
            scores = self.backend.score_options(prompt, letters)
            lp_map = dict(zip(letters, scores))
        except Exception as exc:
            scoring_error = str(exc)
            logger.warning(
                "PriDe: score_options failed for question %s — %s",
                question_row["question_id"],
                exc,
            )

        if not lp_map:
            logger.warning(
                "PriDe: empty logprobs for question %s — skipping debiasing.",
                question_row["question_id"],
            )
        else:
            adjusted_letter = apply_debiased_choice_from_defaults(
                self._calibration_state,
                lp_map,
                letters=tuple(letters),
                eps_prob=1e-12,
            )
            adj_parse = ParseResult(
                final_choice=adjusted_letter,
                status=PARSE_OK,
                raw_text=None,
                normalized_text=adjusted_letter,
                reason="pride_eq8",
            )
            score_adjusted = self._score(adj_parse, question_row["correct_option"])

        # PriDe scores purely via score_options() — it never calls
        # backend.generate(), so there is no raw model text to record. The
        # debiased letter and its score are attached below as pride_* columns.
        row = self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            model_response=None,
            parsed_result=None,
            score_result=None,
            error=scoring_error,
        )
        row["pride_inference_mode"] = "eq8_transfer"
        row["pride_adjusted_choice"] = adjusted_letter
        row["peprior_json"] = json.dumps(self._calibration_state.peprior_probs)
        row["option_logprob_json"] = json.dumps(lp_map) if lp_map else None
        if score_adjusted is not None:
            row["score_status"] = score_adjusted.status
            row["is_correct"] = score_adjusted.is_correct

        return row

    def _build_prompt(self, question_row: Any) -> str:
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            options=self._build_options(question_row),
        )

    def _require_four_options(self, question_row: Any) -> None:
        options = self._build_options(question_row)
        if list(options.keys()) != list(OPTION_LETTERS):
            raise ValueError(
                "PriDe requires four valid A-D options in v0.1; "
                f"question {question_row.get('question_id', '<unknown>')!r} "
                f"has valid options {list(options)}."
            )

    async def run_many_async(self, question_rows: Sequence[Any]) -> list[dict]:
        """Async inference path for PriDe.

        Pattern for logprob-dependent methods:
        1. Run any preflight/calibration synchronously before the async loop.
           Concurrent calibration is a v0.2 item.
        2. Check self.backend.supports_score_options() and fall back to
           super().run_many_async() for backends that don't support logprobs.
        3. Use score_options_async() directly on self.backend._raw_client
           for the main inference loop.
        4. Keep _score_question_async() signature identical to run_one()
           so the output contract is preserved.
        """
        if not self.backend.supports_score_options():
            return await super().run_many_async(question_rows)

        self._ensure_calibration()

        rows = question_rows.to_dict(orient="records")
        results = await asyncio.gather(
            *[self._score_question_async(row, i) for i, row in enumerate(rows)]
        )
        return list(results)

    async def _score_question_async(
        self,
        question_row: Any,
        sample_index: int,
    ) -> dict:
        self._ensure_calibration()

        options = self._build_options(question_row)
        self._require_four_options(question_row)
        letters = list(options.keys())
        prompt = self._build_prompt(question_row)

        adjusted_letter: str | None = None
        score_adjusted = None
        lp_map: dict[str, float] = {}
        scoring_error: str | None = None

        try:
            from choicebench.clients.vllm_client import VLLMClient
            raw: VLLMClient = self.backend._raw_client
            logprob_dict = await raw.score_options_async(prompt, letters)
            scores = [logprob_dict.get(opt, -100.0) for opt in letters]
            lp_map = dict(zip(letters, scores))
        except Exception as exc:
            scoring_error = str(exc)
            logger.warning(
                "PriDe: score_options_async failed for question %s — %s",
                question_row["question_id"],
                exc,
            )

        if not lp_map:
            logger.warning(
                "PriDe: empty logprobs for question %s — skipping debiasing.",
                question_row["question_id"],
            )
        else:
            adjusted_letter = apply_debiased_choice_from_defaults(
                self._calibration_state,
                lp_map,
                letters=tuple(letters),
                eps_prob=1e-12,
            )
            adj_parse = ParseResult(
                final_choice=adjusted_letter,
                status=PARSE_OK,
                raw_text=None,
                normalized_text=adjusted_letter,
                reason="pride_eq8",
            )
            score_adjusted = self._score(adj_parse, question_row["correct_option"])

        row = self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            model_response=None,
            parsed_result=None,
            score_result=None,
            error=scoring_error,
        )
        row["pride_inference_mode"] = "eq8_transfer"
        row["pride_adjusted_choice"] = adjusted_letter
        row["peprior_json"] = json.dumps(self._calibration_state.peprior_probs)
        row["option_logprob_json"] = json.dumps(lp_map) if lp_map else None
        if score_adjusted is not None:
            row["score_status"] = score_adjusted.status
            row["is_correct"] = score_adjusted.is_correct

        return row

    async def _cyclic_rollout_prob_matrix_async(
        self,
        question_row: Any,
    ) -> np.ndarray:
        canon = self._build_options(question_row)
        letters = list(canon.keys())
        permutations = PermutationRunner._generate_permutations(canon)
        prompts = [
            PermutationRunner._build_permuted_prompt(
                question_row,
                perm,
                self._prompts["direct_mcq"],
            )
            for perm in permutations
        ]

        from choicebench.clients.vllm_client import VLLMClient
        raw: VLLMClient = self.backend._raw_client

        async def _score_one(p: str) -> np.ndarray:
            try:
                logprob_dict = await raw.score_options_async(p, letters)
                lp_map = {opt: logprob_dict.get(opt, -100.0) for opt in letters}
                return logprob_map_to_label_distribution(lp_map, letters=letters)
            except Exception as exc:
                logger.warning("PriDe calibration: score_options_async failed — %s", exc)
                return np.ones(len(letters), dtype=np.float64) / len(letters)

        rows = await asyncio.gather(*[_score_one(p) for p in prompts])

        n_success = sum(
            1 for r in rows
            if not np.allclose(r, np.ones(len(letters)) / len(letters))
        )
        if n_success == 0:
            qid = question_row.get("question_id", "<unknown>")
            raise RuntimeError(
                f"PriDe calibration: score_options_async() failed for all "
                f"{len(prompts)} permutations of calibration question {qid!r}."
            )

        return np.stack(rows, axis=0).astype(np.float64)
