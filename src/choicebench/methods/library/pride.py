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

vLLM-path caveat: on the vLLM backend, score_options is read from the top-20
generation logprobs, so any option letter NOT present in that top-20 is floored
to -100.0 (treated as near-impossible). For a model that spreads probability
mass thinly, a real option can be silently floored and lose. The HuggingFace
backend reads the true full-vocabulary logit and is not subject to this floor,
so HF and vLLM "PriDe" runs can produce different priors / debiased answers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from choicebench.clients.types import FAILURE_STATUS, SUCCESS_STATUS
from choicebench.constants import letters_for
from choicebench.parsing.types import PARSE_OK, ParseResult
from choicebench.pipeline.prompt_builder import build_direct_mcq_prompt
from choicebench.methods.base import ExperimentRunner
from choicebench.methods.library.permutation import PermutationRunner
from choicebench.methods.library.pride_math import (
    CalibrationState,
    apply_debiased_choice_from_defaults,
    average_prior_probability_vectors,
    calibration_state_from_sidecar,
    calibration_state_uniform,
    equation7_prior_from_rollouts,
    logprob_map_to_label_distribution,
)

logger = logging.getLogger(__name__)

_SIDE_SCHEMA_VERSION = 4


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

    Calibration sidecar reuse key (PF-9): a cached prior is reused only when the
    model_name slug, ``calibration_benchmark``, sorted calibration question ids,
    ``calibration_seed``, ``temperature``, ``max_tokens``, and ``prompt_version``
    all match. It does NOT capture model weights — two local checkpoints that
    share a basename (``org/model`` → ``org_model``) still collide on the sidecar
    path, so use distinct ``run_id`` / ``calibration_runs_dir`` for those.
    """

    requires_score_options: bool = True
    # This runner has a global calibration step, so it is subject to the
    # modal-k compatibility gate (see choicebench.pride_gate). The orchestrator
    # checks this flag; per-question methods leave it False.
    applies_modal_k_gate: bool = True

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
            model_label: str | None = None,
            *,
            calibration_n: int = 50,
            calibration_seed: int = 42,
            calibration_benchmark: str = "",
            calibration_runs_dir: Path | None = None,
            calibration_questions: list[dict] | None = None,
            preflight_questions: list[dict] | None = None,
            modal_k: int = 4,
            gate_summary: dict | None = None,
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
        if model_label is not None:
            kw["model_label"] = model_label
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

        # PriDe operates on a single option count (enforced upstream by the
        # modal-k gate). All calibration and evaluation questions must have
        # exactly this many options; the letter set is derived from it.
        self._modal_k = int(modal_k)
        self._modal_letters: tuple[str, ...] = tuple(letters_for(self._modal_k))
        self._gate_summary = dict(gate_summary) if gate_summary else None

        self._calibration_ready: bool = False
        self._calibration_state: CalibrationState = calibration_state_uniform(self._modal_letters)
        # PF-4: count permutation rollouts that fell back to a uniform
        # distribution during calibration. PriDe's per-eval-row score is a
        # single score_options call (no per-row permutations), so the
        # uniform-fallback degradation lives in calibration; these totals are
        # stamped onto every eval row so a degraded calibration is visible in
        # the CSV without scrolling the log.
        self._calibration_n_perm_failed: int = 0
        self._calibration_n_perm_total: int = 0

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
        # The orchestrator always passes a DataFrame, so normalize to records
        # exactly like base.ExperimentRunner.run_many does (FC-2 / MF-B).
        rows = question_rows.to_dict(orient="records")
        self._ensure_calibration()
        return [self.run_one(row, i) for i, row in enumerate(rows)]

    def _ensure_calibration(self) -> None:
        if self._calibration_ready:
            return

        # Only calibrate on questions that have exactly the modal option count,
        # so the per-question Eq.(7) priors are all the same length and average
        # cleanly. The gate guarantees the *evaluation* set is modal-k, but the
        # calibration split is separate, so filter it here too.
        eligible = [r for r in self._calibration_questions if self._has_modal_k_options(r)]
        dropped = len(self._calibration_questions) - len(eligible)
        if dropped:
            logger.info(
                "PriDe calibration: dropped %d/%d calibration question(s) whose "
                "option count != modal k=%d.",
                dropped, len(self._calibration_questions), self._modal_k,
            )
        cal_qids, cal_rows = _pick_calibration_rows(
            eligible,
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
                    # PF-9: also key on the generation settings the prior was fit
                    # under, so a sidecar is never silently reused across runs with
                    # a different temperature / max_tokens / prompt_version.
                    and blob.get("temperature") == self.temperature
                    and blob.get("max_tokens") == self.max_tokens
                    and blob.get("prompt_version") == self.prompt_version
                    # A prior fit at a different option count is not reusable.
                    and int(blob.get("n_options", -1)) == self._modal_k
                ):
                    self._calibration_state = calibration_state_from_sidecar(blob, self._modal_letters)
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
            self._calibration_state = calibration_state_uniform(self._modal_letters)
        else:
            for row in cal_rows:
                self._require_modal_k_options(row)
            prior_vectors: list[np.ndarray] = []
            for row in cal_rows:
                roll_mat = self._cyclic_rollout_prob_matrix(row)
                prior_vectors.append(equation7_prior_from_rollouts(roll_mat))

            pep_global = average_prior_probability_vectors(prior_vectors)
            self._calibration_state = CalibrationState(
                peprior_probs={
                    self._modal_letters[i]: float(pep_global[i])
                    for i in range(len(self._modal_letters))
                },
                epsilon=1e-12,
                estimation_question_ids=tuple(sorted_ids),
            )

        self._calibration_ready = True

        sidecar_payload = {
            "schema_version": _SIDE_SCHEMA_VERSION,
            "version": self._calibration_state.version,
            "calibration_seed": self._calibration_seed,
            # Generation settings the prior was fit under — part of the reuse key
            # (PF-9). model weights are NOT captured (only the model_name slug in
            # the filename), so two checkpoints sharing a basename still collide.
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_version": self.prompt_version,
            "n_options": len(self._modal_letters),
            "calibration_question_ids": list(sorted_ids),
            "peprior_probs": {
                L: float(self._calibration_state.peprior_probs.get(L, 0.0))
                for L in self._modal_letters
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

        self._calibration_n_perm_total += len(prompts)
        self._calibration_n_perm_failed += len(prompts) - n_success

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
        self._require_modal_k_options(question_row)
        letters = list(options.keys())
        prompt = self._build_prompt(question_row)

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

        return self._build_debiased_row(
            question_row, sample_index, prompt, letters, lp_map, scoring_error
        )

    def _build_debiased_row(
            self,
            question_row: Any,
            sample_index: int,
            prompt: str,
            letters: list[str],
            lp_map: dict[str, float],
            scoring_error: str | None,
    ) -> dict:
        """Shared row assembly for run_one() / _score_question_async().

        Both inference paths differ only in *how* lp_map (this question's
        per-letter logprobs) is obtained — sync backend.score_options() vs.
        async score_options_async(). Everything downstream (Eq. 8 debiasing,
        scoring, result-row assembly) is identical, so it lives here once
        instead of twice.
        """
        adjusted_letter: str | None = None
        score_adjusted = None

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
        # PF-4: calibration-rollout permutation failures (see __init__ note).
        row["n_permutations_total"] = self._calibration_n_perm_total
        row["n_permutations_failed"] = self._calibration_n_perm_failed
        # Mirror cyclic_logprob.py: the debiased letter is PriDe's authoritative
        # answer, so it must populate parsed_choice — the column both built-in
        # metrics (Accuracy, MAD) read. Without this PriDe reports 0.0 / NaN.
        if adjusted_letter is not None:
            row["parsed_choice"] = adjusted_letter
            row["parse_status"] = PARSE_OK
        # PriDe never calls generate(), so _build_result_row's transport_status
        # defaults to None (no call was made). answer_status is set explicitly
        # here (FC-4 / PF-3) since parsed_result=None was passed above (the row
        # is patched with parsed_choice/parse_status directly instead) —
        # success when a debiased answer was produced, failure when
        # score_options yielded nothing to debias.
        row["answer_status"] = SUCCESS_STATUS if adjusted_letter is not None else FAILURE_STATUS
        if score_adjusted is not None:
            row["score_status"] = score_adjusted.status
            row["is_correct"] = score_adjusted.is_correct

        self._stamp_gate(row)
        return row

    def _build_prompt(self, question_row: Any) -> str:
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            options=self._build_options(question_row),
        )

    def _require_modal_k_options(self, question_row: Any) -> None:
        options = self._build_options(question_row)
        if list(options.keys()) != list(self._modal_letters):
            raise ValueError(
                f"PriDe requires exactly {self._modal_k} valid options "
                f"({list(self._modal_letters)}); question "
                f"{question_row.get('question_id', '<unknown>')!r} has valid "
                f"options {list(options)}. The modal-k gate should have excluded "
                "this question before the run."
            )

    def _has_modal_k_options(self, question_row: Any) -> bool:
        """True if the row resolves to exactly the modal option count."""
        try:
            options = self._build_options(question_row)
        except ValueError:
            return False
        return list(options.keys()) == list(self._modal_letters)

    def _stamp_gate(self, row: dict) -> None:
        """Record the modal-k gate accounting on a result row, if gated."""
        if not self._gate_summary:
            return
        row["gate_modal_k"] = self._gate_summary.get("modal_k")
        row["gate_n_total"] = self._gate_summary.get("n_total")
        row["gate_n_evaluated"] = self._gate_summary.get("n_evaluated")
        row["gate_n_excluded"] = self._gate_summary.get("n_excluded")

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
        self._require_modal_k_options(question_row)
        letters = list(options.keys())
        prompt = self._build_prompt(question_row)

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

        return self._build_debiased_row(
            question_row, sample_index, prompt, letters, lp_map, scoring_error
        )

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
