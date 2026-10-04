# tests/scripts/test_paper_run_flip_rotations_reuse0.py
#
# Rotation-0-reuse flip runs: rotation 0 is IMPORTED from the canonical production file and never sent to a backend;
# only rotations 1..n-1 are generated. Every test uses a scripted fake backend that records the prompts it receives.

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from choicebench.backends.api_backend import APIBackend
from choicebench.backends.dummy_backend import DummyBackend
from choicebench.clients.types import FAILURE_STATUS, SUCCESS_STATUS, ErrorInfo, ModelResponse
from choicebench.config.schema import GenerationKwargsConfig, ModelConfig
from choicebench.methods.library.text_extraction import TextExtractionRunner
from choicebench.methods.library.two_stage import TwoStageRunner
from choicebench.methods.library.visible_llm_matcher import VisibleLLMMatcherRunner
from choicebench.pipeline.prompt_builder import build_direct_mcq_prompt, build_option_matching_prompt, build_rotations

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("flip_reuse0", REPO_ROOT / "scripts/paper/run_flip_rotations_reuse0.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

OPTIONS = ["FTP", "HTTP", "HTTPS", "SMTP"]
CFG = ModelConfig(backend="dummy", model_name_or_path="m", generation_kwargs=GenerationKwargsConfig(max_new_tokens=1024, temperature=0.0))


def _letter_of(prompt: str, text: str):
    m = re.search(rf"^([A-Z])\. {re.escape(text)}\s*$", prompt, flags=re.M)
    return m.group(1) if m else None


class Responder(APIBackend):
    """Async-capable fake: answers via fn(prompt) -> str|None (None = transport failure); records every prompt."""

    def __init__(self, fn):
        self._provider, self._model_name, self._temperature, self._max_tokens, self._seed = "openai", "mock", 0.0, 1024, 42
        self._concurrency_limit = 10
        self.fn, self.prompts = fn, []

    async def generate_batch(self, prompts):
        self.prompts.extend(prompts)
        out = []
        for p in prompts:
            text = self.fn(p)
            out.append(ModelResponse(provider="openai", model_name="mock", status=SUCCESS_STATUS if text is not None else FAILURE_STATUS,
                                     latency_seconds=0.25, raw_text=text, actual_upstream_provider="deepinfra",
                                     error=None if text is not None else ErrorInfo("ProviderMismatchError", "pinned deepinfra", False, "request")))
        return out


def _rt(cls, method, pv="v1"):
    return cls(backend=DummyBackend(), method_name=method, split_name="test", prompt_version=pv, prompts_dir=REPO_ROOT / "prompts",
               run_id="t", seed=42, benchmark_name="mmlu")


def _q(qid, options=OPTIONS, correct=2):
    return dict(run_id="prod_run", experiment_id="e", condition_id="cond_1", benchmark_name="mmlu", question_id=qid, split_name="test",
                subject="computer_security", prompt_version="v1", provider="openai", model_name="gpt-4.1-mini", temperature=0.0,
                max_tokens=1024, seed=42, question_text="Which protocol is primarily used to securely browse websites?",
                choices_json=json.dumps([{"text": t, "source_index": i} for i, t in enumerate(options)]),
                correct_index=correct, correct_option="ABCDE"[correct], n_choices=len(options))


def _two_stage_prod(tmp_path, n=2, options=OPTIONS, family="two_stage", tamper_prompt=False, wrong_parse=False):
    pv = "v1_reasoning" if family == "reasoning_two_stage" else "v1"
    rn = _rt(TwoStageRunner, family, pv)
    rows = []
    for i in range(n):
        q = _q(f"q{i}", options)
        opts = rn._build_options(q)
        rot0 = build_rotations(opts)[0]
        prompt = build_option_matching_prompt(template=rn._prompts["option_matching"], question=q["question_text"], free_text="HTTPS", options=rot0.mapping)
        q.update(method_name=family, prompt_version=pv, free_text_response="HTTPS", prompt=("TAMPERED" if tamper_prompt else prompt),
                 raw_text="C", parsed_choice=("A" if wrong_parse else "C"), transport_status="success")
        rows.append(q)
    path = tmp_path / f"{family}_prod.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _answer_two_stage(prompt):
    return _letter_of(prompt, "HTTPS") or "Z"


def _run(path, family, out, backend, **kw):
    return mod.run(production_csv=path, family=family, model_config=CFG, run_id="flip_t", output_path=out, backend=backend, **kw)


def _rows(out):
    return pd.read_csv(out).to_dict(orient="records")


class TestRotationZeroIsReusedNeverSent:
    def test_only_rotations_1_to_n_minus_1_are_sent_and_rotation_zero_prompt_is_absent(self, tmp_path):
        path = _two_stage_prod(tmp_path)
        b = Responder(_answer_two_stage)
        _run(path, "two_stage", tmp_path / "o.csv", b)
        assert len(b.prompts) == 2 * 3                      # 2 questions x (4 - 1) rotations
        prod_prompts = set(pd.read_csv(path)["prompt"])
        assert not (set(b.prompts) & prod_prompts)          # rotation 0 (the production prompt) is never sent

    def test_combined_trace_has_n_entries_production_first_and_correct_canonical_answers(self, tmp_path):
        path = _two_stage_prod(tmp_path)
        _run(path, "two_stage", tmp_path / "o.csv", Responder(_answer_two_stage))
        r = _rows(tmp_path / "o.csv")[0]
        assert json.loads(r["per_rotation_choices_json"]) == ["C", "C", "C", "C"]   # option text tracked through every rotation
        assert json.loads(r["per_rotation_source_json"]) == ["production", "generated", "generated", "generated"]
        assert json.loads(r["generated_rotation_indices_json"]) == [1, 2, 3]
        assert len(json.loads(r["per_rotation_raw_text_json"])) == 4 and json.loads(r["per_rotation_raw_text_json"])[0] == "C"

    def test_a_three_option_question_generates_two_rotations(self, tmp_path):
        path = _two_stage_prod(tmp_path, n=1, options=["FTP", "HTTP", "HTTPS"])
        b = Responder(_answer_two_stage)
        _run(path, "two_stage", tmp_path / "o.csv", b)
        assert len(b.prompts) == 2 and len(json.loads(_rows(tmp_path / "o.csv")[0]["per_rotation_choices_json"])) == 3

    def test_provenance_columns_identify_the_reused_production_file(self, tmp_path):
        path = _two_stage_prod(tmp_path, family="reasoning_two_stage")
        _run(path, "reasoning_two_stage", tmp_path / "o.csv", Responder(_answer_two_stage))
        r = _rows(tmp_path / "o.csv")[0]
        assert r["rotation0_source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert r["rotation0_source_file"] == str(path) and r["rotation0_source_run_id"] == "prod_run"
        assert r["rotation0_source_condition_id"] == "cond_1" and r["method_name"] == "reasoning_two_stage"
        assert r["run_id"] == "flip_t" and r["model_name"] == "gpt-4.1-mini" and r["benchmark_name"] == "mmlu"
        assert json.loads(r["per_rotation_upstream_provider_json"]) == [None, "deepinfra", "deepinfra", "deepinfra"]

    def test_failed_transport_is_a_null_entry_not_a_dropped_rotation(self, tmp_path):
        path = _two_stage_prod(tmp_path, n=1)
        calls = {"n": 0}

        def flaky(p):
            calls["n"] += 1
            return None if calls["n"] == 2 else _answer_two_stage(p)

        _run(path, "two_stage", tmp_path / "o.csv", Responder(flaky))
        r = _rows(tmp_path / "o.csv")[0]
        assert json.loads(r["per_rotation_choices_json"]) == ["C", "C", None, "C"]
        assert json.loads(r["per_rotation_transport_status_json"])[2] == FAILURE_STATUS


class TestGuards:
    def test_refuses_when_the_production_prompt_is_not_the_rotation_zero_prompt(self, tmp_path):
        path = _two_stage_prod(tmp_path, tamper_prompt=True)
        with pytest.raises(ValueError, match="rotation 0"):
            _run(path, "two_stage", tmp_path / "o.csv", Responder(_answer_two_stage))

    def test_refuses_when_production_parse_differs_from_the_rerun_parse(self, tmp_path):
        path = _two_stage_prod(tmp_path, wrong_parse=True)
        with pytest.raises(ValueError, match="parsed_choice"):
            _run(path, "two_stage", tmp_path / "o.csv", Responder(_answer_two_stage))

    def test_refuses_a_production_file_of_the_wrong_family(self, tmp_path):
        path = _two_stage_prod(tmp_path)
        with pytest.raises(ValueError, match="reasoning_two_stage"):
            _run(path, "reasoning_two_stage", tmp_path / "o.csv", Responder(_answer_two_stage))

    def test_output_may_not_be_the_production_file(self, tmp_path):
        path = _two_stage_prod(tmp_path)
        with pytest.raises(ValueError, match="production"):
            _run(path, "two_stage", path, Responder(_answer_two_stage))

    def test_resume_skips_completed_questions_and_sends_nothing(self, tmp_path):
        path = _two_stage_prod(tmp_path)
        out = tmp_path / "o.csv"
        _run(path, "two_stage", out, Responder(_answer_two_stage), chunk_size=1)
        b = Responder(_answer_two_stage)
        assert _run(path, "two_stage", out, b, chunk_size=1) == 0 and b.prompts == []
        assert len(_rows(out)) == 2


class TestTextExtraction:
    def _prod(self, tmp_path):
        rn = _rt(TextExtractionRunner, "text_extraction")
        q = _q("q0")
        rot0 = build_rotations(rn._build_options(q))[0]
        q.update(method_name="text_extraction", prompt=build_direct_mcq_prompt(template=rn._prompts["text_extraction"], question=q["question_text"], options=rot0.mapping, subject=q["subject"]),
                 raw_text="HTTPS", parsed_choice="A", transport_status="success")    # the buggy production parse (letter parser)
        path = tmp_path / "te_prod.csv"
        pd.DataFrame([q]).to_csv(path, index=False)
        return path

    def test_rotation_zero_is_reparsed_with_the_cascade_and_the_original_parse_is_kept(self, tmp_path):
        path = self._prod(tmp_path)
        _run(path, "text_extraction", tmp_path / "o.csv", Responder(lambda p: "HTTPS"))
        r = _rows(tmp_path / "o.csv")[0]
        assert json.loads(r["per_rotation_choices_json"]) == ["C", "C", "C", "C"]
        assert r["rotation0_original_parsed_choice"] == "A" and "cascade" in r["rotation0_parser"]
        assert json.loads(r["per_rotation_raw_text_json"]) == ["HTTPS"] * 4

    def test_every_generated_rotation_uses_the_cascade_not_the_letter_parser(self, tmp_path):
        path = self._prod(tmp_path)
        _run(path, "text_extraction", tmp_path / "o.csv", Responder(lambda p: "B"))   # a bare letter is not option text
        assert json.loads(_rows(tmp_path / "o.csv")[0]["per_rotation_choices_json"])[1:] == [None, None, None]


class TestVisibleLlmMatcher:
    def _prod(self, tmp_path):
        rn = _rt(VisibleLLMMatcherRunner, "visible_llm_matcher")
        q = _q("q0")
        rot0 = build_rotations(rn._build_options(q))[0]
        q.update(method_name="visible_llm_matcher", reused_extracted_text="HTTPS",
                 prompt=build_option_matching_prompt(template=rn._prompts["option_matching"], question=q["question_text"], free_text="HTTPS", options=rot0.mapping),
                 raw_text="C", parsed_choice="C", transport_status="success")
        path = tmp_path / "vis_prod.csv"
        pd.DataFrame([q]).to_csv(path, index=False)
        flip = tmp_path / "te_flip.csv"
        pd.DataFrame([dict(question_id="q0", model_name="gpt-4.1-mini", benchmark_name="mmlu", method_name="text_extraction",
                           per_rotation_raw_text_json=json.dumps(["HTTPS", "HTTPS", None, "HTTPS"]))]).to_csv(flip, index=False)
        return path, flip

    def test_rotation_zero_input_is_the_production_text_and_later_rotations_use_the_flip_run_texts(self, tmp_path):
        path, flip = self._prod(tmp_path)
        b = Responder(_answer_two_stage)
        _run(path, "visible_llm_matcher", tmp_path / "o.csv", b, text_extraction_flip_csv=flip)
        assert len(b.prompts) == 2                           # rotation 2's stage-1 text failed upstream: skipped, no call
        r = _rows(tmp_path / "o.csv")[0]
        assert json.loads(r["per_rotation_choices_json"]) == ["C", "C", None, "C"]
        assert json.loads(r["per_rotation_source_json"]) == ["production", "generated", "skipped_no_stage1_text", "generated"]

    def test_requires_the_text_extraction_flip_file(self, tmp_path):
        path, _ = self._prod(tmp_path)
        with pytest.raises(ValueError, match="text_extraction_flip_csv"):
            _run(path, "visible_llm_matcher", tmp_path / "o.csv", Responder(_answer_two_stage))


class TestRotationZeroReplayForThePilot:
    def test_replay_sends_only_rotation_zero_prompts_and_reports_agreement(self, tmp_path):
        path = _two_stage_prod(tmp_path, n=3)
        b = Responder(lambda p: "C")
        summary = mod.replay_rotation0(production_csv=path, family="two_stage", model_config=CFG, run_id="replay_t",
                                       output_path=tmp_path / "replay.csv", n=2, backend=b)
        assert len(b.prompts) == 2 and set(b.prompts) <= set(pd.read_csv(path)["prompt"])
        assert summary == {"n": 2, "agree": 2, "agreement": 1.0}
        assert list(pd.read_csv(tmp_path / "replay.csv")["agree"]) == [True, True]


class TestFailureVisibility:
    def test_each_failed_rotation_records_its_error_type(self, tmp_path):
        path = _two_stage_prod(tmp_path, n=1)
        calls = {"n": 0}

        def flaky(p):
            calls["n"] += 1
            return None if calls["n"] == 2 else _answer_two_stage(p)

        _run(path, "two_stage", tmp_path / "o.csv", Responder(flaky))
        assert json.loads(_rows(tmp_path / "o.csv")[0]["per_rotation_error_json"]) == [None, None, "ProviderMismatchError", None]

    def test_a_chunk_that_mostly_fails_aborts_and_writes_nothing(self, tmp_path):
        path = _two_stage_prod(tmp_path, n=4)                    # 4 questions x 3 generated = 12 calls, all failing
        out = tmp_path / "o.csv"
        with pytest.raises(RuntimeError, match="transport failure rate"):
            _run(path, "two_stage", out, Responder(lambda p: None), chunk_size=4)
        assert not out.exists()

    def test_a_few_failures_below_the_threshold_do_not_abort(self, tmp_path):
        path = _two_stage_prod(tmp_path, n=4)
        calls = {"n": 0}

        def one_failure(p):
            calls["n"] += 1
            return None if calls["n"] == 1 else _answer_two_stage(p)

        assert _run(path, "two_stage", tmp_path / "o.csv", Responder(one_failure), chunk_size=4) == 4
