# tests/runners/test_prompt_render_matrix.py
#
# BATCH 8 — A5: prompt-render matrix (validates F4 completely).
#
# Matrix: every shipped runner × every shipped prompt bundle renders
# without KeyError, and {subject} substitution behaves as declared:
#   * direct-mcq-family templates accept subject everywhere;
#   * free_text / option_matching builders deliberately do NOT take a
#     subject — a bundle putting {subject} there fails CLOSED (KeyError)
#     rather than rendering the literal "None" or silently dropping it.
#
# The declared raise is pinned, not an accident: it is the same
# fail-closed philosophy ratified with F4.

from pathlib import Path

import pytest

from tests.runners.conftest import MockBackend
from choicebench.backends.dummy_backend import DummyBackend
from choicebench.benchmarks.base import make_normalized_row
from choicebench.methods.direct_mcq import DirectMCQRunner
from choicebench.methods.library.cyclic_logprob import CyclicLogprobRunner
from choicebench.methods.library.direct_logprob import DirectLogprobRunner
from choicebench.methods.library.permutation import PermutationRunner
from choicebench.methods.library.pride import PriDeRunner
from choicebench.methods.library.shuffled_baseline import ShuffledBaselineRunner
from choicebench.methods.library.two_stage import TwoStageRunner
from choicebench.pipeline.prompt_builder import build_rotations

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _REPO_ROOT / "prompts"

_BUNDLES = ["v1", "pride_repro"]


def _row() -> dict:
    return make_normalized_row(
        subject="abstract_algebra",
        question_text="What is 2 + 2?",
        choices=["3", "4", "5", "6"],
        correct_index=1,
    )


def _runner_kwargs(**overrides) -> dict:
    kwargs = dict(
        backend=DummyBackend(), split_name="test", prompts_dir=_PROMPTS_DIR,
        run_id="rendermatrix", temperature=0.0, max_tokens=16, seed=42,
    )
    kwargs.update(overrides)
    return kwargs


def _direct_mcq_runner(bundle: str):
    return DirectMCQRunner(method_name="direct_mcq", prompt_version=bundle,
                           **_runner_kwargs())


def _permutation_runner(bundle: str):
    return PermutationRunner(method_name="cyclic_permutation",
                             prompt_version=bundle, **_runner_kwargs())


def _cyclic_logprob_runner(bundle: str):
    return CyclicLogprobRunner(method_name="cyclic_logprob",
                               prompt_version=bundle, **_runner_kwargs())


def _pride_runner(bundle: str):
    return PriDeRunner(method_name="pride", prompt_version=bundle,
                       **_runner_kwargs())


def _direct_logprob_runner(bundle: str):
    return DirectLogprobRunner(method_name="direct_logprob",
                               prompt_version=bundle, **_runner_kwargs())


def _shuffled_baseline_runner(bundle: str):
    return ShuffledBaselineRunner(
        method_name="shuffled_baseline", prompt_version=bundle,
        backend=MockBackend(responses=["B"]), prompts_dir=_PROMPTS_DIR,
        run_id="rendermatrix", split_name="test", temperature=0.0,
        max_tokens=16, seed=42,
    )


def _two_stage_runner(bundle: str):
    return TwoStageRunner(
        method_name="two_stage", prompt_version=bundle,
        backend=MockBackend(responses=["4", "B"]), prompts_dir=_PROMPTS_DIR,
        run_id="rendermatrix", split_name="test", temperature=0.0,
        max_tokens=16, seed=42, fallback_on_parse_failure=False,
    )


def _cyclic_logprob_prompts(bundle: str) -> list[str]:
    from choicebench.pipeline.prompt_builder import build_permuted_prompt

    runner = _cyclic_logprob_runner(bundle)
    options = runner._build_options(_row())
    return [
        build_permuted_prompt(_row(), rot.mapping, runner._prompts["direct_mcq"])
        for rot in build_rotations(options)
    ]


_RENDERERS = {
    "direct_mcq": lambda bundle: _direct_mcq_runner(bundle)._build_prompt(_row()),
    "cyclic_permutation": lambda bundle: [
        _permutation_runner(bundle)._build_permuted_prompt(
            _row(), rot.mapping, _permutation_runner(bundle)._prompts["direct_mcq"]
        )
        for rot in build_rotations(_permutation_runner(bundle)._build_options(_row()))
    ],
    "cyclic_logprob": _cyclic_logprob_prompts,
    "pride": lambda bundle: _pride_runner(bundle)._build_prompt(_row()),
    "direct_logprob": lambda bundle: (
        _direct_logprob_runner(bundle)._build_prompt(_row())
    ),
    "shuffled_baseline": lambda bundle: _shuffled_baseline_runner(bundle).run_one(
        _row(), 0
    )["prompt"],
    "two_stage": lambda bundle: _two_stage_runner(bundle).run_one(
        _row(), 0
    )["prompt"],
}


class TestShippedRenderMatrix:
    """Every runner × shipped bundle must render without error."""

    @pytest.mark.parametrize("bundle", _BUNDLES)
    @pytest.mark.parametrize("runner_name",
                             ["direct_mcq", "pride", "direct_logprob"])
    def test_direct_mcq_family_renders(self, runner_name, bundle):
        out = _RENDERERS[runner_name](bundle)
        assert isinstance(out, str) and "{question}" not in out
        assert "Options:" in out

    @pytest.mark.parametrize("bundle", _BUNDLES)
    def test_permutation_runners_render_all_rotations(self, bundle):
        for name in ("cyclic_permutation", "cyclic_logprob"):
            prompts = _RENDERERS[name](bundle)
            assert len(prompts) == 4
            for prompt in prompts:
                assert "{question}" not in prompt

    @pytest.mark.parametrize("bundle", _BUNDLES)
    def test_generation_runners_render_via_run_one(self, bundle):
        shuffled = _RENDERERS["shuffled_baseline"](bundle)
        two_stage = _RENDERERS["two_stage"](bundle)
        assert "{question}" not in shuffled and "{options}" not in shuffled
        assert "{free_text}" not in two_stage and "{question}" not in two_stage


class TestSubjectSubstitutionContract:
    def test_subject_placeholder_bundles_render_subject_in_direct_mcq_paths(
        self, tmp_path
    ):
        """A bundle with {subject} in ALL templates renders the subject on
        every direct-mcq-family path (F4 fix), including permutation
        rotations."""
        bundle_dir = tmp_path / "all_subject"
        bundle_dir.mkdir()
        for name in ("direct_mcq", "free_text", "option_matching"):
            template = (_PROMPTS_DIR / "v1" / f"{name}.txt").read_text()
            (bundle_dir / f"{name}.txt").write_text(
                template.replace("Question:", "Subject line about {subject}.\nQuestion:")
            )
        row = _row()

        prompt = DirectMCQRunner(
            method_name="direct_mcq", prompt_version="all_subject",
            prompts_dir=tmp_path, run_id="r", split_name="test",
            temperature=0.0, max_tokens=8, seed=1, backend=DummyBackend(),
        )._build_prompt(row)
        assert "about abstract algebra" in prompt

        # Permutation rotations carry the subject too.
        runner = PermutationRunner(
            method_name="cyclic_permutation", prompt_version="all_subject",
            prompts_dir=tmp_path, run_id="r", split_name="test",
            temperature=0.0, max_tokens=8, seed=1, backend=DummyBackend(),
        )
        options = runner._build_options(row)
        for rot in build_rotations(options):
            rendered = runner._build_permuted_prompt(
                row, rot.mapping, runner._prompts["direct_mcq"]
            )
            assert "about abstract algebra" in rendered

    def test_free_text_and_option_matching_fail_closed_on_subject_bundle(
        self, tmp_path
    ):
        """Declared behavior: those two builders take no subject kwarg; a
        bundle referencing {subject} there raises KeyError (never renders
        the literal 'None')."""
        bundle_dir = tmp_path / "all_subject"
        bundle_dir.mkdir()
        for name in ("direct_mcq", "free_text", "option_matching"):
            template = (_PROMPTS_DIR / "v1" / f"{name}.txt").read_text()
            (bundle_dir / f"{name}.txt").write_text(
                template.replace("Question:", "Subject line about {subject}.\nQuestion:")
            )

        two_stage = TwoStageRunner(
            method_name="two_stage", prompt_version="all_subject",
            backend=MockBackend(responses=["4", "B"]), prompts_dir=tmp_path,
            run_id="r", split_name="test", temperature=0.0, max_tokens=8,
            seed=1, fallback_on_parse_failure=False,
        )
        with pytest.raises(KeyError):
            two_stage.run_one(_row(), 0)
