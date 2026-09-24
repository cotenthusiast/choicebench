# tests/pipeline/test_prompt_builder_optional_templates.py
#
# load_prompt_templates() must keep requiring the 3 original templates
# (direct_mcq/free_text/option_matching) for every bundle -- but methods
# introduced after those 3 (e.g. independent_hypothesis) need their own
# named template too, and forcing every existing bundle to grow one just
# because ONE new method needs it would break every config that doesn't
# use that method. Extra .txt files in a version directory should load
# opportunistically, keyed by filename stem, without being required.

import pytest

from choicebench.pipeline.prompt_builder import load_prompt_templates


def _write_required_templates(version_dir):
    version_dir.mkdir(parents=True)
    (version_dir / "direct_mcq.txt").write_text("dm {question} {options}")
    (version_dir / "free_text.txt").write_text("ft {question}")
    (version_dir / "option_matching.txt").write_text("om {question} {free_text} {options}")


def test_bundle_without_extra_templates_is_unaffected(tmp_path):
    version_dir = tmp_path / "v1"
    _write_required_templates(version_dir)
    templates = load_prompt_templates("v1", tmp_path)
    assert set(templates) == {"direct_mcq", "free_text", "option_matching"}


def test_extra_template_loads_opportunistically(tmp_path):
    version_dir = tmp_path / "v1_ihs"
    _write_required_templates(version_dir)
    (version_dir / "independent_hypothesis.txt").write_text("Question: {question}\nHypothesis: {option_text}")
    templates = load_prompt_templates("v1_ihs", tmp_path)
    assert "independent_hypothesis" in templates
    assert templates["independent_hypothesis"] == "Question: {question}\nHypothesis: {option_text}"


def test_required_templates_are_still_required(tmp_path):
    version_dir = tmp_path / "v1_incomplete"
    version_dir.mkdir(parents=True)
    (version_dir / "direct_mcq.txt").write_text("dm")
    # free_text.txt / option_matching.txt deliberately missing
    with pytest.raises(FileNotFoundError, match="Missing prompt template"):
        load_prompt_templates("v1_incomplete", tmp_path)
