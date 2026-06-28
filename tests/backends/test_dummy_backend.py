# tests/backends/test_dummy_backend.py

import pytest

from choicebench.backends.dummy_backend import DummyBackend


def test_identity_properties():
    backend = DummyBackend()
    assert backend.provider == "dummy"
    assert backend.model_name == "dummy"
    assert backend.supports_logprobs is True


def test_generate_returns_default_text():
    backend = DummyBackend()
    assert backend.generate("anything") == "The answer is A."


def test_generate_returns_custom_text():
    backend = DummyBackend(fixed_text="Final answer: C")
    assert backend.generate("anything") == "Final answer: C"


def test_score_options_returns_scores_in_request_order():
    backend = DummyBackend()
    scores = backend.score_options("prompt", ["B", "A", "C"])
    assert scores == [-1.5, -0.1, -2.0]


def test_score_options_unknown_option_gets_floor():
    backend = DummyBackend()
    assert backend.score_options("prompt", ["Z"]) == [-5.0]


def test_score_options_custom_scores():
    backend = DummyBackend(fixed_scores={"A": 0.0, "B": -1.0})
    assert backend.score_options("p", ["A", "B"]) == [0.0, -1.0]


def test_score_options_empty_raises():
    backend = DummyBackend()
    with pytest.raises(ValueError, match="options must not be empty"):
        backend.score_options("prompt", [])
