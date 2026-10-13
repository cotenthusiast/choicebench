# tests/scripts/test_paper_replay_text_extraction_canonical.py
#
# Making the corrected text_extraction results canonical WITHOUT new inference: the saved raw generations are primed into
# a run's response cache so the fixed runner re-executes through the normal pipeline (and gets a real manifest), every request
# a cache hit. These tests pin that no request can reach the provider and that the replay is verified against the originals.

import asyncio
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from choicebench.backends.api_backend import APIBackend
from choicebench.clients.types import FAILURE_STATUS, SUCCESS_STATUS, ErrorInfo, ModelResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("replay_te", REPO_ROOT / "scripts/paper/replay_text_extraction_canonical.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


class CountingClient:
    provider, model_name = "openai", "m"
    _upstream_provider, _allow_fallbacks = "deepinfra", False

    def __init__(self):
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        return ModelResponse(provider="openai", model_name="m", status=FAILURE_STATUS, latency_seconds=0.0,
                             error=ErrorInfo("LiveCallError", "a live request escaped the replay cache", False, "request"))


def _backend(tmp_path, client):
    return APIBackend("openai", "m", client, tmp_path / "cache", 0.0, 1024, None, 10, "model_x")


def _rows():
    return [dict(prompt="P1", raw_text="HTTPS", finish_reason="stop", actual_upstream_provider="deepinfra", transport_status="success"),
            dict(prompt="P2", raw_text="FTP", finish_reason="length", actual_upstream_provider=None, transport_status="success"),
            dict(prompt="P3", raw_text=None, finish_reason=None, actual_upstream_provider=None, transport_status="failure")]


class TestPrimeCache:
    def test_every_successful_original_row_becomes_a_cache_hit_and_no_live_call_is_made(self, tmp_path):
        client = CountingClient()
        be = _backend(tmp_path, client)
        assert mod.prime_cache(be, _rows()) == 2
        out = asyncio.run(be.generate_batch(["P1", "P2"]))
        assert client.calls == 0
        assert [r.raw_text for r in out] == ["HTTPS", "FTP"] and [r.finish_reason for r in out] == ["stop", "length"]
        assert all(r.status == SUCCESS_STATUS for r in out)

    def test_a_failed_original_row_is_not_primed_so_it_can_never_masquerade_as_a_success(self, tmp_path):
        client = CountingClient()
        be = _backend(tmp_path, client)
        mod.prime_cache(be, _rows())
        out = asyncio.run(be.generate_batch(["P3"]))
        assert client.calls == 1 and out[0].status == FAILURE_STATUS

    def test_the_provider_pin_is_part_of_the_key_so_a_differently_pinned_client_misses(self, tmp_path):
        be = _backend(tmp_path, CountingClient())
        mod.prime_cache(be, _rows())
        other = CountingClient()
        other._upstream_provider = "someone-else"
        be2 = APIBackend("openai", "m", other, tmp_path / "cache", 0.0, 1024, None, 10, "model_x")
        asyncio.run(be2.generate_batch(["P1"]))
        assert other.calls == 1


def _orig(**kw):
    base = dict(question_id=["a", "b"], prompt=["P1", "P2"], raw_text=["x", "y"], finish_reason=["stop", "stop"],
                parsed_choice=["A", "B"], is_correct=[True, False])
    base.update(kw)
    return pd.DataFrame(base)


def _replayed(**kw):
    base = dict(question_id=["b", "a"], prompt=["P2", "P1"], raw_text=["y", "x"], finish_reason=["stop", "stop"],
                transport_status=["success", "success"], parsed_choice=["C", "D"], is_correct=[False, True])
    base.update(kw)
    return pd.DataFrame(base)


def _corrected(**kw):
    base = dict(question_id=["a", "b"], parsed_choice=["D", "C"], is_correct=[True, False])
    base.update(kw)
    return pd.DataFrame(base)


class TestVerifyReplay:
    def test_a_faithful_replay_has_no_problems_regardless_of_row_order(self):
        assert mod.verify_replay(_orig(), _replayed(), _corrected()) == []

    def test_a_changed_raw_generation_is_reported(self):
        assert any("raw_text" in p for p in mod.verify_replay(_orig(), _replayed(raw_text=["y", "CHANGED"]), _corrected()))

    def test_a_failed_replayed_row_is_reported(self):
        assert any("transport" in p for p in mod.verify_replay(_orig(), _replayed(transport_status=["success", "failure"]), _corrected()))

    def test_a_parse_that_differs_from_the_independent_corrected_derivation_is_reported(self):
        assert any("parsed_choice" in p for p in mod.verify_replay(_orig(), _replayed(parsed_choice=["C", "A"]), _corrected()))

    def test_a_missing_or_extra_question_is_reported(self):
        assert any("question_id" in p for p in mod.verify_replay(_orig(), _replayed().head(1), _corrected()))

    def test_null_parses_compare_equal_to_null_parses(self):
        assert mod.verify_replay(_orig(), _replayed(parsed_choice=[None, "D"]), _corrected(parsed_choice=["D", None])) == []
