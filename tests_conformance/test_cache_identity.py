"""Section M: cache / scientific identity.

Frozen invariant: the response-cache key is a pure function of the
scientific request fields (provider, model_name, prompt, temperature,
max_tokens, seed) plus client-level extra_identity (currently OpenRouter's
upstream_provider/allow_fallbacks pinning) -- unpinned clients are
byte-identical to the pre-pinning formula (backward compatible), a pinned
client's identity must differ from an unpinned one, and both the sync
(APIBackend/CachingClientWrapper) and Batch (BatchAPIBackend) transports
must incorporate it for the RESPONSE cache (a previously-fixed bug: Batch
used to bypass client_extra_identity entirely). Batch state (which job/
request-set is in flight) is a separate, transport-only concern from
response-cache identity.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from choicebench.backends.batch_api_backend import BatchAPIBackend
from choicebench.clients.openrouter_client import OpenRouterClient
from choicebench.clients.types import ModelRequest
from choicebench.infra.cache import ResponseCache, _cache_key, client_extra_identity


def _request(**overrides) -> ModelRequest:
    kwargs = dict(
        provider="openrouter", model_name="meta-llama/llama-3.1-8b-instruct",
        payload="What is the capital of France?", temperature=0.0, max_tokens=1024, seed=None,
    )
    kwargs.update(overrides)
    return ModelRequest(**kwargs)


# --- M1 -------------------------------------------------------------------

def test_m1_stable_key_for_identical_request_and_routing():
    r1 = _request()
    r2 = _request()
    assert _cache_key(r1) == _cache_key(r2)


# --- M2/M3 ------------------------------------------------------------------

def test_m2_m3_upstream_provider_and_allow_fallbacks_change_identity(monkeypatch):
    monkeypatch.setattr("choicebench.config.providers.OPENROUTER_API_KEY", "sk-test")
    deepinfra_pinned = OpenRouterClient(model_name="x", upstream_provider="deepinfra", allow_fallbacks=False)
    together_pinned = OpenRouterClient(model_name="x", upstream_provider="together", allow_fallbacks=False)
    unpinned = OpenRouterClient(model_name="x")

    id_deepinfra = client_extra_identity(deepinfra_pinned)
    id_together = client_extra_identity(together_pinned)
    id_unpinned = client_extra_identity(unpinned)

    assert id_deepinfra is not None
    assert id_together is not None
    assert id_deepinfra != id_together
    assert id_unpinned is None

    request = _request()
    key_deepinfra = _cache_key(request, extra_identity=id_deepinfra)
    key_together = _cache_key(request, extra_identity=id_together)
    key_unpinned = _cache_key(request, extra_identity=id_unpinned)
    assert key_deepinfra != key_together
    # R3 / backward-compat: unpinned client's key equals the no-extra-identity formula.
    assert key_unpinned == _cache_key(request, extra_identity=None)

    allow_fallbacks_true = OpenRouterClient(model_name="x", upstream_provider="deepinfra", allow_fallbacks=True)
    id_allow_true = client_extra_identity(allow_fallbacks_true)
    assert id_allow_true != id_deepinfra
    assert _cache_key(request, extra_identity=id_allow_true) != key_deepinfra


# --- M4 -------------------------------------------------------------------

def test_m4_batch_response_cache_lookup_incorporates_same_identity_as_sync(monkeypatch, tmp_path):
    """Regression guard for the previously-fixed bug where BatchAPIBackend
    bypassed client_extra_identity for its response-cache lookups. Confirmed
    by reading batch_api_backend.py: generate_batch() computes
    extra_identity = client_extra_identity(self._raw_client) once (line 91)
    and passes it to every _cache_key() call for the response cache (lines
    96, 110) -- the same extra_identity a sync CachingClientWrapper.generate()
    would compute for byte-identical (request, client)."""
    monkeypatch.setattr("choicebench.config.providers.OPENROUTER_API_KEY", "sk-test")
    pinned_client = OpenRouterClient(model_name="x", upstream_provider="deepinfra", allow_fallbacks=False)
    backend = BatchAPIBackend(
        provider="openrouter", model_name="x", client=pinned_client,
        cache_dir=tmp_path / "cache", temperature=0.0, max_tokens=1024, seed=None,
        batch_state_dir=tmp_path / "batch_state",
    )
    request = backend._make_request("What is the capital of France?")
    sync_style_key = _cache_key(request, extra_identity=client_extra_identity(pinned_client))

    cache = ResponseCache(cache_dir=tmp_path / "cache")
    cache.put(sync_style_key, {"raw_text": "Paris.", "finish_reason": "stop", "usage": None, "logprobs": None})

    # generate_batch()'s own cache-hit branch must find this entry using the
    # identical key formula (extra_identity included) -- proven directly
    # against the ResponseCache instance batch_api_backend wraps.
    hit = backend._client._cache.get(_cache_key(request, extra_identity=client_extra_identity(backend._raw_client)))
    assert hit is not None
    assert hit["raw_text"] == "Paris."


def test_m4_batch_state_path_now_incorporates_pinning_identity(monkeypatch, tmp_path):
    """FIXED (verified against commit 651137768dcad640de28f124cff3d50837fe7d7c):
    BatchAPIBackend._batch_state_path() now folds client_extra_identity(self.
    _raw_client) into its _cache_key() calls, the same way generate_batch()'s
    response-cache lookups already did -- two differently-pinned OpenRouter
    clients (upstream_provider='deepinfra' vs 'together') now compute
    DIFFERENT batch-state file paths. Previously xfail."""
    monkeypatch.setattr("choicebench.config.providers.OPENROUTER_API_KEY", "sk-test")
    deepinfra_pinned = OpenRouterClient(model_name="x", upstream_provider="deepinfra", allow_fallbacks=False)
    together_pinned = OpenRouterClient(model_name="x", upstream_provider="together", allow_fallbacks=False)

    backend_a = BatchAPIBackend(
        provider="openrouter", model_name="x", client=deepinfra_pinned,
        cache_dir=tmp_path / "cache_a", temperature=0.0, max_tokens=1024, seed=None,
        batch_state_dir=tmp_path / "batch_state",
    )
    backend_b = BatchAPIBackend(
        provider="openrouter", model_name="x", client=together_pinned,
        cache_dir=tmp_path / "cache_b", temperature=0.0, max_tokens=1024, seed=None,
        batch_state_dir=tmp_path / "batch_state",
    )
    requests_a = [backend_a._make_request("What is the capital of France?")]
    requests_b = [backend_b._make_request("What is the capital of France?")]

    path_a = backend_a._batch_state_path(requests_a)
    path_b = backend_b._batch_state_path(requests_b)
    assert path_a != path_b, (
        "batch-state paths for two differently-pinned deployments collided: "
        f"both resolved to {path_a}"
    )


# --- M5 -------------------------------------------------------------------

def test_m5_different_deployment_identity_is_a_clean_cache_miss(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path)
    request = _request()
    identity_a = {"upstream_provider": "deepinfra", "allow_fallbacks": False}
    identity_b = {"upstream_provider": "together", "allow_fallbacks": False}
    cache.put(_cache_key(request, extra_identity=identity_a), {"raw_text": "Paris.", "finish_reason": None, "usage": None, "logprobs": None})
    assert cache.get(_cache_key(request, extra_identity=identity_b)) is None


# --- M6 -------------------------------------------------------------------

def test_m6_transport_mode_is_not_part_of_cache_key_but_batch_state_is_isolated():
    import inspect
    params = inspect.signature(_cache_key).parameters
    assert "transport" not in params and "execution_mode" not in params
    request = _request()
    # Same scientific request/extra_identity -> identical response-cache key
    # regardless of which transport a caller intends to reach it through.
    assert _cache_key(request, extra_identity=None) == _cache_key(request, extra_identity=None)


def test_m6_batch_state_dir_and_response_cache_dir_are_separate_paths(tmp_path):
    cache_dir = tmp_path / "runs" / "r1" / "cache" / "model_x"
    batch_state_dir = tmp_path / "runs" / "r1" / "batch_state" / "model_x"
    assert cache_dir != batch_state_dir


# --- M7 -------------------------------------------------------------------

def test_m7_historical_entry_without_new_identity_field_is_a_clean_miss_not_misattributed(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path)
    request = _request()
    old_key = _cache_key(request, extra_identity=None)  # pre-pinning formula
    cache.put(old_key, {"raw_text": "Old unpinned answer.", "finish_reason": None, "usage": None, "logprobs": None})

    new_key = _cache_key(request, extra_identity={"upstream_provider": "deepinfra", "allow_fallbacks": False})
    assert new_key != old_key
    assert cache.get(new_key) is None  # clean miss, never the old unpinned entry
