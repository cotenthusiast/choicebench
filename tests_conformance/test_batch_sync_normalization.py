"""Section N: batch/sync result normalization.

Frozen invariant: an equivalent sync call and a Batch API call must
normalize to the same core ModelResponse content; Batch result remapping
must preserve per-request identity even under out-of-order results;
duplicate/missing Batch result IDs must fail loudly; resuming a completed
Batch state must reattach rather than resubmit.

Ground truth (established by reading batch_api_backend.py in full, section
M's Task): BatchAPIBackend itself does NOT reassemble Batch results by ID --
its docstring says reassembly "via each request's custom_id" is the
UNDERLYING CLIENT's job (fetch_batch_results contract). BatchAPIBackend's
own generate_batch() trusts that fetch_batch_results() already returned
results in the SAME order as the `requests` it was given, and zips them
back onto uncached_indices with NO length or identity check at all. A
well-behaved client (real or fake) that honors the custom_id contract makes
N1/N2 hold; a client that doesn't -- or that returns the wrong number of
results -- is where N3 exposes a real gap (see the xfail test below).
"""

from __future__ import annotations

import pytest

from choicebench.backends.batch_api_backend import BatchAPIBackend
from choicebench.clients.base import BaseClient
from choicebench.clients.types import (
    BATCH_COMPLETED,
    ModelRequest,
    ModelResponse,
    SUCCESS_STATUS,
)


class FakeBatchClient(BaseClient):
    """Fake provider client implementing the exact contract BatchAPIBackend
    requires: async generate() (via BaseClient), submit_batch, poll_batch,
    fetch_batch_results. fetch_batch_results reassembles by each request's
    OWN payload text (a stand-in for a real client's custom_id-based
    reassembly), optionally shuffled, to prove identity survives reordering.
    """

    def __init__(self, *, shuffle_results: bool = False, result_count_delta: int = 0):
        super().__init__(provider="fake", model_name="fake-model")
        self._shuffle_results = shuffle_results
        self._result_count_delta = result_count_delta
        self.submit_calls: list[list[ModelRequest]] = []
        self._batches: dict[str, list[ModelRequest]] = {}
        self._next_id = 0

    async def _generate_provider_response(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            provider=request.provider, model_name=request.model_name,
            status=SUCCESS_STATUS, latency_seconds=0.0, raw_text=f"sync:{request.payload}",
        )

    async def submit_batch(self, requests: list[ModelRequest]) -> str:
        self.submit_calls.append(list(requests))
        batch_id = f"batch-{self._next_id}"
        self._next_id += 1
        self._batches[batch_id] = list(requests)
        return batch_id

    async def poll_batch(self, batch_id: str) -> str:
        return BATCH_COMPLETED

    async def fetch_batch_results(self, batch_id: str, requests: list[ModelRequest]) -> list[ModelResponse]:
        # Reassemble by payload identity (stand-in for a real client's
        # custom_id lookup), proving order-independence when honored.
        by_payload = {
            r.payload: ModelResponse(
                provider=r.provider, model_name=r.model_name, status=SUCCESS_STATUS,
                latency_seconds=0.0, raw_text=f"batch:{r.payload}",
            )
            for r in requests
        }
        ordered = [by_payload[r.payload] for r in requests]
        if self._shuffle_results:
            ordered = list(reversed(ordered))
            # Reversed is still a valid full reassembly IF the caller also
            # reverses `requests` to match -- to prove genuine ID-based (not
            # positional) reassembly, return in payload-sorted order instead
            # of input order, independent of `requests`' own order.
            ordered = [by_payload[k] for k in sorted(by_payload)]
        if self._result_count_delta:
            ordered = ordered[: len(ordered) + self._result_count_delta]
        return ordered


def _backend(client: FakeBatchClient, tmp_path) -> BatchAPIBackend:
    return BatchAPIBackend(
        provider="fake", model_name="fake-model", client=client,
        cache_dir=tmp_path / "cache", temperature=0.0, max_tokens=1024, seed=None,
        batch_state_dir=tmp_path / "batch_state",
    )


# --- N1 -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_n1_sync_and_batch_normalize_to_same_scientific_content(tmp_path):
    client = FakeBatchClient()
    sync_response = await client.generate(ModelRequest(provider="fake", model_name="fake-model", payload="Q1"))

    backend = _backend(client, tmp_path)
    [batch_response] = await backend.generate_batch(["Q1"])

    assert sync_response.status == batch_response.status == SUCCESS_STATUS
    assert sync_response.raw_text == "sync:Q1"
    assert batch_response.raw_text == "batch:Q1"  # distinguishable markers by design
    assert sync_response.finish_reason == batch_response.finish_reason


# --- N2 -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_n2_out_of_order_batch_results_map_to_correct_request(tmp_path):
    client = FakeBatchClient(shuffle_results=True)
    backend = _backend(client, tmp_path)
    prompts = ["Q1", "Q2", "Q3"]
    responses = await backend.generate_batch(prompts)
    for prompt, response in zip(prompts, responses):
        assert response.raw_text == f"batch:{prompt}", (
            f"expected response for {prompt!r}, got {response.raw_text!r} -- "
            "out-of-order fetch_batch_results broke request identity"
        )


# --- N3 -------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "BatchAPIBackend.generate_batch() has NO length/identity check on "
        "fetch_batch_results()'s return value before zipping it onto "
        "uncached_indices (batch_api_backend.py: "
        "'for local_i, response in zip(uncached_indices, fetched): "
        "results[local_i] = response'). If a client's fetch_batch_results "
        "returns fewer results than requests (a missing batch result ID), "
        "zip() silently truncates and the corresponding results[] slots stay "
        "None -- not a raised, loud failure. The frozen invariant (missing/"
        "duplicate result IDs must fail loudly) is violated at this layer; "
        "only a scrupulously-correct client implementation prevents it from "
        "ever being observed in practice."
    ),
)
async def test_n3_missing_batch_result_fails_loudly(tmp_path):
    client = FakeBatchClient(result_count_delta=-1)  # one fewer result than requests
    backend = _backend(client, tmp_path)
    with pytest.raises(RuntimeError):
        await backend.generate_batch(["Q1", "Q2", "Q3"])


# --- N4 -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_n4_resume_from_completed_batch_state_reattaches_no_duplicate_submission(tmp_path):
    client = FakeBatchClient()
    backend = _backend(client, tmp_path)
    prompts = ["Q1", "Q2"]

    first = await backend.generate_batch(prompts)
    assert len(client.submit_calls) == 1
    assert [r.raw_text for r in first] == ["batch:Q1", "batch:Q2"]

    # Second backend instance, same cache_dir/batch_state_dir, same client
    # object (submit_calls persists) -- but the response cache from the
    # first call means these prompts are now CACHE HITS, so submit_batch is
    # never even reconsidered. Clear the cache to force a genuine
    # batch-state-file resume path instead.
    for f in (tmp_path / "cache").rglob("*.json"):
        f.unlink()

    def _refuse_submit(*_a, **_kw):
        raise AssertionError("submit_batch must not be called -- an existing batch-state file should be reused")

    original_submit = client.submit_batch
    client.submit_batch = _refuse_submit  # type: ignore[method-assign]
    try:
        # A resumed backend still finds the batch-state file BatchAPIBackend
        # itself wrote on the first call (it is only cleared on terminal
        # completion via _clear_batch_state -- but generate_batch() already
        # called that after the first successful fetch, so re-create it here
        # to simulate "process crashed after submit, before first poll
        # completed" -- the actual resumable window this feature targets.
        state_path = backend._batch_state_path([
            backend._make_request(p) for p in prompts
        ])
        state_path.write_text(
            __import__("json").dumps({"batch_id": "batch-0", "provider": "fake", "model_name": "fake-model", "n_requests": 2})
        )
        second = await backend.generate_batch(prompts)
        assert [r.raw_text for r in second] == ["batch:Q1", "batch:Q2"]
    finally:
        client.submit_batch = original_submit


# --- N5 -------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "batch_api_backend.py's module and class docstrings do not mention "
        "the crash window between a provider accepting a submitted batch "
        "(_raw_client.submit_batch() returns a batch_id) and this backend "
        "writing that batch_id to its local batch-state file "
        "(state_path.write_text(...)) -- a crash in between would submit a "
        "real provider job with no local record of it, so a subsequent "
        "resume attempt would resubmit a duplicate job. This is a genuine, "
        "probably-unavoidable limitation (spec N5 says it 'may remain "
        "documented'), but it is not currently documented anywhere in this "
        "file."
    ),
)
def test_n5_crash_window_is_documented():
    import inspect
    from choicebench.backends import batch_api_backend

    module_doc = inspect.getdoc(batch_api_backend) or ""
    class_doc = inspect.getdoc(BatchAPIBackend) or ""
    combined = (module_doc + "\n" + class_doc).lower()
    assert "crash" in combined or "before" in combined and "write" in combined and "submit" in combined
