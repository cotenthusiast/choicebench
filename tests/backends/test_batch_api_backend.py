# tests/backends/test_batch_api_backend.py
#
# BatchAPIBackend implements the SAME generate_batch(prompts) -> list[ModelResponse]
# contract every runner's run_many_async() already calls on APIBackend -- but
# backed by a real provider Batch API (submit once, poll, fetch) instead of
# concurrent per-request dispatch. Every existing runner that already has a
# working run_many_async path therefore gains batch execution for free by
# swapping the backend, with zero runner-level changes.
#
# Tested against a FAKE raw client double (submit_batch/poll_batch/
# fetch_batch_results), never a real provider SDK -- the provider-specific
# clients have their own test files.

import pytest

from choicebench.backends.batch_api_backend import BatchAPIBackend
from choicebench.clients.types import (
    BATCH_COMPLETED,
    BATCH_FAILED,
    BATCH_IN_PROGRESS,
    ErrorInfo,
    ModelRequest,
    ModelResponse,
    SUCCESS_STATUS,
)


class _FakeBatchClient:
    """Records calls; scripted poll_batch return sequence; results keyed by
    request payload so fetch_batch_results can answer regardless of the
    (arbitrary, provider-realistic) order requests are passed in."""

    def __init__(self, poll_sequence, result_by_payload, result_count_delta: int = 0):
        self.provider = "fake"
        self.model_name = "fake-model"
        self._poll_sequence = list(poll_sequence)
        self._result_by_payload = result_by_payload
        self._result_count_delta = result_count_delta
        self.submit_batch_calls: list[list[ModelRequest]] = []
        self.poll_batch_calls: list[str] = []
        self.fetch_batch_results_calls: list[str] = []
        self._next_batch_id = 0

    async def submit_batch(self, requests: list[ModelRequest]) -> str:
        self.submit_batch_calls.append(requests)
        batch_id = f"batch_{self._next_batch_id}"
        self._next_batch_id += 1
        return batch_id

    async def poll_batch(self, batch_id: str) -> str:
        self.poll_batch_calls.append(batch_id)
        return self._poll_sequence.pop(0)

    async def fetch_batch_results(self, batch_id: str, requests: list[ModelRequest]) -> list[ModelResponse]:
        self.fetch_batch_results_calls.append(batch_id)
        results = [
            ModelResponse(
                provider=req.provider, model_name=req.model_name, status=SUCCESS_STATUS,
                latency_seconds=0.0, raw_text=self._result_by_payload[req.payload],
            )
            for req in requests
        ]
        if self._result_count_delta < 0:
            results = results[: len(results) + self._result_count_delta]
        elif self._result_count_delta > 0 and results:
            results = results + [results[-1]] * self._result_count_delta
        return results


async def _noop_sleep(_seconds: float) -> None:
    return None


def _make_backend(tmp_path, client, poll_interval_seconds=0.0, sleep_fn=None):
    return BatchAPIBackend(
        provider="fake", model_name="fake-model", client=client,
        cache_dir=tmp_path / "cache", temperature=0.0, max_tokens=16, seed=42,
        batch_state_dir=tmp_path / "batch_state",
        poll_interval_seconds=poll_interval_seconds,
        sleep_fn=sleep_fn or _noop_sleep,
    )


class TestGenerateBatchHappyPath:
    @pytest.mark.asyncio
    async def test_submits_exactly_one_batch_for_all_prompts(self, tmp_path):
        client = _FakeBatchClient(
            poll_sequence=[BATCH_COMPLETED],
            result_by_payload={"q1": "A", "q2": "B", "q3": "C"},
        )
        backend = _make_backend(tmp_path, client)

        results = await backend.generate_batch(["q1", "q2", "q3"])

        assert len(client.submit_batch_calls) == 1
        assert len(client.submit_batch_calls[0]) == 3
        assert [r.raw_text for r in results] == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_results_returned_in_original_prompt_order_not_provider_order(self, tmp_path):
        """Provider batch APIs return results in ARBITRARY order -- the fake
        double deliberately answers by payload lookup, not list position, so
        this only passes if BatchAPIBackend reassembles correctly."""
        client = _FakeBatchClient(
            poll_sequence=[BATCH_COMPLETED],
            result_by_payload={"first": "R1", "second": "R2", "third": "R3"},
        )
        backend = _make_backend(tmp_path, client)

        results = await backend.generate_batch(["first", "second", "third"])

        assert [r.raw_text for r in results] == ["R1", "R2", "R3"]

    @pytest.mark.asyncio
    async def test_polls_until_completed(self, tmp_path):
        client = _FakeBatchClient(
            poll_sequence=[BATCH_IN_PROGRESS, BATCH_IN_PROGRESS, BATCH_COMPLETED],
            result_by_payload={"q1": "A"},
        )
        backend = _make_backend(tmp_path, client)

        await backend.generate_batch(["q1"])

        assert len(client.poll_batch_calls) == 3


class TestGenerateBatchCaching:
    @pytest.mark.asyncio
    async def test_a_fully_cached_prompt_set_never_submits_a_batch(self, tmp_path):
        client = _FakeBatchClient(poll_sequence=[BATCH_COMPLETED], result_by_payload={"q1": "A"})
        backend = _make_backend(tmp_path, client)
        await backend.generate_batch(["q1"])  # populate the cache
        assert len(client.submit_batch_calls) == 1

        results = await backend.generate_batch(["q1"])  # second call: should be a pure cache hit

        assert len(client.submit_batch_calls) == 1  # still just the one from before
        assert results[0].raw_text == "A"

    @pytest.mark.asyncio
    async def test_only_uncached_prompts_are_submitted(self, tmp_path):
        client = _FakeBatchClient(
            poll_sequence=[BATCH_COMPLETED, BATCH_COMPLETED],
            result_by_payload={"q1": "A", "q2": "B"},
        )
        backend = _make_backend(tmp_path, client)
        await backend.generate_batch(["q1"])  # caches q1 only

        results = await backend.generate_batch(["q1", "q2"])

        # Second submit_batch call must contain only the uncached prompt.
        assert len(client.submit_batch_calls) == 2
        assert [r.payload for r in client.submit_batch_calls[1]] == ["q2"]
        assert [r.raw_text for r in results] == ["A", "B"]


class TestGenerateBatchResumability:
    @pytest.mark.asyncio
    async def test_a_second_backend_instance_reattaches_to_an_in_flight_batch_id(self, tmp_path):
        """Simulates a process restart: a fresh BatchAPIBackend, same
        batch_state_dir, same prompt set -- must poll the EXISTING batch_id
        rather than submitting a duplicate job."""
        client = _FakeBatchClient(
            poll_sequence=[BATCH_IN_PROGRESS],
            result_by_payload={"q1": "A"},
        )
        backend1 = _make_backend(tmp_path, client)

        # Simulate backend1 submitting, then the process dying before the
        # batch completes: submit the job, write state, but never poll to
        # completion (leave the state file behind for the "restart").
        requests = [backend1._make_request("q1")]
        await backend1._resolve_or_submit_batch(requests)
        assert len(client.submit_batch_calls) == 1

        client2 = _FakeBatchClient(
            poll_sequence=[BATCH_COMPLETED],
            result_by_payload={"q1": "A"},
        )
        backend2 = _make_backend(tmp_path, client2)

        results = await backend2.generate_batch(["q1"])

        assert len(client2.submit_batch_calls) == 0  # reattached, did not resubmit
        assert results[0].raw_text == "A"

    @pytest.mark.asyncio
    async def test_batch_state_file_is_cleared_after_a_successful_fetch(self, tmp_path):
        client = _FakeBatchClient(poll_sequence=[BATCH_COMPLETED], result_by_payload={"q1": "A"})
        backend = _make_backend(tmp_path, client)

        await backend.generate_batch(["q1"])

        state_dir = tmp_path / "batch_state"
        assert list(state_dir.glob("*.json")) == []


class TestGenerateBatchResultCardinality:
    """Confirmed conformance gap: fetch_batch_results()'s return value had
    no length check before being zipped onto uncached_indices -- a client
    returning fewer results than requested silently truncated (the missing
    slots stayed None) instead of raising."""

    @pytest.mark.asyncio
    async def test_fewer_results_than_requests_raises(self, tmp_path):
        client = _FakeBatchClient(
            poll_sequence=[BATCH_COMPLETED],
            result_by_payload={"q1": "A", "q2": "B", "q3": "C"},
            result_count_delta=-1,  # one fewer result than requested
        )
        backend = _make_backend(tmp_path, client)

        with pytest.raises(RuntimeError, match="3 request"):
            await backend.generate_batch(["q1", "q2", "q3"])

    @pytest.mark.asyncio
    async def test_more_results_than_requests_also_raises(self, tmp_path):
        client = _FakeBatchClient(
            poll_sequence=[BATCH_COMPLETED],
            result_by_payload={"q1": "A", "q2": "B"},
            result_count_delta=1,
        )
        backend = _make_backend(tmp_path, client)

        with pytest.raises(RuntimeError):
            await backend.generate_batch(["q1", "q2"])

    @pytest.mark.asyncio
    async def test_exact_matching_count_still_works(self, tmp_path):
        """Regression guard: the cardinality check itself must not reject
        a correctly-sized result set."""
        client = _FakeBatchClient(
            poll_sequence=[BATCH_COMPLETED],
            result_by_payload={"q1": "A", "q2": "B"},
        )
        backend = _make_backend(tmp_path, client)

        results = await backend.generate_batch(["q1", "q2"])
        assert [r.raw_text for r in results] == ["A", "B"]


class TestGenerateBatchFailure:
    @pytest.mark.asyncio
    async def test_a_failed_batch_job_produces_a_failure_response_per_request(self, tmp_path):
        client = _FakeBatchClient(poll_sequence=[BATCH_FAILED], result_by_payload={})
        backend = _make_backend(tmp_path, client)

        results = await backend.generate_batch(["q1", "q2"])

        assert len(results) == 2
        assert all(not r.is_success() for r in results)
        assert all(isinstance(r.error, ErrorInfo) for r in results)

    @pytest.mark.asyncio
    async def test_a_failed_batch_is_not_cached_and_state_is_cleared_for_retry(self, tmp_path):
        client = _FakeBatchClient(poll_sequence=[BATCH_FAILED], result_by_payload={})
        backend = _make_backend(tmp_path, client)
        await backend.generate_batch(["q1"])

        state_dir = tmp_path / "batch_state"
        assert list(state_dir.glob("*.json")) == []

        # A retry must submit fresh, not reattach to the dead batch.
        client2 = _FakeBatchClient(poll_sequence=[BATCH_COMPLETED], result_by_payload={"q1": "A"})
        backend2 = _make_backend(tmp_path, client2)
        results = await backend2.generate_batch(["q1"])
        assert len(client2.submit_batch_calls) == 1
        assert results[0].raw_text == "A"


class _FakePinnedBatchClient(_FakeBatchClient):
    """A batch client double carrying OpenRouter-style upstream pinning
    attributes, to prove BatchAPIBackend's own _cache_key calls (which
    bypass CachingClientWrapper.generate() entirely) still fold in
    client_extra_identity -- a regression the claim-7 cache-identity fix
    initially missed, since it only touched CachingClientWrapper.generate()."""

    def __init__(self, *args, upstream_provider, allow_fallbacks=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._upstream_provider = upstream_provider
        self._allow_fallbacks = allow_fallbacks


class TestGenerateBatchUpstreamProviderIdentity:
    @pytest.mark.asyncio
    async def test_different_upstream_pinning_never_shares_a_cache_entry(self, tmp_path):
        cache_dir = tmp_path / "cache"

        deepinfra_client = _FakePinnedBatchClient(
            poll_sequence=[BATCH_COMPLETED], result_by_payload={"q1": "A"},
            upstream_provider="deepinfra",
        )
        deepinfra_backend = BatchAPIBackend(
            provider="openrouter", model_name="meta-llama/llama-3.1-8b-instruct",
            client=deepinfra_client, cache_dir=cache_dir, temperature=0.0, max_tokens=16,
            seed=None, batch_state_dir=tmp_path / "batch_state_a",
            poll_interval_seconds=0.0, sleep_fn=_noop_sleep,
        )
        await deepinfra_backend.generate_batch(["q1"])

        together_client = _FakePinnedBatchClient(
            poll_sequence=[BATCH_COMPLETED], result_by_payload={"q1": "B"},
            upstream_provider="together",
        )
        together_backend = BatchAPIBackend(
            provider="openrouter", model_name="meta-llama/llama-3.1-8b-instruct",
            client=together_client, cache_dir=cache_dir, temperature=0.0, max_tokens=16,
            seed=None, batch_state_dir=tmp_path / "batch_state_b",
            poll_interval_seconds=0.0, sleep_fn=_noop_sleep,
        )
        results = await together_backend.generate_batch(["q1"])

        # Sharing one cache_dir must NOT make the together-pinned backend
        # see the deepinfra-pinned response as a cache hit.
        assert len(together_client.submit_batch_calls) == 1
        assert results[0].raw_text == "B"

    @pytest.mark.asyncio
    async def test_different_upstream_pinning_never_shares_a_batch_state_path(self, tmp_path):
        """Confirmed conformance gap: _batch_state_path() called _cache_key
        with NO extra_identity at all -- unlike generate_batch()'s own
        response-cache lookups. Two differently-pinned clients would
        compute the IDENTICAL state path, so a restart could poll/resume
        the WRONG pinned deployment's in-flight batch job."""
        state_dir = tmp_path / "batch_state"

        deepinfra_client = _FakePinnedBatchClient(
            poll_sequence=[BATCH_IN_PROGRESS], result_by_payload={},
            upstream_provider="deepinfra",
        )
        deepinfra_backend = BatchAPIBackend(
            provider="openrouter", model_name="meta-llama/llama-3.1-8b-instruct",
            client=deepinfra_client, cache_dir=tmp_path / "cache_a", temperature=0.0, max_tokens=16,
            seed=None, batch_state_dir=state_dir,
            poll_interval_seconds=0.0, sleep_fn=_noop_sleep,
        )
        deepinfra_path = deepinfra_backend._batch_state_path(
            [deepinfra_backend._make_request("q1")]
        )

        together_client = _FakePinnedBatchClient(
            poll_sequence=[BATCH_IN_PROGRESS], result_by_payload={},
            upstream_provider="together",
        )
        together_backend = BatchAPIBackend(
            provider="openrouter", model_name="meta-llama/llama-3.1-8b-instruct",
            client=together_client, cache_dir=tmp_path / "cache_a", temperature=0.0, max_tokens=16,
            seed=None, batch_state_dir=state_dir,
            poll_interval_seconds=0.0, sleep_fn=_noop_sleep,
        )
        together_path = together_backend._batch_state_path(
            [together_backend._make_request("q1")]
        )

        assert deepinfra_path != together_path
