# experiments/visible_llm_matcher/repairs/group1_historical_arc/harness/repair_infra.py
#
# Shared execution/cache/provenance infrastructure for all 6 method
# repairs. Not a port of historical code — this is new, harness-specific
# plumbing.
#
# Cache isolation: every repair call goes through a brand-new cache
# directory (REPAIR_CACHE_ROOT, created fresh, never shared with
# two-stage-prompting's own .cache/responses/). Since the repaired prompt
# text itself is byte-different from the historical contaminated prompt
# (drops the D line entirely, "three options" wording), the cache KEY
# (sha256 of provider+model+prompt+temperature+max_tokens+seed) would
# never collide with anything in the historical cache even if shared — but
# per instruction, a dedicated namespace is used anyway rather than relying
# on that as the only safeguard.
#
# Freshness verification, corrected after investigation: ModelResponse.
# timestamp_utc is NEVER populated anywhere in choicebench (every client —
# openai/gemini/groq/together/anthropic/vllm — hardcodes timestamp_utc=None
# in both the success and failure path; there is no code path that ever
# sets it to a real value, confirmed by reading every clients/*_client.py).
# An earlier version of this module checked it as a freshness signal, which
# would have made every call — cached or not — look "not fresh" and
# incorrectly block genuine repairs. latency_seconds is real (BaseClient.
# generate() measures it with time.perf_counter() on every live call — see
# clients/base.py), but a cache HIT replays whatever latency_seconds was
# stored from the ORIGINAL real call, so latency > 0 alone cannot
# distinguish "fresh right now" from "fresh when this was first cached
# and replayed since". The reliable signal is therefore structural, not
# metadata-based: REPAIR_CACHE_ROOT starts empty for every new
# (method, model) namespace, so counting cache files before and after a
# batch of calls, and asserting the count grew by exactly the number of
# successful calls, is a positive, file-system-level proof that the live
# client path (not CachingClientWrapper's cache-hit branch) was taken for
# every one of them.

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from choicebench.backends.api_backend import APIBackend
from choicebench.clients.base import BaseClient
from choicebench.clients.types import ModelResponse

REPO_ROOT = Path(__file__).resolve().parents[5]
REPAIR_CACHE_ROOT = REPO_ROOT / "experiments" / "visible_llm_matcher" / "repairs" / "group1_historical_arc" / ".repair_cache"
STAGING_ROOT = REPO_ROOT / "experiments" / "visible_llm_matcher" / "repairs" / "group1_historical_arc" / "staged_repairs"
PROVENANCE_ROOT = REPO_ROOT / "experiments" / "visible_llm_matcher" / "repairs" / "group1_historical_arc" / "provenance"


class FreshCallRequiredError(RuntimeError):
    """Raised when a repair batch's cache-file growth doesn't match the
    number of successful calls made — evidence that at least one response
    was served from a pre-existing cache entry rather than a genuine live
    call. A repair harness invocation must never silently accept a cached
    response as evidence of a genuine repair call."""


def assert_fresh_call(response: ModelResponse, *, question_id: str) -> None:
    """Weak, per-response sanity check only: a successful response must
    have positive latency (every live call does; see module docstring for
    why this alone is not sufficient to prove freshness — see
    count_cache_entries()/assert_cache_grew_by() for the authoritative
    check, applied per repair batch in run_repair.py).
    """
    if not response.is_success():
        return
    if not response.latency_seconds or response.latency_seconds <= 0:
        raise FreshCallRequiredError(
            f"question_id={question_id!r}: response has latency_seconds="
            f"{response.latency_seconds!r}, which no live call (cached or "
            "not) should ever produce — investigate before proceeding."
        )


def count_cache_entries(cache_namespace: str) -> int:
    """Count cache files currently on disk for one (method, model)
    namespace under REPAIR_CACHE_ROOT."""
    ns_dir = REPAIR_CACHE_ROOT / cache_namespace
    if not ns_dir.is_dir():
        return 0
    return sum(1 for _ in ns_dir.rglob("*.json"))


def assert_cache_grew_by(cache_namespace: str, *, before: int, expected_new_entries: int) -> None:
    """Authoritative freshness proof for a completed repair batch: the
    dedicated cache namespace must have gained exactly expected_new_entries
    files. Fewer means at least one call was served from a pre-existing
    cache entry (should be structurally impossible given the repaired
    prompt text is unique, but this is the positive, independent check
    rather than trusting that reasoning alone). More would mean an
    unexpected extra call was made — also worth failing loudly on.
    """
    after = count_cache_entries(cache_namespace)
    grew_by = after - before
    if grew_by != expected_new_entries:
        raise FreshCallRequiredError(
            f"cache_namespace={cache_namespace!r}: expected the repair cache "
            f"to grow by exactly {expected_new_entries} new entries, but it "
            f"grew by {grew_by} (before={before}, after={after}). This means "
            "at least one call in this batch did not take the live client "
            "path — do not accept this batch as a genuine repair."
        )


def build_repair_backend(
    *,
    provider: str,
    model_name: str,
    client: BaseClient,
    temperature: float,
    max_tokens: int,
    seed: int,
    concurrency_limit: int,
    cache_namespace: str,
) -> APIBackend:
    """Construct an APIBackend pointed at the dedicated repair cache dir.

    Args:
        client: An already-constructed provider client (OpenAIClient,
            GeminiClient, GroqClient, TogetherAIClient) — this function does
            not read API keys itself; see run_repair.py for that.
        cache_namespace: e.g. "text_extraction__gpt-4.1-mini" — kept
            distinct per (method, model) so repair caches for different
            cells never collide with each other either.
    """
    cache_dir = REPAIR_CACHE_ROOT / cache_namespace
    return APIBackend(
        provider,
        model_name,
        client,
        cache_dir,
        temperature,
        max_tokens,
        seed,
        concurrency_limit,
        cache_namespace,
    )


@dataclass(frozen=True, slots=True)
class ProtocolDiffRecord:
    """Structured record of exactly what differs from the historical
    protocol for one repair cell — written to PROVENANCE_ROOT so every
    repair's "one permitted correction" claim is independently auditable.
    """

    cell_id: str
    method: str
    model: str
    benchmark: str
    source_repo: str
    source_commit: str
    source_config_path: str
    source_run_id: str
    prompt_template_historical: str
    prompt_template_repaired: str
    repaired_question_ids: list[str]
    correction_description: str
    stage1_disposition: str  # "not_applicable" | "reused_verified_clean" | "rerun_fresh"
    stage1_disposition_reason: str

    def write(self) -> Path:
        PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)
        path = PROVENANCE_ROOT / f"{self.cell_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path
