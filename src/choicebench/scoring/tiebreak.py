# src/choicebench/scoring/tiebreak.py

"""Canonical deterministic tie-break utility (protocol: choicebench-tiebreak-v1).

Any method that can produce multiple equally-winning canonical options
(cyclic majority vote, independent-hypothesis score ties, semantic-matching/
text-extraction cascade collisions, PriDe exact-score ties, ...) resolves
the tie through this single utility rather than a per-method RNG. The
result is a pure function of (seed, benchmark, question, method, the set of
tied canonical option identities) -- deliberately excluding model/provider,
displayed letter, permutation/rotation index, and execution order, so the
same tie always resolves the same way no matter which physical position an
option is displayed at or which model produced it.

Uses BLAKE2b instead of Python's random module or built-in hash() so the
result is reproducible across processes, machines, and Python versions
without depending on random.Random's string-seeding behavior.
"""

from __future__ import annotations

import hashlib
from typing import Collection

TIEBREAK_NAMESPACE = "choicebench-tiebreak-v1"

_DIGEST_SIZE = 8


def resolve_tie(
        *,
        seed: int,
        benchmark_id: str,
        question_id: str,
        method_name: str,
        tied_canonical_ids: Collection[int],
) -> int:
    """Deterministically pick one canonical option ID from a tied set.

    Args:
        seed: Experiment seed.
        benchmark_id: Benchmark name (e.g. "mmlu", "arc_challenge").
        question_id: Canonical content-hash question identifier.
        method_name: Registry method name producing this tie.
        tied_canonical_ids: The tied options' stable canonical identities
            (e.g. ``source_index``) -- never display letters, never a
            permutation/rotation index. Order does not matter: the result
            depends only on which IDs are present, not how they're given.

    Returns:
        The winning canonical option ID (always a member of
        ``tied_canonical_ids``).

    Raises:
        ValueError: if ``tied_canonical_ids`` is empty.
    """
    ids = sorted(tied_canonical_ids)
    if not ids:
        raise ValueError("resolve_tie requires at least one candidate ID.")
    if len(ids) == 1:
        return ids[0]

    key = "|".join([
        TIEBREAK_NAMESPACE,
        str(int(seed)),
        str(benchmark_id),
        str(question_id),
        str(method_name),
        ",".join(str(i) for i in ids),
    ])
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=_DIGEST_SIZE).digest()
    index = int.from_bytes(digest, byteorder="big", signed=False) % len(ids)
    return ids[index]
