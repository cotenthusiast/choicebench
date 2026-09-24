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

import collections
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


def majority_vote_with_tiebreak(
        choices: list[str | None],
        *,
        label_to_source_index: dict[str, int],
        seed: int,
        benchmark_id: str,
        question_id: str,
        method_name: str,
) -> str | None:
    """Collapse several per-vote canonical letters to one, via majority vote.

    Shared by any method that collects multiple votes for the same question
    (cyclic permutation's per-rotation votes, a two-stage rotation rerun's
    per-rotation stage-2 votes, ...) and must resolve ties the same way
    everywhere: through resolve_tie() over each tied letter's stable
    source_index, never first-arrival order or the display letter itself.

    Args:
        choices: Canonical letters, one per vote, with None for any vote
            that failed to produce an answer.
        label_to_source_index: This question's letter->source_index map.
        seed: Experiment seed, part of the tie-break key.
        benchmark_id: Benchmark name, part of the tie-break key.
        question_id: This question's canonical ID, part of the tie-break key.
        method_name: Registry method name, part of the tie-break key.

    Returns:
        The most frequent letter, or None if no valid votes exist.
    """
    cleaned = [x for x in choices if x is not None]
    if not cleaned:
        return None

    counts = collections.Counter(cleaned)
    top = counts.most_common(2)
    if len(top) == 1 or top[0][1] != top[1][1]:
        return top[0][0]

    max_count = top[0][1]
    tied_letters = [letter for letter, count in counts.items() if count == max_count]
    tied_ids = [label_to_source_index[letter] for letter in tied_letters]
    winning_id = resolve_tie(
        seed=seed, benchmark_id=benchmark_id, question_id=question_id,
        method_name=method_name, tied_canonical_ids=tied_ids,
    )
    id_to_label = {v: k for k, v in label_to_source_index.items()}
    return id_to_label[winning_id]
