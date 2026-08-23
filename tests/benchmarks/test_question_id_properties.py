# tests/benchmarks/test_question_id_properties.py
#
# BATCH 8 — A6: dataset normalization / identity property battery.
#
# Encodes the ratified question_id contract (P3-F2 acceptance):
#   * ids are a pure function of the payload string
#     "{subject}|{question}|c0|...|cN" (sha256, first 16 hex chars);
#   * injectivity holds on PAYLOAD STRINGS — two structurally different
#     tuples whose joined payloads coincide share an id BY DESIGN; that
#     residual is fail-closed downstream (duplicate-id rejection at
#     preparation) and is pinned here as documentation;
#   * unicode-heavy content round-trips without mangling.

import hashlib
import itertools
import random
import string

from choicebench.benchmarks.base import make_normalized_row

_ALPHABET = (
    string.ascii_letters + string.digits + " |äöü漢字emoji→é"
    "\t'\"\\/*"
)


def _expected_id(subject: str, question: str, choices: list[str]) -> str:
    content = f"{subject}|{question}|" + "|".join(str(c) for c in choices)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _random_payload(rng: random.Random) -> tuple[str, str, list[str]]:
    def _text(max_len: int = 40) -> str:
        return "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(1, max_len)))

    n = rng.randint(2, 6)
    return _text(20), _text(), [_text(15) for _ in range(n)]


class TestQuestionIdProperties:
    def test_ids_are_deterministic_pure_functions_of_payload(self):
        rng = random.Random(20260823)
        for _ in range(200):
            subject, question, choices = _random_payload(rng)
            row_a = make_normalized_row(subject, question, choices, 0)
            row_b = make_normalized_row(subject, question, choices, 0)
            assert row_a["question_id"] == row_b["question_id"]
            assert row_a["question_id"] == _expected_id(subject, question, choices)

    def test_payload_string_injectivity_over_fuzz_corpus(self):
        """Distinct payload strings must yield distinct ids (no hash
        collisions in practice); equal payload strings necessarily share
        an id. Injectivity domain = payload STRINGS, not tuples."""
        rng = random.Random(42)
        seen: dict[str, str] = {}
        for _ in range(500):
            subject, question, choices = _random_payload(rng)
            content = f"{subject}|{question}|" + "|".join(choices)
            qid = make_normalized_row(subject, question, choices, 0)["question_id"]
            if content in seen:
                assert seen[content] == qid
            else:
                seen[content] = qid
        # 500 distinct unicode-heavy payloads -> distinct ids.
        assert len(set(seen.values())) == len(seen)

    def test_delimiter_injection_shares_id_and_is_fail_closed(self):
        """P3-F2 documented caveat: '|' inside text fields can merge two
        structurally different tuples into one payload string. The shared
        id is the accepted design; preparation rejects realized duplicate
        ids instead of silently scoring both."""
        a = make_normalized_row("a|b", "q", ["x", "y"], 0)
        b = make_normalized_row("a", "b|q", ["x", "y"], 0)
        assert a["question_id"] == b["question_id"]
        assert a["question_id"] == _expected_id("a|b", "q", ["x", "y"])

    def test_unicode_content_preserved_verbatim_in_row(self):
        subject = "philosophie"
        question = "La «nan» → 漢字?"
        choices = ["café ✓", "naïve — maybe", "日本語", "🎉"]
        row = make_normalized_row(subject, question, choices, 2)
        assert row["question_text"] == question
        assert row["subject"] == subject
        assert row["correct_answer_text"] == "日本語"
        import json

        texts = [r["text"] for r in json.loads(row["choices_json"])]
        assert texts == choices


class TestPathologicalLabelPolicies:
    def test_arc_duplicate_labels_first_wins_documented(self):
        from choicebench.benchmarks.arc import normalize_row

        row = {
            "id": "p", "question": "q",
            "choices": {"text": ["first!", "second!", "third"],
                        "label": ["B", "B", "A"]},
            "answerKey": "B",
        }
        normalized = normalize_row(row)
        assert normalized["correct_answer_text"] == "first!"

    def test_truthfulqa_multiple_gold_first_wins_documented(self):
        from choicebench.benchmarks.truthful_qa import normalize_row

        row = {
            "question": "q",
            "mc1_targets": {"choices": ["gold-first", "x", "also-gold"],
                            "labels": [1, 0, 1]},
        }
        normalized = normalize_row(row)
        assert normalized["correct_answer_text"] == "gold-first"


class TestNullOptionPolicyInvariants:
    def test_mmlu_pro_never_emits_nan_for_any_null_position(self):
        """Fuzz: nulls at any distractor position are dropped with correct
        gold remapping; null gold raises."""
        import json

        import pytest

        from choicebench.benchmarks.mmlu_pro import normalize_row

        rng = random.Random(7)
        for _ in range(100):
            n = rng.randint(3, 8)
            gold = rng.randrange(n)
            options = [f"opt{i}" for i in range(n)]
            null_positions = set()
            for i in range(n):
                if i != gold and rng.random() < 0.4:
                    options[i] = None
                    null_positions.add(i)
            row = {"category": "s", "question": "q", "options": options,
                   "answer_index": gold}
            normalized = normalize_row(row)
            texts = [r["text"] for r in json.loads(normalized["choices_json"])]
            assert "nan" not in texts
            assert len(texts) == n - len(null_positions)
            kept_before_gold = sum(
                1 for i in range(gold) if i not in null_positions
            )
            assert normalized["correct_index"] == kept_before_gold
            assert normalized["correct_answer_text"] == f"opt{gold}"

    def test_mmlu_pro_null_gold_always_rejects(self):
        import pytest

        from choicebench.benchmarks.mmlu_pro import normalize_row

        for position in range(4):
            options = ["a", "b", "c", "d"]
            options[position] = None
            with pytest.raises(ValueError, match="null gold"):
                normalize_row({"category": "s", "question": "q",
                               "options": options, "answer_index": position})
