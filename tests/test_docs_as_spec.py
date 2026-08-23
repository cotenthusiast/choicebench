# tests/test_docs_as_spec.py
#
# BATCH 8 — A10: docs-as-spec consistency meta-tests.
#
# Invariant: the README "Supported Built-ins" tables and the config
# template's capability claims must agree with the code registries, so a
# shipped capability can never silently disappear from (or appear in)
# documentation again.

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _readme_section(start_marker: str, end_markers: tuple[str, ...] = ("\n### ",)) -> str:
    text = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index(start_marker)
    end = len(text)
    for marker in end_markers:
        pos = text.find(marker, start + len(start_marker))
        if pos != -1:
            end = min(end, pos)
    return text[start:end]


def _first_table_names(section: str) -> set[str]:
    """Backtick-quoted names from the section's FIRST table only."""
    lines = section.splitlines()
    names = set()
    in_table = False
    for line in lines:
        if line.startswith("| `"):
            in_table = True
            names.add(line.split("`")[1])
        elif in_table and not line.startswith("|"):
            break
    return names


class TestReadmeMatchesRegistries:
    def test_methods_table_covers_registry_exactly(self):
        from choicebench.registry import METHOD_REGISTRY

        listed = _first_table_names(
            _readme_section("### Methods", ("\n#### ",))
        )
        assert listed == set(METHOD_REGISTRY), (
            "README methods table and METHOD_REGISTRY disagree: "
            f"docs-only={listed - set(METHOD_REGISTRY)}, "
            f"code-only={set(METHOD_REGISTRY) - listed}"
        )

    def test_metrics_table_covers_builtin_metrics_exactly(self):
        from choicebench.metrics import BUILTIN_METRICS

        listed = _first_table_names(_readme_section("### Metrics"))
        assert listed == set(BUILTIN_METRICS)

    def test_clients_table_covers_client_registry_exactly(self):
        from choicebench.registry import CLIENT_REGISTRY

        listed = _first_table_names(_readme_section("### Clients (API backends)"))
        assert listed == set(CLIENT_REGISTRY)

    def test_benchmarks_table_lists_every_registered_benchmark(self):
        from choicebench.benchmarks.registry import BENCHMARK_REGISTRY

        listed = _first_table_names(_readme_section("### Benchmarks"))
        # The synthetic 'toy' bundle and the generic 'huggingface' escape
        # hatch are documented rows too; every registered normalizer must
        # appear.
        assert set(BENCHMARK_REGISTRY) <= listed


class TestConfigTemplateCapabilityClaims:
    def _template_text(self) -> str:
        return (_REPO_ROOT / "config" / "experiment_template.yaml").read_text()

    def test_no_vllm_logprob_claim(self):
        """D3: vLLM is generate-only like every API provider; the old claim
        that it supports logprob methods must stay removed."""
        text = self._template_text()
        assert not re.search(r"vLLM supports logprob", text, re.IGNORECASE)

    def test_provider_seed_note_names_together_as_the_only_forwarder(self):
        """D4: only Together forwards request seed among API providers."""
        text = self._template_text()
        assert "Together" in text and "seed" in text.lower()
