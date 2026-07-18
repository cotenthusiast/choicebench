import asyncio
import json
from enum import Enum
from pathlib import Path

import pandas as pd
import pytest

from choicebench.config.paths import validate_run_id
from choicebench.config.schema import (
    BenchmarkConfig, ExperimentConfig, GenerationKwargsConfig, MethodConfig,
    ModelConfig, RunConfig,
)
from choicebench.datasets import (
    DatasetArtifactError, DatasetSpec, dataset_content_digest,
    dataset_sample_identities, spec_for_benchmark, write_prepared_dataset,
)
from choicebench.identity import canonical_json, redact_text, sanitize_url, stable_digest
from choicebench.infra.artifacts import FileLock
from choicebench.infra.cache import ResponseCache
from choicebench.infra.checkpoint import CheckpointManager
from choicebench.io.writers import validate_result_artifact, write_run_results
from choicebench.manifest import (
    CANONICALIZATION_VERSION, PROTOCOL_VERSION, ManifestCompatibilityError,
    make_manifest, validate_manifest,
)
from choicebench.provenance import resolve_hf_model_identity


class _Mode(Enum):
    A = "a"


def _rows(question='["x", "y"]', qid="q1"):
    return pd.DataFrame([{
        "question_id": qid, "subject": "s", "question_text": question,
        "choice_a": "Alpha", "choice_b": "Beta", "correct_option": "A",
    }])


def test_canonicalization_adversarial_types_unicode_and_urls():
    assert canonical_json({"mode": _Mode.A, "items": {1, "1"}})
    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_json({1: "a", "1": "b"})
    assert stable_digest({"x": "e\u0301"}) == stable_digest({"x": "é"})
    assert sanitize_url("https://HOST:443/v1/") == "https://host/v1"
    assert sanitize_url("https://h/v1?api-version=1") != sanitize_url("https://h/v1?api-version=2")
    assert stable_digest({"url": "https://h/v1?token=one"}) == stable_digest({"url": "https://h/v1?token=two"})
    assert sanitize_url("https://blob.test/data?sv=1&se=soon&sp=r&sig=one") == sanitize_url(
        "https://blob.test/data?sv=2&se=later&sp=rw&sig=two"
    )


@pytest.mark.parametrize("key", [
    "session_token", "id_token", "auth", "X-Auth-Token", "aws_access_key_id",
    "client_secret", "Authorization",
])
def test_extended_credential_names_are_refused_in_identity_payloads(key):
    # Refusal (not silent redaction) keeps the value out of every manifest
    # while also making it impossible for two different configs to collapse
    # into one identity because a value was blanked before hashing.
    with pytest.raises(ValueError, match="Credential-named key"):
        canonical_json({"nested": [{key: "DO-NOT-LEAK"}]})


@pytest.mark.parametrize("key", [
    "token", "secret", "secret_strength", "access_token_count", "signature",
])
def test_secret_shaped_scientific_params_keep_distinct_identities(key):
    # CB-4: secret-shaped but scientifically meaningful method parameters must
    # affect scientific identity — different values, different digests.
    assert stable_digest({"params": {key: "value-A"}}) != stable_digest({"params": {key: "value-B"}})


def test_untrusted_exception_text_is_sanitized():
    message = "Authorization: Bearer abc123 at https://user:pw@host/v1?api_key=secret&api-version=2"
    redacted = redact_text(message)
    assert "abc123" not in redacted and "pw" not in redacted and "secret" not in redacted
    assert "api-version=2" in redacted


def test_dataset_digest_preserves_runtime_text_and_overlap_is_semantic(tmp_path):
    assert dataset_content_digest(_rows('["x", "y"]')) != dataset_content_digest(_rows('[ "x",  "y" ]'))
    left = _rows("  Café\tquestion ")
    right = _rows("Cafe\u0301 question", qid="other")
    right["subject"] = "different metadata"
    right[["choice_a", "choice_b"]] = right[["choice_b", "choice_a"]].to_numpy()
    assert dataset_sample_identities(left) == dataset_sample_identities(right)

    duplicate = pd.concat([_rows(), _rows()], ignore_index=True)
    with pytest.raises(DatasetArtifactError, match="duplicate question_id"):
        write_prepared_dataset(duplicate, tmp_path, DatasetSpec("x", "test"))


def test_local_model_content_is_identity_bound(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("one")
    first = resolve_hf_model_identity(str(model), None)
    (model / "config.json").write_text("two")
    second = resolve_hf_model_identity(str(model), None)
    assert first["content_digest"] != second["content_digest"]


def test_generic_huggingface_runtime_spec_matches_prepare_cli_addressing():
    from choicebench.cli.prepare_data import _resolve_output_stem

    config = BenchmarkConfig("huggingface", hf_path="cais/mmlu", hf_subset="all")
    spec = spec_for_benchmark(config)
    assert spec.benchmark == "mmlu"
    assert spec.output_name == _resolve_output_stem(None, "cais/mmlu", "all") == "mmlu"


def test_packaged_prompt_resources_are_byte_identical_to_source_prompts():
    root = Path(__file__).resolve().parents[1]
    source = root / "prompts"
    packaged = root / "src" / "choicebench" / "resources" / "prompts"
    source_files = sorted(path.relative_to(source) for path in source.rglob("*.txt"))
    assert source_files == sorted(path.relative_to(packaged) for path in packaged.rglob("*.txt"))
    for relative in source_files:
        assert (source / relative).read_bytes() == (packaged / relative).read_bytes()


def test_manifest_enforces_protocol_and_canonicalization_versions():
    manifest = make_manifest({"value": 1})
    validate_manifest(manifest)
    assert manifest["payload"]["protocol_version"] == PROTOCOL_VERSION
    assert manifest["payload"]["canonicalization_version"] == CANONICALIZATION_VERSION
    manifest["payload"]["protocol_version"] = "unknown"
    with pytest.raises(ManifestCompatibilityError, match="protocol"):
        validate_manifest(manifest)


def test_result_and_cache_tampering_are_detected(tmp_path):
    metadata = {
        "experiment_id": "exp", "condition_id": "cond", "dataset_artifact_id": "ds",
        "dataset_selection_id": "sel", "model_id": "model", "method_id": "method",
        "prompt_id": "prompt", "benchmark_split": "test",
    }
    result = write_run_results(
        [{"question_id": "q1", "is_correct": False, **metadata}], tmp_path,
        "r", "direct", "model", "toy", condition_id="cond",
    )
    frame = pd.read_csv(result)
    frame.loc[0, "is_correct"] = True
    frame.to_csv(result, index=False)
    with pytest.raises(RuntimeError, match="integrity"):
        validate_result_artifact(result)

    cache = ResponseCache(tmp_path / "cache")
    cache.put("ab-key", {"raw_text": "A"})
    path = cache._path("ab-key")
    record = json.loads(path.read_text())
    record["payload"]["raw_text"] = "B"
    path.write_text(json.dumps(record))
    assert cache.get("ab-key") is None


def test_batch_mismatch_does_not_advance_checkpoint(tmp_path):
    import choicebench.cli.run_experiment as run

    class Runner:
        run_id = "r"
        backend = object()

        def run_many(self, rows):
            return []

    manager = CheckpointManager(
        tmp_path / "checkpoints", "r", "m", "model", "toy",
        condition_id="cond", experiment_id="exp", selection_id="sel",
    )
    with pytest.raises(RuntimeError, match="returned 0 result rows"):
        asyncio.run(run.run_method("m", Runner(), pd.DataFrame([{"question_id": "q1"}]), manager, tmp_path, 1))
    assert manager.load() is None


@pytest.mark.parametrize("value", ["../escape", "/absolute", "a/b", "", ".", "..", "white space"])
def test_run_id_path_traversal_is_rejected(value):
    with pytest.raises(ValueError, match="run ID"):
        validate_run_id(value)


def test_second_process_writer_lock_is_refused(tmp_path):
    path = tmp_path / "run.lock"
    with FileLock(path, "first"):
        with pytest.raises(RuntimeError, match="held by"):
            with FileLock(path, "second"):
                pass


def _config(models, methods, run, name="display"):
    return ExperimentConfig(
        name, models, [BenchmarkConfig("toy", n_samples=2)], methods,
        ["accuracy"], run,
    )


def test_operational_settings_and_grid_order_do_not_change_experiment_identity():
    import choicebench.cli.run_experiment as run

    models = [
        ModelConfig("api", "same", provider="openai", generation_kwargs=GenerationKwargsConfig(1)),
        ModelConfig("api", "same", provider="groq", generation_kwargs=GenerationKwargsConfig(1)),
    ]
    methods = [MethodConfig("direct_mcq"), MethodConfig("two_stage")]
    first = run.build_execution_plan(_config(models, methods, RunConfig(seed=4, checkpoint_every_n=1, concurrency_limit=1)))
    second = run.build_execution_plan(_config(
        list(reversed(models)), list(reversed(methods)),
        RunConfig(seed=4, resume=False, checkpoint_every_n=999, concurrency_limit=99),
        name="renamed",
    ))
    assert first.manifest["experiment_id"] == second.manifest["experiment_id"]
    assert {c["condition_id"] for c in first.conditions.values()} == {
        c["condition_id"] for c in second.conditions.values()
    }


def test_omitted_and_explicit_method_defaults_are_identity_equivalent():
    import choicebench.cli.run_experiment as run

    model = [ModelConfig("dummy", "same")]
    omitted = run.build_execution_plan(_config(model, [MethodConfig("two_stage")], RunConfig(seed=4)))
    explicit = run.build_execution_plan(_config(
        model, [MethodConfig("two_stage", params={"fallback_on_parse_failure": False})], RunConfig(seed=4),
    ))
    assert next(iter(omitted.conditions.values()))["method_id"] == next(
        iter(explicit.conditions.values())
    )["method_id"]


def test_machine_paths_are_replaced_by_logical_content_identity(tmp_path):
    import choicebench.cli.run_experiment as run

    external = tmp_path / "machine-specific" / "settings.json"
    external.parent.mkdir()
    external.write_text('{"value": 1}')
    config = _config(
        [ModelConfig("dummy", "same")],
        [MethodConfig("direct_mcq", params={"settings_path": str(external)})],
        RunConfig(seed=4),
    )
    rendered = json.dumps(run.build_execution_plan(config).manifest)
    assert str(tmp_path) not in rendered
    assert "settings.json" in rendered
