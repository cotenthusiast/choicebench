import importlib
import asyncio
import shutil
import json
from pathlib import Path

import pytest
import pandas as pd

from choicebench.config.schema import (
    BenchmarkConfig, ExperimentConfig, GenerationKwargsConfig, MethodConfig, ModelConfig, RunConfig,
)
from choicebench.io.writers import write_run_results

ROOT = Path(__file__).resolve().parents[1]


def _module():
    import choicebench.cli.run_experiment as module
    return importlib.reload(module)


def _config(models, methods):
    return ExperimentConfig("grid", models, [BenchmarkConfig("toy", n_samples=2)], methods, ["accuracy"], RunConfig(seed=42))


def test_same_method_name_different_params_and_same_model_name_different_configs_do_not_collide():
    mod = _module()
    config = _config(
        [
            ModelConfig("api", "same", provider="openai", generation_kwargs=GenerationKwargsConfig(1)),
            ModelConfig("api", "same", provider="openai", generation_kwargs=GenerationKwargsConfig(2)),
        ],
        [MethodConfig("direct_mcq", params={"variant": 1}), MethodConfig("direct_mcq", params={"variant": 2})],
    )
    plan = mod.build_execution_plan(config)
    ids = [c["condition_id"] for c in plan.conditions.values()]
    assert len(ids) == 4 and len(set(ids)) == 4
    assert len({c["model_id"] for c in plan.conditions.values()}) == 2
    assert len({c["method_id"] for c in plan.conditions.values()}) == 2


def test_same_model_display_name_different_provider_backend_settings_do_not_collide():
    mod = _module()
    config = _config(
        [ModelConfig("api", "same", provider="openai"), ModelConfig("api", "same", provider="groq")],
        [MethodConfig("direct_mcq")],
    )
    plan = mod.build_execution_plan(config)
    assert len({c["model_id"] for c in plan.conditions.values()}) == 2
    assert len({c["condition_id"] for c in plan.conditions.values()}) == 2


def test_identical_duplicate_conditions_are_rejected():
    mod = _module()
    config = _config([ModelConfig("dummy", "same")], [MethodConfig("direct_mcq"), MethodConfig("direct_mcq")])
    with pytest.raises(mod.ConfigurationError, match="Duplicate scientific condition"):
        mod.build_execution_plan(config)


def test_plan_manifest_never_serializes_secret_values():
    mod = _module()
    # Embedded URL credentials are sanitized out of the persisted manifest.
    config = _config(
        [ModelConfig("api", "same", provider="openai", base_url="https://user:pass@example.test/v1?token=x")],
        [MethodConfig("direct_mcq")],
    )
    rendered = json.dumps(mod.build_execution_plan(config).manifest)
    assert "user:pass" not in rendered and "token=x" not in rendered


def test_credential_named_method_param_is_refused_not_redacted():
    # CB-4: an unambiguous credential key in scientific configuration is a
    # hard error (fail closed) instead of being silently blanked before
    # hashing, which previously collapsed distinct configs into one identity.
    mod = _module()
    config = _config(
        [ModelConfig("dummy", "same")],
        [MethodConfig("direct_mcq", params={"api_key": "super-secret"})],
    )
    with pytest.raises(ValueError, match="Credential-named key"):
        mod.build_execution_plan(config)


def test_secret_shaped_scientific_param_changes_condition_identity():
    # CB-4: a method param merely *named* like a secret ("token") is ordinary
    # scientific configuration — different values must produce different
    # condition and experiment identities so they can never resume into one
    # another.
    mod = _module()
    plans = [
        mod.build_execution_plan(_config(
            [ModelConfig("dummy", "same")],
            [MethodConfig("direct_mcq", params={"token": value})],
        ))
        for value in ("value-A", "value-B")
    ]
    first, second = plans
    assert first.manifest["experiment_id"] != second.manifest["experiment_id"]
    first_ids = {item["condition_id"] for item in first.conditions.values()}
    second_ids = {item["condition_id"] for item in second.conditions.values()}
    assert first_ids.isdisjoint(second_ids)


def test_prompt_content_change_changes_experiment_and_condition_identity(tmp_path, monkeypatch):
    mod = _module()
    source = ROOT / "prompts"
    copied = tmp_path / "prompts"
    shutil.copytree(source, copied)
    monkeypatch.setattr(mod, "PROMPTS_DIR", copied)
    config = _config([ModelConfig("dummy", "same")], [MethodConfig("direct_mcq")])
    first = mod.build_execution_plan(config)
    prompt = copied / "v1" / "direct_mcq.txt"
    prompt.write_text(prompt.read_text() + "\nChanged.")
    second = mod.build_execution_plan(config)
    assert first.manifest["experiment_id"] != second.manifest["experiment_id"]
    assert next(iter(first.conditions.values()))["condition_id"] != next(iter(second.conditions.values()))["condition_id"]


def test_runner_prompt_snapshot_is_manifest_bound_and_tamper_evident(tmp_path):
    mod = _module()
    config = _config([ModelConfig("dummy", "same")], [MethodConfig("direct_mcq")])
    plan = mod.build_execution_plan(config)
    mod._ensure_run_prompt_snapshot(tmp_path, plan.manifest)
    prompt = plan.manifest["payload"]["prompts"]
    snapshot = tmp_path / prompt["run_snapshot_path"] / prompt["version"] / "direct_mcq.txt"
    assert snapshot.read_text() == prompt["files"]["direct_mcq"]["content"]
    snapshot.write_text(snapshot.read_text() + "tampered")
    with pytest.raises(mod.ConfigurationError, match="prompt snapshot is corrupt"):
        mod._ensure_run_prompt_snapshot(tmp_path, plan.manifest)


def test_completed_result_is_validated_and_skipped(tmp_path, monkeypatch):
    mod = _module()
    config = _config([ModelConfig("dummy", "same")], [MethodConfig("direct_mcq")])
    plan = mod.build_execution_plan(config)
    condition = next(iter(plan.conditions.values()))
    condition["experiment_id"] = plan.manifest["experiment_id"]
    questions = plan.selections[0].questions
    rows = pd.DataFrame({
        "question_id": questions["question_id"].astype(str),
        "condition_id": condition["condition_id"], "experiment_id": condition["experiment_id"],
        "dataset_selection_id": condition["selection_id"], "model_id": condition["model_id"],
        "method_id": condition["method_id"], "dataset_artifact_id": condition["artifact_id"],
        "prompt_id": condition["prompt_id"], "benchmark_split": condition["split"],
    })
    result = tmp_path / "results" / f"{condition['condition_id']}.csv"
    result.parent.mkdir()
    write_run_results(
        rows.to_dict(orient="records"), tmp_path, "r", "direct_mcq", "same", "toy",
        condition_id=condition["condition_id"],
    )
    monkeypatch.setattr(mod, "build_backend", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must skip")))
    asyncio.run(mod._run_model(
        config.methods[0], config.models[0], config.benchmarks[0], questions, None,
        config, "r", tmp_path, tmp_path / "checkpoints", {}, condition,
    ))
    result.unlink()
    result.with_suffix(".artifact.json").unlink()
    write_run_results(
        rows.iloc[:-1].to_dict(orient="records"), tmp_path, "r", "direct_mcq", "same", "toy",
        condition_id=condition["condition_id"],
    )
    with pytest.raises(mod.ConfigurationError, match="incomplete"):
        asyncio.run(mod._run_model(
            config.methods[0], config.models[0], config.benchmarks[0], questions, None,
            config, "r", tmp_path, tmp_path / "checkpoints", {}, condition,
        ))


def test_api_do_sample_does_not_change_experiment_identity():
    """CB-7: do_sample is a HuggingFace-only knob the API backend never
    consumes, so two otherwise identical API configs must share one identity
    (and therefore resume into one another). HF identities still bind it."""
    mod = _module()
    plans = [
        mod.build_execution_plan(_config(
            [ModelConfig("api", "gpt-x", provider="openai",
                         generation_kwargs=GenerationKwargsConfig(do_sample=do_sample))],
            [MethodConfig("direct_mcq")],
        ))
        for do_sample in (False, True)
    ]
    first, second = plans
    assert first.manifest["experiment_id"] == second.manifest["experiment_id"]
