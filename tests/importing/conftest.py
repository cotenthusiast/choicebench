from pathlib import Path

import pandas as pd
import pytest

from choicebench.cli import evaluate_run
from choicebench.datasets import dataset_content_digest
from choicebench.identity import integrity_digest
from choicebench.infra.artifacts import atomic_write_text
from choicebench.io.readers import read_manifest_results
from choicebench.io.writers import validate_result_artifact, write_run_results
from choicebench.manifest import (
    build_manifest_payload,
    ensure_manifest,
    initial_run_state,
    make_manifest,
    write_run_state,
)
from choicebench.metrics import BUILTIN_METRICS
from choicebench.provenance import implementation_identity


EXPECTED_EXPERIMENT_ID = "exp_d1b8b8ddecd1f16f"
EXPECTED_EVALUATION_ID = "eval_a1be80926996d03a"


@pytest.fixture
def synthetic_v2_run(tmp_path, monkeypatch) -> tuple[Path, dict, str]:
    questions = pd.DataFrame(
        [
            {
                "question_id": "q1",
                "subject": "arithmetic",
                "question_text": "What is 2 + 2?",
                "choice_a": "3",
                "choice_b": "4",
                "choice_c": "5",
                "choice_d": "6",
                "correct_option": "B",
                "correct_answer_text": "4",
            },
            {
                "question_id": "q2",
                "subject": "geography",
                "question_text": "What is the capital of France?",
                "choice_a": "Berlin",
                "choice_b": "Madrid",
                "choice_c": "Paris",
                "choice_d": "Rome",
                "correct_option": "C",
                "correct_answer_text": "Paris",
            },
        ]
    )
    prompt_content = "Question: {question_text}\nChoices:\n{choices}\nAnswer:"
    selection_id = "sel_legacyfixture"
    artifact_id = "dataset_legacyfixture"
    prompt_id = "prompt_legacyfixture"
    model_id = "model_legacyfixture"
    method_id = "method_legacyfixture"
    condition_id = "cond_legacyfixture"

    dataset_record = {
        "selection_id": selection_id,
        "artifact_id": artifact_id,
        "benchmark": "toy",
        "split": "test",
        "prepared_path": "toy/test.csv",
        "prepared_content_digest": "prepared-legacy-fixture",
        "prepared_metadata": {"schema_version": "choicebench.dataset-artifact.v1"},
        "selected_content_digest": dataset_content_digest(questions),
        "selected_question_ids": ["q1", "q2"],
        "selected_sample_identities": ["sample_q1", "sample_q2"],
        "run_snapshot_path": f"artifacts/datasets/{selection_id}.csv",
        "row_count": 2,
    }
    prompt_record = {
        "prompt_id": prompt_id,
        "version": "legacy-v1",
        "files": {
            "direct_mcq": {
                "content": prompt_content,
                "sha256": integrity_digest(prompt_content),
            }
        },
        "run_snapshot_path": f"artifacts/prompts/{prompt_id}",
    }
    model_record = {
        "model_id": model_id,
        "config": {"backend": "dummy", "model_name_or_path": "dummy-model"},
        "resolved_model": None,
    }
    method_record = {
        "method_id": method_id,
        "config": {"name": "direct_mcq", "params": {}, "preflight": None},
        "implementation": {"qualified_name": "legacy.fixture:DirectMCQ"},
    }
    condition_record = {
        "condition_id": condition_id,
        "benchmark_name": "toy",
        "split": "test",
        "selection_id": selection_id,
        "artifact_id": artifact_id,
        "model_id": model_id,
        "method_id": method_id,
        "prompt_id": prompt_id,
        "prompt_snapshot_path": prompt_record["run_snapshot_path"],
        "result_path": f"results/{condition_id}.csv",
        "checkpoint_path": f"checkpoints/{condition_id}.json",
        "result_metadata_path": f"results/{condition_id}.artifact.json",
        "model_display_name": "dummy-model",
        "resolved_model": None,
        "identity": {"fixture": "native-v2"},
    }
    source = {
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "dirty": False,
        "dirty_tracked_digest": None,
        "source_tree_digest": "source-tree-legacy-fixture",
    }
    environment = {
        "choicebench_version": "0.2.0",
        "python": "3.12.0",
        "implementation": "CPython",
        "dependencies": {"pandas": "2.2.0", "pytest": "8.3.0"},
    }
    payload = build_manifest_payload(
        config={"run": {"seed": 7}, "metrics": ["accuracy"]},
        datasets=[dataset_record],
        prompts=prompt_record,
        models=[model_record],
        methods=[method_record],
        conditions=[condition_record],
        source=source,
        environment=environment,
    )
    payload["evaluation"] = [
        {
            "name": "accuracy",
            "implementation": implementation_identity(BUILTIN_METRICS["accuracy"]),
        }
    ]
    manifest = make_manifest(payload)
    assert manifest["experiment_id"] == EXPECTED_EXPERIMENT_ID

    run_dir = tmp_path / "native-v2-fixture"
    ensure_manifest(run_dir, manifest)
    atomic_write_text(
        run_dir / dataset_record["run_snapshot_path"],
        questions.to_csv(index=False),
    )
    atomic_write_text(
        run_dir
        / prompt_record["run_snapshot_path"]
        / prompt_record["version"]
        / "direct_mcq.txt",
        prompt_content,
    )

    rows = []
    for question, parsed_choice in zip(
        questions.to_dict("records"), ("B", "A"), strict=True
    ):
        rows.append(
            {
                **question,
                "raw_text": f"The answer is {parsed_choice}",
                "parsed_choice": parsed_choice,
                "parse_status": "parse_ok",
                "normalized_text": parsed_choice,
                "parse_reason": "Answer successfully parsed",
                "score_status": "scored",
                "is_correct": parsed_choice == question["correct_option"],
                "transport_status": "ok",
                "experiment_id": manifest["experiment_id"],
                "condition_id": condition_id,
                "dataset_artifact_id": artifact_id,
                "dataset_selection_id": selection_id,
                "model_id": model_id,
                "method_id": method_id,
                "prompt_id": prompt_id,
                "benchmark_split": "test",
                "model_name": "dummy-model",
                "method_name": "direct_mcq",
            }
        )
    result_path = write_run_results(
        rows,
        run_dir,
        run_dir.name,
        "direct_mcq",
        "dummy-model",
        "toy",
        condition_id,
    )
    artifact = validate_result_artifact(result_path)
    state = initial_run_state(manifest)
    state["conditions"][condition_id] = {
        "status": "completed",
        "result_path": condition_record["result_path"],
        "result_sha256": artifact["file_sha256"],
    }
    write_run_state(run_dir, state)

    monkeypatch.setattr(evaluate_run, "RUNS_DIR", run_dir.parent)
    frame, loaded = read_manifest_results(run_dir)
    report = evaluate_run.build_evaluation_report(
        run_dir.name, frame, loaded, reparse=False
    )
    assert report["evaluation_id"] == EXPECTED_EVALUATION_ID

    return run_dir, manifest, EXPECTED_EVALUATION_ID
