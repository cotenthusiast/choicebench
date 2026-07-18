import pandas as pd
import pytest

from choicebench.config.schema import BenchmarkConfig, MethodConfig, PreflightConfig
from choicebench.datasets import DatasetSpec, write_prepared_dataset
import choicebench.preflight as preflight


def _rows(qid, question):
    return pd.DataFrame([{"question_id": qid, "subject": "s", "question_text": question,
                          "choice_a": "a", "choice_b": "b", "correct_option": "A"}])


def test_requested_calibration_split_is_loaded_and_overlap_is_content_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "PROCESSED_DIR", tmp_path)
    write_prepared_dataset(_rows("eval", "Eval?"), tmp_path, DatasetSpec("toy", "test", output_name="toy"))
    validation = write_prepared_dataset(_rows("cal", "Cal?"), tmp_path, DatasetSpec("toy", "validation", output_name="toy"))
    method = MethodConfig("pride", preflight=PreflightConfig("benchmark", "validation", 1))
    result = preflight.load_preflight(method, BenchmarkConfig("toy", "test"), 42,
                                      eval_artifact_id="different", eval_sample_identities=set(), return_selection=True)
    assert result.split == "validation" and result.artifact_id == validation.artifact_id
    eval_ids = set(result.sample_identities)
    with pytest.raises(ValueError, match="overlap"):
        preflight.load_preflight(method, BenchmarkConfig("toy", "test"), 42,
                                 eval_artifact_id="different", eval_sample_identities=eval_ids, return_selection=True)


def test_overlap_identity_ignores_source_specific_question_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "PROCESSED_DIR", tmp_path)
    validation = _rows("calibration-id", "Same question?")
    write_prepared_dataset(validation, tmp_path, DatasetSpec("toy", "validation", output_name="toy"))
    from choicebench.datasets import dataset_sample_identities
    eval_copy = _rows("evaluation-id", "Same question?")
    method = MethodConfig("pride", preflight=PreflightConfig("benchmark", "validation", 1))
    with pytest.raises(ValueError, match="overlap"):
        preflight.load_preflight(
            method, BenchmarkConfig("toy", "test"), 42,
            eval_artifact_id="different",
            eval_sample_identities=set(dataset_sample_identities(eval_copy)),
            return_selection=True,
        )
