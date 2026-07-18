# External Results Importer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `$subagent-driven-development` (recommended) or `$executing-plans` to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Add a reusable, provenance-preserving importer that validates
externally generated row-level results, publishes them as immutable
ChoiceBench-native-format imported result artifacts, and evaluates eligible
realizations without claiming ChoiceBench performed the original inference.

**Architecture:** Manifest v3 separates stable scientific conditions from
source-sensitive realizations while manifest-v2 readers remain byte-compatible.
The paper-agnostic importer parses strict YAML, scans exact CSV logical records,
validates rows against an explicitly classified dataset reference, builds
identity and lineage, and atomically publishes a complete staged run. A thin
Stage 1 profile translates the immutable freeze into the same generic types.

**Tech Stack:** Python 3.10+, dataclasses, `pathlib`, `csv`-compatible parsing,
pandas, PyYAML, SHA-256/canonical JSON through `choicebench.identity`, pytest,
setuptools, build, and Twine. The implementation adds no runtime dependency.

## Global Constraints

- Approved design: `docs/superpowers/specs/2026-07-18-external-results-importer-design.md`.
- Branch/worktree: `feat/external-results-importer` in
  `/home/cotenthusiast/Projects/choicebench-external-results-importer`.
- Preserve the existing 808-test baseline; before every task or review-fix
  commit run the targeted tests and `python -m pytest -q`, with zero failures.
- Use TDD for every behavior-changing task: add a failing behavioral test, run
  it and observe the expected failure, implement the smallest slice, rerun
  targeted tests, then the full suite, then commit. Test-only compatibility
  characterization (Task 1) and documentation/installed-package
  characterization (Task 18) intentionally begin with passing gates.
- Keep manifest-v2 validation, reading, evaluation identity, resume, and result
  ownership behavior intact for legacy runs. Only a pre-existing v2 run may
  continue writing v2-compatible native rows; every newly created run is v3 and
  records structured result origin explicitly.
- Keep semantic `condition_id` aligned with ChoiceBench's existing benchmark
  selection x model x method x prompt x seed/preflight/protocol identity.
  Source bytes, parsing, mapping, status, scope, authorization, repair, and
  lineage belong only to `realization_id` and downstream identities.
- Result-artifact identity is computed only after exact result bytes exist and
  is stored in the v2 sidecar/run state, never in the immutable v3 manifest or
  CSV. Pre-ownership transformation payloads and lineage-node IDs must not
  depend on their child realization/result/experiment.
- The acyclic publication order is: semantic dataset/model/method/prompt and
  condition records; source/evidence plus pre-ownership validation/lineage
  digests; realization; manifest/experiment; exact evidence index and
  realization-validation artifact; exact result CSV; result-artifact sidecar;
  final realization-keyed run state. Only downstream records may reference an
  earlier identity or exact file checksum.
- A manifest-v3 result path is
  `results/<realization_id>.csv` with sidecar
  `results/<realization_id>.artifact.json`.
- Published runs are immutable. A repair or offline transformation consumes a
  verified base run as read-only input and publishes a different run ID,
  experiment ID, realization ID, and result artifact. It never appends to,
  replaces, or mutates the base run.
- Keep the generic importer free of paper names, the 100-cell matrix, ARC IDs,
  Gemini/IHS/PriDe identities, and historical filename rules. All such knowledge
  belongs in `importing/profiles/stage1_paper_freeze.py` and profile tests.
- The Stage 1 freeze is read-only external input. Never create, rewrite, rename,
  chmod, or delete anything below
  `/home/cotenthusiast/Projects/model-generalization/paper_data_freeze`.
- Tests use programmatically generated small synthetic fixtures. Do not copy
  historical result rows, benchmark snapshots, cache trees, or reports into Git.
- Do not run model inference or any approved, held, excluded, or forensic queue.
- Explicit user-selected absolute source/output paths are valid after
  canonicalization. Specifications cannot choose output roots. Reject traversal,
  unsafe symlinks, recursive directory import, secrets, executable objects, and
  divergent overwrite.
- Every high-risk gate named below gets an independent read-only review before
  the next task. Resolve confirmed findings and rerun that gate's commands.
- Commit only the files named by the task. Never combine two task commits.

## Target file map

### New reusable importer modules

- `src/choicebench/importing/__init__.py` — stable public importer exports.
- `src/choicebench/importing/schema.py` — strict dataclass schema and YAML loader.
- `src/choicebench/importing/csv_adapter.py` — deterministic UTF-8 logical-record
  scanner and CSV adapter.
- `src/choicebench/importing/dataset_reference.py` — expected-question snapshot,
  trust classification, and derivation verification.
- `src/choicebench/importing/identity.py` — semantic condition, realization,
  lineage-node, origin assignment, and import-specification identities.
- `src/choicebench/importing/validation.py` — question-keyed row validation,
  evidence findings, extension-field preservation, and evaluable row creation.
- `src/choicebench/importing/authorization.py` — typed authorization validation.
- `src/choicebench/importing/overlays.py` — immutable repair and offline
  transformation derivation.
- `src/choicebench/importing/evidence.py` — run-local content-addressed evidence
  and validation artifacts.
- `src/choicebench/importing/transaction.py` — same-filesystem staged publication
  and verify-only existing-run behavior.
- `src/choicebench/importing/engine.py` — generic validate/import orchestration and
  machine-readable report.
- `src/choicebench/importing/profiles/__init__.py` — profile registry.
- `src/choicebench/importing/profiles/stage1_paper_freeze.py` — Stage 1-only
  translation, trust chain, statuses, queue ledgers, and schema maps.
- `src/choicebench/cli/import_results.py` — installed CLI with lazy workspace
  bootstrap.

### Existing modules changed deliberately

- `src/choicebench/manifest.py` — version-dispatched v2/v3 validation, normalized
  manifest view, v3 run state, and full child-digest/path verification.
- `src/choicebench/io/writers.py` — v2 writer compatibility plus non-circular v2
  result-artifact sidecars for v3 realizations.
- `src/choicebench/io/readers.py` — version-dispatched publication reader and
  explicit realization selection.
- `src/choicebench/cli/evaluate_run.py` — legacy evaluation-v1 compatibility plus
  realization-aware evaluation-v2.
- `src/choicebench/cli/run_experiment.py` — new native runs emit manifest v3 and
  native realization/origin metadata; existing v2 resumes stay v2.
- `src/choicebench/infra/checkpoint.py` — realization-bound checkpoint-v2 with
  legacy checkpoint-v1 resume compatibility.
- `src/choicebench/infra/artifacts.py` — no-replace directory publication and
  directory fsync helper.
- `pyproject.toml` — `choicebench-import-results` entry point only; no dependency
  or package-version change.
- `README.md`, `CHANGELOG.md`, and `docs/external-results-import.md` — user and
  adapter documentation.

### New and expanded tests

- `tests/importing/conftest.py` — synthetic generic and miniature Stage 1 freeze
  builders.
- `tests/importing/test_manifest_v2_compat.py`
- `tests/importing/test_manifest_v3.py`
- `tests/importing/test_schema.py`
- `tests/importing/test_csv_adapter.py`
- `tests/importing/test_dataset_reference.py`
- `tests/importing/test_identity.py`
- `tests/importing/test_result_artifact.py`
- `tests/importing/test_validation.py`
- `tests/importing/test_evidence_status.py`
- `tests/importing/test_authorization.py`
- `tests/importing/test_overlays.py`
- `tests/importing/test_evidence.py`
- `tests/importing/test_transaction.py`
- `tests/importing/test_engine.py`
- `tests/importing/test_evaluation.py`
- `tests/importing/test_native_v3.py`
- `tests/importing/test_stage1_profile.py`
- `tests/importing/test_cli.py`
- Expand `tests/test_publication_identity.py`, `tests/test_condition_grid.py`,
  `tests/io/test_readers.py`, `tests/io/test_writers.py`,
  `tests/scripts/test_evaluate_run.py`, `tests/scripts/test_reset_run.py`,
  `tests/scripts/test_build_backend.py`, `tests/test_pride_reproduction_wiring.py`,
  `tests/test_release_adversarial.py`, and `tests/test_wheel_smoke.py` only where
  each task says.

## Required review protocol

For every task, the implementer commits only after targeted and full tests pass.
For Tasks 1, 3, 5, 6, 7, 10, 11, 13, 14, 16, 17, and 19, dispatch a fresh read-only
reviewer after the commit. Give the reviewer only the approved specification,
the task text, and the commit diff. Do not advance until all confirmed Critical
and Important findings are resolved in a focused follow-up commit and the same
validation is rerun.

## Specification acceptance-test traceability

The approved specification's 65 acceptance cases are assigned before any
production edit. The implementing worker must retain these numbered case IDs in
test docstrings or parametrization IDs so the final compliance reviewer can
prove that no requirement was lost:

- Cases 1-6: Tasks 2, 7, 11, and 12
  (`test_schema.py`, `test_result_artifact.py`, `test_evidence.py`,
  `test_transaction.py`, and `test_engine.py`) cover valid import,
  deterministic identity, exact no-op, divergent refusal, checksum mismatch,
  and source mutation.
- Cases 7-17: Tasks 3, 4, 8, and 9 (`test_csv_adapter.py`,
  `test_dataset_reference.py`, `test_validation.py`, and
  `test_evidence_status.py`) cover question membership, gold/options,
  variable-option and three-option rows, null/NaN, predictions, correctness,
  columns, and method extensions.
- Cases 18-22: Tasks 9 and 13 (`test_evidence_status.py` and
  `test_evaluation.py`) cover complete/qualified/partial/malformed evidence,
  imported evaluation, and non-metric accounting.
- Cases 23-27: Task 10 (`test_authorization.py` and `test_overlays.py`) covers
  valid repair, unauthorized/conflicting replacements, repair identity/lineage,
  and offline-transformation lineage.
- Cases 28-31: Tasks 15 and 16 (`test_stage1_profile.py`) cover profile
  translation, hosted PriDe exclusion, held Gemini ARC IHS, and strict
  separation of approved and forensic queues.
- Cases 32-36: Tasks 17-19 (`test_cli.py`, `test_wheel_smoke.py`, and
  `test_release_adversarial.py`) cover dry-run, real import, outside-repository
  operation, installed wheel/sdist behavior, and adversarial security.
- Cases 37-46: Tasks 2, 9-11, 13, and 17 cover orthogonal statuses, audit-only
  identity exclusion, explicit new origin/legacy-only inference, evidence CAS,
  absolute path safety, exact malformed evidence, evidence-only bases,
  non-self-authorizing overlays, and derived-status recomputation.
- Cases 47-52: Tasks 5, 7, 10, and 13 cover the four identity layers,
  condition-sharing alternatives, exact-byte result identity, constituent and
  per-row origins, and offline derivation versus prediction origin.
- Cases 53-57: Tasks 4, 10, 15, and 16 cover typed inference/offline authority,
  the exact six-condition/18-question-cell offline authority, distinct dataset
  trust classes, and the frozen-input trust chain with its upstream limitation.
- Cases 58-62: Tasks 2, 3, and 16 cover every identity-bearing CSV setting,
  strict UTF-8/BOM/dialect/newline behavior, exact embedded-newline spans, and
  the three MMLU duplicate pairs with pre/post derivation digests.
- Cases 63-65: Tasks 5, 7, 11, and 12 cover acyclic publication identity,
  crash-safe no-replace publication, staging invisibility, and existing-run
  verify-only behavior.

The mapping is a minimum, not permission to omit cross-layer integration tests
listed in the individual tasks.

---

### Task 1: Freeze manifest-v2 compatibility before broad changes

**Files:**

- Create: `tests/importing/__init__.py`
- Create: `tests/importing/conftest.py`
- Create: `tests/importing/test_manifest_v2_compat.py`
- Inspect only: `src/choicebench/manifest.py`, `src/choicebench/io/writers.py`,
  `src/choicebench/io/readers.py`, `src/choicebench/cli/evaluate_run.py`

**Interfaces:**

- Consumes unchanged v0.2 functions: `make_manifest`, `validate_manifest`,
  `initial_run_state`, `write_run_results`, `read_manifest_results`, and
  `build_evaluation_report`.
- Produces `synthetic_v2_run(tmp_path, monkeypatch) -> tuple[Path, dict, str]`,
  a deterministic two-question native run with explicit source/environment
  records and a hard-coded expected experiment/evaluation identity.

- [ ] **Step 1: Confirm the untouched v0.2 baseline**

  ```bash
  python -m pytest -q
  ```

  Expected: exactly 808 tests pass with zero failures. Record the duration and
  commit SHA. If this does not hold, stop and diagnose the starting state before
  writing any plan task files.

- [ ] **Step 2: Add characterization tests before production changes**

  Create a literal two-question dataset/prompt/model/method/condition fixture.
  Pass fixed `source` and `environment` to `build_manifest_payload`, write the
  manifest, snapshots, result, sidecar, and state through the current v2 APIs,
  and assert the exact current IDs. The test names and assertions must be:

  ```python
  def test_v2_fixture_validates_without_rewrite(synthetic_v2_run):
      run_dir, manifest, _ = synthetic_v2_run
      before = {p.relative_to(run_dir): p.read_bytes()
                for p in run_dir.rglob("*") if p.is_file()}
      validate_manifest(manifest)
      frame, loaded = read_manifest_results(run_dir)
      after = {p.relative_to(run_dir): p.read_bytes()
               for p in run_dir.rglob("*") if p.is_file()}
      assert loaded["schema_version"] == "choicebench.manifest.v2"
      assert frame["question_id"].astype(str).tolist() == ["q1", "q2"]
      assert after == before


  def test_v2_evaluation_identity_and_shape_are_frozen(
      synthetic_v2_run, monkeypatch
  ):
      run_dir, manifest, expected_evaluation_id = synthetic_v2_run
      monkeypatch.setattr(evaluate_run, "RUNS_DIR", run_dir.parent)
      frame, _ = read_manifest_results(run_dir)
      report = evaluate_run.build_evaluation_report(
          run_dir.name, frame, manifest, reparse=False
      )
      assert report["schema_version"] == "choicebench.evaluation.v1"
      assert report["evaluation_id"] == expected_evaluation_id
      assert set(report["conditions"]) == {"cond_legacyfixture"}


  def test_v2_missing_result_origin_implies_native_only_in_compat_view(
      synthetic_v2_run,
  ):
      _, manifest, _ = synthetic_v2_run
      assert "result_origin" not in manifest["payload"]["conditions"][0]
  ```

  The fixture helper must assert its hard-coded experiment/evaluation IDs so an
  implementer cannot update expected values casually after a regression.

- [ ] **Step 3: Run the compatibility tests against unmodified v0.2 code**

  Run:

  ```bash
  python -m pytest tests/importing/test_manifest_v2_compat.py -v
  ```

  Expected: all characterization tests PASS. If they do not, correct the
  synthetic fixture; do not change production behavior in this task.

- [ ] **Step 4: Reconfirm the baseline**

  Run:

  ```bash
  python -m pytest -q
  ```

  Expected: the original 808 tests plus the new compatibility tests pass, with
  zero failures.

- [ ] **Step 5: Commit the compatibility boundary**

  ```bash
  git add tests/importing/__init__.py tests/importing/conftest.py \
    tests/importing/test_manifest_v2_compat.py
  git commit -m "test: freeze manifest v2 compatibility"
  ```

**Intermediate gate:** A fresh reviewer compares the synthetic run and exact
assertions with current v2 construction, reading, and evaluation. No manifest-v3
work begins until this test-only commit is approved.

---

### Task 2: Add the strict, paper-agnostic import specification

**Files:**

- Create: `src/choicebench/importing/__init__.py`
- Create: `src/choicebench/importing/schema.py`
- Create: `tests/importing/test_schema.py`
- Inspect: `src/choicebench/config/schema.py`, `src/choicebench/identity.py`

**Interfaces:**

```python
ImportState = Literal["validated", "imported", "failed"]
EvidenceStatus = Literal[
    "complete", "qualified", "partial", "malformed", "recoverable", "failed"
]
ScopeDisposition = Literal[
    "included", "excluded_from_paper_matrix", "held", "superseded"
]
PredictionOrigin = Literal[
    "native_inference",
    "external_historical_inference",
    "external_repair_inference",
]
DerivationOrigin = Literal[
    "native_execution", "external_import", "repair_overlay", "offline_transformation"
]


@dataclass(frozen=True)
class CsvDialectSpec:
    encoding: Literal["utf-8"] = "utf-8"
    bom_policy: Literal["forbid", "strip_utf8_bom"] = "forbid"
    decoding_errors: Literal["strict"] = "strict"
    delimiter: str = ","
    quote_character: str = '"'
    escape_character: str | None = None
    double_quote: bool = True
    line_terminators: tuple[str, ...] = ("crlf", "lf", "cr")
    mixed_line_terminators: Literal["allow", "forbid"] = "allow"
    final_record_without_terminator: Literal["allow", "forbid"] = "allow"
    blank_record_policy: Literal["reject"] = "reject"
    skip_initial_space: bool = False
    header: Literal["first_logical_record"] = "first_logical_record"
    strict_syntax: bool = True


@dataclass(frozen=True)
class NumericColumnSpec:
    source_column: str
    value_type: Literal["integer", "float"]
    null_allowed: bool
    finite_only: bool = True


@dataclass(frozen=True)
class OptionMappingSpec:
    mode: Literal["ordered_columns", "structured_json"]
    ordered_columns: tuple[str, ...]
    structured_column: str | None
    structured_label_key: str | None
    structured_text_key: str | None


@dataclass(frozen=True)
class SourceArtifactSpec:
    source_id: str
    path: Path                 # audit-only lookup location
    logical_path: str          # stable identity-bearing location
    expected_sha256: str
    format: Literal["csv"]
    format_version: str
    classification: Literal[
        "raw", "canonical", "derived", "repaired", "aggregate_only"
    ]
    dialect: CsvDialectSpec
    columns: Mapping[str, str]
    expected_columns: tuple[str, ...]
    ignored_columns: Mapping[str, str]
    null_values: tuple[str, ...]
    numeric_columns: tuple[NumericColumnSpec, ...]
    option_mapping: OptionMappingSpec
    extra_field_policy: Literal["preserve_unmapped", "reject_unmapped"]
    preserve_namespace: str
    source_run_id: str | None
    source_repository: str | None
    source_commit: str | None
    notes: Mapping[str, Any]


@dataclass(frozen=True)
class DatasetReferenceSpec:
    dataset_id: str
    benchmark_name: str
    split: str
    reference_kind: Literal[
        "independent_input_snapshot", "profile_derived_reference_snapshot"
    ]
    trust_label: str
    source_ids: tuple[str, ...]
    selection_source_id: str
    expected_question_ids: tuple[str, ...]
    selection_seed: int | None
    selection_n_samples: int | None
    subject_filter: tuple[str, ...]
    selection_unknown_reasons: Mapping[str, str]
    columns: Mapping[str, str]
    revision: str | None
    fingerprint: str | None
    derivation: Mapping[str, Any]
    limitations: tuple[str, ...]
    native_compatibility_identity: Mapping[str, Any] | None


@dataclass(frozen=True)
class ImportModelSpec:
    model_key: str
    display_name: str
    backend: str | None
    provider: str | None
    revision: str | None
    effective_parameters: Mapping[str, Any]
    unknown_reasons: Mapping[str, str]
    native_compatibility_identity: Mapping[str, Any] | None


@dataclass(frozen=True)
class ImportMethodSpec:
    method_key: str
    name: str
    effective_parameters: Mapping[str, Any]
    implementation: Mapping[str, Any] | None
    unknown_reasons: Mapping[str, str]
    native_compatibility_identity: Mapping[str, Any] | None


@dataclass(frozen=True)
class ImportPromptSpec:
    prompt_key: str
    template_identity: str | None
    template_digest: str | None
    template_contents: Mapping[str, str] | None
    unknown_reason: str | None
    native_compatibility_identity: Mapping[str, Any] | None


@dataclass(frozen=True)
class ResultOriginSpec:
    derivation_origin: DerivationOrigin
    default_prediction_origin: PredictionOrigin | None
    per_question_prediction_origins: Mapping[str, PredictionOrigin]


@dataclass(frozen=True)
class ImportConditionSpec:
    condition_key: str
    source_ids: tuple[str, ...]
    dataset_id: str
    model_key: str
    method_key: str
    prompt_key: str
    seed: int | None
    calibration_identity: Mapping[str, Any] | None
    preflight_identity: Mapping[str, Any] | None
    protocol_settings: Mapping[str, Any]
    generation_parameters: Mapping[str, Any]
    unknown_reasons: Mapping[str, str]
    expected_question_ids: tuple[str, ...]
    evidence_status: EvidenceStatus
    scope_disposition: ScopeDisposition
    executable: bool | None
    qualifications: tuple[Mapping[str, Any], ...]
    limitations: tuple[Mapping[str, Any], ...]
    damaged_question_ids: tuple[str, ...]
    recoverable_question_ids: tuple[str, ...]
    result_origin: ResultOriginSpec


@dataclass(frozen=True)
class AuthorizationSpec:
    authorization_id: str
    authorization_type: Literal["inference_repair", "offline_transformation"]
    source_id: str
    condition_question_reasons: Mapping[str, Mapping[str, str]]
    authority: str
    purpose: str
    executable: bool
    input_evidence_digests: Mapping[str, str]
    expected_snapshot_digests: Mapping[str, str]


@dataclass(frozen=True)
class OverlaySpec:
    overlay_id: str
    base_run_path: Path        # audit-only lookup location
    base_condition_digest: str
    base_realization_id: str
    base_realization_digest: str
    base_evidence_digests: Mapping[str, str]
    base_validation_artifact_sha256: str
    base_result_sha256: str | None
    source_id: str
    authorization_id: str
    replacement_reasons: Mapping[str, str]
    result_origin: ResultOriginSpec
    lineage_notes: Mapping[str, Any]
    implementation: Mapping[str, Any]
    input_digest: str
    preownership_output_digest: str
    expected_evidence_status: EvidenceStatus


@dataclass(frozen=True)
class ImportSpec:
    schema_version: Literal["choicebench.import-spec.v1"]
    import_name: str
    sources: tuple[SourceArtifactSpec, ...]
    datasets: tuple[DatasetReferenceSpec, ...]
    models: tuple[ImportModelSpec, ...]
    methods: tuple[ImportMethodSpec, ...]
    prompts: tuple[ImportPromptSpec, ...]
    conditions: tuple[ImportConditionSpec, ...]
    authorizations: tuple[AuthorizationSpec, ...]
    overlays: tuple[OverlaySpec, ...]
    metrics: tuple[str, ...]  # validated strictly against BUILTIN_METRICS
    provenance: Mapping[str, Any]
    audit: Mapping[str, Any]


def load_import_spec(path: Path) -> ImportSpec: ...
def stable_import_projection(spec: ImportSpec) -> dict[str, Any]: ...
def import_spec_digest(spec: ImportSpec) -> str: ...
```

The same file defines strict enums/dataclasses for dataset/model/method/prompt,
the orthogonal `import_state`, `evidence_status`, `scope_disposition`, structured
origins, expected question IDs, qualifications, and overlay references. The
schema has no output-root or output-path field.

- [ ] **Step 1: Write failing strict-schema tests**

  Parameterize tests for: a valid minimal YAML; every unknown top-level/nested
  key; non-mapping YAML; missing/duplicate references; strict booleans and finite
  numerics; invalid SHA-256; invalid enum; arbitrary YAML object tag; recursive
  credential-named keys; an attempted `output_root`; single-byte distinct ASCII
  delimiter/quote/escape; fixed UTF-8/strict decoding; and explicit unknown
  provenance as `{value: null, reason: "not recorded by producer"}`.
  Reject every metric not present in the closed `BUILTIN_METRICS` registry,
  especially `module:Class`, dotted import paths, entry points, and filesystem
  paths. Import specifications can never select a dynamic import target.
  Cover ordered-column and structured-JSON option declarations, typed finite
  numeric policies, and preserve/reject extra-field policies; reject incomplete
  or contradictory option declarations and duplicate numeric-column rules.
  Validate optional native-compatibility payloads against exact current
  dataset/model/method/prompt allowed keys and recomputed full/short identities;
  reject arbitrary claimed IDs or partial/mismatched payloads. Stage 1 fixtures
  leave these fields null rather than fabricating native equivalence.

  Add these identity assertions:

  ```python
  def test_audit_locations_do_not_change_import_spec_digest(
      minimal_spec, minimal_overlay_spec
  ):
      moved = replace(
          minimal_spec,
          audit={"source_path": "/different/host", "imported_at": "later"},
      )
      assert import_spec_digest(moved) == import_spec_digest(minimal_spec)

      moved_source = replace(
          minimal_spec,
          sources=(replace(minimal_spec.sources[0], path=Path("/other/freeze/source.csv")),),
      )
      assert import_spec_digest(moved_source) == import_spec_digest(minimal_spec)

      moved_base = replace(
          minimal_overlay_spec,
          overlays=(replace(minimal_overlay_spec.overlays[0],
                            base_run_path=Path("/other/home/runs/base")),),
      )
      assert import_spec_digest(moved_base) == import_spec_digest(minimal_overlay_spec)


  @pytest.mark.parametrize(
      "change",
      ["mapping", "dialect", "numeric_policy", "option_policy",
       "extra_field_policy", "status", "scope"],
  )
  def test_stable_mapping_fields_change_import_spec_digest(minimal_spec, change):
      changed = mutate_identity_field(minimal_spec, change)
      assert import_spec_digest(changed) != import_spec_digest(minimal_spec)
  ```

- [ ] **Step 2: Run tests and observe the missing module**

  Run:

  ```bash
  python -m pytest tests/importing/test_schema.py -v
  ```

  Expected: FAIL during collection with
  `ModuleNotFoundError: No module named 'choicebench.importing'`.

- [ ] **Step 3: Implement strict dataclass parsing and stable projection**

  Use `yaml.safe_load`, explicit allowed-key sets, the existing credential-key
  detection/canonicalization rules, and `integrity_digest`. Reject rather than
  coerce booleans, integers, non-finite values, enum spellings, paths in stable
  provenance, and unknown keys. Export only stable public types from
  `importing/__init__.py`.

- [ ] **Step 4: Run targeted and full tests**

  ```bash
  python -m pytest tests/config/test_schema.py tests/importing/test_schema.py -v
  python -m pytest -q
  ```

  Expected: all schema tests pass; the original 808 tests remain green.

- [ ] **Step 5: Commit**

  ```bash
  git add src/choicebench/importing/__init__.py \
    src/choicebench/importing/schema.py tests/importing/test_schema.py
  git commit -m "feat: add strict external import specification"
  ```

---

### Task 3: Implement exact UTF-8 CSV logical-record scanning

**Files:**

- Create: `src/choicebench/importing/csv_adapter.py`
- Create: `tests/importing/test_csv_adapter.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class OpenedSource:
    source_id: str
    audit_path: Path
    logical_path: str
    data: bytes
    sha256: str


@dataclass(frozen=True)
class LogicalRecordSpan:
    index: int
    start: int
    end: int
    terminator: bytes


@dataclass(frozen=True)
class SourceRow:
    values: Mapping[str, str | None]
    span: LogicalRecordSpan
    raw_sha256: str


@dataclass(frozen=True)
class AdaptedTable:
    columns: tuple[str, ...]
    rows: tuple[SourceRow, ...]
    source_sha256: str


class SourceAdapter(Protocol):
    def parse(
        self, source: OpenedSource, declaration: SourceArtifactSpec, *, strict: bool
    ) -> AdaptedTable: ...


def scan_csv_logical_records(
    data: bytes, dialect: CsvDialectSpec
) -> tuple[LogicalRecordSpan, ...]: ...


def parse_csv_source(
    source: OpenedSource, declaration: SourceArtifactSpec, *, strict: bool
) -> AdaptedTable: ...
```

- [ ] **Step 1: Write byte-literal failing tests**

  Use `Path.write_bytes` under `tmp_path`, never a committed CSV, for UTF-8 BOM,
  invalid UTF-8, LF/CRLF/CR, mixed terminators, forbidden mixed terminators,
  final record without terminator, blank records, doubled quotes, explicit
  escape characters, quoted embedded CR/LF/CRLF, and malformed unclosed quotes.
  Assert exact spans and row hashes:

  ```python
  def test_embedded_newline_span_is_exact():
      data = b'id,text\r\nq1,"line one\r\nline two"\r\nq2,end\n'
      spans = scan_csv_logical_records(data, CsvDialectSpec())
      assert data[spans[1].start:spans[1].end] == \
          b'q1,"line one\r\nline two"\r\n'
      assert spans[1].terminator == b"\r\n"


  def test_literal_nan_is_not_implicit_null(opened_csv, source_spec):
      table = parse_csv_source(opened_csv(b"id,value\nq1,NaN\n"), source_spec,
                               strict=True)
      assert table.rows[0].values["value"] == "NaN"
  ```

  Also cover header order, duplicate header refusal, extra columns in normal and
  strict modes, explicit ignored-column reasons, empty string versus declared
  null, each typed numeric policy and malformed numeric strings, ordered-column
  and structured-JSON choices, three-option rows, and six-option rows. Assert
  effective CLI strictness may tighten but never loosen the declared extra-field
  policy and that the effective policy is realization-identity-bearing.

- [ ] **Step 2: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_csv_adapter.py -v
  ```

  Expected: FAIL because `choicebench.importing.csv_adapter` is absent.

- [ ] **Step 3: Implement the binary scanner and adapter**

  Scan original bytes before decoding. Recognize ASCII quote/escape bytes and
  CRLF longest-first; separators inside quoted fields remain field bytes. Decode
  each exact logical span with strict UTF-8, then parse using the declared CSV
  semantics. Never use pandas in this adapter and never normalize malformed
  evidence.

- [ ] **Step 4: Run targeted and full tests**

  ```bash
  python -m pytest tests/importing/test_csv_adapter.py -v
  python -m pytest -q
  ```

  Expected: every dialect/raw-span case passes; no baseline regression.

- [ ] **Step 5: Commit**

  ```bash
  git add src/choicebench/importing/csv_adapter.py \
    tests/importing/test_csv_adapter.py
  git commit -m "feat: add deterministic CSV source adapter"
  ```

**Intermediate gate:** An adversarial read-only reviewer checks the state
machine against embedded newlines, escaped quotes, BOM offsets, malformed EOF,
and every identity-bearing dialect field.

---

### Task 4: Build expected-dataset references and trust-qualified snapshots

**Files:**

- Create: `src/choicebench/importing/dataset_reference.py`
- Create: `tests/importing/test_dataset_reference.py`
- Inspect: `src/choicebench/datasets.py`, `src/choicebench/pipeline/options.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ExpectedDataset:
    dataset_id: str
    benchmark_name: str
    split: str
    artifact_id: str
    artifact_digest: str
    selection_id: str
    selection_digest: str
    frame: pd.DataFrame
    selected_question_ids: tuple[str, ...]
    reference_kind: Literal[
        "independent_input_snapshot", "profile_derived_reference_snapshot"
    ]
    trust_label: str
    question_set_digest: str
    snapshot_digest: str
    derivation: Mapping[str, Any]
    derivation_digest: str
    limitations: tuple[str, ...]


def build_expected_dataset(
    declaration: DatasetReferenceSpec,
    opened_sources: Mapping[str, OpenedSource],
) -> ExpectedDataset: ...


def write_expected_snapshot(staged_run: Path, dataset: ExpectedDataset) -> dict: ...
def validate_expected_snapshot(run_dir: Path, record: Mapping[str, Any]) -> None: ...
```

- [ ] **Step 1: Write failing trust and variable-option tests**

  Cover independently supplied input plus selection IDs; profile-derived
  reference requiring two declared independent source groups; cross-source
  question/gold/option disagreement refusal; full derivation digest; unknown
  publisher revision limitation; duplicate selected ID refusal; missing selected
  ID refusal; stable order; six options; and ARC-style three options without a
  phantom empty fourth option. Recompute and verify full imported dataset-
  artifact and selection digests plus their short IDs; the selection projection
  follows ChoiceBench's existing artifact/content/sample-identity/seed/filter
  semantics using explicit unknown/null values where historical sampling inputs
  are unavailable.

  Assert two differently located/mapped/trust-classified references that produce
  identical normalized semantic rows have the same artifact/selection IDs but
  different reference/derivation digests. Changing gold, ordered option text,
  membership, or selection order changes the applicable semantic IDs.

  ```python
  def test_reference_kinds_are_not_equivalent(independent_decl, derived_decl, sources):
      independent = build_expected_dataset(independent_decl, sources)
      derived = build_expected_dataset(derived_decl, sources)
      assert independent.reference_kind == "independent_input_snapshot"
      assert derived.reference_kind == "profile_derived_reference_snapshot"
      assert independent.derivation_digest != derived.derivation_digest
      assert "not independently authenticated" in derived.limitations
  ```

- [ ] **Step 2: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_dataset_reference.py -v
  ```

  Expected: FAIL because the dataset-reference module is absent.

- [ ] **Step 3: Implement snapshot construction and self-validation**

  Use `build_option_map`, `correct_option_for_row`,
  `dataset_content_digest`, and existing atomic JSON/CSV helpers. Populate the
  run snapshot from the declared reference, never result-side fields. Preserve
  reference kind/trust/limitations in the realization-facing reference record,
  not in the semantic dataset artifact. Build full
  dataset-artifact and selection records with the same semantic boundaries and
  full-digest/short-ID pattern as current ChoiceBench dataset ownership; do not
  put source paths, parsing/mapping policy, trust classification, or audit fields
  into either semantic identity. The dataset artifact binds normalized semantic
  question/gold/ordered-option content plus benchmark/split; selection binds the
  exact selected semantic rows/question sample identities and known selection
  semantics. Thus a source mapping/trust/provenance change that yields identical
  semantic rows changes the realization but not artifact/selection/condition;
  an actual semantic snapshot or selection change changes all applicable
  semantic identities.

- [ ] **Step 4: Run targeted and full tests**

  ```bash
  python -m pytest tests/pipeline/test_options.py \
    tests/importing/test_dataset_reference.py -v
  python -m pytest -q
  ```

  Expected: trust distinctions, cross-source checks, and variable options pass;
  no baseline regression.

- [ ] **Step 5: Commit**

  ```bash
  git add src/choicebench/importing/dataset_reference.py \
    tests/importing/test_dataset_reference.py
  git commit -m "feat: add trusted import dataset references"
  ```

---

## Concrete compatibility resolution discovered during planning

Current `build_execution_plan()` hashes the deterministic relative
`prompt_snapshot_path` (`artifacts/prompts/<prompt_id>`) into `condition_id`.
The approved specification simultaneously requires v3 conditions to retain the
existing scientific ID and to exclude operational paths. Removing this field
would change every native condition ID. The implementation must therefore keep
this one prompt-ID-derived, machine-independent value as a frozen compatibility
discriminator in the v3 semantic condition payload. Tests prove it is exactly
derived from `prompt_id`; arbitrary/absolute paths remain forbidden. No other
path field is grandfathered. This is the only architecture clarification in the
plan and is directly required by inspected v0.2 code.

---

### Task 5: Separate semantic condition, realization, lineage, and origin identity

**Files:**

- Create: `src/choicebench/importing/identity.py`
- Create: `tests/importing/test_identity.py`
- Inspect: `src/choicebench/identity.py`,
  `src/choicebench/cli/run_experiment.py:980`

**Interfaces:**

```python
@dataclass(frozen=True)
class ImportSemanticRecords:
    dataset_artifact: Mapping[str, Any]
    selection: Mapping[str, Any]
    model: Mapping[str, Any]
    method: Mapping[str, Any]
    prompt: Mapping[str, Any]
    condition: Mapping[str, Any]


def make_semantic_condition(
    *, identity: Mapping[str, Any], fields: Mapping[str, Any]
) -> dict[str, Any]: ...


def build_import_semantic_identity(
    *,
    condition: ImportConditionSpec,
    dataset: ExpectedDataset,
    model: ImportModelSpec,
    method: ImportMethodSpec,
    prompt: ImportPromptSpec,
) -> ImportSemanticRecords: ...


def make_lineage_component(
    *,
    operation_type: str,
    question_id: str,
    parent_digests: Sequence[str],
    source_digests: Sequence[str],
    authorization_digest: str | None,
    implementation: Mapping[str, Any],
    parameters: Mapping[str, Any],
    input_digest: str,
    preownership_output_digest: str,
    prediction_origin: str,
) -> dict[str, Any]: ...


def make_result_origin(
    *,
    derivation_origin: Literal[
        "native_execution", "external_import", "repair_overlay",
        "offline_transformation"
    ],
    row_assignments: Sequence[tuple[str, str, str]],
) -> dict[str, Any]: ...


def importer_implementation_identity(
    *, adapter: Callable[..., Any], validator: Callable[..., Any]
) -> dict[str, Any]: ...


def make_realization(
    *,
    condition_id: str,
    condition_digest: str,
    identity: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> dict[str, Any]: ...
```

`row_assignments` entries are `(question_id, prediction_origin,
prediction_lineage_id)`. Records contain full SHA-256 digests plus short IDs.

- [ ] **Step 1: Write failing layered-identity tests**

  Use one base scientific identity and parameterize changes. Assert:

  ```python
  @pytest.mark.parametrize(
      "field",
      ["source_sha256", "mapping", "dialect", "evidence_status", "scope",
       "authorization_digest", "overlay_sha256", "prediction_origins"],
  )
  def test_realization_changes_do_not_change_condition(base_records, field):
      original_condition, original_realization = base_records
      changed_condition, changed_realization = mutate_realization(base_records, field)
      assert changed_condition["condition_id"] == original_condition["condition_id"]
      assert changed_realization["realization_id"] != original_realization["realization_id"]


  @pytest.mark.parametrize(
      "field", ["selection_id", "model_id", "method_id", "prompt_id", "seed",
                "protocol_settings"]
  )
  def test_scientific_change_changes_condition(base_records, field):
      changed = mutate_semantic_condition(base_records[0], field)
      assert changed["condition_id"] != base_records[0]["condition_id"]
  ```

  Also test base/repair/transformation sharing one condition; rejection of bare
  `mixed`; origin set/count and ordered mapping; offline rematching retaining
  underlying prediction origin; lineage component exclusion of child IDs,
  paths, timestamps, and final CSV hashes; audit changes having no effect;
  importer package/version plus adapter/validator source identities present and
  identity-bearing; the grandfathered prompt path preserving the exact current
  `condition_id`; and arbitrary path injection refusal.

  Add a generic-spec integration fixture that builds dataset-artifact,
  selection, external model, historical method, prompt, and condition records
  from the Task 2 dataclasses. Assert the final condition payload has the exact
  current keys (`benchmark` with artifact/selection IDs, `preflight`,
  `model_id`, `method_id`, `prompt_id`, frozen prompt discriminator, `seed`, and
  applicable `protocol_settings`). Scientific input changes alter the proper
  child/full/condition digests; source bytes, dialect, mapping, status, scope,
  and audit-only changes leave that payload fixed and alter only realization
  and downstream identities. Unknown historical implementation/revision/prompt
  values remain explicit null+reason records and are never replaced by native
  ChoiceBench identities.

  Freeze these canonical child payload contracts in literal tests:

  ```python
  imported_dataset_payload = {
      "schema_version": "choicebench.semantic-dataset.v1",
      "benchmark": benchmark_name,
      "split": split,
      "content_digest": normalized_semantic_content_digest,
  }
  imported_selection_payload = {
      "artifact_id": artifact_id,
      "content_digest": selected_semantic_content_digest,
      "sample_identities": ordered_sample_identities,
      "seed": selection_seed_or_explicit_unknown,
      "n_samples": selection_n_samples_or_explicit_unknown,
      "subject_filter": sorted_subject_filter,
  }
  imported_model_payload = {
      "schema_version": "choicebench.semantic-model.v1",
      "backend": backend_or_explicit_unknown,
      "provider": provider_or_explicit_unknown,
      "model": display_name,
      "revision": revision_or_explicit_unknown,
      "effective_parameters": canonical_parameters,
  }
  imported_method_payload = {
      "schema_version": "choicebench.semantic-method.v1",
      "name": exact_historical_name,
      "effective_params": canonical_parameters,
      "preflight": preflight_identity,
      "implementation": implementation_or_explicit_unknown,
  }
  imported_prompt_payload = {
      "schema_version": "choicebench.semantic-prompt.v1",
      "template_identity": identity_or_explicit_unknown,
      "template_digest": digest_or_explicit_unknown,
      "template_contents": contents_or_explicit_unknown,
  }
  ```

  Full digests cover these exact payloads; short prefixes remain `ds`, `sel`,
  `model`, `method`, and `prompt`. These imported payloads are the documented
  fallback when no exact current-native identity is recoverable. When a
  synthetic external specification supplies a fully validated
  `native_compatibility_identity` for every child, use the exact current native
  canonical payload instead, compare its child and condition identities with
  `build_execution_plan()`, and require equality.
  When historical fields are unknown or the imported semantic-dataset schema is
  necessarily content-derived rather than current `DatasetSpec`-derived, require
  explicit documented divergence while preserving the same condition payload
  structure and never pretending equivalence. Mapping, trust, and source
  provenance remain realization-only in either case.

  Parameterize every `CsvDialectSpec` field individually through final
  realization construction and require each parsing-affecting change to alter
  `realization_id` while preserving `condition_id`.

- [ ] **Step 2: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_identity.py -v
  ```

  Expected: FAIL because the importer identity module is absent.

- [ ] **Step 3: Implement canonical full/short identity records**

  Reuse `canonicalize`, `stable_digest`, `integrity_digest`, and `short_id`.
  Reuse `choicebench.provenance.implementation_identity` for registered importer,
  adapter, validator, and transformation callables; never accept their active
  code identity from untrusted YAML.
  Explicitly allow only the declared identity keys at every layer. Compute in
  this order: semantic condition; parent-derived lineage components;
  pre-ownership output; row-origin mapping; realization. Never accept a current
  experiment ID, result-artifact ID, final CSV SHA, output path, or audit field
  in realization identity.

  Build the imported dataset/model/method/prompt records with the same semantic
  payload boundaries and full-digest/short-ID pattern as native records.
  Historical method identity is declared provenance, not a claim that the
  current ChoiceBench method implementation produced the rows.

- [ ] **Step 4: Run targeted and full tests**

  ```bash
  python -m pytest tests/test_condition_grid.py tests/importing/test_identity.py -v
  python -m pytest -q
  ```

  Expected: all layered-identity cases pass and every existing native short ID
  test remains unchanged.

- [ ] **Step 5: Commit**

  ```bash
  git add src/choicebench/importing/identity.py tests/importing/test_identity.py
  git commit -m "feat: separate condition and realization identity"
  ```

**Intermediate gate:** Independent identity/provenance review. The reviewer must
construct the identity dependency graph and confirm there is no child/self edge
and no source/provenance field in the semantic condition.

---

### Task 6: Introduce manifest v3 while preserving manifest v2 exactly

**Files:**

- Modify: `src/choicebench/manifest.py`
- Create: `tests/importing/test_manifest_v3.py`
- Modify: `tests/importing/test_manifest_v2_compat.py`
- Modify: `tests/test_publication_identity.py`

**Interfaces:**

```python
PROTOCOL_VERSION = "choicebench.protocol.v2"  # frozen public legacy default
PROTOCOL_V3_VERSION = "choicebench.protocol.v3"
MANIFEST_SCHEMA_VERSION = "choicebench.manifest.v2"  # frozen public legacy default
MANIFEST_V3_SCHEMA_VERSION = "choicebench.manifest.v3"
RUN_STATE_SCHEMA_VERSION = "choicebench.run-state.v2"  # frozen public legacy default
RUN_STATE_V3_SCHEMA_VERSION = "choicebench.run-state.v3"


@dataclass(frozen=True)
class ManifestView:
    manifest: Mapping[str, Any]
    schema_version: str
    semantic_conditions: Mapping[str, Mapping[str, Any]]
    realizations: Mapping[str, Mapping[str, Any]]
    realization_ids_by_condition: Mapping[str, tuple[str, ...]]
    legacy_v2: bool


def make_manifest_v3(
    payload: Mapping[str, Any], *, audit: Mapping[str, Any] | None = None
) -> dict[str, Any]: ...


def validate_manifest_v2(manifest: Mapping[str, Any]) -> None: ...
def validate_manifest_v3(manifest: Mapping[str, Any]) -> None: ...
def validate_manifest(manifest: Mapping[str, Any]) -> None: ...
def normalize_manifest(manifest: Mapping[str, Any]) -> ManifestView: ...
```

Keep current `PROTOCOL_VERSION`, `MANIFEST_SCHEMA_VERSION`,
`RUN_STATE_SCHEMA_VERSION`, `build_manifest_payload()`, and `make_manifest()` as
v2-compatible public entry points even after Task 14. Importer and new native
creation call explicit v3 functions/constants. `normalize_manifest()` synthesizes exactly one in-memory
native realization for v2 without rewriting bytes or changing stored IDs.

- [ ] **Step 1: Write failing v3 and expanded v2 compatibility tests**

  Test `semantic_conditions` and `realizations` as separate unique tables;
  condition full-digest recomputation; realization full-digest recomputation;
  stable `semantic_grid_digest`; realization-to-condition full-digest binding;
  fixed realization result/sidecar/validation paths; semantic validation digest;
  explicit structured origin; rejection
  of result-artifact ID/final CSV SHA in a manifest; audit exclusion from
  experiment identity; unsafe paths; duplicate IDs; and v3 run state keyed by
  realization.

  Extend the Task 1 tests to ensure v2 normalization preserves stored
  experiment/condition/result paths, infers native origin only in memory, and
  leaves all run bytes unchanged.

- [ ] **Step 2: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_manifest_v3.py \
    tests/importing/test_manifest_v2_compat.py tests/test_publication_identity.py -v
  ```

  Expected: new v3 tests fail because the symbols/schema are absent; every Task
  1 v2 test still passes.

- [ ] **Step 3: Implement exact schema dispatch**

  Dispatch only on the exact `schema_version`, never on missing fields. Preserve
  current v2 validation code as `validate_manifest_v2`. Add child-ID/full-digest
  recomputation for v3, safe fixed paths, separate audit integrity, and
  realization-keyed v3 `initial_run_state`/`validate_run_state`. Add normalized
  access helpers instead of rewriting callers prematurely.

- [ ] **Step 4: Run compatibility, publication, and full tests**

  ```bash
  python -m pytest tests/importing/test_manifest_v3.py \
    tests/importing/test_manifest_v2_compat.py tests/test_publication_identity.py \
    tests/test_release_adversarial.py -v
  python -m pytest -q
  ```

  Expected: v2 byte/identity tests and v3 structural tests pass; no baseline
  failures.

- [ ] **Step 5: Commit**

  ```bash
  git add src/choicebench/manifest.py tests/importing/test_manifest_v3.py \
    tests/importing/test_manifest_v2_compat.py tests/test_publication_identity.py
  git commit -m "feat: add manifest v3 compatibility layer"
  ```

**Intermediate gate:** Independent manifest review covering v2 dispatch,
full-child-digest verification, path safety, audit exclusion, and the absence of
result-artifact identity from the immutable manifest.

---

### Task 7: Publish non-circular realization result artifacts

**Files:**

- Modify: `src/choicebench/io/writers.py`
- Create: `tests/importing/test_result_artifact.py`
- Modify: `tests/io/test_writers.py`
- Modify: `tests/test_publication_identity.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PreparedResultArtifact:
    result_path: str
    metadata_path: str
    csv_bytes: bytes
    metadata: Mapping[str, Any]


def prepare_manifest_result(
    results: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    realization_id: str,
) -> PreparedResultArtifact: ...


def publish_manifest_result(
    prepared: PreparedResultArtifact, *, run_dir: Path
) -> tuple[Path, Mapping[str, Any], bool]: ...


def validate_result_artifact(
    result_path: Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    realization: Mapping[str, Any] | None = None,
) -> dict[str, Any]: ...
```

The returned boolean means an identical artifact already existed. Preserve the
legacy/programmatic `write_run_results()` contract and result-artifact-v1
validation for v2.

Result-artifact-v2 metadata contains exact experiment/condition/realization full
digests; result file SHA-256/size/row and ordered-column/question digests;
dataset artifact/selection/expected-snapshot digests; model/method/prompt full
digests; import-spec and source-byte digests; evidence-index digest;
realization-validation semantic digest and exact file SHA-256; authorization and
lineage digests; structured derivation origin; constituent prediction-origin
set/count and ordered per-question origin/lineage digest; result-artifact full
digest/short ID; and a sidecar self-digest. Audit locations/timestamps are not
included. Imported realizations require the import/source/evidence bindings;
native-v3 realizations encode those fields as explicit not-applicable records
with reasons and still require dataset, validation, origin, lineage, and result
bindings. They are never silently omitted.

- [ ] **Step 1: Write failing result-artifact-v2 tests**

  Test exact CSV rendering in memory; mandatory uniform condition, realization,
  experiment, dataset/model/method/prompt, split, `prediction_origin`, and
  `prediction_lineage_id`; result ID/digest bound to the already fixed
  experiment/realization and exact file bytes; ordered columns/question/origin
  digests; no result ID inside CSV/manifest; correct realization paths; full
  sidecar self-validation; identical no-op; one-byte divergence refusal before
  writing; missing half-pair refusal; and v1 allowed only for legacy v2. Add a
  tampering/mismatch case for every v2 metadata binding above and compare it
  against the manifest realization, validation artifact, evidence index, exact
  CSV, and run state rather than merely trusting the sidecar's own digest.

  ```python
  def test_result_identity_is_computed_after_manifest(v3_manifest, native_rows):
      prepared = prepare_manifest_result(
          native_rows, manifest=v3_manifest,
          realization_id=native_rows[0]["realization_id"],
      )
      assert "result_artifact_id" not in v3_manifest
      assert prepared.metadata["experiment_id"] == v3_manifest["experiment_id"]
      assert prepared.metadata["file_sha256"] == hashlib.sha256(
          prepared.csv_bytes
      ).hexdigest()
      assert prepared.metadata["result_artifact_id"].startswith("result_")
  ```

- [ ] **Step 2: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_result_artifact.py \
    tests/io/test_writers.py tests/test_publication_identity.py -v
  ```

  Expected: new API tests fail; legacy writer tests pass.

- [ ] **Step 3: Implement preparation, v2 sidecar, and collision refusal**

  Render once to UTF-8 bytes, validate rows, compute the sidecar payload and
  `result_artifact_digest`, then add its short ID and sidecar integrity digest.
  `publish_manifest_result` may write only inside a staging tree or to an absent
  pair. If either target exists, validate and compare the entire proposed graph;
  return no-op only for exact equality.

- [ ] **Step 4: Run targeted and full tests**

  ```bash
  python -m pytest tests/importing/test_result_artifact.py \
    tests/io/test_writers.py tests/test_publication_identity.py -v
  python -m pytest -q
  ```

  Expected: v1/v2 sidecar dispatch and non-circular identity cases pass; no
  baseline regression.

- [ ] **Step 5: Commit**

  ```bash
  git add src/choicebench/io/writers.py tests/importing/test_result_artifact.py \
    tests/io/test_writers.py tests/test_publication_identity.py
  git commit -m "feat: publish realization result artifacts"
  ```

**Intermediate gate:** Independent provenance/collision review. Require a
reviewer-produced dependency graph proving
condition -> realization -> experiment -> CSV/result artifact, never the
reverse.

---

### Task 8: Validate source rows by question identity

**Files:**

- Create: `src/choicebench/importing/validation.py`
- Create: `tests/importing/test_validation.py`
- Inspect: `src/choicebench/pipeline/options.py`,
  `src/choicebench/scoring/scorer.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ValidationFinding:
    code: str
    source_id: str
    field: str | None
    question_id: str | None
    record_span: LogicalRecordSpan | None
    raw_row_sha256: str | None
    message: str


@dataclass(frozen=True)
class ValidatedSourceRows:
    source_id: str
    rows_by_question_id: Mapping[str, SourceRow]
    expected_question_ids: tuple[str, ...]
    findings: tuple[ValidationFinding, ...]
    findings_digest: str


def validate_source_rows(
    table: AdaptedTable,
    *,
    condition: ImportConditionSpec,
    expected: ExpectedDataset,
) -> ValidatedSourceRows: ...
```

- [ ] **Step 1: Write failing membership and scientific-content tests**

  Add one focused test per required failure: missing required field; null ID;
  duplicate ID with both raw spans; missing ID; unexpected ID; gold mismatch;
  option count mismatch; ordered option text mismatch; correct-option mismatch;
  invalid parsed prediction for the row's actual option set; recomputable
  correctness mismatch; malformed numeric/NaN/infinity; model/benchmark/split/
  method ownership mismatch; status inconsistent with observed coverage; and
  source-schema/extra-column violation.

  Parameterize option coverage with three-, four-, and six-option rows. Assert no
  input row is dropped, padded, deduplicated, reordered, repaired, or converted
  from malformed to valid.

- [ ] **Step 2: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_validation.py -v
  ```

  Expected: FAIL because `validation.py` is absent.

- [ ] **Step 3: Implement the fail-closed validator**

  Join only by exact string `question_id`. Compare all source-provided
  question/gold/option fields with the expected snapshot, but never let source
  values override it. Accumulate all deterministic findings in source-record
  order and raise a single actionable validation error for undeclared defects.
  Sanitize source text in messages while preserving field, question, span, and
  digest evidence.

- [ ] **Step 4: Run targeted and full tests**

  ```bash
  python -m pytest tests/importing/test_validation.py \
    tests/pipeline/test_options.py tests/scoring/test_scorer.py -v
  python -m pytest -q
  ```

  Expected: all question-keyed failures are detected and the baseline remains
  green.

- [ ] **Step 5: Commit**

  ```bash
  git add src/choicebench/importing/validation.py \
    tests/importing/test_validation.py
  git commit -m "feat: validate imported rows by question identity"
  ```

---

### Task 9: Preserve evidence status, malformed bytes, and extension fields

**Files:**

- Modify: `src/choicebench/importing/validation.py`
- Create: `tests/importing/test_evidence_status.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class RealizationValidation:
    evaluable_rows: tuple[Mapping[str, Any], ...]
    findings: tuple[ValidationFinding, ...]
    validation_digest: str
    computed_evidence_status: Literal[
        "complete", "qualified", "partial", "malformed", "recoverable", "failed"
    ]
    evaluable: bool
    qualifications: tuple[Mapping[str, Any], ...]
    limitations: tuple[Mapping[str, Any], ...]
    defect_question_ids: tuple[str, ...]


@dataclass(frozen=True)
class PreparedValidationArtifact:
    relative_path: str
    json_bytes: bytes
    file_sha256: str
    validation_digest: str


def normalize_realization_rows(
    validated: ValidatedSourceRows,
    *,
    condition: ImportConditionSpec,
    expected: ExpectedDataset,
    experiment_id: str | None = None,
    realization_id: str | None = None,
) -> RealizationValidation: ...


def prepare_realization_validation_artifact(
    validation: RealizationValidation,
    *,
    realization: Mapping[str, Any],
    evidence_records: Sequence[Mapping[str, Any]],
) -> PreparedValidationArtifact: ...


def write_realization_validation_artifact(
    staged_run: Path, prepared: PreparedValidationArtifact
) -> tuple[Path, bool]: ...


def validate_realization_validation_artifact(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    realization: Mapping[str, Any],
    expected_sha256: str,
) -> Mapping[str, Any]: ...
```

The first call before manifest construction produces a canonical pre-ownership
payload/digest. The second call after experiment and realization IDs are fixed
injects ownership without changing predictions or evidence decisions.

- [ ] **Step 1: Write failing status/extension tests**

  Cover complete, qualified, partial, malformed, recoverable, and failed
  evidence; complete+excluded and malformed+excluded; qualifications preserved
  beside metrics; exact declared defect IDs and reasons; undeclared defect
  refusal; no evaluable rows for partial/malformed/recoverable/failed/excluded/
  held/superseded or `aggregate_only` sources; unknown/extra fields preserved in namespaced
  `external_fields_json`; explicit ignored fields and reasons; credential-shaped
  field refusal; literal null/NaN preservation; and method-specific nested JSON
  round-trip.

  For malformed evidence assert:

  ```python
  assert finding.record_span == LogicalRecordSpan(2, 19, 47, b"\r\n")
  assert finding.raw_row_sha256 == sha256(source_bytes[19:47]).hexdigest()
  assert validation.evaluable_rows == ()
  ```

  Assert deterministic JSON bytes at
  `artifacts/imports/validation/<realization_id>.json`, self-digest validation,
  exact realization/evidence/finding bindings, checksum change on any finding or
  raw-row digest change, atomic temp-file publication inside staging, identical
  no-op/divergent refusal, and refusal for traversal, unsafe path,
  realization/evidence mismatch, tampering, or a raw-span digest that does not
  reproduce from the archived evidence bytes.

- [ ] **Step 2: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_evidence_status.py -v
  ```

  Expected: FAIL because normalization/status behavior is not implemented.

- [ ] **Step 3: Implement normalization without evidence repair**

  Fill evaluative question text/options/gold only from `ExpectedDataset`; retain
  original mapped and extension fields separately. Recompute evidence status
  from exact coverage/defects and refuse a declaration mismatch. Produce
  evaluable rows only for imported+included complete/qualified realizations.
  Never fabricate request IDs, timestamps, usage, prompts, responses, provider
  fields, commits, revisions, or seeds.

  Prepare one exact validation artifact for every realization, including
  evidence-only realizations. Its semantic `validation_digest` is available to
  realization identity before ownership; after the realization is fixed, its
  exact serialized file SHA-256 is recorded downstream in run state and any
  result sidecar, never back-propagated into the realization/manifest identity.

- [ ] **Step 4: Run targeted and full tests**

  ```bash
  python -m pytest tests/importing/test_validation.py \
    tests/importing/test_evidence_status.py -v
  python -m pytest -q
  ```

  Expected: status eligibility and exact malformed evidence pass; no baseline
  regression.

- [ ] **Step 5: Commit**

  ```bash
  git add src/choicebench/importing/validation.py \
    tests/importing/test_evidence_status.py
  git commit -m "feat: preserve imported evidence and status"
  ```

---

### Task 10: Validate typed authorization and derive immutable overlays

**Files:**

- Create: `src/choicebench/importing/authorization.py`
- Create: `src/choicebench/importing/overlays.py`
- Create: `tests/importing/test_authorization.py`
- Create: `tests/importing/test_overlays.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ValidatedAuthorizationBundle:
    authorization_id: str
    authorization_digest: str
    authorization_type: Literal["inference_repair", "offline_transformation"]
    grants: Mapping[str, Mapping[str, str]]  # condition_digest -> qid -> reason
    authority: str
    purpose: str
    executable: bool
    source_sha256: str
    input_evidence_digests: Mapping[str, str]
    expected_snapshot_digests: Mapping[str, str]


@dataclass(frozen=True)
class ValidatedAuthorization:
    bundle_id: str
    bundle_digest: str
    authorization_type: Literal["inference_repair", "offline_transformation"]
    condition_digest: str
    question_reasons: Mapping[str, str]
    authority: str
    purpose: str
    executable: bool
    source_sha256: str
    input_evidence_digests: Mapping[str, str]
    expected_snapshot_digest: str


@dataclass(frozen=True)
class VerifiedBaseRealization:
    condition_id: str
    condition_digest: str
    realization_id: str
    realization_digest: str
    evidence_index_digest: str
    evidence_source_digests: Mapping[str, str]
    validation_artifact_sha256: str
    result_sha256: str | None
    rows_by_question_id: Mapping[str, Mapping[str, Any]]
    prediction_origins: Mapping[str, str]


@dataclass(frozen=True)
class DerivedRealizationPayload:
    condition_id: str
    condition_digest: str
    preownership_rows: tuple[Mapping[str, Any], ...]
    lineage_components: tuple[Mapping[str, Any], ...]
    result_origin: Mapping[str, Any]
    evidence_status: str
    replacement_question_ids: tuple[str, ...]


def validate_authorization_bundle(
    declaration: AuthorizationSpec,
    *,
    opened_source: OpenedSource,
    condition_digests: Mapping[str, str],
    expected: Mapping[str, ExpectedDataset],
) -> ValidatedAuthorizationBundle: ...


def authorization_for_condition(
    bundle: ValidatedAuthorizationBundle, *, condition_digest: str
) -> ValidatedAuthorization: ...


def derive_overlay(
    *,
    base: VerifiedBaseRealization,
    overlay: OverlaySpec,
    authorization: ValidatedAuthorization,
    overlay_table: AdaptedTable,
    expected: ExpectedDataset,
) -> DerivedRealizationPayload: ...
```

- [ ] **Step 1: Write failing typed-authorization tests**

  `inference_repair` must require exact condition/question membership,
  `queue_disposition=approved`, `execution_authority=authoritative`, and
  `executable=true`. Assert refusal for held, excluded, forensic, absent,
  false-executable, and offline records. `offline_transformation` must require a
  separately opened checksum-verified source with exact IDs/reasons/authority/
  purpose, input-evidence digests, expected-snapshot digest, and
  `inference_executable=false`; an overlay/spec cannot authorize its own IDs.
  Assert the types cannot substitute for one another or bind a different base
  condition/snapshot/evidence graph.

  `derive_overlay` requires exact equality between
  `overlay.base_evidence_digests`, the authorization slice's applicable
  `input_evidence_digests`, and the realization-scoped source/blob digest mapping
  returned by `verify_import_run`; no evidence-index digest alone is accepted as
  a substitute. Refuse missing, extra, or cross-realization evidence digests.

  Test both a one-condition bundle and a single content-addressed six-condition/
  18-pair bundle. The aggregate digest binds every condition digest, question,
  reason, evidence/snapshot digest, purpose, authority, and executable flag;
  each condition-scoped slice retains the same immutable bundle digest and
  cannot add or borrow grants from another condition.

- [ ] **Step 2: Write failing overlay/lineage tests**

  Cover a valid evidence-only base with no result checksum; a valid evaluable
  base; exact replacement subset; unauthorized, duplicate, unexpected, missing,
  and conflicting replacement IDs; multiple overlays targeting one ID;
  gold/options/ownership mismatch; status recomputation; base evidence/result
  checksum preservation; base -> authorization -> overlay/transformation ->
  resulting lineage; mixed historical/native repair origins with exact per-row
  mapping; external repair origin; offline rematching preserving underlying
  prediction origin; base condition/realization/validation-artifact digest
  verification; and transformation pre-ownership I/O/code digests. Require a
  declared origin assignment for every replacement ID: native repair rows use
  `native_inference`, externally generated repair rows use
  `external_repair_inference`, and offline rematching retains the base response's
  prediction origin while recording `offline_transformation` as derivation.
  Reject missing/extra assignments and any bare `mixed` value; identity-bind the
  complete per-row mapping and stable lineage notes.

  `derive_overlay` accepts only a `VerifiedBaseRealization`, never a filesystem
  path or unverified caller dictionary. Unit tests construct this frozen trusted
  value and prove the pure function cannot mutate it. Task 12 integration
  snapshots every base-run file before/after full-graph verification and
  derivation and requires byte equality. The derived payload shares the semantic
  condition only when model/dataset/method/prompt/protocol are unchanged.

- [ ] **Step 3: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_authorization.py \
    tests/importing/test_overlays.py -v
  ```

  Expected: FAIL because authorization/overlay modules are absent.

- [ ] **Step 4: Implement typed verification and pure derivation**

  Bundle verification consumes an independently opened authorization artifact.
  The overlay function is pure with respect to the filesystem and accepts only
  the trusted base value produced later by the engine's full graph verifier. It
  returns a derived pre-ownership payload and never opens/writes a base run or
  performs inference. Use the Task 5 lineage and origin builders, excluding
  child IDs from lineage identity.

- [ ] **Step 5: Run targeted and full tests**

  ```bash
  python -m pytest tests/importing/test_authorization.py \
    tests/importing/test_overlays.py tests/importing/test_identity.py -v
  python -m pytest -q
  ```

  Expected: typed authority, overlay conflicts, mixed origins, and base
  immutability all pass.

- [ ] **Step 6: Commit**

  ```bash
  git add src/choicebench/importing/authorization.py \
    src/choicebench/importing/overlays.py \
    tests/importing/test_authorization.py tests/importing/test_overlays.py
  git commit -m "feat: add authorized immutable import overlays"
  ```

**Intermediate gate:** Independent authorization/overlay review. The reviewer
must attempt self-authorization, cross-type authorization, duplicate/conflicting
replacement, base mutation, and a bare `mixed` origin.

---

### Task 11: Add verified evidence storage and atomic staged publication

**Files:**

- Modify: `src/choicebench/infra/artifacts.py`
- Create: `src/choicebench/importing/evidence.py`
- Create: `src/choicebench/importing/transaction.py`
- Create: `tests/importing/test_evidence.py`
- Create: `tests/importing/test_transaction.py`
- Modify: `tests/test_release_adversarial.py`

**Interfaces:**

```python
def open_verified_source(
    declaration: SourceArtifactSpec,
    *,
    containment_root: Path | None = None,
) -> OpenedSource: ...


def evidence_blob_path(staged_run: Path, sha256: str, format_name: str) -> Path: ...
def write_evidence_blob(
    staged_run: Path, source: OpenedSource, references: Sequence[str]
) -> dict[str, Any]: ...
def validate_evidence_blob(run_dir: Path, record: Mapping[str, Any]) -> None: ...
def write_evidence_index(
    staged_run: Path, records: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]: ...
def validate_evidence_index(
    run_dir: Path, expected_digest: str
) -> Mapping[str, Any]: ...


def fsync_tree(root: Path) -> None: ...
def atomic_publish_directory_no_replace(source: Path, destination: Path) -> None: ...


class ImportTransaction:
    def __init__(self, *, runs_dir: Path, run_id: str): ...
    def __enter__(self) -> "ImportTransaction": ...
    @property
    def staged_run(self) -> Path: ...
    def publish(self, validator: Callable[[Path], None]) -> Path: ...
    def __exit__(self, exc_type, exc, traceback) -> None: ...
```

- [ ] **Step 1: Write failing verified-source/evidence tests**

  Cover independently computed checksum mismatch; source mutation/replacement;
  same opened bytes used for hash and parse; absolute user source; profile-root
  containment; traversal, directory, device, and unsafe symlink refusal; suffix
  derived from validated format; within-run dedup; cross-run independent copy;
  tampered blob/sidecar/index; divergent digest-path refusal; exact malformed row
  evidence; no recursive import; and secret-shaped metadata refusal.

- [ ] **Step 2: Write failing transaction tests**

  Assert the established `.locks/<run-id>.manifest.lock`; same-filesystem sibling
  owner-marked staging; full validator invoked before publication; destination
  absent; real atomic no-replace behavior; fsync calls; failure leaves final path
  absent; stale staging is never recognized as a run; safe cleanup touches only
  its owner-marked directory; existing final run is verify-only and creates no
  staging; identical final graph no-ops; divergent graph refuses unchanged.

  Linux implementation tests should exercise libc `renameat2(...,
  RENAME_NOREPLACE)` through a small private wrapper. Mock absence of that symbol
  and assert fail-closed behavior; do not fall back to `os.replace` or an
  existence-check race.

- [ ] **Step 3: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_evidence.py \
    tests/importing/test_transaction.py tests/test_release_adversarial.py -v
  ```

  Expected: FAIL because evidence/transaction APIs do not exist.

- [ ] **Step 4: Implement same-bytes source opening and run-local CAS**

  Open only declared regular files; read once; hash `data`; pass `data` forward.
  Store only referenced row-level bytes at
  `artifacts/imports/evidence/sha256/<first-two>/<full-digest>.<safe-suffix>`
  inside staging. Sidecars bind byte size/format/references; identical bytes in
  one run reuse the blob. Write each blob, sidecar, and the self-digested
  evidence index through a temporary file, flush/fsync, revalidate exact bytes,
  and atomically rename within staging. Never accept a divergent existing blob,
  sidecar, or index.

- [ ] **Step 5: Implement staged no-replace publication**

  Acquire the manifest lock before examining final state. For a new run, build
  under a sibling staging directory, validate/fsync the complete tree, and call
  the no-replace primitive once. For an existing run, invoke the supplied full
  graph verifier only; exact equality no-ops and divergence raises. A report
  outside the run cannot affect commit state.

- [ ] **Step 6: Run targeted and full tests**

  ```bash
  python -m pytest tests/importing/test_evidence.py \
    tests/importing/test_transaction.py tests/test_release_adversarial.py -v
  python -m pytest -q
  ```

  Expected: source, CAS, crash, symlink, no-replace, and idempotence tests pass;
  no baseline regression.

- [ ] **Step 7: Commit**

  ```bash
  git add src/choicebench/infra/artifacts.py \
    src/choicebench/importing/evidence.py \
    src/choicebench/importing/transaction.py \
    tests/importing/test_evidence.py tests/importing/test_transaction.py \
    tests/test_release_adversarial.py
  git commit -m "feat: stage external imports atomically"
  ```

**Intermediate gate:** Independent filesystem/security review on source opening,
staging containment, owner markers, symlink behavior, no-replace semantics,
cleanup scope, and verify-only final runs.

---

### Task 12: Orchestrate generic dry-run and real imports

**Files:**

- Create: `src/choicebench/importing/engine.py`
- Modify: `src/choicebench/importing/__init__.py`
- Create: `tests/importing/test_engine.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ImportRequest:
    spec: ImportSpec
    run_id: str
    workspace_root: Path
    strict: bool
    overlays: tuple[OverlaySpec, ...] = ()


@dataclass(frozen=True)
class ImportPlan:
    manifest: Mapping[str, Any]
    expected_datasets: Mapping[str, ExpectedDataset]
    realizations: Mapping[str, RealizationValidation]
    evidence_records: tuple[Mapping[str, Any], ...]
    report_identity: Mapping[str, Any]


@dataclass(frozen=True)
class VerifiedImportRun:
    manifest: Mapping[str, Any]
    manifest_digest: str
    realizations: Mapping[str, VerifiedBaseRealization]


@dataclass(frozen=True)
class ImportCounts:
    conditions: Mapping[str, int]       # exact keys listed below
    realizations: Mapping[str, int]
    rows: Mapping[str, int]
    evidence_status: Mapping[str, int]
    scope_disposition: Mapping[str, int]
    prediction_origin: Mapping[str, int]
    derivation_origin: Mapping[str, int]
    source_classification: Mapping[str, int]
    column_disposition: Mapping[str, int]
    authorization: Mapping[str, int]


@dataclass(frozen=True)
class DefectReport:
    total_findings: int
    by_code: Mapping[str, int]
    question_ids_by_code: Mapping[str, tuple[str, ...]]
    findings_digest: str


@dataclass(frozen=True)
class ChecksumRecord:
    logical_id: str
    expected_sha256: str | None
    actual_sha256: str
    matches: bool | None


@dataclass(frozen=True)
class ArtifactChecksumRecord:
    logical_path: str
    file_sha256: str
    size: int
    semantic_digest: str | None


@dataclass(frozen=True)
class ImportChecksumReport:
    all_referenced_sources_match: bool
    sources: Mapping[str, ChecksumRecord]
    expected_snapshots: Mapping[str, ArtifactChecksumRecord]
    evidence_index: ArtifactChecksumRecord | None
    validation_artifacts: Mapping[str, ArtifactChecksumRecord]
    results: Mapping[str, ArtifactChecksumRecord]


@dataclass(frozen=True)
class ResultIdentityReport:
    result_artifact_id: str
    result_artifact_digest: str
    file_sha256: str


@dataclass(frozen=True)
class ImportIdentityProjection:
    import_spec_digest: str
    semantic_grid_digest: str
    experiment_id: str | None
    experiment_digest: str | None
    condition_digests: Mapping[str, str]
    realization_digests: Mapping[str, str]
    result_artifacts: Mapping[str, ResultIdentityReport]
    authorization_bundle_digests: Mapping[str, str]
    lineage_digests: Mapping[str, str]


@dataclass(frozen=True)
class ImportFailureReport:
    code: str
    source_id: str | None
    field: str | None
    question_id: str | None
    record_span: Mapping[str, int | str] | None
    raw_row_sha256: str | None
    sanitized_message: str


@dataclass(frozen=True)
class ImportAuditReport:
    source_location: str | None
    output_root: str | None
    imported_at: str


@dataclass(frozen=True)
class ImportReport:
    schema_version: Literal["choicebench.import-report.v1"]
    import_state: Literal["validated", "imported", "failed"]
    wrote_artifacts: bool
    idempotent_noop: bool
    run_id: str
    experiment_id: str | None
    counts: ImportCounts
    defects: DefectReport
    checksums: ImportChecksumReport
    identity_projection: ImportIdentityProjection
    profile: Mapping[str, Any]
    failures: tuple[ImportFailureReport, ...]
    audit: ImportAuditReport


def build_import_plan(request: ImportRequest) -> ImportPlan: ...
def verify_import_run(
    run_dir: Path, expected: ImportPlan | None = None
) -> VerifiedImportRun: ...
def execute_import(request: ImportRequest, *, dry_run: bool = False) -> ImportReport: ...
def serialize_import_report(report: ImportReport) -> dict[str, Any]: ...
```

The serialized JSON uses these fields exactly at top level. `profile` is `{}`
for generic imports; the Stage 1 profile uses stable `matrix`, `queues`,
`offline_authority`, `datasets`, and `source_schema_count` keys. `failures`
contains deterministic code/source/field/question/span/digest plus sanitized
message entries and is empty on success. Tracebacks, credentials, raw untrusted
text, absolute temporary paths, and timestamps appear in neither failures nor
identity projection; safe machine-local locations/timestamps may appear only
under `audit`.

Nested report-v1 keys are also closed and exact:

```text
counts.conditions: declared, intended, preserved
counts.realizations: planned, evaluable, evidence_only, selected
counts.rows: source, expected, validated, evaluable, defective, replaced
counts.evidence_status: complete, qualified, partial, malformed, recoverable, failed
counts.scope_disposition: included, excluded_from_paper_matrix, held, superseded
counts.prediction_origin: native_inference, external_historical_inference,
                          external_repair_inference
counts.derivation_origin: native_execution, external_import, repair_overlay,
                          offline_transformation
counts.source_classification: raw, canonical, derived, repaired, aggregate_only
counts.column_disposition: mapped, namespaced_preserved, ignored_with_reason
counts.authorization: inference_repair_bundles, offline_transformation_bundles,
                      granted_pairs, executable_pairs, nonexecutable_pairs
defects: total_findings, by_code, question_ids_by_code, findings_digest
checksums: all_referenced_sources_match, sources, expected_snapshots,
           evidence_index, validation_artifacts, results
identity_projection: import_spec_digest, semantic_grid_digest,
                     experiment_id, experiment_digest, condition_digests,
                     realization_digests, result_artifacts,
                     authorization_bundle_digests, lineage_digests
identity_projection.result_artifacts[realization_id]: result_artifact_id,
                     result_artifact_digest, file_sha256
profile: matrix, queues, offline_authority, datasets, source_schema_count
failures[]: code, source_id, field, question_id, record_span,
            raw_row_sha256, sanitized_message
audit: source_location, output_root, imported_at
```

Every mapping has unknown-key rejection in report self-validation. Empty
categories remain explicit zero/empty mappings. Generic success, validation
failure, imported success, exact no-op, repair overlay, offline transformation,
and Stage 1 profile tests assert the entire nested key set and cross-total
invariants; report identity excludes `audit` and `import_state` but includes all
stable semantic/provenance/lineage projections.
`record_span` has exactly `record_index`, `start`, `end`, and
`terminator_hex`. Tests delete, add, and tamper every typed leaf in turn and
require report self-validation failure.

- [ ] **Step 1: Write failing end-to-end synthetic engine tests**

  Cover valid generic CSV import; deterministic IDs; dry run writing nothing;
  real complete/qualified artifacts; partial/malformed/recoverable evidence-only
  realizations; source checksum mismatch and mutation; exact repeated import
  no-op with a byte-identical tree; mapping/source/status/overlay divergence
  refusal; result artifact/run-state linkage; atomic failure; import report
  counts; audit paths excluded from identities; and import/evaluation outside the
  repository using an absolute workspace.

  Assert run state uses `status=completed` for evaluable realizations and
  `status=evidence_only` for preserved ineligible realizations. Imported
  realization records explicitly use `import_state=imported`; a failed
  transaction publishes no realization/run. Dry-run reports use
  `import_state=validated` without changing planned realization IDs.

  Add a `load_import_spec()` -> `build_import_plan()` integration assertion for
  the exact dataset/model/method/prompt/semantic-condition bridge from Task 5,
  including the scientific-versus-realization-only mutation matrix. Assert a
  failed validation serializes `import_state=failed`, both operation booleans
  false, no experiment ID, and structured sanitized failures without publishing
  a run.

  Add base-plus-overlay integration only here, after the full verifier exists:
  verify the immutable base into `VerifiedImportRun`, pass the selected frozen
  realization to Task 10's pure derivation, and publish the derived result under
  a different run/experiment. Snapshot every base byte before/after; cover valid,
  evidence-only, unauthorized, conflict, mixed-origin, offline, and divergent
  cases. No Task 10 function opens the base filesystem directly.

- [ ] **Step 2: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_engine.py -v
  ```

  Expected: FAIL because engine orchestration is absent.

- [ ] **Step 3: Implement planning and dry-run**

  Resolve sources/datasets/references, adapt rows, validate, build conditions and
  realizations, resolve every importer metric only through the closed
  `BUILTIN_METRICS` registry and bind its registered implementation identity,
  then create manifest v3. Dry run executes every read/validation/
  identity step and returns the report without creating workspace directories,
  evidence, staging, results, state, or manifest. It still renders eligible
  final CSV bytes and prepares validation/result sidecars in memory after the
  planned manifest is fixed, so report-v1 includes the same per-realization
  result artifact IDs/digests/file SHA-256 as a real identical import.

- [ ] **Step 4: Implement staged real publication and graph verification**

  Use `ImportTransaction`; write snapshots/evidence/validation artifacts;
  normalize eligible rows with final ownership; prepare/publish result artifacts
  in staging; write final realization-keyed state with exact evidence-index and
  validation-artifact SHA-256 for every realization (plus result identity/digest/
  SHA-256 only when evaluable); validate the entire staged
  graph; publish once. Existing final runs call `verify_import_run` and either
  exact no-op or refuse.

  `verify_import_run` is the single shared full-graph verifier used for existing-
  run idempotence, overlay base input, and Task 13 publication reading. It
  validates manifest/state/snapshots/evidence/index/validation artifacts/results/
  sidecars, derives each realization's exact logical-source/blob SHA-256 mapping,
  and returns immutable trusted values only after the whole graph passes.

- [ ] **Step 5: Run targeted and full tests**

  ```bash
  python -m pytest tests/importing/test_engine.py \
    tests/importing/test_transaction.py tests/importing/test_result_artifact.py -v
  python -m pytest -q
  ```

  Expected: all generic dry-run/import/idempotence/status cases pass; no baseline
  regression.

- [ ] **Step 6: Commit**

  ```bash
  git add src/choicebench/importing/engine.py \
    src/choicebench/importing/__init__.py tests/importing/test_engine.py
  git commit -m "feat: import external results into immutable runs"
  ```

---

### Task 13: Read and evaluate explicit realizations

**Files:**

- Modify: `src/choicebench/io/readers.py`
- Modify: `src/choicebench/cli/evaluate_run.py`
- Create: `tests/importing/test_evaluation.py`
- Modify: `tests/io/test_readers.py`
- Modify: `tests/scripts/test_evaluate_run.py`
- Modify: `tests/test_publication_identity.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class RealizationSelection:
    policy: Literal["single_evaluable_per_condition", "explicit"]
    realization_ids: tuple[str, ...]


@dataclass(frozen=True)
class ManifestResultSet:
    rows: pd.DataFrame
    manifest: Mapping[str, Any]
    view: ManifestView
    state: Mapping[str, Any]
    selection: RealizationSelection
    artifacts: Mapping[str, Mapping[str, Any]]
    accounting: Mapping[str, Mapping[str, Any]]


def read_manifest_result_set(
    run_dir: Path, *, realization_ids: Sequence[str] | None = None
) -> ManifestResultSet: ...


def read_manifest_results(run_dir: Path) -> tuple[pd.DataFrame, dict]: ...


def build_evaluation_report_for_result_set(
    run_id: str,
    result_set: ManifestResultSet,
    *,
    reparse: bool,
) -> dict[str, Any]: ...
```

`read_manifest_results` remains the v2-compatible wrapper. For v3 it may
auto-select only when each semantic condition has at most one eligible
realization; ambiguity is a refusal. Add repeatable CLI
`choicebench-evaluate --realization-id <id>`.

- [ ] **Step 1: Write failing realization-reader tests**

  Cover manifest/state/evidence/snapshot/realization-validation-artifact/sidecar full-graph validation; undeclared
  result refusal; result for evidence-only realization refusal; exact row
  ownership; question membership and uniqueness; and fieldwise join of
  `question_text`, serialized choices, `correct_option`, and correct answer text
  against the expected snapshot. Include one variable-option ARC-style result.

  Add two eligible realizations for one condition and assert default refusal,
  explicit selection of exactly one, unknown/ineligible ID refusal, and no
  concatenation. All unselected/ineligible realizations remain in accounting.

- [ ] **Step 2: Write failing evaluation-v2 tests**

  Assert complete/qualified selected metrics; qualifications/limitations beside
  qualified metrics; partial/malformed/recoverable/failed/excluded/held/
  superseded accounting with null result/metrics; complete+excluded and
  malformed+excluded orthogonality; condition grouping separate from realization
  metrics; selection policy and every accounted digest in evaluation identity;
  audit path/timestamp changes excluded; result byte change included; and
  selection change produces a new evaluation ID. Add an importer-manifest metric
  changed to `module:Class` and prove refusal occurs before `importlib` is called;
  separately retain a legacy native custom-metric compatibility test.

  Report shape must include:

  ```python
  assert report["conditions"][condition_id]["realization_ids"] == [real_a, real_b]
  assert report["realizations"][real_a]["selected"] is True
  assert report["realizations"][real_a]["benchmark_name"] == "arc_challenge"
  assert report["realizations"][real_a]["evidence_status"] == "complete"
  assert report["realizations"][real_a]["scope_disposition"] == "included"
  assert report["realizations"][real_a]["prediction_origins"] == [
      "external_historical_inference"
  ]
  assert report["realizations"][real_a]["option_count_distribution"] == {
      "3": 1,
      "4": 1,
  }
  assert report["realizations"][real_b]["selected"] is False
  assert report["realizations"][malformed]["metrics"] == {}
  ```

  Re-run the Task 1 v2 fixture and require the exact evaluation-v1 identity and
  shape to remain unchanged.

- [ ] **Step 3: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_evaluation.py tests/io/test_readers.py \
    tests/scripts/test_evaluate_run.py \
    tests/importing/test_manifest_v2_compat.py -v
  ```

  Expected: new v3 selection/evaluation tests fail; legacy tests pass.

- [ ] **Step 4: Implement version-dispatched reading**

  Preserve the current v2 path in a private helper. For v3, use
  `verify_import_run` rather than duplicating graph verification, then enforce
  row ownership, status eligibility, and explicit selection over its trusted
  realization records.
  Never call `read_all_run_results`.

- [ ] **Step 5: Implement evaluation-v2 and keep evaluation-v1**

  Branch on manifest schema. Keep the existing v1 identity/report function
  byte-for-byte for legacy input. Build v2 units sorted by full realization
  identity and bind selection, statuses, qualifications, evidence,
  authorization, lineage, result IDs/digests/checksums, metric implementations,
  and optional reparsing implementations.

  For importer-created manifests, instantiate only the manifest-bound built-in
  metric name after rechecking it against `BUILTIN_METRICS` and its recorded
  implementation identity; never pass an importer metric string to the current
  dynamic `_load_metric` path. Preserve the existing trusted native v2/v3 custom-
  metric behavior without making it reachable from an import specification.

- [ ] **Step 6: Run targeted and full tests**

  ```bash
  python -m pytest tests/importing/test_evaluation.py tests/io/test_readers.py \
    tests/scripts/test_evaluate_run.py tests/test_publication_identity.py \
    tests/importing/test_manifest_v2_compat.py -v
  python -m pytest -q
  ```

  Expected: v2 identity compatibility and v3 selection/accounting both pass;
  the full baseline is green.

- [ ] **Step 7: Commit**

  ```bash
  git add src/choicebench/io/readers.py \
    src/choicebench/cli/evaluate_run.py tests/importing/test_evaluation.py \
    tests/io/test_readers.py tests/scripts/test_evaluate_run.py \
    tests/test_publication_identity.py
  git commit -m "feat: evaluate explicit result realizations"
  ```

**Intermediate gate:** Independent reader/evaluation review. Require tests or a
manual adversarial construction demonstrating that two alternatives cannot be
silently combined and that v2 evaluation identity remains exact.

---

### Task 14: Migrate newly created native runs to manifest v3

**Files:**

- Modify: `src/choicebench/cli/run_experiment.py`
- Modify: `src/choicebench/infra/checkpoint.py`
- Modify: `tests/test_condition_grid.py`
- Modify: `tests/infra/test_checkpoint.py`
- Modify: `tests/scripts/test_reset_run.py`
- Modify: `tests/scripts/test_build_backend.py`
- Modify: `tests/test_pride_reproduction_wiring.py`
- Modify: `tests/test_wheel_smoke.py`
- Create: `tests/importing/test_native_v3.py`

**Interfaces:**

```python
class ExecutionPlan:
    selections: Sequence[BenchmarkSelection]
    preflights: Mapping[tuple[int, int], Any]
    semantic_conditions: Mapping[str, Mapping[str, Any]]
    realizations: Mapping[str, Mapping[str, Any]]
    jobs: Mapping[tuple[int, int, int], Mapping[str, Any]]
    manifest: Mapping[str, Any]


def build_execution_plan(
    config: ExperimentConfig,
    *,
    manifest_schema: Literal["v3", "legacy_v2_resume"] = "v3",
) -> ExecutionPlan: ...
```

New checkpoint-v2 records use `checkpoints/<realization_id>.json` and bind
experiment, condition, realization, and selection. Existing checkpoint-v1 and
programmatic behavior remain readable only in legacy-v2 resume mode.
`build_execution_plan(..., manifest_schema="v3")` calls the explicit v3
manifest/state builders; it does not change the legacy public v2 constants or
the default behavior of `make_manifest()` for external programmatic callers.

- [ ] **Step 1: Write failing native-v3 tests before changing the runner**

  Assert a new dry execution plan has manifest v3; unchanged current
  `condition_id`; one `native_execution` realization per condition; explicit
  `native_inference` row assignment/lineage; realization-keyed run state;
  realization-addressed checkpoint/result/validation paths; deterministic native
  validation artifact with no external-evidence claim; result-artifact-v2; and native
  evaluation-v2. Assert no missing result origin in a new manifest.

  Add a pre-existing v2 run/resume test: detect the existing manifest before
  candidate creation, rebuild a v2-compatible plan, verify exact experiment/
  condition/path ownership, resume/skip without rewriting completed bytes, and
  never upgrade the manifest/state/sidecar in place. A new absent run may never
  request `legacy_v2_resume`.

- [ ] **Step 2: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_native_v3.py \
    tests/test_condition_grid.py tests/infra/test_checkpoint.py \
    tests/scripts/test_reset_run.py -v
  ```

  Expected: new-run v3 assertions fail; all existing runner tests pass.

- [ ] **Step 3: Split runtime jobs from semantic/realization manifest records**

  Preserve the current condition identity payload including the single frozen
  prompt-path compatibility discriminator. Build one explicit native realization
  per job. Runtime jobs may merge descriptive condition/realization fields for
  execution, but the manifest tables stay separate and semantic records contain
  no realization back-reference.

- [ ] **Step 4: Inject explicit native row origin and use v3 result publication**

  Prefer the existing `condition_metadata` injection after each runner batch so
  method implementations remain untouched. Inject condition/realization and
  ownership IDs, `prediction_origin=native_inference`, and a non-null stable
  `prediction_lineage_id`. Validate the completed native rows against the bound
  dataset snapshot, write the same self-digested realization-validation artifact
  with an empty external-evidence reference set, then publish through
  `prepare_manifest_result`/`publish_manifest_result`; update realization-keyed
  run state with validation SHA-256 plus result ID/digest/file checksum. Native
  failure/gate accounting retains its existing evidence artifact and emits no
  fake imported source/evidence record.

- [ ] **Step 5: Add versioned checkpoint/resume behavior**

  New checkpoints bind `realization_id`; legacy-v2 resume continues using the
  old condition checkpoint/result contract and implicit native origin. Do not
  combine v2 checkpoint rows with v3 rows. Preserve current reset safety.

- [ ] **Step 6: Run focused native compatibility tests**

  ```bash
  python -m pytest tests/importing/test_native_v3.py \
    tests/test_condition_grid.py tests/infra/test_checkpoint.py \
    tests/scripts/test_reset_run.py tests/scripts/test_build_backend.py \
    tests/test_pride_reproduction_wiring.py tests/test_wheel_smoke.py -v
  ```

  Expected: new native runs are v3, legacy v2 resumes remain byte-compatible,
  and native dummy-backend workflows pass.

- [ ] **Step 7: Run the full suite**

  ```bash
  python -m pytest -q
  ```

  Expected: all original 808 tests plus importer tests pass with zero failures.

- [ ] **Step 8: Commit**

  ```bash
  git add src/choicebench/cli/run_experiment.py \
    src/choicebench/infra/checkpoint.py \
    tests/importing/test_native_v3.py tests/test_condition_grid.py \
    tests/infra/test_checkpoint.py tests/scripts/test_reset_run.py \
    tests/scripts/test_build_backend.py tests/test_pride_reproduction_wiring.py \
    tests/test_wheel_smoke.py
  git commit -m "feat: emit native manifest v3 realizations"
  ```

**Intermediate gate:** Independent compatibility review must exercise one new
v3 native run and one pre-existing v2 resume. Stop if either changes scientific
condition IDs or rewrites legacy bytes.

---

### Task 15: Translate Stage 1 manifests and typed authorities

**Files:**

- Create: `src/choicebench/importing/profiles/__init__.py`
- Create: `src/choicebench/importing/profiles/stage1_paper_freeze.py`
- Extend: `tests/importing/conftest.py`
- Create: `tests/importing/test_stage1_profile.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ProfileTranslation:
    spec: ImportSpec
    report: Mapping[str, Any]


def translate_stage1_paper_freeze(freeze_root: Path) -> ProfileTranslation: ...
```

The profile registry exposes the name `stage1-paper-freeze`. No generic module
may import this profile or contain any Stage 1 constant.

- [ ] **Step 1: Build a programmatic miniature freeze fixture**

  In `tests/importing/conftest.py`, write only synthetic metadata/one-row source
  CSVs under `tmp_path` with the same headers and joins as:
  `expected_matrix.csv`, canonical manifest JSON/CSV, status matrix,
  `artifact_inventory.csv`, `frozen_artifact_index.csv`, approved/held/excluded/
  forensic queues, report, and checksum ledger. Exercise the authoritative join
  chain explicitly:

  ```text
  canonical_results_manifest.canonical_artifact_id
      -> artifact_inventory.artifact_id
      -> frozen raw path and checksum
      -> frozen_artifact_index canonical path and cell_id
  ```

  A missing, duplicate, or cross-cell link at any hop fails closed. Generate
  checksum values only for the miniature equivalents of files covered by the
  real ledger, including `reports/canonical_freeze_report.md`; intentionally
  leave only artifact inventory/index unledgered and assert those two remain
  independently hashed corroboration. Assert a one-byte canonical-report change
  fails its ledger check. Include one included
  complete, qualified, recoverable, partial, and malformed cell; both hosted
  PriDe exclusion combinations; one approved inference repair; one held record;
  one excluded record; and one forensic-only record.

- [ ] **Step 2: Write failing translation/authority tests**

  Require joins by explicit `cell_id`/`canonical_artifact_id`, never filename.
  Test exact status mapping:

  ```python
  STATUS_MAP = {
      "canonical_complete": "complete",
      "canonical_qualified": "qualified",
      "recoverable_from_existing_artifacts": "recoverable",
      "incomplete_requires_inference": "partial",
      "malformed_requires_inference": "malformed",
  }
  ```

  `excluded_from_paper_matrix` is not an evidence-status input to this map. For
  the two hosted PriDe rows, assign only
  `scope_disposition=excluded_from_paper_matrix`, then run ordinary row/defect
  validation to compute MMLU `complete` and ARC `malformed`. Assert these two
  dimensions remain independent and the records are preserved but not intended.

  Assert approved/authoritative/executable inference authority is separate from
  687-style held false, excluded false, and forensic-only records. For every
  queue row, expand exact `(cell_id, question_id)` pairs; require membership in
  the canonical manifest's missing/damaged IDs as applicable; require the three
  queue ledgers to be pairwise disjoint; and validate `queue_disposition`,
  `execution_authority`, `executable`, and `supersedes_for_execution`. Assert all
  828 classified queue pairs equal 138 approved + 687 held + 3 excluded and the
  776 forensic pairs never become authority.

  Assert method identities and all eight exact canonical source-header schemas
  map explicitly, with every column assigned to mapped, namespaced-preserved, or
  explicitly ignored-with-reason. Cover `semantic_matching_v1`,
  `text_extraction`, `two_stage_v1/v2/v3`, `independent_hypothesis`,
  `cyclic_generation_majority`, `PriDe`, and historical `baseline` without
  renaming/merging.

- [ ] **Step 3: Write the exact offline-authority test**

  The fixture must include these six historical semantic-matching cell IDs and
  authorize the same three question IDs for each cell:

  ```python
  RECOVERABLE_CELLS = (
      "cbp__gemini-2-5-flash__arc_challenge__semantic_matching_v1",
      "cbp__gpt-4-1-mini__arc_challenge__semantic_matching_v1",
      "cbp__llama-3-1-8b-instant__arc_challenge__semantic_matching_v1",
      "cbp__meta-llama-llama-3-1-8b-instruct__arc_challenge__semantic_matching_v1",
      "cbp__qwen-qwen2-5-7b-instruct-turbo__arc_challenge__semantic_matching_v1",
      "cbp__qwen-qwen2-5-7b-instruct__arc_challenge__semantic_matching_v1",
  )
  ```

  ```python
  RECOVERABLE_QUESTION_IDS = (
      "79e8c959bbeb74a0",
      "ad6b5d46ae54842c",
      "c30e75b011696a95",
  )
  assert report["offline_authority"]["condition_count"] == 6
  assert report["offline_authority"]["question_cell_count"] == 18
  assert report["offline_authority"]["inference_executable"] is False
  ```

  Assert each cell has exactly `RECOVERABLE_QUESTION_IDS` in
  `recoverable_question_ids`; preserve every question reason; and bind the
  canonical source SHA-256/base-evidence digest, full semantic condition digest,
  expected-snapshot digest, allowed semantic-rematching purpose, and
  `inference_executable=false`. Verify the authority is a deterministic
  content-addressed projection of checksum-covered canonical-manifest JSON
  entries cross-checked fieldwise with the CSV, and that
  `arc_question_audit.csv` cannot be its trust anchor.

  Emit one `AuthorizationSpec` bundle containing the six condition-key grants,
  not six unrelated self-authorizing records. Assert its aggregate bundle digest
  is reproduced by all six condition-scoped validated slices.

- [ ] **Step 4: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_stage1_profile.py -v
  ```

  Expected: FAIL because the profile modules are absent.

- [ ] **Step 5: Implement manifest/checksum/queue translation**

  Independently hash every opened authoritative file against
  `checksums.sha256` when it has a ledger entry. Root identity/status/source
  authority in the checksum-covered canonical manifest and independently
  computed canonical/source-artifact bytes. Treat `artifact_inventory.csv`,
  `frozen_artifact_index.csv`, and reports as corroborating cross-reference
  evidence when the ledger does not cover them; record their independently
  computed audit digests but never promote them to a stronger trust anchor. Use
  exact method/header-signature mapping tables in the
  profile; do not inspect filenames for identity. Preserve original absolute
  paths only as audit metadata and freeze-relative paths as stable provenance.

- [ ] **Step 6: Run targeted and full tests**

  ```bash
  python -m pytest tests/importing/test_stage1_profile.py \
    tests/importing/test_authorization.py tests/importing/test_schema.py -v
  python -m pytest -q
  ```

  Expected: profile translation, exclusions, queue separation, offline authority,
  and paper-agnostic core checks pass.

- [ ] **Step 7: Commit**

  ```bash
  git add src/choicebench/importing/profiles/__init__.py \
    src/choicebench/importing/profiles/stage1_paper_freeze.py \
    tests/importing/conftest.py tests/importing/test_stage1_profile.py
  git commit -m "feat: translate Stage 1 import manifests"
  ```

---

### Task 16: Implement the Stage 1 expected-dataset trust chain

**Files:**

- Modify: `src/choicebench/importing/profiles/stage1_paper_freeze.py`
- Extend: `tests/importing/test_stage1_profile.py`

**Interfaces:**

```python
def build_stage1_expected_datasets(
    freeze_root: Path,
    checksum_ledger: Mapping[str, str],
) -> tuple[ExpectedDataset, ExpectedDataset, Mapping[str, Any]]: ...
```

- [ ] **Step 1: Extend the synthetic freeze with input/split artifacts**

  Create small synthetic ARC raw/normalized plus selection/metadata and MMLU
  raw/normalized plus selection/metadata files under these exact Stage 1
  relative paths:

  ```text
  raw/local_model_generalization/data/raw/arc_challenge_raw.csv
  raw/local_model_generalization/data/processed/arc_challenge_normalized.csv
  raw/local_model_generalization/data/splits/arc_challenge/robustness_ids.json
  raw/local_model_generalization/data/splits/arc_challenge/robustness_metadata.json
  raw/local_model_generalization/data/raw/mmlu_raw.csv
  raw/local_model_generalization/data/processed/mmlu_normalized.csv
  raw/local_model_generalization/data/splits/benchmark/robustness_ids.json
  raw/local_model_generalization/data/splits/benchmark/robustness_metadata.json
  ```

  ARC includes one three-option row. MMLU includes the exact three duplicate IDs
  from the approved spec, each occurring twice with identical parsed fields.
  Make selection order deliberately differ from normalized source order so the
  source-order keep-first rule and final selection-order projection are tested
  separately. Add every byte checksum to the synthetic ledger.

- [ ] **Step 2: Write failing ARC/MMLU trust-chain tests**

  Assert `reference_kind=independent_input_snapshot` and
  `trust_label=checksum_verified_freeze_internal`; raw/normalized/selection/
  metadata checksum verification; result independence; selection membership;
  stable snapshot order; variable ARC options; unknown upstream Hugging Face
  revision/authenticity limitation; and byte-identical copies/result agreement
  as corroboration only.

  For MMLU assert source-order, field-identical duplicate pairs, exact duplicate
  row digests, keep-first semantics equivalent to
  `drop_duplicates(subset="question_id", keep="first")`, and identity-bound
  pre/post ordered digests. Mutate the second occurrence of each pair in turn and
  require fail-closed conflict. A different duplicate policy must change the
  derivation/realization identity.

- [ ] **Step 3: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_stage1_profile.py \
    -k "dataset or duplicate or option or trust" -v
  ```

  Expected: new trust-chain assertions fail.

- [ ] **Step 4: Implement the frozen-input derivation**

  Verify the ledger against the exact opened bytes. Use the generic CSV adapter
  and dataset-reference builder. In registered profile code, deterministically
  derive the comparable semantic fields and question IDs from the raw rows and
  compare them fieldwise with the checksum-bound normalized rows; report this as
  a ChoiceBench revalidation, not proof that the archived normalizer was the
  producer. Reproduce the archived MMLU first-occurrence rule, but first require
  every duplicate occurrence to agree on all parsed fields. Record the archived
  script path/checksum as provenance evidence and the ChoiceBench implementation
  identity as the active verification/transformation identity. Never execute
  the archived script or any other freeze content. Preserve the limitation that
  this proves internal freeze consistency, not upstream publisher authenticity.

- [ ] **Step 5: Run focused, profile, and full tests**

  ```bash
  python -m pytest tests/importing/test_stage1_profile.py \
    tests/importing/test_dataset_reference.py \
    tests/importing/test_csv_adapter.py -v
  python -m pytest -q
  ```

  Expected: trust labels, three-option ARC, exact duplicate derivation, and
  profile translation pass; no baseline regression.

- [ ] **Step 6: Commit**

  ```bash
  git add src/choicebench/importing/profiles/stage1_paper_freeze.py \
    tests/importing/test_stage1_profile.py
  git commit -m "feat: verify Stage 1 dataset trust chain"
  ```

**Intermediate gate:** Independent Stage 1 review against the immutable freeze.
The reviewer independently checks ledger coverage, selected-ID counts, all three
MMLU duplicate pairs, variable ARC options, 100/2 scope split, status counts,
138/687/3 queue counts, and six-cell/18-row offline authority. Review is dry-run
and read-only.

---

### Task 17: Add the installed CLI, reports, and security regression suite

**Files:**

- Create: `src/choicebench/cli/import_results.py`
- Modify: `pyproject.toml`
- Create: `tests/importing/test_cli.py`
- Modify: `tests/test_release_adversarial.py`

**Interfaces:**

```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace: ...
def write_import_report(path: Path, report: Mapping[str, Any]) -> None: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

At module scope import only stdlib modules and `choicebench` package metadata;
do not import `choicebench.config.paths`, the engine, profiles, readers, or any
provider. `main()` parses and validates `--output-root`, sets
`CHOICEBENCH_HOME`, and only then imports workspace-dependent modules.

- [ ] **Step 1: Write failing subprocess CLI tests**

  Cover generic spec import; `--dry-run` and `--validate-only` equivalence;
  `--strict`; absolute `--output-root`; `CHOICEBENCH_HOME` fallback; current
  directory fallback; required safe run ID; profile dispatch; repeated
  `--overlay`; human summary; machine report; exact no-op; divergent refusal;
  sanitized errors; and help text. Run output-root tests in fresh subprocesses
  with `PYTHONPATH` controlled so module caching cannot hide import order.

  Machine report assertions use stable fields:

  ```python
  assert report["counts"]["scope_disposition"]["included"] >= 1
  assert report["counts"]["evidence_status"]["complete"] >= 1
  assert report["wrote_artifacts"] is expected_write
  assert report["idempotent_noop"] is expected_noop
  assert report["failures"] == []
  assert report["audit"]["source_location"] == str(source.resolve())
  assert "audit" not in report["identity_projection"]
  ```

- [ ] **Step 2: Extend adversarial/security tests before implementation**

  Add YAML executable-tag refusal; credential metadata; source traversal;
  specification-controlled output; unsafe source/output symlinks; recursive
  directory; source change after validation; malicious CSV error text;
  malformed byte sequences; result/evidence/manifest/state tampering; divergent
  report overwrite; overlay self-authorization; and response-cache forest
  refusal. Explicitly accept user-selected absolute source and output roots.

- [ ] **Step 3: Run and observe failure**

  ```bash
  python -m pytest tests/importing/test_cli.py \
    tests/test_release_adversarial.py -v
  ```

  Expected: CLI tests fail because the module/entry point is absent.

- [ ] **Step 4: Implement bootstrap parsing and command dispatch**

  Positional input is a generic spec path unless `--profile` is present, in
  which case it is the profile root. Required options/aliases are:

  ```text
  choicebench-import-results INPUT --run-id ID
      [--profile stage1-paper-freeze]
      [--dry-run | --validate-only]
      [--strict]
      [--output-root PATH]
      [--overlay PATH ...]
      [--report PATH]
  ```

  Output-root selection comes only from CLI/environment. Canonicalize it, refuse
  unsafe symlinks, set the environment, then lazily import and execute. Reports
  are atomic; an identical existing report is a no-op and divergent content is
  refused. Audit locations/timestamps remain outside report identity.

- [ ] **Step 5: Register the command without changing dependencies/version**

  Add exactly:

  ```toml
  choicebench-import-results = "choicebench.cli.import_results:main"
  ```

  Do not change `version = "0.2.0"` or package data.

- [ ] **Step 6: Run targeted and full tests**

  ```bash
  python -m pytest tests/importing/test_cli.py \
    tests/test_release_adversarial.py -v
  python -m pytest -q
  ```

  Expected: CLI/environment/security tests pass and the full baseline is green.

- [ ] **Step 7: Commit**

  ```bash
  git add src/choicebench/cli/import_results.py pyproject.toml \
    tests/importing/test_cli.py tests/test_release_adversarial.py
  git commit -m "feat: add external results import command"
  ```

**Intermediate gate:** Independent adversarial review of CLI import order, path
trust boundaries, YAML/CSV handling, report collision behavior, secret rejection,
and absence of inference execution.

---

### Task 18: Document and package the importer

**Files:**

- Create: `docs/external-results-import.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `tests/test_wheel_smoke.py`

**Interfaces:** No new runtime interface. Documentation examples use only the
installed `choicebench-import-results` and `choicebench-evaluate` commands.

- [ ] **Step 1: Extend the installed wheel/sdist smoke test first**

  In each installed environment, assert importer modules and console entry point
  exist; run `--help`; create a two-question generic CSV/spec outside the
  repository; dry-run; real import; exact repeated no-op; and evaluation. Assert
  the installed wheel/sdist contains no run, report, historical, cache, or test
  fixture data.

- [ ] **Step 2: Run and observe failure**

  ```bash
  python -m pytest tests/test_wheel_smoke.py -v
  ```

  Expected: PASS because Tasks 2-17 already supplied and registered the runtime
  functionality. Treat this as a packaging characterization gate before the
  documentation-only changes; if it fails, diagnose and fix the responsible
  earlier runtime/packaging task in a focused commit rather than hiding the
  problem in documentation.

- [ ] **Step 3: Write complete user/adapter documentation**

  `docs/external-results-import.md` must cover: external versus native inference;
  supported UTF-8 CSV/dialect fields; every import-spec section; generic and
  dry-run examples; `CHOICEBENCH_HOME`/absolute output; checksums and four
  identity layers; orthogonal statuses; expected-dataset trust; method-specific
  extension data; mixed prediction origins; immutable new-run overlays; typed
  inference/offline authorization; evaluation selection/accounting; security
  boundary; limitations; future adapter protocol; and a complete two-question
  example.

  Link the guide from README. Add an `Unreleased` changelog entry consistent
  with the existing newest-first format. State explicitly that no historical
  data or model inference is included. Do not change the package version.

- [ ] **Step 4: Run documentation-facing and full tests**

  ```bash
  python -m pytest tests/test_wheel_smoke.py tests/importing/test_cli.py -v
  python -m pytest -q
  ```

  Expected: wheel and sdist workflows work outside the repository; full suite
  passes.

- [ ] **Step 5: Commit**

  ```bash
  git add docs/external-results-import.md README.md CHANGELOG.md \
    tests/test_wheel_smoke.py
  git commit -m "docs: document and package external result imports"
  ```

---

### Task 19: Full validation, independent reviews, push, and unmerged PR

**Files:**

- Modify only files required to resolve confirmed review findings.
- Create outside Git: Stage 1 JSON reports, temporary workspaces/builds, and PR
  body.

- [ ] **Step 1: Review the complete diff and repository cleanliness**

  ```bash
  git status --short
  git diff --check origin/choicebench...HEAD
  git diff --stat origin/choicebench...HEAD
  git diff --name-only origin/choicebench...HEAD
  ```

  Expected: only source, tests, documentation, and small synthetic test support;
  no imported runs, reports, historical data, caches, credentials, build output,
  or temporary files.

- [ ] **Step 2: Run all focused importer/compatibility/security tests**

  ```bash
  python -m pytest tests/importing tests/io/test_readers.py \
    tests/io/test_writers.py tests/infra/test_checkpoint.py \
    tests/test_condition_grid.py tests/test_publication_identity.py \
    tests/test_release_adversarial.py tests/scripts/test_evaluate_run.py \
    tests/scripts/test_reset_run.py -v
  ```

  Expected: all focused tests pass with zero failures.

- [ ] **Step 3: Run the full ChoiceBench suite**

  ```bash
  python -m pytest tests/ -v
  ```

  Expected: the existing 808 tests plus all new tests pass; record the exact
  final count and duration for the PR.

- [ ] **Step 4: Confirm the repository has no configured lint/type gate**

  ```bash
  rg -n "ruff|mypy|pyright|black|flake8|pylint" pyproject.toml \
    .github/workflows tests
  ```

  Expected: no configured command. Do not invent a new formatter/type-check gate
  in this PR; report that pytest is the repository's configured CI check.

- [ ] **Step 5: Build and check wheel/sdist outside the repository**

  ```bash
  IMPORT_BUILD_DIST="$(mktemp -d)"
  python -m build --outdir "$IMPORT_BUILD_DIST"
  python -m twine check "$IMPORT_BUILD_DIST"/*
  python -m pytest tests/test_wheel_smoke.py -v
  ```

  Expected: build exits 0; Twine reports both distributions `PASSED`; installed
  wheel and sdist generic import/evaluation smoke passes outside the repository.

- [ ] **Step 6: Record a read-only freeze inventory before Stage 1 validation**

  ```bash
  FREEZE_ROOT=/home/cotenthusiast/Projects/model-generalization/paper_data_freeze
  FREEZE_METADATA_BEFORE="$(mktemp)"
  FREEZE_CONTENT_BEFORE="$(mktemp)"
  find "$FREEZE_ROOT" -printf '%y\t%P\t%m\t%U\t%G\t%s\t%T@\t%l\n' \
    | sort > "$FREEZE_METADATA_BEFORE"
  find "$FREEZE_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$FREEZE_CONTENT_BEFORE"
  ```

  Expected: metadata, symlink targets, and content digests for the entire freeze
  are written outside both repositories. Do not execute any freeze script and
  do not change file permissions or metadata.

- [ ] **Step 7: Run the full Stage 1 strict dry run**

  ```bash
  STAGE1_DRY_REPORT_DIR="$(mktemp -d)"
  STAGE1_DRY_REPORT="$STAGE1_DRY_REPORT_DIR/stage1-dry-run.json"
  choicebench-import-results "$FREEZE_ROOT" \
    --profile stage1-paper-freeze \
    --run-id paper-stage1-validation \
    --dry-run --strict \
    --report "$STAGE1_DRY_REPORT"
  ```

  Validate the report with:

  ```bash
  jq -e '
    .profile.matrix.intended_cells == 100 and
    .profile.matrix.core_method_cells == 84 and
    .profile.matrix.local_pride_cells == 4 and
    .profile.matrix.ihs_cells == 12 and
    .profile.matrix.excluded_preserved_cells == 2 and
    .counts.evidence_status.complete == 58 and
    .counts.evidence_status.qualified == 5 and
    .counts.evidence_status.recoverable == 6 and
    .counts.evidence_status.partial == 4 and
    .counts.evidence_status.malformed == 29 and
    .profile.matrix.intended_evidence.complete == 57 and
    .profile.matrix.intended_evidence.qualified == 5 and
    .profile.matrix.intended_evidence.recoverable == 6 and
    .profile.matrix.intended_evidence.partial == 4 and
    .profile.matrix.intended_evidence.malformed == 28 and
    .profile.queues.approved_executable_question_cells == 138 and
    .profile.queues.held_nonexecutable_question_cells == 687 and
    .profile.queues.excluded_nonexecutable_question_cells == 3 and
    .profile.queues.forensic_question_cells == 776 and
    .profile.queues.classified_question_cells == 828 and
    .profile.queues.pairwise_disjoint == true and
    .profile.queues.approved_authoritative_executable == true and
    .profile.queues.held_executable == false and
    .profile.queues.excluded_executable == false and
    .profile.source_schema_count == 8 and
    .profile.offline_authority.condition_count == 6 and
    .profile.offline_authority.question_cell_count == 18 and
    .profile.offline_authority.inference_executable == false and
    .profile.datasets.arc.reference_kind == "independent_input_snapshot" and
    .profile.datasets.arc.trust_label == "checksum_verified_freeze_internal" and
    .profile.datasets.arc.selected_question_count == 1000 and
    .profile.datasets.arc.option_count_distribution["3"] == 3 and
    .profile.datasets.arc.option_count_distribution["4"] == 997 and
    .profile.datasets.mmlu.reference_kind == "independent_input_snapshot" and
    .profile.datasets.mmlu.trust_label == "checksum_verified_freeze_internal" and
    .profile.datasets.mmlu.selected_question_count == 1000 and
    .profile.datasets.mmlu.pre_dedup_row_count == 1003 and
    .profile.datasets.mmlu.post_dedup_row_count == 1000 and
    .profile.datasets.mmlu.duplicate_question_ids ==
      ["79686d32dfe155ea", "2f7aa3c7ebb98cfe", "74f7227e190200ac"] and
    (.profile.datasets.mmlu.pre_dedup_digest | length) == 64 and
    (.profile.datasets.mmlu.post_dedup_digest | length) == 64 and
    .profile.datasets.arc.publisher_authenticated == false and
    .profile.datasets.mmlu.publisher_authenticated == false and
    .checksums.all_referenced_sources_match == true
  ' "$STAGE1_DRY_REPORT"
  ```

  Expected: importer reports `validated`, writes no run, and every assertion is
  true. Held/excluded records remain non-executable. The profile performs the
  same sealed invariants through registered ChoiceBench code; it does not run
  `scripts/validate_stage1_freeze.py` or any other external executable content.

- [ ] **Step 8: Perform an isolated real Stage 1 import and evaluation**

  ```bash
  STAGE1_IMPORT_ROOT="$(mktemp -d)"
  STAGE1_IMPORT_REPORT_DIR="$(mktemp -d)"
  STAGE1_IMPORT_REPORT="$STAGE1_IMPORT_REPORT_DIR/stage1-import.json"
  choicebench-import-results "$FREEZE_ROOT" \
    --profile stage1-paper-freeze \
    --run-id paper-stage1-import \
    --strict \
    --output-root "$STAGE1_IMPORT_ROOT" \
    --report "$STAGE1_IMPORT_REPORT"
  CHOICEBENCH_HOME="$STAGE1_IMPORT_ROOT" \
    choicebench-evaluate --run-id paper-stage1-import
  STAGE1_EVAL_REPORT="$(find "$STAGE1_IMPORT_ROOT/reports" -maxdepth 1 \
    -type f -name 'paper-stage1-import_*_metrics.json' -print -quit)"
  jq -e '
    any(.realizations[];
      .selected == true and .evidence_status == "complete" and
      .scope_disposition == "included" and (.metrics | length) > 0 and
      (.prediction_origins | all(. == "external_historical_inference"))) and
    any(.realizations[];
      .selected == true and .evidence_status == "qualified" and
      .scope_disposition == "included" and (.qualifications | length) > 0 and
      (.metrics | length) > 0) and
    any(.realizations[];
      (.evidence_status == "partial" or .evidence_status == "malformed") and
      (.metrics | length) == 0) and
    any(.realizations[];
      .benchmark_name == "arc_challenge" and .selected == true and
      .option_count_distribution["3"] == 3 and (.metrics | length) > 0) and
    all(.realizations[];
      if (.scope_disposition == "excluded_from_paper_matrix" or
          .scope_disposition == "held" or
          .scope_disposition == "superseded" or
          .evidence_status == "failed")
      then (.metrics | length) == 0 else true end)
  ' "$STAGE1_EVAL_REPORT"
  ```

  Expected: a complete immutable run exists only below the temporary root;
  included complete and qualified realizations have metrics; incomplete/
  malformed/recoverable/excluded records are accounted without metrics; at
  least one evaluated ARC realization contains all three variable-option rows;
  origins remain external; no inference is invoked. Record experiment,
  realization, result, and evaluation IDs and the evaluation report path.

- [ ] **Step 9: Verify the freeze stayed unchanged**

  ```bash
  FREEZE_METADATA_AFTER="$(mktemp)"
  FREEZE_CONTENT_AFTER="$(mktemp)"
  find "$FREEZE_ROOT" -printf '%y\t%P\t%m\t%U\t%G\t%s\t%T@\t%l\n' \
    | sort > "$FREEZE_METADATA_AFTER"
  find "$FREEZE_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$FREEZE_CONTENT_AFTER"
  cmp "$FREEZE_METADATA_BEFORE" "$FREEZE_METADATA_AFTER"
  cmp "$FREEZE_CONTENT_BEFORE" "$FREEZE_CONTENT_AFTER"
  ```

  Expected: both `cmp` commands exit 0. The import root and reports remain
  outside Git.

- [ ] **Step 10: Run independent specification-compliance review**

  Give a fresh read-only reviewer the approved specification, implementation
  plan, full diff, and validation evidence. Require a line-by-line verdict on
  generic core, v2/v3 compatibility, provenance/identity, statuses, overlays,
  evaluation, Stage 1 profile, CLI, tests, docs, and boundaries. Resolve all
  confirmed Critical/Important findings in focused commits; rerun affected tests;
  return current HEAD/evidence to that reviewer until it gives an explicit
  approval on the current commit.

- [ ] **Step 11: Run independent code-quality review**

  A different fresh reviewer inspects cohesion, API/type consistency, error
  clarity, duplication, maintenance cost, performance/memory on Stage 1, and
  paper leakage. Resolve confirmed blockers, rerun focused/full tests, and obtain
  that reviewer's explicit approval of the resulting current commit.

- [ ] **Step 12: Run independent adversarial/security review**

  A different reviewer attacks YAML/CSV parsing, byte spans, checksum races,
  paths/symlinks, staging/no-replace, idempotence, secret handling, manifest/
  sidecar/state tampering, authorization escalation, base-run mutation, and
  malformed evidence normalization. Resolve blockers and rerun security plus
  full tests, then obtain that reviewer's explicit approval of current HEAD.

- [ ] **Step 13: Run fresh-context final review**

  Give a final reviewer only the user requirements, approved spec/plan, final
  diff, test/build/Twine/Stage 1 evidence, and earlier resolved-finding commits.
  Require a clear PR-ready/not-ready verdict. Do not proceed on Critical or
  Important findings. If it finds any, resolve them in focused commits, repeat
  affected/full verification, and send the new current HEAD back to the same
  fresh-context reviewer for a renewed verdict. Repeat until it explicitly says
  PR-ready for the exact commit that will be pushed.

- [ ] **Step 14: Repeat final verification after all review fixes**

  Confirm the specification, quality, security, and final reviewers all approved
  the current HEAD, then repeat Steps 1–9. If this verification causes any code,
  test, or documentation change, invalidate all four approvals and repeat Steps
  10–13 before proceeding. Expected: approvals and validation evidence refer to
  the exact same commit.

- [ ] **Step 15: Reconfirm remote target and branch ancestry**

  ```bash
  git fetch origin
  git remote show origin
  git merge-base --is-ancestor origin/choicebench HEAD
  git status --short --branch
  ```

  Expected: worktree clean; feature contains the synchronized integration base.
  Confirm the repository's actual integration/default target from remote state
  rather than assuming its name. If it differs from the branch used to start
  this work or ancestry is false, stop and report instead of rebasing/retargeting
  silently.

- [ ] **Step 16: Push the feature branch**

  ```bash
  git push -u origin feat/external-results-importer
  ```

  Expected: GitHub confirms the remote feature branch. Do not force-push.

- [ ] **Step 17: Create the unmerged pull request**

  Create `/tmp/choicebench-external-results-pr.md` with `apply_patch`, containing
  motivation; architecture; imported-versus-native semantics; generic importer;
  Stage 1 profile; overlay/authorization design; security; tests and exact final
  count; Stage 1 dry-run and real-import evidence; build/Twine; limitations;
  explicit no historical data/inference statement; base/feature branch; and
  exact commit hashes. Then run:

  ```bash
  gh pr create \
    --base <confirmed-integration-branch> \
    --head feat/external-results-importer \
    --title "feat: add provenance-preserving external results importer" \
    --body-file /tmp/choicebench-external-results-pr.md
  ```

  Expected: GitHub returns a PR number and URL. Verify with `gh pr view`. Leave
  it open; do not merge, tag, or release. If authentication prevents creation,
  preserve the pushed branch and report the exact GitHub CLI error without
  claiming a PR exists.

---

## Plan completion condition

Implementation is complete only when every task/checkbox is satisfied, all
independent reviews pass, the full validation evidence is current, the feature
branch is pushed, and GitHub confirms an open unmerged PR. Generated runs,
reports, historical data, caches, temporary build/workspaces, and the external
PR-body file remain intentionally excluded from Git.
