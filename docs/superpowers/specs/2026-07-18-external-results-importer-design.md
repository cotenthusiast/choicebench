# External Results Importer Design

**Date:** 2026-07-18
**Status:** Approved design; implementation not started
**Target:** ChoiceBench after v0.2.0
**Feature branch:** `feat/external-results-importer`
**Base:** `origin/choicebench` at `a01d85b1297bd7b557e0d2a16ad51c8c1c38e866`

## Purpose

ChoiceBench needs a first-class way to import row-level experiment results that
were generated outside ChoiceBench. The imported data must participate in
ChoiceBench's deterministic identity, provenance, manifest, result ownership,
integrity, status accounting, reading, and evaluation systems without implying
that ChoiceBench performed the original model inference.

The first real consumer is the immutable Stage 1 paper-data freeze at
`/home/cotenthusiast/Projects/model-generalization/paper_data_freeze`. The core
design is intentionally paper-agnostic. A thin profile translates the Stage 1
manifests into the same generic import specifications accepted for any other
external source.

This stage imports and validates existing row-level evidence. It does not run a
model, execute repair queues, alter historical predictions, regenerate the
freeze, or copy the 3.4 GiB raw/cache evidence forest.

## Existing architecture and design constraints

ChoiceBench v0.2 is manifest-first:

- canonical JSON plus SHA-256 supplies deterministic identities;
- dataset artifacts and selected rows have verified content identities;
- models, methods, prompts, conditions, and experiments have distinct
  identities;
- a condition exclusively owns `results/<condition_id>.csv` and its sidecar;
- immutable manifests bind the condition grid and artifact paths;
- run state accounts for operational completion, gating, and failure;
- publication-grade readers revalidate manifests, snapshots, result sidecars,
  ownership fields, and question coverage;
- evaluation verifies metric implementation identity and binds consumed result
  bytes and condition accounting into the evaluation identity;
- `CHOICEBENCH_HOME`, or the current directory when it is unset, is the
  writable workspace;
- installed commands are strict `argparse` entry points named
  `choicebench-*`.

The importer extends these systems. It must not use the legacy filename-based
result aggregator, fabricate an inference configuration, or route rows through
a fake model backend.

Two existing boundaries require importer-specific hardening:

1. The generic result writer can overwrite a same-condition file after only
   verifying its sidecar. Import transactions must instead perform a full
   identical-content no-op or refuse divergent content.
2. The existing reader validates result question IDs but does not compare
   result-side question text, options, and gold labels with the archived input
   snapshot. Imported evaluable rows must be joined to a trusted snapshot and
   checked field by field before they can become evaluable imported result
   artifacts.

## Chosen architecture

The importer creates ordinary ChoiceBench run directories and manifests. It
adds explicit import records, evidence artifacts, provenance dimensions, and
lineage while reusing existing identity, locking, atomic-write, manifest,
sidecar, reader, and evaluator facilities.

The main components are:

1. **Strict import schema** — dataclass-based YAML loading with the same
   unknown-field rejection and primitive validation style as experiment
   configuration.
2. **Source-adapter protocol** — an internal boundary that returns a stable
   tabular representation plus exact source-column information. CSV is the
   first implementation.
3. **Expected-dataset loader** — resolves the declared question set and trusted
   question/gold/option data, creates a ChoiceBench run snapshot, and binds its
   digest to the import identity.
4. **Row validator and normalizer** — joins by `question_id`, validates source
   evidence, and creates ChoiceBench-native-format rows only when the evidence
   is eligible for evaluation.
5. **Evidence store** — atomically archives only referenced row-level source
   bytes in a content-addressed store and reuses identical blobs.
6. **Lineage/overlay engine** — validates repair and offline-transformation
   overlays without modifying the base import.
7. **Import transaction** — locks a run, constructs and verifies the manifest,
   performs collision-safe writes, and produces a machine-readable report.
8. **Manifest-aware evaluation extensions** — account for imported evidence
   and compute metrics only for eligible conditions.
9. **Stage 1 profile** — translates sealed paper manifests and queue ledgers
   into generic specifications without adding paper knowledge to the core.
10. **Installed CLI** — `choicebench-import-results` with a profile switch,
    dry-run mode, strict validation, output-root selection, and reports.

Likely module boundaries are:

```text
src/choicebench/importing/
  schema.py
  adapters.py
  validation.py
  evidence.py
  lineage.py
  engine.py
  profiles/stage1_paper_freeze.py
src/choicebench/cli/import_results.py
```

Exact filenames may change during planning if existing module cohesion calls
for a smaller layout. The adapter/profile boundary and the absence of
paper-specific branches in the reusable engine are mandatory.

## Provenance dimensions

Import outcome, evidence quality, paper/execution scope, and inference origin
are orthogonal. No single status field may collapse them.

### Result origin

Every newly written manifest records exactly one `result_origin`:

- `native_inference`
- `external_historical_inference`
- `external_repair_inference`
- `offline_transformation`

Missing `result_origin` may imply `native_inference` only while reading legacy
manifest schema versions. New native and imported manifests write the field
explicitly. A ChoiceBench-native-format imported result artifact retains an
external origin; the format and evaluator do not change who performed the
inference.

### Import state

`import_state` describes the import operation only:

- `validated` — validation succeeded in a dry-run/validate-only report;
- `imported` — the verified artifact transaction was committed;
- `failed` — validation or the transaction failed.

A failed transaction does not leave a partially published run. Failure details
are recorded in the import report. An already committed run may account for a
source-level failed condition through `evidence_status`; that is distinct from
an importer failure.

### Evidence status

`evidence_status` describes scientific usability:

- `complete`
- `qualified`
- `partial`
- `malformed`
- `recoverable`
- `failed`

Qualifications, limitations, damaged IDs, recoverable IDs, and failure reasons
are explicit structured evidence. The existence or row count of a CSV never
upgrades evidence status.

### Scope disposition

`scope_disposition` describes inclusion and preservation policy:

- `included`
- `excluded_from_paper_matrix`
- `held`
- `superseded`

Excluded, held, failed, and superseded entries may reference genuine preserved
source evidence and lineage. They receive no new evaluable imported result
artifact or metric, but their evidence is not erased.

### Executability

`executable` is an explicit boolean wherever a repair/work authorization is
represented. It is not evaluation eligibility and the importer never interprets
it as permission to run model inference. In the Stage 1 profile, historical
condition import records are non-executing, while queue records reproduce the
authoritative true/false execution classification for later stages.

The evaluator's eligibility predicate is based on successful import,
`evidence_status in {complete, qualified}`, and an included scope disposition.
It does not use `executable`.

## Stable identity and audit provenance

The importer separates identity-bearing semantic provenance from audit-only
provenance.

### Identity-bearing provenance

The deterministic import identity includes:

- import schema and canonicalization versions;
- source format and exact source-byte SHA-256;
- a stable logical source identifier, when declared;
- source run identifier, repository, and commit declarations when known;
- source classification: `raw`, `canonical`, `derived`, `repaired`, or
  `aggregate_only`;
- dataset identity, split, expected question-set digest, snapshot revision or
  fingerprint, and exact run-snapshot content digest;
- model identity, backend/provider, and explicitly known revision;
- method identity, including historical identities that must not be merged;
- prompt/template identity and generation parameters when known;
- source column mapping, parsing policy, null policy, option mapping, and extra
  field policy;
- evidence status, qualifications, limitations, and stable per-row failure
  evidence;
- scope disposition and applicable execution authorization;
- repair authorizations, overlay bytes, replacement IDs/reasons, and stable
  lineage;
- importer or transformation implementation identity;
- the stable projection of the generic import specification.

Changing source bytes, an expected checksum, scientific mapping, identity
metadata, expected dataset, overlay, repaired IDs, or transformation identity
creates a different full import digest and therefore a different condition or
experiment identity.

Imported source, dataset, model, method, prompt, and condition records carry
full SHA-256 digests in addition to short filesystem-safe IDs. Readers verify
the full digest before resolving an imported condition's artifact path.

### Audit-only provenance

The following are recorded for traceability but excluded from import,
condition, experiment, and evaluation identities:

- import timestamp;
- the machine-local absolute source location used during this invocation;
- an absolute `CHOICEBENCH_HOME` or CLI output root;
- temporary paths;
- hostname and other machine-local execution details;
- machine-local import-report location.

Stage 1's historical original path is preserved as audit provenance. Its
freeze-relative canonical/raw path is stable logical provenance. The absolute
path used to locate the freeze on this machine is audit-only.

An import-specification digest is computed from a documented stable projection
that excludes operational locations and audit fields. Moving byte-identical
sources or the specification between machines does not change identity.

Unknown provenance stays explicit. An unknown field contains a null value and a
reason such as `not recorded by producer`; it is never replaced with an
inferred request ID, time, token count, prompt, response, provider field,
commit, seed, revision, or configuration value.

## Generic import specification

The generic schema follows existing strict ChoiceBench configuration patterns.
It contains these logical sections:

- schema/protocol version;
- stable import name and optional source-run declaration;
- one or more source artifacts;
- expected dataset declarations;
- model declarations;
- method declarations;
- prompt/template declarations;
- condition declarations;
- optional repair or transformation overlays;
- metric selection;
- notes and provenance evidence;
- audit-only location bindings.

A source artifact declaration can express:

- user-selected source location;
- expected SHA-256;
- stable logical path/name;
- format and format version;
- source run, repository, and commit declarations;
- raw/canonical/derived/repaired/aggregate-only classification;
- exact or allowed source schema;
- source column mapping;
- null and numeric policies;
- extra-field policy;
- arbitrary safe notes and evidence.

A condition declaration can express:

- source and expected-dataset references;
- model/backend/provider identity;
- benchmark and split identity;
- snapshot revision/fingerprint;
- historical method identity;
- prompt/template identity;
- known generation parameters;
- the four provenance/status dimensions and `executable` where relevant;
- expected question set;
- qualifications, limitations, and exact damaged/recoverable IDs;
- optional overlay references.

Credential-named keys are refused by the same identity canonicalization rules
used elsewhere in ChoiceBench. Arbitrary executable Python objects, pickle, and
source-controlled dynamic import targets are not supported.

## Path and source safety

Absolute paths are not inherently unsafe. These explicitly user-selected paths
are valid after canonicalization and containment/symlink checks:

- an absolute `CHOICEBENCH_HOME`;
- an absolute CLI-selected output root;
- an explicitly declared read-only external source such as the Stage 1 freeze.

The trust boundary is who selects the output, not whether the path begins at the
filesystem root.

An untrusted import specification cannot redirect writes. Output root selection
comes only from `CHOICEBENCH_HOME` or an explicit CLI option. Specifications
contain source locations and logical artifact relationships, but never an
output destination. Every derived output path is a fixed relative path under
the selected root.

The importer:

- resolves and canonicalizes the selected roots before writing;
- rejects `..` traversal or any derived output escaping the output root;
- rejects an output root or run directory that resolves through unsafe
  symlinks;
- refuses source symlinks when their target/containment cannot be safely
  established;
- never follows a source path into a directory tree for recursive import;
- opens only explicitly referenced row-level artifacts;
- computes SHA-256 independently from the opened bytes;
- parses the same bytes that were hashed, avoiding a hash/parse time-of-check
  race;
- rejects a source whose bytes differ from the specification checksum;
- never executes artifact content;
- sanitizes untrusted exception text before logging or reporting it.

## Evidence store and deduplication

Every referenced row-level source is archived atomically in the run's import
evidence store. Storage is addressed by the actual source-byte SHA-256, for
example:

```text
artifacts/imports/evidence/sha256/ce/ced2c5...<full digest>.csv
```

The suffix comes from the validated adapter format, not an untrusted filename.
An integrity sidecar records the digest, size, format, and stable source
references. Identical bytes imported by multiple conditions or specifications
reuse one evidence blob. A pre-existing blob is accepted only after full byte
digest and sidecar validation; divergent overwrite is refused.

Evidence snapshots are written to a temporary file under the destination,
flushed, verified, and atomically renamed. The evidence index is also atomic and
self-digested. No raw repository, response-cache directory, checkpoint forest,
archive collection, or other unreferenced Stage 1 file is copied.

For malformed evidence, the byte-identical source snapshot is authoritative.
The CSV adapter retains each logical record's exact raw byte span, including
quoting, delimiters, line endings, and embedded newlines. The condition
validation artifact records each affected question ID, the byte offsets, the
SHA-256 of those exact source-row bytes, and the validation failure. A
canonical source-row digest may be recorded as additional search/index data,
but never substitutes for the raw-row-byte digest. The importer does not coerce
a malformed prediction into an apparently valid normalized prediction.

## CSV adapter and extension fields

The first adapter supports CSV/tabular row sources. It reads source bytes as
data, never as code, preserves header order, and produces source cells without
allowing pandas' default NaN coercion to erase the distinction between an empty
cell and a literal `NaN` string. The import specification or profile declares
the accepted null representation and numeric parsing rules.

Variable-option questions are supported through either:

- an explicitly mapped structured choices column; or
- an ordered set/pattern of option columns whose present values determine the
  row's option count.

The adapter never assumes four choices. A three-option ARC row remains a
three-option row; an empty trailing option is not converted into a phantom
choice.

Every source column has one disposition:

1. mapped to a ChoiceBench common field;
2. preserved in `external_fields_json` under a source namespace; or
3. explicitly ignored with a documented reason.

Normal mode preserves safe unmapped fields and reports them. Strict mode
requires every column to be declared and rejects extras. Neither mode silently
drops a source column. Credential-named or secret-bearing metadata is rejected
in both modes.

`external_fields_json` preserves the original source field names, string/null
values, and method-specific diagnostics without forcing unrelated methods into
a lossy shared schema. This covers Stage-1/Stage-2 responses, matching output,
option scores and parse flags, permutations, votes, provider finish reasons,
failure counts, historical parse data, IHS per-option fields, and other safe
diagnostics.

Historical method identity is taken from the specification/profile, not renamed
from the source filename or simplified to a current built-in method. The Stage
1 profile preserves at least:

- `semantic_matching_v1`
- `text_extraction`
- `two_stage_v1`
- `two_stage_v2`
- `two_stage_v3`
- `independent_hypothesis`
- `cyclic_generation_majority`
- `PriDe`

Source-row aliases such as `cyclic`, `twostage_semantic_match`, and
`two_prompt` are validation evidence interpreted only through explicit profile
mapping.

## Row-level validation

Validation is by question identity, never by row count alone. All errors name
the condition, source, field, and question ID where possible. No row is silently
dropped, padded, deduplicated, reinterpreted, or repaired.

The validator checks:

- required mapped fields and source schema;
- unique, non-null question IDs;
- exact expected membership for complete/qualified evidence;
- declared subset membership for partial/malformed/recoverable evidence;
- exact missing and unexpected IDs;
- duplicate IDs, including all duplicate locations;
- trusted gold answer equality;
- option count and ordered option identity/text where available;
- correct-option mapping into the actual variable-size option set;
- parsed prediction validity or an exact declared damaged/failure record;
- correctness consistency when recomputable;
- method-specific numeric fields for malformed values, NaN, and infinity;
- row ownership by condition;
- source-row model, benchmark, method, and split consistency when those fields
  are present;
- evidence status against observed coverage and declared damaged rows;
- all source/checksum/specification digests;
- unknown/extra column policy;
- path and symlink safety.

The trusted expected snapshot supplies the evaluative question text, choices,
correct option, and gold answer. Source-provided versions are compared with it
but never override it.

For `complete` and `qualified` evidence, every expected question must have one
valid evaluable prediction. A declared provider failure may occupy a source row,
but that condition is not complete until a valid authorized repair produces a
derived condition.

For `partial`, `malformed`, or `recoverable` evidence, known defects are legal
only when the exact IDs and reasons are declared. Any additional defect fails
validation. These sources can be archived and accounted, but they do not
produce an evaluable imported result artifact.

## ChoiceBench-native-format imported result artifacts

Only imported conditions with:

- `import_state=imported`;
- `evidence_status=complete` or `qualified`; and
- `scope_disposition=included`

produce an evaluable imported result artifact under
`results/<condition_id>.csv`.

Its row ownership fields use the ordinary ChoiceBench experiment, condition,
dataset, model, method, prompt, benchmark, and split identities. Its result
sidecar additionally binds the full import/source/lineage digests and explicitly
states the external `result_origin`. The artifact is ChoiceBench-native in
format and validation only; it never claims ChoiceBench executed the model.

Non-evaluable conditions retain condition records, evidence references, exact
validation artifacts, and lineage. They do not receive placeholder result CSVs
or empty metrics.

## Idempotence and collision safety

An import transaction holds the normal per-run process lock for validation of
existing state through final atomic publication.

Repeating an import with the same stable specification projection, source
bytes, expected snapshot, mapping, status dimensions, and lineage yields the
same full identities. If all existing manifest, evidence, result, and sidecar
bytes validate and match, the command reports an idempotent no-op.

If the run ID, condition ID, source digest path, or report identity already
exists with different verified content, the importer refuses the overwrite and
names the differing identity-bearing sections. The importer does not offer a
silent reset or destructive replacement path.

## Repair overlays and offline transformations

An overlay declaration contains:

- immutable base import/condition full digest;
- mandatory base evidence and condition-validation-artifact checksums;
- a base evaluable-result checksum only when the base condition has one;
- overlay source and independently computed checksum;
- exact replacement question IDs;
- a reference to a separately declared, immutable repair-authorization record;
- one replacement reason per question;
- output source classification;
- result origin;
- transformation or repair implementation identity;
- stable lineage notes.

An overlay cannot declare or expand its own authorization set. Before overlay
validation, the importer independently validates the referenced authorization
artifact, its checksum, its scope to the base condition, its exact authorized
IDs and reasons, and its authority/executable fields. For the Stage 1 profile,
the authorization must resolve to a checksum-verified authoritative queue
record with `queue_disposition=approved`,
`execution_authority=authoritative`, and `executable=true`.

The overlay engine requires every replacement ID to be in that independently
validated authorization record, the expected dataset, and the base expected
set. It rejects duplicate IDs, unauthorized IDs, unexpected IDs, duplicate
overlay rows, multiple overlays that replace the same ID, inconsistent
gold/options/ownership, and conflicting overlay declarations. It validates
replacement rows with the same rules as base rows.

Applying an overlay never changes the base evidence snapshot, base condition,
or optional base result. It creates a derived condition with a new identity and
complete base -> authorization -> overlay -> resulting-artifact lineage. The
specification declares the expected derived evidence status; the importer
recomputes it from validated coverage and remaining defects and refuses a
mismatch. A declaration of complete or qualified succeeds only if all remaining
defects are resolved.

Offline transformations use the same derived-artifact mechanism. They record
input and output checksums plus transformation code identity. A specification
cannot ask ChoiceBench to import and execute arbitrary code. A transformation
is either:

- a registered ChoiceBench implementation whose code identity is computed by
  ChoiceBench; or
- a precomputed external output whose producing code identity/digest is
  declared and whose output is independently validated.

Deterministic rematching of surviving Stage-1 responses is represented as
`result_origin=offline_transformation`, not new model inference.

## Manifest, reader, and evaluation behavior

Imported manifests add an identity-bearing import section and condition-level
orthogonal provenance dimensions. Audit-only provenance is stored in a separate
non-identity section whose exclusion is explicit and validated.

Manifest validation recomputes imported child full digests and verifies their
short IDs and fixed artifact paths. Result-side identity summaries and column
digests are compared with the CSV and manifest rather than merely stored.

The publication-grade reader:

- validates content-addressed evidence snapshots and condition validation
  artifacts;
- validates evaluable imported result artifacts through the normal sidecar and
  row-ownership path;
- refuses result artifacts for ineligible conditions;
- returns evaluable rows plus manifest accounting for all imported conditions;
- preserves compatibility with legacy native v0.2 manifests.

Evaluation:

- computes configured metrics for included complete/qualified imported
  conditions;
- includes qualifications and limitations beside qualified metrics;
- accounts for partial, malformed, recoverable, failed, excluded, held, and
  superseded evidence without metrics;
- reports condition counts by each orthogonal dimension rather than one lossy
  status tally;
- never estimates metrics for missing, invalid, excluded, or held rows;
- continues to validate dataset snapshots, metric implementation identity, row
  ownership, and result checksums.

Evaluation identity includes only stable semantic provenance and lineage:

- experiment and imported condition full digests;
- evidence status and scope disposition;
- stable qualification/limitation digest;
- result/evidence/lineage digests as applicable;
- configured metric and postprocessing implementation identities;
- exact consumed evaluable result checksums.

It excludes timestamps, absolute paths, output roots, report locations,
hostnames, and temporary paths.

## CLI and reports

The installed command follows current project naming conventions:

```bash
choicebench-import-results SPEC.yaml --run-id external-study

choicebench-import-results SPEC.yaml \
  --run-id external-study \
  --dry-run \
  --strict \
  --report /tmp/external-study-validation.json

choicebench-import-results /path/to/paper_data_freeze \
  --profile stage1-paper-freeze \
  --run-id paper-stage1 \
  --dry-run
```

Required options/behavior include:

- real import;
- `--dry-run`/`--validate-only` aliases with no result/evidence writes;
- `--output-root` as an explicit trusted alternative to `CHOICEBENCH_HOME`;
- a required safe run ID;
- `--strict` source-column/schema validation;
- repeated explicit overlay arguments or overlays in the specification;
- human-readable summaries;
- optional machine-readable JSON report;
- mandatory independent checksum verification;
- sanitized, actionable errors.

`--output-root` is parsed before importing modules that resolve
`CHOICEBENCH_HOME`. An absolute user-selected output root is valid. The import
specification cannot set or override it.

The machine report includes:

- stable import/specification/experiment IDs;
- audit-only input/output locations;
- counts by import state, evidence status, scope disposition, and origin;
- evaluable versus evidence-only condition counts;
- row coverage and exact defect IDs;
- source/checksum/schema results;
- extra-column dispositions;
- overlay authorizations and lineage;
- Stage 1 queue classifications when the profile is used;
- whether the operation wrote artifacts or was an idempotent no-op.

## Stage 1 paper profile

The profile is selected through the same command:

```bash
choicebench-import-results FREEZE_ROOT \
  --profile stage1-paper-freeze \
  --run-id stage1-paper-import
```

It reads the current sealed manifests and validator-enforced invariants. It does
not regenerate them from `build_stage1_audit.py`, which predates the final
local-only PriDe amendment.

Authoritative inputs include, as relevant:

- `manifests/expected_matrix.csv`
- `manifests/canonical_results_manifest.json` and CSV cross-check
- `manifests/cell_status_matrix.csv`
- `manifests/frozen_artifact_index.csv`
- `manifests/approved_rerun_queue.csv`
- `manifests/held_or_declined_reruns.csv`
- `manifests/paper_scope_excluded_reruns.csv`
- `reports/canonical_freeze_report.md`
- `checksums/checksums.sha256`

The profile joins by explicit `cell_id` and `canonical_artifact_id`; it does not
infer scientific identity from filenames. Freeze-relative paths are resolved
beneath the explicitly selected absolute freeze root and checked against
traversal and symlinks.

The profile produces 102 preserved condition/evidence records:

- exactly 100 have `scope_disposition=included` and constitute the intended
  paper matrix;
- exactly two hosted-Qwen PriDe records have
  `scope_disposition=excluded_from_paper_matrix` and are never counted as
  intended paper cells.

This permits the hosted Qwen PriDe MMLU record to remain complete and excluded,
while the hosted Qwen PriDe ARC record remains malformed and excluded.

The intended-matrix evidence counts must be exactly:

- 57 complete;
- 5 qualified;
- 6 recoverable;
- 4 partial, mapped explicitly from Stage 1
  `incomplete_requires_inference`;
- 28 malformed.

Across all 102 preserved records, the two exclusions add one complete hosted
PriDe MMLU record and one malformed hosted PriDe ARC record. Thus the preserved
evidence totals are 58 complete, 5 qualified, 6 recoverable, 4 partial, and 29
malformed, while the scope totals remain 100 included and two excluded.

The profile verifies all source checksums used by its generated specifications.
It preserves the eight distinct historical CSV schemas and their method-specific
fields.

Queue authority is explicit:

- `approved_rerun_queue.csv`: 138 question-cells, executable true;
- `held_or_declined_reruns.csv`: 687 Gemini ARC IHS question-cells,
  executable false and held;
- `paper_scope_excluded_reruns.csv`: three hosted PriDe question-cells,
  executable false and excluded;
- `rerun_queue.csv`: historical forensic evidence only and never execution
  authority.

The profile imports no repair output and executes none of these queues. It only
records the authorization boundary needed for future overlays.

For expected datasets, the profile uses explicit manifest-selected frozen
canonical evidence, constructs benchmark/split question snapshots, and
cross-checks question/gold/option identity across conditions. The generated
generic specifications contain the resulting explicit expected question sets
and digests; the reusable core receives no paper-specific inference rule.

## Security boundaries

The importer maintains ChoiceBench's fail-closed posture:

- no pickle or executable-object deserialization;
- no execution of artifact content;
- no arbitrary dynamic transformation import from untrusted YAML;
- no source-provided checksum trust without independent hashing;
- no untrusted specification-controlled output path;
- no traversal or unsafe symlink resolution;
- no credential-named or secret-bearing persisted metadata;
- no divergent overwrite;
- no weakening of canonicalization, manifest, dataset, or evaluation checks;
- no recursive import of repositories, caches, archives, or directory forests;
- sanitized untrusted error text;
- process locking and atomic writes for every published artifact/index.

Checksums prove internal byte consistency, not producer authenticity. Declared
source repository/model/provider facts remain declared unless independently
resolved from supplied immutable evidence.

## Testing strategy

Implementation follows test-driven development with small synthetic fixtures.
No historical dataset or large artifact is copied into the repository.

Focused tests cover:

1. Valid generic CSV import.
2. Deterministic import identity.
3. Idempotent repeated import.
4. Divergent overwrite refusal.
5. Source checksum mismatch.
6. Source mutation after specification creation.
7. Duplicate question IDs.
8. Missing question IDs.
9. Unexpected question IDs.
10. Gold-answer mismatch.
11. Variable option counts.
12. Three-option ARC-style rows.
13. Null and NaN behavior.
14. Invalid parsed predictions.
15. Correctness mismatch.
16. Unknown and extra fields in normal/strict modes.
17. Method-specific extra-field preservation.
18. Complete imported evidence.
19. Qualified imported evidence.
20. Partial and malformed imported evidence.
21. Imported-condition evaluation.
22. Incomplete-condition accounting.
23. Base plus valid repair overlay.
24. Unauthorized repair-ID rejection.
25. Conflicting repair-overlay rejection.
26. Repair lineage and identity.
27. Offline-transformation lineage.
28. Paper-profile manifest translation.
29. Hosted PriDe exclusions are not intended paper cells.
30. Held Gemini ARC IHS remains non-executable.
31. Approved queue is not confused with the forensic queue.
32. CLI dry-run.
33. CLI real import.
34. Import and evaluation outside the repository.
35. Wheel/sdist installation retains importer functionality.
36. Security/adversarial cases matching existing release tests.
37. Orthogonal status combinations, including complete+excluded and
    malformed+excluded.
38. Audit paths/timestamps do not change import or evaluation identity.
39. New manifests always write explicit `result_origin`.
40. Legacy manifests alone may infer native origin.
41. Content-addressed evidence deduplication and divergent-blob refusal.
42. Explicit absolute source/output roots are accepted while spec-controlled
    output and traversal are rejected.
43. Malformed row evidence remains byte/digest exact and non-evaluable.
44. Repair overlays on evidence-only bases do not require a nonexistent base
    result checksum.
45. Overlay-declared authorization cannot self-authorize replacement IDs.
46. Declared derived evidence status must equal the recomputed status.

Existing native-run tests remain unchanged or receive compatibility coverage.
The full suite must continue to pass.

## Stage 1 end-to-end validation

After synthetic tests pass:

1. Run the Stage 1 profile in dry-run mode across the entire intended matrix.
2. Confirm 100 intended and two excluded preserved cells.
3. Confirm the exact evidence-status and queue counts listed above.
4. Confirm held and excluded work stays non-executable.
5. Confirm every referenced canonical source checksum.
6. Perform a real import into a temporary absolute `CHOICEBENCH_HOME` outside
   both repositories.
7. Verify the freeze has no filesystem changes.
8. Use native ChoiceBench reading/evaluation to demonstrate:
   - one complete imported condition;
   - one qualified imported condition;
   - one incomplete or malformed evidence-only condition;
   - one variable-option ARC condition.
9. Write the end-to-end report outside the repository.

Generated imports, reports, build artifacts, temporary workspaces, and
historical evidence remain uncommitted.

## Documentation and packaging

User documentation will explain:

- what external import means and why it is not native inference;
- supported CSV format and extension-field preservation;
- import schema fields and examples;
- generic, dry-run, strict, output-root, and Stage 1 profile usage;
- deterministic identity versus audit provenance;
- checksum and idempotence behavior;
- orthogonal status dimensions and evaluation eligibility;
- repair and offline-transformation lineage;
- security boundaries and limitations;
- how to implement a future adapter/profile.

A concise changelog entry will be added because the repository records notable
features there. The package version will not change.

Before the pull request, validation includes targeted tests, the full suite,
repository lint/type/static checks that actually exist, security/adversarial
tests, `python -m build`, `python -m twine check dist/*`, the existing isolated
wheel/sdist smoke test, the Stage 1 dry run, the isolated real import, and
representative native evaluation.

## Review workflow

The work follows these independent gates:

1. architecture reconstruction and design review;
2. independent written-specification review;
3. test-driven implementation by a fresh implementer where tasks are safely
   separable;
4. independent specification-compliance review;
5. independent code-quality review;
6. independent adversarial/security review;
7. fresh-context final review of the complete diff and validation evidence.

Reviewers are read-only and do not inherit implementer conclusions. Confirmed
blockers are fixed and relevant validation is rerun before advancing.

## Out of scope

This stage does not:

- modify Stage 1 source artifacts;
- run model inference;
- execute any repair, held, or excluded queue;
- start the final 2x2 experiment;
- rewrite the paper;
- alter historical predictions;
- implement unrelated evaluation modules;
- create a general data-platform abstraction;
- import response caches, checkpoints, or raw repository forests;
- commit imported outputs or historical data;
- change the package version;
- merge the eventual pull request or create a release/tag.

## Success criteria

The design is successful when a strict, reusable specification can import
external CSV rows into collision-safe ChoiceBench manifests and
ChoiceBench-native-format imported result artifacts; every artifact explicitly
retains external inference origin; non-evaluable evidence remains preserved and
honestly accounted; deterministic identities exclude machine-local audit data;
repair lineage is immutable and authorized; the Stage 1 profile reproduces all
sealed counts without modifying the freeze; native reading/evaluation and
installed-package workflows work outside the repository; and the feature is
submitted on its isolated branch as an unmerged pull request.
