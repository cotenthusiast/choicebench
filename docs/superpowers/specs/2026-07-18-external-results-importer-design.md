# External Results Importer Design

**Date:** 2026-07-18
**Status:** Independently reviewed; awaiting implementation-planning approval
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
- in manifest v2, a condition exclusively owns
  `results/<condition_id>.csv` and its sidecar;
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
   snapshot. Imported evaluable rows must be joined to a declared expected
   snapshot with an explicit trust classification and checked field by field
   before they can become evaluable imported result artifacts.

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
3. **Expected-dataset loader** — resolves the declared question set and
   trust-qualified question/gold/option data, creates a ChoiceBench run snapshot,
   and binds its digest to the realization identity.
4. **Identity/realization builder** — preserves the existing scientific
   `condition_id` while assigning every imported, repaired, transformed, or
   native execution record a separate immutable `realization_id`.
5. **Row validator and normalizer** — joins by `question_id`, validates source
   evidence, and creates ChoiceBench-native-format rows only when the evidence
   is eligible for evaluation.
6. **Evidence store** — atomically archives only referenced row-level source
   bytes in a content-addressed store and reuses identical blobs.
7. **Lineage/overlay engine** — validates repair and offline-transformation
   overlays without modifying the base import.
8. **Import transaction** — locks a run, constructs and verifies the manifest,
   performs collision-safe writes, and produces a machine-readable report.
9. **Manifest-aware evaluation extensions** — account for imported evidence
   and compute metrics only for eligible conditions.
10. **Stage 1 profile** — translates sealed paper manifests and queue ledgers
   into generic specifications without adding paper knowledge to the core.
11. **Installed CLI** — `choicebench-import-results` with a profile switch,
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

### Structured result origin and derivation

A single scalar origin is insufficient because a repaired realization can
contain predictions from several producers. Every newly written manifest uses
a structured `result_origin` record with separate derivation and prediction
origin information:

```yaml
result_origin:
  derivation_origin: repair_overlay
  prediction_origins:
    - external_historical_inference
    - native_inference
  ordered_row_origin_digest: <sha256>
  lineage_component_origin_digest: <sha256>
```

`derivation_origin` describes how the realization was created:

- `native_execution`
- `external_import`
- `repair_overlay`
- `offline_transformation`

`prediction_origin` describes who or what produced each prediction:

- `native_inference`
- `external_historical_inference`
- `external_repair_inference`

Every evaluable row has a non-null `prediction_origin` and
`prediction_lineage_id`. The realization record and result sidecar contain the
sorted set/counts of constituent origins plus a digest of the ordered per-row
assignment. When more than one origin occurs, the per-row mapping is mandatory;
the importer never substitutes an uninformative `mixed` label.

An offline transformation is a derivation, not new model inference. A rematched
row retains the prediction origin of its underlying model response while its
lineage points to an `offline_transformation` component and its implementation
identity. Every lineage node (base evidence, repair evidence, authorization,
and transformation) records its own origin.

Missing `result_origin` may imply a homogeneous native realization only while
reading legacy manifest-v2 records. All newly written native and imported
manifest-v3 realizations record the structured field explicitly. A
ChoiceBench-native-format imported result artifact retains its external
prediction origins; the format and evaluator do not change who performed the
inference.

### Import state

`import_state` describes the import operation only:

- `validated` — validation succeeded in a dry-run/validate-only report;
- `imported` — the verified artifact transaction was committed;
- `failed` — validation or the transaction failed.

A failed transaction does not leave a partially published final run. It may
leave only a clearly marked unpublished staging directory after process or host
failure; readers never resolve staging paths. Failure details are recorded in
the import report. An already committed run may account for a source-level
failed condition through `evidence_status`; that is distinct from an importer
failure.

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

## Identity layers and audit provenance

The importer separates semantic condition identity, realization identity,
result artifact identity, evaluation identity, and audit-only provenance. These
layers are related but never interchangeable.

### Semantic condition identity

`condition_id` remains the scientific grid-cell identity already constructed by
ChoiceBench v0.2. Its canonical payload contains only scientifically meaningful
condition metadata:

- dataset artifact and exact selection identities, benchmark, and split;
- model identity, including the backend/provider and known consumed generation
  settings or revision;
- method identity, including the historical method name, effective parameters,
  preflight identity, and method implementation identity where recoverable;
- prompt/template identity;
- seed, calibration/preflight identity, and applicable protocol settings.

This matches the current model, method, prompt, dataset-selection, and condition
construction rather than defining an importer-only scientific identity. The
canonical payload has a full `condition_digest`; `condition_id` is its existing
filesystem-safe short form.

Source CSV bytes, CSV dialect/mapping, importer code, evidence status, scope,
authorization, overlay bytes, repair IDs, and lineage are not semantic
condition fields. Changing any of them alone leaves `condition_id` unchanged.
Changing the dataset selection, model, method, prompt, generation semantics,
seed, or protocol settings changes the semantic condition.

### Import/result realization identity

A `realization_id` identifies one immutable evidence/provenance record that
asserts or derives results for a semantic condition. Multiple historical,
repaired, transformed, or native realizations may share one `condition_id`.
The canonical realization payload includes:

- the full semantic `condition_digest`;
- stable import-specification digest;
- logical source records, source classifications, formats, exact source-byte
  SHA-256 values, and stable source provenance;
- expected-input snapshot and question-set digests;
- adapter/importer implementation identity, complete CSV dialect, source
  mapping, null/numeric/option/extra-field policies;
- validation-findings digest, evidence status, qualification/limitation/defect
  digests, and scope disposition;
- structured result origin and ordered prediction-origin assignment digest;
- parent realization/evidence/result digests;
- typed authorization digest;
- overlay source bytes, replacement IDs/reasons, and transformation
  input/pre-ownership-output/code digests.

The output digest in a realization is never the final ChoiceBench CSV digest.
It is either the checksum of a separately supplied precomputed overlay/output
source, or a canonical transformation payload produced before ChoiceBench row
ownership is injected. That canonical payload excludes the child
`realization_id`, result-artifact and experiment identities, output paths,
timestamps, and machine locations. The final CSV checksum appears only in the
post-manifest result-artifact identity, sidecar, and run state.

Lineage-component IDs included in the realization are likewise computed only
from their operation type, parent/source/evidence/authorization digests,
implementation identity, stable parameters, question ID, and pre-ownership
input/output digests. They never include the child realization, child result
artifact, or current experiment identity. The realization can therefore bind
the ordered row-to-lineage-component mapping without a self-reference.

`import_state` is not identity-bearing: a dry-run realization that moves from
`validated` to atomically committed `imported` keeps the same planned identity.
A failed transaction publishes no realization.

Changing source bytes, source checksum, CSV dialect, source mapping, stable
provenance, importer/adapter implementation, evidence classification, scope,
authorization, overlay, repaired IDs, prediction-origin assignment, or
transformation always changes `realization_id`. It changes `condition_id` only
when the scientific condition payload also changes.

A repaired or transformed output that implements the same intended dataset x
model x method x prompt protocol is a derived realization of the same semantic
condition. If the repair changes the model, prompt, method algorithm, generation
semantics, dataset selection, or protocol, it belongs to a new semantic
condition. Deterministic rematching that reconstructs the declared
`semantic_matching_v1` protocol from preserved Stage-1 responses shares the
base semantic condition; a different matcher definition would change the
method and condition identities.

### Manifest and experiment identity

Manifest schema v3 separates `semantic_conditions` from `realizations`.
Semantic-condition records contain no realization back-reference in their
condition identity; grouping is derived from the realization table. Realizations
reference a semantic `condition_id` and own fixed result/evidence
paths plus an expected result ownership/schema contract. The immutable manifest
does not contain the result-artifact ID or digest, which cannot be known until
the exact result bytes exist. New native runs normally have one realization per
condition; imported runs may retain multiple alternative or derived
realizations. Run state is keyed by `realization_id`, not by semantic condition.

The existing `experiment_id` remains the immutable manifest/run identity. Its
identity payload includes the semantic grid and the selected realization
records, so changing source bytes or lineage changes the experiment identity
without changing the shared scientific condition. A separate
`semantic_grid_digest` over datasets/models/methods/prompts/semantic conditions
supports cross-realization comparison. Legacy manifest-v2 records retain their
existing experiment and condition IDs and are normalized in memory as one
native realization per condition.

### Result artifact identity

An evaluable CSV has a separate `result_artifact_id` and full
`result_artifact_digest`. The digest binds:

- semantic condition and realization full digests;
- manifest experiment ID;
- exact CSV SHA-256, row count, ordered columns, and columns digest;
- row-ownership summary and ordered question-identity digest;
- structured prediction-origin counts and ordered assignment digest;
- evidence and lineage digests.

The result ID is calculated after the exact CSV bytes are produced. It is stored
in the integrity sidecar and run state, not inside the CSV or immutable manifest.
This is an explicit two-phase boundary: first the manifest/experiment fixes the
semantic conditions, realizations, output contracts, and safe paths; then the
atomic result publication computes and records the result-artifact identity.
The result-artifact digest may therefore bind the already fixed experiment ID
without an identity cycle. Manifest-v3 result paths are fixed by realization,
`results/<realization_id>.csv` and
`results/<realization_id>.artifact.json`; a second realization therefore cannot
collide with the shared semantic condition. Legacy v2 paths remain
`results/<condition_id>.csv`.

### Evaluation identity

Evaluation units are realizations grouped under semantic conditions. Alternative
realizations are never silently concatenated. The evaluation identity binds:

- experiment digest and explicit realization-selection policy;
- every accounted semantic condition and realization full digest;
- evidence/scope/qualification/limitation and stable lineage digests;
- applicable result artifact ID/digest and exact consumed CSV SHA-256;
- metric, parser/scorer/postprocessing, and evaluator implementation identities.

Ineligible evidence-only realizations remain identity-bound accounting units
with null result fields. Reports expose both
`conditions[condition_id].realization_ids` and realization-level metrics/status.
Changing source/mapping/provenance changes the realization and applicable
result/evaluation identities even when the semantic condition is unchanged.

Imported sources, datasets, models, methods, prompts, semantic conditions,
realizations, and results carry full SHA-256 digests in addition to short IDs.
Readers verify the full digest before resolving any imported artifact path.

### Audit-only provenance

The following are recorded for traceability but excluded from semantic
condition, realization, result-artifact, experiment, and evaluation identities:

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
- realization and structured-origin declarations;
- typed authorization declarations;
- optional repair or transformation overlays;
- metric selection;
- notes and provenance evidence;
- audit-only location bindings.

A source artifact declaration can express:

- user-selected source location;
- expected SHA-256;
- stable logical path/name;
- format and format version;
- the complete identity-bearing decoding and CSV-dialect declaration;
- source run, repository, and commit declarations;
- raw/canonical/derived/repaired/aggregate-only classification;
- exact or allowed source schema;
- source column mapping;
- null and numeric policies;
- extra-field policy;
- arbitrary safe notes and evidence.

An expected-dataset declaration also states its reference kind and trust basis,
as defined below, rather than representing every reference snapshot as equally
trusted.

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
references. Identical bytes referenced by multiple source declarations or
conditions within the same run reuse one evidence blob. A pre-existing blob is
accepted only after full byte digest and sidecar validation; divergent overwrite
is refused. Separate immutable runs archive their own content-addressed copy even
when bytes match; the initial design deliberately avoids a mutable global store
or cross-run hardlink trust boundary.

Evidence snapshots are written to a temporary file under the destination,
flushed, verified, and atomically renamed. The evidence index is also atomic and
self-digested. No raw repository, response-cache directory, checkpoint forest,
archive collection, or other unreferenced Stage 1 file is copied.

For malformed evidence, the byte-identical source snapshot is authoritative.
The CSV adapter retains each logical record's exact raw byte span, including
quoting, delimiters, line endings, and embedded newlines. The
realization-validation artifact records each affected question ID, the byte
offsets, the SHA-256 of those exact source-row bytes, and the validation failure. A
canonical source-row digest may be recorded as additional search/index data,
but never substitutes for the raw-row-byte digest. The importer does not coerce
a malformed prediction into an apparently valid normalized prediction.

## CSV adapter and extension fields

The first adapter supports CSV/tabular row sources. It reads source bytes as
data, never as code, preserves header order, and produces source cells without
allowing pandas' default NaN coercion to erase the distinction between an empty
cell and a literal `NaN` string. The import specification or profile declares
the accepted null representation and numeric parsing rules.

### Deterministic decoding and dialect

Parsing behavior is fully declared and identity-bearing in the realization
payload because it determines logical records and cell values. The initial CSV
adapter has these explicit defaults:

```yaml
csv:
  encoding: utf-8
  bom_policy: forbid
  decoding_errors: strict
  delimiter: ","
  quote_character: '"'
  escape_character: null
  double_quote: true
  line_terminators: [crlf, lf, cr]
  mixed_line_terminators: allow
  final_record_without_terminator: allow
  blank_record_policy: reject
  skip_initial_space: false
  header: first_logical_record
  strict_syntax: true
```

The initial adapter accepts only `encoding: utf-8`. `bom_policy` may instead be
explicitly set to `strip_utf8_bom`; stripping is limited to one leading UTF-8
BOM and the byte offsets still refer to the original source. Decoding always
fails closed on invalid byte sequences; a replacement-character or ignore
policy is not supported. Delimiter, quote, and non-null escape characters are
restricted in the first adapter to distinct single-byte ASCII characters.
`double_quote` controls whether two consecutive quote characters inside a
quoted field represent one literal quote.

The listed line terminators are the only recognized record separators and are
recognized only outside a quoted field. Their order is canonical and longest
first, so CRLF is one terminator rather than CR followed by LF. The mixed-line
policy, acceptance of an unterminated final logical record, and blank-record
policy are explicit. Raw spans include the record terminator when one is
present. A binary logical-record scanner uses the declared quote, escape, and
doubled-quote rules before text decoding, retains start/end byte offsets in the
original source, and treats newlines inside quoted fields as field bytes. It
then decodes and parses those exact spans under the same declaration.
Consequently the malformed-row raw-byte-span guarantee remains well-defined for
quoted records containing embedded CR, LF, or CRLF. The fixed UTF-8 encoding
declaration and every declared BOM, delimiter, quoting, escaping,
doubled-quote, terminator, mixed-terminator, final/blank-record, whitespace,
header, decoding-error, and strictness field are identity-bearing. A future
adapter version that supports another encoding necessarily produces a distinct
realization identity, though not a different semantic condition.

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

## Expected-dataset reference trust

The reusable core distinguishes two reference kinds:

- `independent_input_snapshot` — benchmark input artifacts and selection records
  that existed independently of the result rows being imported. The declaration
  binds their exact checksums, selection identity, transformation chain, and any
  independently known publisher revision/fingerprint. Validation may claim
  question, option, and gold consistency against this snapshot, while separately
  stating the authenticity limit of its provenance chain.
- `profile_derived_reference_snapshot` — a reference reconstructed only from
  result-side evidence because no independent input artifact is available. It
  requires at least two explicitly declared independent source groups when
  possible, exact cross-source agreement for question/gold/options, a complete
  derivation record, conflict refusal, and a stable derivation digest. It may
  support internal consistency and membership validation, but reports must not
  claim independent benchmark truth or upstream dataset authenticity.

The reference kind, source/checksum chain, declared trust level, cross-source
policy, and derivation digest are realization- and evaluation-identity-bearing.
They do not change semantic dataset identity unless the selected questions or
their semantic content changes. Profiles may not silently upgrade a
result-derived reference to an independent snapshot merely because several
result files agree.

## Row-level validation

Validation is by question identity, never by row count alone. All errors name
the condition, realization, source, field, and question ID where possible. No
imported result row is silently dropped, padded, deduplicated, reinterpreted, or
repaired. Any declared expected-dataset derivation is a separate, identity-bound
input transformation and cannot be used to excuse duplicate result rows.

The validator checks:

- required mapped fields and source schema;
- unique, non-null question IDs;
- exact expected membership for complete/qualified evidence;
- declared subset membership for partial/malformed/recoverable evidence;
- exact missing and unexpected IDs;
- duplicate IDs, including all duplicate locations;
- declared reference-snapshot gold answer equality, qualified by its trust kind;
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

The declared expected snapshot supplies the evaluative question text, choices,
correct option, and gold answer. Source-provided versions are compared with it
but never override it. The report qualifies the resulting validation claim by
the snapshot's declared reference kind and trust chain.

For `complete` and `qualified` evidence, every expected question must have one
valid evaluable prediction. A declared provider failure may occupy a source row,
but that realization is not complete until a valid authorized repair produces a
derived realization whose recomputed coverage supports that status.

For `partial`, `malformed`, or `recoverable` evidence, known defects are legal
only when the exact IDs and reasons are declared. Any additional defect fails
validation. These sources can be archived and accounted, but they do not
produce an evaluable imported result artifact.

## ChoiceBench-native-format imported result artifacts

Only realizations with:

- `import_state=imported`;
- `evidence_status=complete` or `qualified`; and
- `scope_disposition=included`

produce an evaluable imported result artifact under
`results/<realization_id>.csv`.

Its row ownership fields use the ordinary ChoiceBench experiment, condition,
dataset, model, method, prompt, benchmark, and split identities and add
`realization_id`, `prediction_origin`, and `prediction_lineage_id`. The result
sidecar additionally binds the full realization, result-artifact, source,
authorization, lineage, and ordered row-origin digests and explicitly records
the structured `result_origin`. It is a ChoiceBench-native-format imported
result artifact in format and validation only; it never claims ChoiceBench
executed the model.

Non-evaluable realizations retain their semantic-condition reference, evidence
references, exact validation artifacts, and lineage. They do not receive
placeholder result CSVs or empty metrics.

## Idempotence and collision safety

An import transaction holds the normal per-run process lock for validation of
existing state through final atomic publication.

For a new run, it reuses ChoiceBench's parent-level
`.locks/<run-id>.manifest.lock`, creates a uniquely named sibling staging
directory on the same filesystem as `runs/<run-id>`, and writes the complete
manifest, content-addressed evidence, validation artifacts, evaluable results,
sidecars, and final run state there. Every file and cross-reference is validated
from the staged tree; files and directories are flushed/fsynced where the
platform supports it. Publication is one atomic, no-replace directory rename
from the staged tree to the previously absent final run path while the lock is
held. The implementation never uses a replacement rename for a run directory.
If no safe atomic no-replace publication is available on the platform, it fails
closed rather than falling back to incremental final-path writes.

Validation failure removes only the importer-owned staging tree when safe. A
crash may leave an owner-marked staging tree, but it is not a run, is never
enumerated by readers/evaluators, and can be verified and garbage-collected by a
separate safe maintenance operation. Because the content-addressed evidence
store is inside that staging run, no externally published blob points at an
uncommitted run. Machine-readable reports written outside the run are published
separately and cannot make a failed run appear committed.

For an already existing final run, the importer performs verify-only behavior:
it validates the entire manifest, state, evidence, result, and sidecar graph. An
exact match is an idempotent no-op; any mismatch is a refusal. It never stages
over or replaces an existing run.

Repeating an import with the same stable specification projection, source
bytes, expected snapshot, mapping, status dimensions, origins, authorization,
and lineage yields the same semantic condition, realization, result-artifact,
experiment, and evaluation identities. If all existing manifest, evidence,
result, and sidecar bytes validate and match, the command reports an idempotent
no-op.

If a run ID, realization ID, result-artifact path, source-digest path, or report
identity already exists with different verified content, the importer refuses
the overwrite and names the differing identity-bearing sections. A shared
semantic `condition_id` is not itself a collision: it may legitimately group
several separately addressed realizations. The importer does not offer a silent
reset or destructive replacement path.

## Repair overlays and offline transformations

An overlay declaration contains:

- immutable base realization and semantic-condition full digests;
- mandatory base evidence and realization-validation-artifact checksums;
- a base evaluable-result checksum only when the base realization has one;
- overlay source and independently computed checksum;
- exact replacement question IDs;
- an authorization type and reference to a separately declared, immutable,
  checksum-verified authorization record;
- one replacement reason per question;
- output source classification;
- structured result origin and per-row prediction origins;
- transformation or repair implementation identity;
- stable lineage notes.

An overlay cannot declare or expand its own authorization set. Before overlay
validation, the importer independently validates the referenced authorization
artifact, its checksum, its scope to the base semantic condition and realization,
its exact authorized IDs and reasons, its authority, and the declared operation
type. The transformation/overlay specification may only reference authorization;
it cannot serve as the authority for its own IDs.

Two authorization types are distinct and non-interchangeable:

- `inference_repair` authorizes new model inference. For the Stage 1 profile it
  must resolve every condition/question pair to the checksum-verified
  `approved_rerun_queue.csv` record with `queue_disposition=approved`,
  `execution_authority=authoritative`, and `executable=true`. Held, excluded,
  forensic, absent, or false-executable records cannot authorize inference.
- `offline_transformation` authorizes deterministic processing of preserved
  evidence without model execution. Its immutable authorization source binds
  the exact condition/question IDs, reasons, authority, allowed transformation
  purpose, input-evidence digests, and expected-dataset digest. It explicitly
  records `inference_executable=false`. It neither requires nor fabricates an
  approved-rerun-queue entry.

An authorization record can permit only its named operation type. An
`offline_transformation` record cannot authorize inference, and an
`inference_repair` record does not implicitly authorize unrelated rematching or
postprocessing.

The overlay engine requires every replacement ID to be in that independently
validated authorization record, the expected dataset, and the base expected
set. It rejects duplicate IDs, unauthorized IDs, unexpected IDs, duplicate
overlay rows, multiple overlays that replace the same ID, inconsistent
gold/options/ownership, and conflicting overlay declarations. It validates
replacement rows with the same rules as base rows.

Applying an overlay never changes the base evidence snapshot, semantic
condition, base realization, or optional base result. It creates a derived
realization and result artifact with new identities and complete base ->
authorization -> overlay/transformation -> resulting-artifact lineage. It keeps
the shared semantic condition when the scientific protocol is unchanged. The
specification declares the expected derived evidence status; the importer
recomputes it from validated coverage and remaining defects and refuses a
mismatch. A declaration of complete or qualified succeeds only if all remaining
defects are resolved.

For a repair that combines historical base rows with newly generated rows, the
derived realization records `derivation_origin=repair_overlay`, retains
`external_historical_inference` on unchanged rows, assigns `native_inference` or
`external_repair_inference` to each replacement row as applicable, and records
the full constituent set/counts and ordered mapping. It never erases component
origins or reduces them to `mixed`.

Offline transformations use the same derived-artifact mechanism. They record
input and pre-ownership output checksums plus transformation code identity under
the non-circular rules above. The final ChoiceBench CSV checksum is added only
after the realization and experiment are fixed. A specification cannot ask
ChoiceBench to import and execute arbitrary code. A transformation is either:

- a registered ChoiceBench implementation whose code identity is computed by
  ChoiceBench; or
- a precomputed external output whose producing code identity/digest is
  declared and whose output is independently validated.

Deterministic rematching of surviving Stage-1 responses is represented with
`derivation_origin=offline_transformation`, while the rematched row retains the
origin of the inference response being rematched. It is not new model inference.

### Stage 1 offline-transformation authority

The six recoverable ARC `semantic_matching_v1` cells are authorized separately
from the rerun queue. The Stage 1 profile derives and content-addresses an
immutable `offline_transformation` authorization record from the
checksum-covered recoverable entries in
`manifests/canonical_results_manifest.json` (cross-checked against its CSV form).
The record identifies that manifest as the user-designated Stage 1 authority and
names these exact cells:

- `cbp__gemini-2-5-flash__arc_challenge__semantic_matching_v1`
- `cbp__gpt-4-1-mini__arc_challenge__semantic_matching_v1`
- `cbp__llama-3-1-8b-instant__arc_challenge__semantic_matching_v1`
- `cbp__meta-llama-llama-3-1-8b-instruct__arc_challenge__semantic_matching_v1`
- `cbp__qwen-qwen2-5-7b-instruct-turbo__arc_challenge__semantic_matching_v1`
- `cbp__qwen-qwen2-5-7b-instruct__arc_challenge__semantic_matching_v1`

The profile's manifest translation binds each historical `cell_id` to its full
ChoiceBench semantic `condition_digest`; authorization validation checks both
identities. For each it authorizes exactly these three question IDs:

- `79e8c959bbeb74a0`
- `ad6b5d46ae54842c`
- `c30e75b011696a95`

Thus it binds exactly six conditions and 18 condition/question pairs, their
manifest reasons, base-source checksums, expected-snapshot digest, the allowed
semantic-rematching purpose, and `inference_executable=false`. The generated
generic transformation specification references this independent authorization
record and digest; it cannot add IDs. `arc_question_audit.csv` may corroborate
row-level evidence but, because it is not itself covered by the freeze checksum
ledger, it is not the authorization trust anchor.

## Manifest, reader, and evaluation behavior

Manifest v3 retains the existing models, methods, prompts, datasets, semantic
conditions, and experiment semantics, and adds an identity-bearing realization
table. Each realization references one semantic condition and carries the
source, validation, orthogonal provenance dimensions, structured origin,
authorization, evidence, lineage, and planned result contract/path. After an
evaluable artifact is atomically written, its sidecar and run-state entry—not
the immutable manifest—reference its separate result-artifact identity.
Audit-only provenance is stored in a separate non-identity section whose
exclusion is explicit and validated.

Manifest validation recomputes imported child full digests and verifies their
short IDs and fixed artifact paths. Result-side identity summaries and column
digests are compared with the CSV and manifest rather than merely stored.

The publication-grade reader:

- validates content-addressed evidence snapshots and realization validation
  artifacts;
- validates evaluable imported result artifacts through the normal sidecar and
  row-ownership path;
- refuses result artifacts for ineligible realizations;
- selects evaluation realizations explicitly and never concatenates alternative
  realizations that share a semantic condition;
- returns evaluable rows plus manifest accounting for all imported semantic
  conditions and realizations;
- preserves compatibility with legacy native v0.2 manifests.

Evaluation:

- computes configured metrics for explicitly selected included
  complete/qualified imported realizations;
- includes qualifications and limitations beside qualified metrics;
- accounts for partial, malformed, recoverable, failed, excluded, held, and
  superseded evidence without metrics;
- reports semantic-condition and realization counts by each orthogonal dimension
  rather than one lossy status tally;
- never estimates metrics for missing, invalid, excluded, or held rows;
- continues to validate dataset snapshots, metric implementation identity, row
  ownership, and result checksums.

Evaluation identity includes only stable semantic provenance and lineage:

- experiment and semantic-condition full digests;
- explicit realization-selection policy and every accounted realization digest;
- evidence status and scope disposition;
- stable qualification/limitation digest;
- result-artifact, evidence, authorization, structured-origin, and lineage
  digests as applicable;
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

- stable import-specification, semantic-grid, condition, realization,
  result-artifact (when evaluable), and experiment IDs;
- audit-only input/output locations;
- counts by import state, evidence status, scope disposition, and origin;
- evaluable versus evidence-only realization and semantic-condition counts;
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

Inference-repair queue authority is explicit:

- `approved_rerun_queue.csv`: 138 question-cells, executable true;
- `held_or_declined_reruns.csv`: 687 Gemini ARC IHS question-cells,
  executable false and held;
- `paper_scope_excluded_reruns.csv`: three hosted PriDe question-cells,
  executable false and excluded;
- `rerun_queue.csv`: historical forensic evidence only and never execution
  authority.

These queue classifications authorize or decline new model inference only; they
do not authorize offline transformations. The profile imports no repair output
and executes none of these queues. It records the typed authorization boundaries
needed for future overlays and the separate offline-transformation authority
defined above.

### Stage 1 expected-dataset trust anchor

Stage 1 does not need to reconstruct expected questions from result CSVs. Its
trust anchor is the class of pre-inference benchmark-input and split artifacts
under `raw/local_model_generalization/data/`, all bound by
`checksums/checksums.sha256`:

- ARC raw/normalized inputs:
  `data/raw/arc_challenge_raw.csv` and
  `data/processed/arc_challenge_normalized.csv`;
- ARC selection and metadata:
  `data/splits/arc_challenge/robustness_ids.json` and
  `robustness_metadata.json`;
- MMLU raw/normalized inputs:
  `data/raw/mmlu_raw.csv` and `data/processed/mmlu_normalized.csv`;
- MMLU selection and metadata:
  `data/splits/benchmark/robustness_ids.json` and
  `robustness_metadata.json`.

The profile classifies these as `independent_input_snapshot` because they are
benchmark inputs independent of the result rows, with the more precise trust
label `checksum_verified_freeze_internal`. It verifies the freeze checksum
ledger against the opened bytes, parses the raw/normalized data under an
explicit adapter declaration, verifies the split-selection and metadata
digests, applies the declared selection derivation, constructs the ChoiceBench
snapshot, and then cross-checks result-side question, gold, and option evidence.

The ARC normalized input resolves its 1,000 unique selected IDs directly and
retains 997 four-option and three three-option rows. The MMLU normalized input
contains 1,003 rows for its 1,000 unique selected IDs because each of
`79686d32dfe155ea`, `2f7aa3c7ebb98cfe`, and `74f7227e190200ac` occurs twice.
This is part of the frozen input evidence, not silently invalidated or ignored.
For MMLU the profile reproduces the archived split-construction rule from
`raw/local_model_generalization/scripts/prepare_data.py`: preserve source order
and keep the first occurrence under
`drop_duplicates(subset="question_id", keep="first")`. Before applying it, the
profile requires every duplicate occurrence to agree exactly on all parsed
fields; any conflict fails closed. The registered profile implementation
identity, ordered pre-dedup row digest, exact duplicate-ID/row digests,
first-occurrence policy, and ordered post-dedup digest are identity-bearing.
After that declared derivation, every selected MMLU ID resolves exactly once and
all 1,000 selected rows have four options.

This chain establishes internal freeze consistency and independence from result
CSVs. It does not establish upstream publisher authenticity: the freeze does not
record a verifiable Hugging Face revision/commit/fingerprint or publisher-signed
checksum for these bytes. The profile therefore preserves unknown upstream
revision fields, reports that limitation, and never labels the snapshot as
publisher-authenticated. Byte-identical copies elsewhere in the freeze and
cross-result agreement are corroboration only, not the trust basis. If these
input artifacts were absent, the profile would have to use the weaker
`profile_derived_reference_snapshot` rules and correspondingly limited claims.

The generated generic specifications contain the exact reference kind, trust
label, source and selection checksums, transformation/selection derivation,
expected question sets, and snapshot digests. The reusable core receives no
paper-specific inference rule.

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
39. New manifests always write explicit structured `result_origin`.
40. Legacy manifests alone may infer native origin.
41. Content-addressed evidence deduplication and divergent-blob refusal.
42. Explicit absolute source/output roots are accepted while spec-controlled
    output and traversal are rejected.
43. Malformed row evidence remains byte/digest exact and non-evaluable.
44. Repair overlays on evidence-only bases do not require a nonexistent base
    result checksum.
45. Overlay-declared authorization cannot self-authorize replacement IDs.
46. Declared derived evidence status must equal the recomputed status.
47. Source bytes/mapping/provenance change realization and result/evaluation
    identities without changing an otherwise identical semantic condition.
48. Scientifically meaningful model/dataset/method/prompt changes do change the
    semantic condition.
49. Base, repaired, and transformed realizations share a semantic condition
    when protocol semantics are unchanged, and alternatives are never
    concatenated for evaluation.
50. Result-artifact identity changes when exact result bytes change.
51. A repaired artifact records constituent origins and the exact per-row origin
    mapping; no bare `mixed` origin is accepted.
52. Offline transformation preserves underlying prediction origin while
    recording a separate derivation origin and component lineage.
53. `inference_repair` requires an exact approved executable queue binding and
    cannot use held, excluded, forensic, or offline authority.
54. `offline_transformation` requires its separate checksum-verified authority,
    cannot self-authorize, and does not require executable inference authority.
55. The Stage 1 offline authority contains exactly six semantic-matching cells,
    18 condition/question pairs, and the three declared ARC IDs per cell.
56. Independent-input and profile-derived dataset references retain distinct
    trust labels, derivations, and validation claims.
57. The Stage 1 profile anchors expected data to the checksum-verified frozen
    benchmark inputs/splits and reports the unknown upstream revision.
58. Every decoding/dialect field participates in realization identity.
59. UTF-8 BOM forbid/strip behavior and strict decoding-error refusal.
60. Delimiter, quote, escape, doubled-quote, whitespace, header, and strict CSV
    behavior, including invalid-declaration rejection.
61. CRLF, LF, CR, and mixed-line policies, with exact raw byte spans for quoted
    records containing embedded newlines.
62. The Stage 1 MMLU snapshot verifies the three exact duplicate pairs, rejects
    conflicting duplicates, applies stable first-occurrence deduplication, and
    identity-binds both pre- and post-dedup ordered row digests.
63. Manifest, realization, lineage-component, transformation-output, and
    result-artifact identities can be computed in order with no self-reference.
64. Failure/crash before the no-replace staging rename leaves no published final
    run, and readers ignore owner-marked staging directories.
65. Existing final runs are verify-only: exact graphs no-op and any divergent
    graph is refused without staging over the run.

Existing native-run tests remain unchanged or receive compatibility coverage.
The full suite must continue to pass.

## Stage 1 end-to-end validation

After synthetic tests pass:

1. Run the Stage 1 profile in dry-run mode across the entire intended matrix.
2. Confirm 100 intended and two excluded preserved cells.
3. Confirm the exact evidence-status and queue counts listed above.
4. Confirm held and excluded work stays non-executable.
5. Confirm every referenced canonical source checksum.
6. Verify both frozen benchmark-input/split chains, their trust labels, all
   2,000 selected IDs, and the three variable-option ARC rows.
7. Verify the separate offline-transformation authorization has exactly six
   conditions and 18 authorized condition/question pairs and grants no inference
   execution.
8. Perform a real import into a temporary absolute `CHOICEBENCH_HOME` outside
   both repositories.
9. Verify the freeze has no filesystem changes.
10. Use native ChoiceBench reading/evaluation to demonstrate:
   - one complete imported condition;
   - one qualified imported condition;
   - one incomplete or malformed evidence-only condition;
   - one variable-option ARC condition.
11. Write the end-to-end report outside the repository.

Generated imports, reports, build artifacts, temporary workspaces, and
historical evidence remain uncommitted.

## Documentation and packaging

User documentation will explain:

- what external import means and why it is not native inference;
- supported CSV format and extension-field preservation;
- import schema fields and examples;
- generic, dry-run, strict, output-root, and Stage 1 profile usage;
- deterministic identity versus audit provenance;
- semantic condition, realization, result-artifact, and evaluation identity;
- checksum and idempotence behavior;
- orthogonal status dimensions and evaluation eligibility;
- mixed prediction origins, repair lineage, and separately authorized offline
  transformations;
- expected-dataset trust kinds and deterministic CSV decoding/dialect behavior;
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
ChoiceBench-native-format imported result artifacts; semantic condition,
realization, result-artifact, and evaluation identities have the specified
separation; every artifact retains structured derivation and exact constituent
prediction origins; non-evaluable evidence remains preserved and honestly
accounted; deterministic identities exclude machine-local audit data; repair
and offline-transformation lineage uses the correct immutable typed authority;
dataset trust and CSV parsing behavior are explicit; the Stage 1 profile
reproduces all sealed counts without modifying the freeze; native
reading/evaluation and installed-package workflows work outside the repository;
and the feature is submitted on its isolated branch as an unmerged pull request.
