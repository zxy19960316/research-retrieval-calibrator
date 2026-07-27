# M1-T01R Intent-to-Canonical-arXiv Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute inline with a red/green test cycle. Each checkbox is one verifiable action.

**Goal:** Close M1-T01 review gaps so a raw question deterministically becomes a provenance-complete `IntentDraft`, clarification result or frozen intent, and a canonical, unencoded arXiv `QueryPlan`.

**Architecture:** M0 historical evidence validation verifies only immutable M0 facts; the project-document validator owns every current phase-state transition. A single fake-provider seam validates structured output exactly once, assigns `LLM_FAKE` provenance to generated terms, then derives clarification or a frozen intent. Query construction is pure, fails closed for unmapped Chinese, and returns only canonical arXiv query text.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, Ruff, mypy, GitHub Actions.

## Global Constraints

- Update existing Draft PR #6 only; do not create a PR or start M1-T02.
- No HTTP, Atom parsing, cache, pagination, real arXiv request, database, platform API, CNKI, or real model call.
- Only `all`, `ti`, `abs`, `cat` field prefixes and `AND`, `OR`, `ANDNOT` operators are allowed in M1-T01 query text.
- A required unmapped Chinese term raises `UNMAPPED_RETRIEVAL_TERM`; ASCII stripping and `research` fallback are forbidden.
- Commit C contains code, tests, validator repair, and this plan. Commit D contains only refreshed evidence and `STATUS.md`.

---

### Task 1: Separate M0 historical validation from live phase counts

**Files:**
- Modify: `scripts/validate_phase.py`
- Modify: `tests/contract/test_evidence_report.py`
- Modify: `tests/contract/test_project_docs_validation.py`

**Interfaces:** `validate_report_file("M0")` accepts a historical M0 `COMPLETE 4/4` report regardless of legal later M1 count; `validate_project_docs.py` accepts M1 `READY 0/4`, `IN_PROGRESS 1/4..3/4`, and `COMPLETE 4/4` with M2 `READY`, while rejecting regressions and premature M2/M3 states.

- [ ] Write failing parametrized tests for legal M1 counts, M1 completion/M2 ready, M0 regression, wrong M0 total, premature M2, and premature M3.
- [ ] Run `py -3.12 -m pytest tests/contract/test_evidence_report.py tests/contract/test_project_docs_validation.py -q`; observe red before the validator repair.
- [ ] Change `_validate_status` in `scripts/validate_phase.py` to require only M0 `COMPLETE 4/4` plus immutable evidence ancestry/hashes; do not enumerate M1 states.
- [ ] Add the live-state matrix to `validate_status_text` in `scripts/validate_project_docs.py`.
- [ ] Re-run the tests and `py -3.12 scripts/validate_phase.py M0`; both must pass at live M1 `IN_PROGRESS 1/4`.

### Task 2: Implement raw-question preparation with strict fake provenance

**Files:**
- Modify: `app/adapters/llm.py`
- Modify: `app/core/intent.py`
- Modify: `app/models/planning.py`
- Modify: `app/adapters/__init__.py`
- Modify: `app/models/__init__.py`
- Modify: `tests/unit/test_intent_clarification.py`
- Modify: `tests/contract/test_m1_t01_contracts.py`

**Interfaces:**

```python
def prepare_intent(
    request: StructuredRequest,
    provider: LLMProvider,
) -> IntentPreparationResult: ...

class RetrievalTerm(BaseModel):
    original_text: str
    retrieval_text_en: str
    source: TermSource
    target_field: IntentField
```

- [ ] Write failing tests for English complete input, Chinese traceable candidates, exactly one provider invocation, invalid output, an incomplete draft returning questions, and a complete draft freezing.
- [ ] Run `py -3.12 -m pytest tests/unit/test_intent_clarification.py tests/contract/test_m1_t01_contracts.py -q`; observe red because the orchestration interface is absent.
- [ ] Implement `prepare_intent` with exactly one `provider.generate_structured(request)` call followed by `validate_llm_candidate`.
- [ ] Convert candidate terms to `RetrievalTerm` records with `TermSource.LLM_FAKE`, construct `IntentDraft` evidence, call gaps/questions, and freeze only when complete.
- [ ] For every candidate/schema failure raise `PlanningError("INVALID_INTENT_STRUCTURE")`; never guess or fill missing terms.
- [ ] Re-run the tests; all must pass with fake provenance intact.

### Task 3: Enforce mapped retrieval terms and canonical arXiv grammar

**Files:**
- Modify: `app/core/query_planner.py`
- Modify: `app/models/planning.py`
- Modify: `tests/unit/test_query_planner.py`
- Modify: `tests/contract/test_m1_t01_contracts.py`

**Interfaces:**

```python
class QueryExpansion(BaseModel):
    target_field: IntentField
    term_en: str
    source: TermSource

def build_query_plan(
    project_id: str,
    intent: ResearchIntent,
    *,
    positive_expansions: Sequence[QueryExpansion] | None = None,
    generated_at_utc: datetime | None = None,
) -> QueryPlan: ...
```

- [ ] Write failing tests for mapped Chinese, unmapped Chinese rejection, mixed Chinese-English preservation, all twelve query grammar strings, accepted/excluded expansions, `linear` versus `nonlinear`, injected clock, and default current UTC clock.
- [ ] Run `py -3.12 -m pytest tests/unit/test_query_planner.py tests/contract/test_m1_t01_contracts.py -q`; observe red while bare clauses/fallbacks/fixed timestamp remain.
- [ ] Represent every required query term as original text plus mapped English retrieval text; reject an unmapped required Chinese term with `UNMAPPED_RETRIEVAL_TERM`.
- [ ] Construct every clause as `all:"term"` (or another allowed field) and join only with the allowed uppercase operators; preserve canonical text without URL encoding.
- [ ] Add non-conflicting `QueryExpansion` terms only to target-field MEDIUM/WIDE queries. Excluded terms appear only in retained conflict provenance, never in query text.
- [ ] Use `generated_at_utc or datetime.now(UTC)`; make plan identity independent of the observed wall clock.
- [ ] Re-run the tests; all twelve queries must conform.

### Task 4: Make term evidence exact and commit implementation C

**Files:**
- Modify: `app/models/planning.py`
- Modify: `tests/contract/test_m1_t01_contracts.py`
- Modify: all Task 1-3 files
- Create: this plan file

**Interfaces:** For every `IntentField`, normalized declared terms and normalized evidence terms must be equal and evidence must not contain duplicates. Source values remain distinct (`ORIGINAL_INPUT`, `USER_CLARIFICATION`, `DETERMINISTIC_RULE`, `LLM_FAKE`).

- [ ] Write failing tests for a multi-term field missing one evidence record, undeclared evidence, duplicate evidence, and mixed sources.
- [ ] Run `py -3.12 -m pytest tests/contract/test_m1_t01_contracts.py -q`; observe red before exact-set enforcement.
- [ ] Validate each field with `declared_terms == evidence_terms` and `len(evidence_terms) == len(evidence_items)` after normalization.
- [ ] Run `py -3.12 -m pytest tests/unit/test_intent_clarification.py tests/unit/test_query_planner.py tests/contract/test_m1_t01_contracts.py tests/contract/test_evidence_report.py -q`; it must pass.
- [ ] Commit all implementation, test, validator, and plan files as `fix: close M1-T01 intent planning gaps` (commit C).

### Task 5: Regenerate evidence and update existing Draft PR #6

**Files:**
- Modify: `evaluation/reports/m1-t01-intent-query-plan.json`
- Modify: `STATUS.md`

**Interfaces:** The report references full commit C, hashes all required C blobs (including files listed in the acceptance request), records actual red/green command results, and labels external work `not_run`. `STATUS.md` stays M1 `IN_PROGRESS 1/4` and names M1-T02 only as the future action, not started work.

- [ ] Run the seven required commands: focused pytest, full pytest, Ruff, mypy, project-doc validation, historical M0 validation, and `pip check`; every exit must be 0.
- [ ] Hash `git show <C>:<path>` for every required report input, write the report with full C SHA and actual counts, and update status language for M1-T01R.
- [ ] Commit only the report and `STATUS.md` as `chore: refresh M1-T01R evidence` (commit D).
- [ ] Push the existing branch and edit Draft PR #6's description with full C/D/head hashes, scope exclusions, automated result, and explicit `not_run` boundaries.

## Self-Review

- Coverage: Tasks 1-5 implement all requested validator, preparation, retrieval-term, grammar, expansion, timestamp, evidence, commit, PR, and verification requirements.
- No placeholders: every task names exact files, interfaces, test targets, red/green commands, and resulting commit.
- Type consistency: `prepare_intent` consumes `StructuredRequest`/`LLMProvider` and returns `IntentPreparationResult`; `build_query_plan` consumes a frozen intent plus `QueryExpansion` values and returns `QueryPlan`.

---

## M1-T01R2 exclusion and lifecycle closure

**Goal:** Close the remaining deterministic query-planning review gaps without starting M1-T02.

**Files:** `app/core/text_normalization.py`, `app/core/intent.py`, `app/core/query_planner.py`, `app/models/planning.py`, `scripts/validate_phase.py`, the M1 unit/contract tests, and this plan.

- [ ] Add red tests showing that normalized exclusions remove fixed synonyms and bridge terms, do not substring-match `linear` against `nonlinear`, reject unmapped Chinese, and reject exact conflicts with core object/task/method/scope terms.
- [ ] Centralize NFKC, trim, whitespace collapse, and casefold in a dependency-free normalizer; route retrieval mapping, exclusion comparison, and evidence equality through it.
- [ ] Build a mapped English exclusion set that retains original audit text and produces provenance-bearing conflicts for deterministic rules, fake-provider synonyms, and original core terms.
- [ ] Restrict `QueryExpansion` and `candidate_synonyms` to object/task/method/scope; map candidate synonyms to `query_expansions` with `LLM_FAKE` provenance, fail unsupported fields as `INVALID_QUERY_PLAN`, and fail unmapped Chinese as `UNMAPPED_RETRIEVAL_TERM`.
- [ ] Enforce `m1-t01.v2`, unique query text, exactly one query per branch/breadth, branch-weight equality, supported positive-expansion fields, and exclusion absence in `QueryPlan` model validation.
- [ ] Add an M0 report gate test proving a live `M0 IN_PROGRESS 3/4` status is rejected after an immutable M0 report exists while legal M1 `0/4` through `4/4` transitions remain valid.
- [ ] Run the focused red commands first; after implementation run focused tests, full regression, Ruff, mypy, document validation, historical M0 validation, and `pip check` before creating the implementation/evidence commit pair.

---

## M1-T01R3 breadth semantics and exclusion degradation

**Goal:** Make the four M1-T01 query branches structurally prove that WIDE broadens MEDIUM while preserving auditable exclusion input, without beginning M1-T02.

**Architecture:** `QueryExpression` is a strict serializable OR-group structure: each non-empty tuple is a mandatory group and terms inside a group are alternatives. `QueryPlan` stores one expression for each query ID and verifies its rendered canonical arXiv text, branch anchor, unique text, and NARROW/MEDIUM/WIDE required-group monotonicity. Exclusions become records containing original user text plus canonical English retrieval text; the planner applies their canonical values but retains the original input for conflict audit.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, Ruff, mypy, GitHub Actions.

## Global Constraints

- Continue existing Draft PR #6 on `agent/m1-t01-intent-query-plan`; do not create a branch or PR.
- Preserve M1-T01 `m1-t01.v2`, twelve English canonical unencoded arXiv query texts, IDs, branch weights, fake-provider boundary, M0 immutable-evidence validation, and intent preparation behavior.
- Do not implement HTTP, Atom, pagination, caching, real arXiv access, model download/inference, CNKI, platform work, or M1-T02.
- Use implementation commit G `fix: align M1 query breadth semantics`, then evidence-only commit H `chore: refresh merge-ready M1-T01 evidence`; H contains only `evaluation/reports/m1-t01-intent-query-plan.json` and `STATUS.md`.

---

### Task 1: Write and preserve the R3 red tests

**Files:**
- Modify: `tests/unit/test_query_planner.py`

**Interfaces:** Tests consume `QueryPlan.expressions: dict[str, QueryExpression]`, `QueryExpression.required_group_count`, `serialize()`, `terms()`, `has_anchor()`, `ExclusionTerm.original_text`, and `ExclusionTerm.canonical_text_en`.

- [ ] Add `test_breadth_expressions_are_monotonic_serialized_and_anchor_each_branch` with the exact per-branch assertions `NARROW >= MEDIUM > WIDE`, WIDE anchors, rendered text equality, and counts DIRECT=2, PROBLEM=2, METHOD=1, BRIDGE=2.
- [ ] Add a parameterized test for `framework`, `application`, `methodology`, and `benchmark`; each exclusion must still produce twelve unique texts and omit its term.
- [ ] Add the exhausted six-bridge-term test; it must assert `PlanningError.code == "INVALID_QUERY_PLAN"` and `reason == "bridge vocabulary exhausted"`.
- [ ] Add a Chinese `"\u6846\u67b6"` exclusion test asserting the stored `(original_text, canonical_text_en)` equals `("\u6846\u67b6", "framework")`.
- [ ] Run `py -3.12 -m pytest tests/unit/test_query_planner.py -q`; record the expected current red result: no expression metadata, empty former-WIDE groups, unstable bridge error, and absent audit record.

### Task 2: Add strict expression and exclusion-audit contracts

**Files:**
- Modify: `app/models/planning.py`
- Modify: `tests/contract/test_m1_t01_contracts.py`

**Interfaces:**

```python
class QueryExpression(BaseModel):
    required_groups: tuple[tuple[str, ...], ...]
    anchor_terms: tuple[str, ...]
    @property
    def required_group_count(self) -> int: ...
    def terms(self) -> tuple[str, ...]: ...
    def has_anchor(self) -> bool: ...
    def serialize(self) -> str: ...

class ExclusionTerm(BaseModel):
    original_text: str
    canonical_text_en: str
    source: TermSource
```

- [ ] Make both models `extra="forbid"`, reject empty groups/anchors and duplicate canonical exclusions, and render each group as `all:"..."` joined by uppercase `AND`/`OR` only.
- [ ] Change `QueryPlan.exclusions` to `list[ExclusionTerm]` and add `expressions: dict[str, QueryExpression]`.
- [ ] Extend `QueryPlan.validate_first_round_structure` to require exactly one rendered expression for each query, matching text, canonical exclusion absence, WIDE term-subset-of-MEDIUM, and the four-branch group-count relation.
- [ ] Run `py -3.12 -m pytest tests/contract/test_m1_t01_contracts.py tests/unit/test_query_planner.py -q`; expect all focused contract tests to pass after the planner task.

### Task 3: Replace append-based WIDE construction with the four explicit matrices

**Files:**
- Modify: `app/core/query_planner.py`
- Modify: `app/core/intent.py`

**Interfaces:** `build_query_plan()` returns the contract from Task 2. `build_exclusion_set()` returns deterministic `ExclusionTerm` records. `PlanningError(code, reason=None)` exposes `code` and `reason` without changing existing code-only callers.

- [ ] Add the retrieval mapping `"\u6846\u67b6": "framework"` so a Chinese exclusion is retained exactly while it is compared through canonical English.
- [ ] Remove `_WIDE_TERMS` and build each expression directly: DIRECT `(object_group, task_group, method_group)` / `(object_group, task_group OR method_group)`; PROBLEM `(object_group, task_group, scope_group)` / `(object_group, task_group OR scope_group)`; METHOD `(method_group, task_group)` / `(method_group)`; BRIDGE `(object_group, task_group, method_group, bridge_group)` / `(object_group OR task_group, method_group OR bridge_group)`.
- [ ] Keep NARROW on the first term of each required group; MEDIUM uses the corresponding expanded groups. WIDE may merge existing groups but may not add a mandatory topic or a former `_WIDE_TERMS` group.
- [ ] Filter optional deterministic bridge terms by canonical exclusions. If at least one remains, build BRIDGE queries; if none remain, raise `PlanningError("INVALID_QUERY_PLAN", "bridge vocabulary exhausted")` before serialization.
- [ ] Reject canonical duplicate exclusions deterministically; preserve every accepted original/canonical/source record and use its canonical value for conflict detection.
- [ ] Run `py -3.12 -m pytest tests/unit/test_query_planner.py -q`; expected result: all query-planner tests pass and each former fixed WIDE term exclusion still yields a complete plan.

### Task 4: Validate, commit, and refresh immutable R3 evidence

**Files:**
- Modify for commit G: `app/core/intent.py`, `app/core/query_planner.py`, `app/models/planning.py`, `tests/unit/test_query_planner.py`, `tests/contract/test_m1_t01_contracts.py`, this plan.
- Modify for commit H only: `evaluation/reports/m1-t01-intent-query-plan.json`, `STATUS.md`.

- [ ] Run `py -3.12 -m pytest tests/unit/test_intent_clarification.py tests/unit/test_query_planner.py tests/contract/test_m1_t01_contracts.py tests/contract/test_evidence_report.py -q`, then `py -3.12 -m pytest -q`, `py -3.12 -m ruff check app evaluation scripts tests`, `py -3.12 -m mypy app evaluation scripts`, `py -3.12 scripts/validate_project_docs.py`, `py -3.12 scripts/validate_phase.py M0`, and `py -3.12 -m pip check`; every command must exit 0.
- [ ] Commit only implementation/tests/plan as `fix: align M1 query breadth semantics`; save its full SHA as `validated_implementation_commit` and hash all changed implementation blobs from that commit.
- [ ] Write R3 evidence with actual red/green results and the six required review repairs: breadth monotonicity, fewer WIDE groups, no mandatory WIDE terms, safe optional-term exclusion degradation, stable bridge exhaustion, and original exclusion audit. Keep every real external operation `not_run` and M1-T02 not started.
- [ ] Commit only the evidence report and `STATUS.md` as `chore: refresh merge-ready M1-T01 evidence`, push the existing branch, and update PR #6 while keeping it Draft. Record rollback as `git revert <H>` then `git revert <G>`.

## Self-Review

- Coverage: the explicit matrix covers all four branches and all three breadths; expression metadata proves breadth instead of counting raw `AND` strings; exclusion records preserve Chinese original text and canonical English comparison; every listed fixed WIDE exclusion degrades safely; bridge exhaustion has a stable business error.
- Placeholder scan: no task requires HTTP, a real model, external retrieval, or M1-T02 work.
- Type consistency: `QueryPlan.expressions` keys are query IDs; `QueryExpression.serialize()` is the only text renderer; `ExclusionTerm.canonical_text_en` is the sole comparison key while `original_text` remains audit data.
