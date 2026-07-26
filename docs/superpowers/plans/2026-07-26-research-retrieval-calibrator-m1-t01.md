# M1-T01 Intent IR and Query Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deterministically turn a user research request into a validated pre-freeze intent draft, at most three high-value clarification questions, and a first-round arXiv `QueryPlan` with exactly four branches and twelve English queries.

**Architecture:** New M1-only Pydantic contracts in `app/models/planning.py` keep a mutable `IntentDraft` separate from M0's frozen `ResearchIntent`. Pure functions in `app/core/intent.py` calculate gaps, reject malformed LLM-fake candidates, and freeze only complete drafts; `app/core/query_planner.py` expands a frozen intent through fixed four-branch, three-breadth templates. No source adapter, HTTP request, model inference, persistence, API, ranking, or M1-T02 work is introduced.

**Tech Stack:** Python 3.12, Pydantic 2, pytest 8, Ruff, mypy, stdlib `hashlib`/`unicodedata`.

## Global Constraints

- Work from `agent/m1-t01-intent-query-plan`, created from main commit `9eb8d8d4e4482ff2a05346a8fdad5b86780ea6cf`.
- Only implement M1-T01; arXiv requests, M1-T02, M2 ranking, model download/inference, platform APIs, databases, and CNKI remain out of scope.
- Preserve M0 contracts; make only the minimum validator adjustment necessary for an M1 `IN_PROGRESS 1/4` status to retain historical M0 validation.
- Normalise comparison strings with Unicode NFKC, trim, collapsed whitespace, and `casefold`; do exact-token comparisons only, never substring removal.
- All LLM behavior is a synchronous `Protocol` plus deterministic tests/fakes. Malformed fake output raises a stable coded error and is never guessed or repaired.
- Record actual red and green command outcomes separately in an evidence report. All real external and human actions remain explicitly `not_run`.

---

## File map

- `app/models/planning.py`: M1 intent-draft, clarification, term-provenance, and query-plan contracts.
- `app/models/__init__.py`: exports the public M1 planning contracts.
- `app/adapters/llm.py`: minimal structured-output protocol and strict fake-output boundary.
- `app/core/intent.py`: pure normalisation, candidate validation, gap scoring, question selection, and freeze guard.
- `app/core/query_planner.py`: fixed branch/breadth templates, English lexical mapping, conflict filtering, stable IDs, and deterministic weights.
- `tests/unit/test_intent_clarification.py`: red/green coverage for gaps, questions, freeze boundary, and fake rejection.
- `tests/unit/test_query_planner.py`: red/green coverage for deterministic twelve-query planning and exclusion safety.
- `tests/contract/test_m1_t01_contracts.py`: strict Pydantic and public-contract invariants.
- `scripts/validate_phase.py`, `tests/contract/test_evidence_report.py`: allow a legal M1 in-progress status while retaining M0 evidence immutability checks.
- `docs/superpowers/plans/2026-07-26-research-retrieval-calibrator-m1-t01.md`: this executable plan.
- `evaluation/reports/m1-t01-intent-query-plan.json`: machine-readable final evidence, referencing implementation commit A only.
- `STATUS.md`: records M1 `IN_PROGRESS 1/4` and next action `M1-T02` only after all checks pass.

## Public interfaces

```python
class IntentDraft(BaseModel):
    original_input: str
    object_terms: list[str] = []
    task_terms: list[str] = []
    method_terms: list[str] = []
    scope_terms: list[str] = []
    exclusions: list[str] = []
    method_constraint: MethodConstraint | None = None
    accepted_paper_roles: set[str] = set()
    source_language: Literal["zh", "en", "mixed"]
    revision: int
    field_evidence: dict[IntentField, list[TermEvidence]]

def build_intent_gaps(draft: IntentDraft, field_confidences: Mapping[IntentField, float] = {}) -> list[IntentGap]: ...
def build_clarification_questions(draft: IntentDraft, gaps: Sequence[IntentGap]) -> list[ClarificationQuestion]: ...
def freeze_research_intent(draft: IntentDraft, frozen_at: datetime) -> ResearchIntent: ...
def validate_llm_candidate(payload: dict[str, object]) -> LLMCandidate: ...
def build_query_plan(project_id: str, intent: ResearchIntent) -> QueryPlan: ...
```

`priority_score = 0.65 * missingness_score + 0.35 * ambiguity_score`; fields marked explicit receive no question even if a supplied fake confidence is low. Descending priority breaks ties in this fixed order: `object`, `task`, `method`, `scope`, `accepted_paper_roles`, `exclusions`.

`QueryPlan` uses source `arxiv`, round `1`, planning config `m1-t01.v1`, branch weights `0.35/0.25/0.20/0.20`, and branch budget divided equally across its three queries. The timestamp is a documented deterministic planning timestamp so equal normalised inputs have byte-identical plans; evidence timestamps use actual UTC time.

### Task 1: Define M1 planning contracts and the LLM boundary

**Files:**

- Create: `app/models/planning.py`
- Create: `app/adapters/__init__.py`
- Create: `app/adapters/llm.py`
- Modify: `app/models/__init__.py`
- Test: `tests/contract/test_m1_t01_contracts.py`

- [ ] **Step 1: Write failing contracts**

Add tests that import `IntentDraft`, `IntentGap`, `ClarificationQuestion`, `QueryPlan`, and `validate_llm_candidate`; assert `extra="forbid"`, blank-after-trim term rejection, score range enforcement, `QueryPlan` rejection of a non-arXiv or non-first-round plan, and stable `INVALID_INTENT_STRUCTURE` for an unknown fake field.

- [ ] **Step 2: Preserve red output**

Run `py -3.12 -m pytest tests/contract/test_m1_t01_contracts.py -q`; expect collection/import failure because M1 planning modules do not yet exist. Save command, exit code, and test summary for the report.

- [ ] **Step 3: Implement strict contracts**

Use `ConfigDict(extra="forbid")` on every M1 model. Implement `TermEvidence(term, source)` where source is `original_input`, `user_clarification`, `deterministic_rule`, or `llm_fake`; require evidence terms to match a declared term after normalisation. Define `QueryPlan` around existing M0 `Query` objects and validate exactly four `QueryBranch` values, each with all three `QueryBreadth` values, exactly twelve unique stable IDs, `language="en"`, empty `revision_sources`, positive query weights summing to one, and matching frozen intent revision.

`StructuredRequest` contains only `original_input` and the six allowed target fields. `LLMProvider.generate_structured(request) -> dict[str, object]` performs no I/O. `LLMCandidate` admits only candidate terms/synonyms, allowed field confidences in `[0, 1]`, and language `zh|en|mixed`; Pydantic validation failures are wrapped in `PlanningError("INVALID_INTENT_STRUCTURE")`.

- [ ] **Step 4: Verify contracts green**

Run `py -3.12 -m pytest tests/contract/test_m1_t01_contracts.py -q`; expect all new contract tests to pass.

### Task 2: Implement pure intent clarification and freeze guard

**Files:**

- Create: `app/core/intent.py`
- Test: `tests/unit/test_intent_clarification.py`

- [ ] **Step 1: Write failing behavior tests**

Cover: explicit object and method are not queried; a missing high-ambiguity field outranks other gaps; maximum is three; exact ties use the declared order; incomplete draft raises `INCOMPLETE_INTENT`; a draft cannot be used where a frozen `ResearchIntent` is accepted; valid fake candidates are normalised with provenance; invalid candidate confidence/language/unknown fields raises `INVALID_INTENT_STRUCTURE`.

- [ ] **Step 2: Preserve red output**

Run `py -3.12 -m pytest tests/unit/test_intent_clarification.py -q`; expect collection/import failure due to absent core module.

- [ ] **Step 3: Implement deterministic pure functions**

Implement `normalise_term(value)` as NFKC → trim → whitespace collapse → casefold. Implement `build_intent_gaps` using the stated numeric formula and bounded scores. Use fixed bilingual templates for `ClarificationQuestion`; skip `already_explicit=True` gaps, sort deterministically, and return at most three. `freeze_research_intent` verifies all required research fields plus method constraint and accepted roles, then constructs the existing M0 `ResearchIntent`; no inheritance or coercion makes an `IntentDraft` a frozen intent.

- [ ] **Step 4: Verify focused green**

Run `py -3.12 -m pytest tests/unit/test_intent_clarification.py -q`; expect all tests to pass.

### Task 3: Build deterministic four-route query plans

**Files:**

- Create: `app/core/query_planner.py`
- Test: `tests/unit/test_query_planner.py`

- [ ] **Step 1: Write failing planner tests**

Construct a frozen English intent and assert equal inputs produce equal plans; assert four branches × three widths = twelve queries; each query has the intent revision, `en`, no revision sources, a unique deterministic ID, positive weight, and total weight one. Assert `DIRECT_INTERSECTION` contains object/task/method, `PROBLEM_DOMAIN` omits a required method term, `METHOD_DOMAIN` weakens object restriction, and `BRIDGE_DOMAIN` draws only from the fixed bridge vocabulary. Add a Chinese shielding/transfer-learning intent and assert an English query is emitted. Test exact `linear` exclusion removes only `linear`, preserves `nonlinear`, and records the conflict source.

- [ ] **Step 2: Preserve red output**

Run `py -3.12 -m pytest tests/unit/test_query_planner.py -q`; expect import failure because the planner is absent.

- [ ] **Step 3: Implement fixed template expansion**

Use only deterministic templates. `NARROW` joins strongest phrases by `AND`; `MEDIUM` joins core terms by `AND` and fixed synonyms by `OR`; `WIDE` retains the intersection core while safely reducing constraints. Problem uses object/task/scope, method uses method plus task/evaluation, and bridge uses the fixed versioned terms `application`, `adaptation`, `transfer`, `framework`, `methodology`, `benchmark`.

Apply exact normalised exclusion filtering only to positive expansion candidates. Retain every exclusion and append `TermConflict` provenance; do not remove substrings. A small fixed Chinese-to-English lexical mapping supports the test terms; unmapped Chinese text does not reach generated English arXiv queries. Generate query IDs from canonical JSON hashed with SHA-256; use only the hash-derived value and deterministic input-derived generated timestamp.

- [ ] **Step 4: Verify focused green**

Run `py -3.12 -m pytest tests/unit/test_query_planner.py -q`; expect all tests to pass.

### Task 4: Preserve M0 validation, execute full verification, and commit implementation A

**Files:**

- Modify: `scripts/validate_phase.py`
- Modify: `tests/contract/test_evidence_report.py`
- Modify: `docs/superpowers/plans/2026-07-26-research-retrieval-calibrator-m1-t01.md`
- Modify: all Task 1–3 files

- [ ] **Step 1: Add failing M0-history test**

Create a test showing the M0 evidence validator accepts a current M1 header/table state `IN_PROGRESS`, `1/4` while still requiring M0 `COMPLETE 4/4` and all later phases blocked. Run `py -3.12 -m pytest tests/contract/test_evidence_report.py -q`; expect the new test to fail against the old M0-only status snapshot rule.

- [ ] **Step 2: Apply the minimum validator change**

Permit exactly the legal M1-T01 post-state in `_validate_status`; retain M0 commit/blob/hash validation and reject all other later-phase changes. Do not change any M0 data model or evidence schema.

- [ ] **Step 3: Run the required green suite**

Run, recording actual exit codes and counts:

```powershell
py -3.12 -m pytest tests/unit/test_intent_clarification.py tests/unit/test_query_planner.py tests/contract/test_m1_t01_contracts.py -q
py -3.12 -m pytest -q
py -3.12 -m ruff check app evaluation scripts tests
py -3.12 -m mypy app evaluation scripts
py -3.12 scripts/validate_project_docs.py
py -3.12 scripts/validate_phase.py M0
py -3.12 -m pip check
```

- [ ] **Step 4: Commit implementation A**

Stage only implementation, tests, plan, and minimum historical-validator change. Commit with `feat: implement M1 intent and query planning`, then capture its full SHA as `validated_implementation_commit`. Do not include `STATUS.md` or the M1 evidence report in this commit.

### Task 5: Produce evidence commit B, update status, publish, and open draft PR

**Files:**

- Create: `evaluation/reports/m1-t01-intent-query-plan.json`
- Modify: `STATUS.md`

- [ ] **Step 1: Generate non-self-referential evidence**

Report `task_id=M1-T01`, `phase=M1`, baseline main commit, implementation commit A, actual UTC timestamp, Python version, preserved red-light and green-light command records, SHA-256 hashes of the exact test inputs, and the six mandated `not_run` boundaries: arXiv network, real model download/inference, platform API/deployment, CNKI, human judgement, and real source-pool freeze. State explicitly that this task has no network requests, model calls, M1-T02, or M2 objects.

- [ ] **Step 2: Update status only after green verification**

Set current phase/status to `M1`/`IN_PROGRESS`; set M0 to `COMPLETE 4/4`, M1 to `IN_PROGRESS 1/4`, M2 to `BLOCKED_BY_M1`; make `M1-T02` the next action. Do not mark M1 complete.

- [ ] **Step 3: Commit evidence B and revalidate**

Commit only the report and `STATUS.md` with `chore: record M1-T01 evidence`. Rerun the full required green suite, including historical M0 validation, and confirm `git status --short` is empty.

- [ ] **Step 4: Push and create a draft PR**

Push the task branch, open a draft PR titled `feat: implement M1 intent and query planning`, and include baseline SHA, draft/frozen boundary, scoring formula, 4×3/12 constraints, exclusion rule, fake-rejection test, red/green result, automated/not-run evidence, status, and `git revert <evidence B>` rollback instructions. Report CI state without claiming it passed until observed.

## Acceptance checklist

- [ ] All 23 requested TDD cases are represented across focused and contract tests.
- [ ] Every new Pydantic boundary forbids extra fields and blank terms, and all invalid fake paths fail with stable codes.
- [ ] Same normalised frozen intent produces byte-identical 12-query plan with the required routes, breadths, language, provenance, IDs, and weights.
- [ ] No external request, model download/inference, platform API, CNKI, persistence, M1-T02, or M2 object exists in the diff.
- [ ] Actual red/green evidence is machine readable and does not self-reference evidence commit B.
- [ ] M0 phase validation remains green after the legal M1 status update.
- [ ] Rollback is `git revert <evidence-B-sha>` followed by `git revert <implementation-A-sha>`; no reset, rebase, or force push.
