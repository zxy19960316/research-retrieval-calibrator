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
