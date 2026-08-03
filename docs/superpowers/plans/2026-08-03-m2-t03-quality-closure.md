# M2-T03 Evidence Classification Quality Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the M2-T03 deterministic classification contract boundary without claiming real classification quality or starting M2-T04/M2-T05.

**Architecture:** Keep public evidence records closed and provenance-bound, but make the deterministic fake provider select an immutable internal match from the same title/abstract evidence fragment that supplies the slot, support level, and excerpt. Add machine-readable quality diagnostics and CI validation while keeping completion scope contract-only and scoring ineligible.

**Tech Stack:** Python 3.12, Pydantic, pytest, Ruff, mypy, JSON evidence artifacts, GitHub Actions.

## Global Constraints

- Use merge commits only; no rebase, squash merge, force-push, or shared-history rewrite.
- Do not run or download embedding/reranker/classification models; deterministic fake evidence remains explicitly fake.
- Do not modify M2-T01/M2-T02 formal result bytes or feed fake output into M2-T04 scoring.
- Keep STATUS at M2 `IN_PROGRESS 2/5`, next task `M2-T03 quality closure`, and M3 `BLOCKED_BY_M2 0/5`.
- Preserve the fixed M1 candidate and M2-T02 input hashes and keep real-model/human-review evidence `not_run`.

---

### Task 1: Establish the merged baseline and contract-only status

**Files:**
- Modify: `STATUS.md`
- Create: `docs/superpowers/plans/2026-08-03-m2-t03-quality-closure.md`

- [ ] **Step 1: Verify the merge baseline**

Run:

```powershell
git show --no-patch --pretty=raw origin/main
git merge-base --is-ancestor b198a122494656ec68a54ea68c643313d11ebb63 origin/main
```

Expected: merge commit `1b837f160b40ff47d36f63757e30c2876f8c6f5e`; ancestor check exits 0.

- [ ] **Step 2: Record the non-completion state**

Keep the phase row and evidence prose equivalent to:

```text
M2 = IN_PROGRESS 2/5
next task = M2-T03 quality closure
M3 = BLOCKED_BY_M2 0/5
completion_scope = contract_only
scoring_eligible = false
```

- [ ] **Step 3: Validate the baseline**

Run `python scripts/validate_project_docs.py` and `git diff --check`. Both must pass before code changes.

- [ ] **Step 4: Commit the merge resolution**

```powershell
git add agent.md STATUS.md docs/phases/M2-ranking-and-round1-selection.md scripts/validate_project_docs.py tests/contract/test_project_docs_validation.py docs/superpowers/plans/2026-08-03-m2-task-order-and-handoff-fix.md
git commit --no-edit
```

Expected: a non-squash merge commit with no conflict markers.

### Task 2: Add failing grounded-excerpt regression tests

**Files:**
- Modify: `tests/unit/test_m2_t03_evidence_classification_provider.py`
- Modify: `tests/contract/test_m2_t03_evidence_classification.py`

**Interfaces:** The tests exercise `DeterministicFakeEvidenceClassifier.classify()` and the existing public `EvidenceClassificationRecord` fields without adding a public matched-marker field.

- [ ] **Step 1: Add failing tests**

Cover these exact cases:

```python
def test_title_only_marker_uses_title_excerpt() -> None: ...
def test_title_marker_does_not_fallback_to_unrelated_abstract() -> None: ...
def test_generic_system_dataset_results_application_words_do_not_classify_alone() -> None: ...
def test_support_level_is_bound_to_the_selected_sentence() -> None: ...
def test_equal_low_specificity_matches_are_rejected() -> None: ...
```

Also retain the existing direct/indirect/hypothetical, title-only, empty-abstract, unknown-enum, external-assertion, exact-substring, order-independence, duplicate-ID, and provider-failure tests.

- [ ] **Step 2: Run only the new tests and confirm red**

```powershell
python -m pytest -q tests/unit/test_m2_t03_evidence_classification_provider.py -k "title_only_marker or unrelated_abstract or generic or selected_sentence or low_specificity"
```

Expected: the new cases fail against the old excerpt/fallback behavior.

### Task 3: Bind classification to a matched evidence fragment

**Files:**
- Modify: `app/adapters/evidence_classification.py`
- Modify: `app/core/evidence_classification.py` only if validation helpers need the new rejection path
- Test: `tests/unit/test_m2_t03_evidence_classification_provider.py`

**Interfaces:** Add a provider-private immutable structure:

```python
@dataclass(frozen=True)
class _EvidenceMatch:
    slot: EvidenceSlot
    support_level: SupportLevel
    matched_marker: str
    matched_source: Literal["title", "abstract"]
    supporting_excerpt: str
    specificity: int
```

- [ ] **Step 1: Split title/abstract into candidate fragments**

Use sentence-like fragments with source name and start offset; preserve exact source text and reject empty fragments.

- [ ] **Step 2: Match markers with bounded rules**

Use `re.IGNORECASE` word-boundary patterns for English inflections such as `benchmark(?:s|ed|ing)?`, and explicit phrase matching for Chinese markers. Treat generic `system`, `dataset`, `results`, and `application` as non-decisive alone.

- [ ] **Step 3: Rank candidate matches deterministically**

Order by match score, specificity, title-before-abstract source priority, excerpt offset, and enum order. Equal low-specificity matches and marker-free candidates must yield `REJECTED`; never select a slot solely to fill all 33 records.

- [ ] **Step 4: Remove arbitrary excerpt fallback**

Delete any behavior equivalent to `source[:MAX_SUPPORTING_EXCERPT_LENGTH]` when no slot marker is bound to the excerpt. The selected excerpt must be the exact fragment used for slot and support decisions.

- [ ] **Step 5: Run the focused tests green**

```powershell
python -m pytest -q tests/unit/test_m2_t03_evidence_classification_provider.py tests/contract/test_m2_t03_evidence_classification.py
```

Expected: all focused classification tests pass, including the new red-to-green cases.

### Task 4: Add quality diagnostics and contract-only report fields

**Files:**
- Modify: `app/models/evidence_classification.py`
- Modify: `scripts/run_m2_t03_evidence_classification.py`
- Modify: `scripts/validate_m2_t03_evidence.py`
- Modify: `evaluation/reports/m2-t03-evidence-classification.json`
- Test: `tests/contract/test_m2_t03_evidence_classification_evidence.py`

- [ ] **Step 1: Add a closed diagnostic schema**

The result execution section must expose a closed diagnostic object containing:

```json
{
  "observed_slot_count": 0,
  "observed_support_level_count": 0,
  "all_records_same_support_level": false,
  "generic_marker_only_count": 0,
  "ambiguous_rejection_count": 0,
  "warnings": []
}
```

Warnings must use stable codes such as `FEWER_THAN_THREE_SLOTS_OBSERVED`, `ALL_RECORDS_SAME_SUPPORT_LEVEL`, `NO_INDIRECT_OR_HYPOTHETICAL_RECORDS`, `GENERIC_MARKER_DOMINANCE`, and `NO_REJECTED_OR_UNCERTAIN_RECORDS`.

- [ ] **Step 2: Add contract-only summary fields**

The report must contain exactly:

```json
{
  "completion_scope": "contract_only",
  "scoring_eligible": false
}
```

Residual risks must state that deterministic fake evidence proves contracts/reproducibility only and cannot supply M2-T04 evidence-slot scores.

- [ ] **Step 3: Extend the validator and tests**

Reject missing/unknown diagnostic fields, `scoring_eligible=true`, real-model/human claims, or M2 status `3/5`; require M2 `2/5` and next task `M2-T03 quality closure`.

### Task 5: Add remote CI M2-T03 evidence validation

**Files:**
- Modify: `.github/workflows/docs-validation.yml`
- Modify: `tests/contract/test_project_docs_validation.py` or add the established workflow contract test file

- [ ] **Step 1: Add the workflow command**

Insert:

```yaml
- name: Validate completed M2-T03 evidence
  run: python scripts/validate_m2_t03_evidence.py
```

- [ ] **Step 2: Add workflow contract assertions**

Assert the exact validator path/name is present when M2-T03 is reported complete and that later task changes do not remove the historical validator.

- [ ] **Step 3: Run the workflow contract test and local docs validator**

```powershell
python -m pytest -q tests/contract/test_project_docs_validation.py
python scripts/validate_project_docs.py
```

### Task 6: Regenerate deterministic fake artifacts and bind hashes

**Files:**
- Modify: `evaluation/source-artifacts/m2-t03-evidence-classification-run.json`
- Modify: `evaluation/source-artifacts/m2-t03-evidence-classification-run-receipt.json`
- Modify: `evaluation/reports/m2-t03-evidence-classification.json`

- [ ] **Step 1: Run only the deterministic fake runner**

```powershell
python scripts/run_m2_t03_evidence_classification.py --execute-deterministic-fake
```

Expected: no model download/import, no embedding/reranker execution, and a receipt with `real_model_run=false`, `human_judged=false`, and the current execution commit/runner hash.

- [ ] **Step 2: Recompute and validate all artifact hashes**

Bind the fixed M1 snapshot/manifest and M2-T02 result/receipt hashes, then run `python scripts/validate_m2_t03_evidence.py`.

- [ ] **Step 3: Verify immutable historical artifacts**

Confirm these hashes remain unchanged:

```text
m2-t02-reranker-candidate-run.json = 0a751dbc35bfa8d07433796113939412bfbdffd4c5c13043c90390bccfc5c35b
m2-t02-reranker.json = 009f4ad441b80a597b955c40ff6d31aac13476adf6574540bf3f4a790192361a
m2-t02-reranker-candidate-run-receipt.json = 0c9ca1339a2385a9c6940f8c53bd4711c62b8faf9f3f4cf98710ff465d56ec9a
```

### Task 7: Run the complete gate set and update PR #13

**Files:**
- Modify: `STATUS.md`
- Modify: `evaluation/reports/m2-t03-evidence-classification.json`
- Modify: PR #13 description through GitHub

- [ ] **Step 1: Run focused, full, static, evidence, and dependency checks**

```powershell
python -m pytest -q tests/contract/test_m2_t03_evidence_classification.py tests/unit/test_m2_t03_evidence_classification_provider.py tests/unit/test_m2_t03_evidence_classification_runner.py tests/contract/test_m2_t03_evidence_classification_evidence.py
python -m pytest -q -m "not packaging"
python -m pytest -q -m packaging
python -m ruff check app evaluation scripts tests
python -m ruff check --select I app evaluation scripts tests
python -m mypy app evaluation scripts
python scripts/validate_project_docs.py
python scripts/validate_phase.py M0
python scripts/validate_m1_evidence.py
python scripts/validate_m2_t01_evidence.py
python scripts/validate_m2_t02_evidence.py
python scripts/validate_m2_t03_evidence.py
python -m pip check
git diff --check
```

- [ ] **Step 2: Confirm final scope**

Do not change M2 to 3/5, do not mark PR #13 ready, do not merge PR #13, and do not start M2-T04.

- [ ] **Step 3: Update PR #13 with final evidence**

The body must state the new HEAD, PR #12 merge commit, merge-from-main commit, classification rules, distributions, rejected/ambiguous counts, warning codes, result/receipt/report hashes, test totals, remote CI run ID, `real_model=not_run`, `human_review=not_run`, `scoring_eligible=false`, and the remaining M2-T03 quality closure work.

- [ ] **Step 4: Push without rewriting history**

```powershell
git push origin agent/m2-t03-evidence-slot-classification
```

Expected: the existing seven PR #13 commits remain intact plus explicit merge/fix/test/diagnostic/CI/evidence/docs commits; PR #13 remains Draft.
