# M2-T03 Scientific Fragment Scanner Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair deterministic title/abstract fragment boundaries so dotted identifiers, decimals, abbreviations, versions, URLs, and DOIs remain intact while genuine sentence endings still isolate evidence support.

**Architecture:** Keep _source_fragments() private and dependency-free. Replace its punctuation regex with a deterministic left-to-right scanner that records stripped fragment text and its original source offset. Make _EvidenceMatch carry the absolute excerpt offset and actual matched marker text, while continuing to derive support only from the selected bounded excerpt.

**Tech Stack:** Python 3.12, stdlib re/dataclasses, pytest, Ruff, mypy, JSON evidence artifacts, GitHub Actions.

## Global Constraints

- Process only PR #13 M2-T03 contract-only closure; do not start M2-T04 or implement M2-T05.
- Do not modify M2-T01/M2-T02 formal artifact bytes or run/download any model.
- Keep fake evidence completion_scope="contract_only", scoring_eligible=false, real_model=not_run, human_review=not_run, human_judged=false.
- Keep STATUS at M2 IN_PROGRESS 2/5, next task M2-T03 quality closure, and M3 BLOCKED_BY_M2 0/5.
- Do not amend, rebase, squash, force-push, or rewrite shared history; each change is an independent commit.
- Preserve source order, exact source offsets, exact-substring excerpts, deterministic output, and input-order independence.
- Do not add a third-party NLP dependency or rules merely to force all slots/support levels/classifications to appear.

---

### Task 1: Add failing scientific boundary tests

Files:
- Modify: tests/unit/test_m2_t03_evidence_classification_provider.py
- Modify: tests/contract/test_m2_t03_evidence_classification.py

Interfaces: Tests use the existing public classifier and the private _source_fragments() scanner only for boundary/offset characterization; no public evidence schema field is added.

- [x] Step 1: Confirm the clean baseline

Run:

~~~powershell
git status --short --branch
git rev-parse HEAD
git merge-base --is-ancestor 1b837f160b40ff47d36f63757e30c2876f8c6f5e HEAD
~~~

Expected: clean branch, HEAD 34fe6c3f573c1e746eafefad4519b0f65e16e5af, ancestor exit 0.

- [ ] Step 2: Add exact boundary tests

Add this helper and tests to the unit provider test module:

~~~python
from app.adapters.evidence_classification import _source_fragments

def _texts(source: str) -> list[str]:
    return [fragment for fragment, _, _ in _source_fragments(source, "abstract")]

def test_dotted_identifier_is_one_fragment_and_exact_source_text() -> None:
    source = "Clinfo.ai: An Open-Source Retrieval-Augmented System"
    fragments = _source_fragments(source, "abstract")
    assert [fragment for fragment, _, _ in fragments] == [source]
    assert "Clinfo.ai" in fragments[0][0]
    assert source[fragments[0][2] : fragments[0][2] + len(fragments[0][0])] == fragments[0][0]

def test_decimal_is_not_truncated() -> None:
    source = "The method achieves a mean Average Precision of 59.70%."
    assert _texts(source) == [source]
    item = _input(title="Evaluation of a retrieval method", abstract=source)
    record = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier()).records[0]
    assert record.supporting_excerpt is not None
    assert "59.70%" in record.supporting_excerpt
    assert "59." not in record.supporting_excerpt.replace("59.70%", "")

@pytest.mark.parametrize("source", [
    "We use e.g. a graph method. It is evaluated.",
    "We use i.e. a graph method. It is evaluated.",
    "The result follows et al. in the cited study. It is evaluated.",
    "Fig. 2 shows the pipeline. It is evaluated.",
    "Dr. Smith presents the method. It is evaluated.",
    "The method is compared vs. a baseline. It is evaluated.",
])
def test_common_abbreviations_do_not_create_internal_fragments(source: str) -> None:
    fragments = _texts(source)
    assert len(fragments) == 2
    assert "It is evaluated." in fragments[1]
    assert fragments[0].startswith(("We use", "The result", "Fig.", "Dr.", "The method"))

@pytest.mark.parametrize("source", [
    "The contract targets v1.2. It is evaluated.",
    "The runner uses Python 3.12. It is evaluated.",
    "The identifier is 10.1000/example.doi. It is evaluated.",
    "The endpoint is example.org/path. It is evaluated.",
])
def test_versions_urls_and_dois_remain_inside_their_fragment(source: str) -> None:
    fragments = _texts(source)
    assert len(fragments) == 2
    assert "It is evaluated." in fragments[1]
    assert fragments[0].startswith(("The contract", "The runner", "The identifier", "The endpoint"))

def test_real_sentence_ending_still_splits_in_order() -> None:
    source = "We implement a retrieval pipeline. It is evaluated on a benchmark."
    assert _texts(source) == [
        "We implement a retrieval pipeline.",
        "It is evaluated on a benchmark.",
    ]

def test_scanner_preserves_order_and_offsets() -> None:
    source = "Python 3.12. We implement a pipeline."
    fragments = _source_fragments(source, "abstract")
    assert [fragment for fragment, _, _ in fragments] == [
        "Python 3.12.",
        "We implement a pipeline.",
    ]
    assert all(
        source[offset : offset + len(fragment)] == fragment
        for fragment, _, offset in fragments
    )
    assert [offset for _, _, offset in fragments] == sorted(
        offset for _, _, offset in fragments
    )
~~~

Retain the existing title/abstract grounding, generic-only rejection, ambiguous rejection, support binding, exact-substring, and reordered-input tests.

- [ ] Step 3: Run only the new regression tests and preserve the red result

Run:

~~~powershell
$py = "C:\Users\94310\AppData\Local\Programs\Python\Python312\python.exe"
& $py -m pytest -q tests/unit/test_m2_t03_evidence_classification_provider.py -k "dotted_identifier or decimal or abbreviations or versions_urls or real_sentence or scanner"
~~~

Expected: failure against the current regex scanner because Clinfo.ai, 59.70, abbreviations, or dotted identifiers split before the true sentence end. After confirming red, commit only the plan/tests:

~~~powershell
git add docs/superpowers/plans/2026-08-03-m2-t03-fragment-scanner-regression.md tests/unit/test_m2_t03_evidence_classification_provider.py tests/contract/test_m2_t03_evidence_classification.py
git commit -m "test: cover scientific fragment boundary cases"
~~~

---

### Task 2: Implement the deterministic fragment scanner

Files:
- Modify: app/adapters/evidence_classification.py
- Test: tests/unit/test_m2_t03_evidence_classification_provider.py

Interfaces: Keep _source_fragments(source, source_name) returning list[tuple[str, Literal["title", "abstract"], int]]; add only private scanner helpers and constants.

- [ ] Step 1: Replace the sentence regex with a left-to-right scanner

Implement these private rules:

~~~python
_PROTECTED_DOT_TOKENS = frozenset(
    {"e.g.", "i.e.", "et al.", "fig.", "dr.", "vs.", "etc.", "al."}
)

def _dotted_token_ending_at(source: str, period_index: int) -> str:
    start = period_index
    while start > 0 and (
        source[start - 1].isalnum() or source[start - 1] == "."
    ):
        start -= 1
    return source[start : period_index + 1]

def _is_sentence_terminal_period(source: str, period_index: int) -> bool:
    if source[period_index] != ".":
        return False
    previous = source[period_index - 1] if period_index else ""
    following = source[period_index + 1] if period_index + 1 < len(source) else ""
    if previous.isalnum() and following.isalnum():
        return False
    token = _dotted_token_ending_at(source, period_index).casefold()
    if token in _PROTECTED_DOT_TOKENS:
        return False
    parts = token[:-1].split(".") if token.endswith(".") else []
    if len(parts) >= 2 and all(
        len(part) == 1 and part.isalpha() for part in parts
    ):
        return False
    return True
~~~

The scanner must finalize only on a genuine terminal period, !, ?, Chinese sentence punctuation, or newline. It must preserve the existing stripped-fragment text and adjust the returned offset by removed leading whitespace. Periods between ASCII alphanumeric characters cover decimals, versions, DOI/domain segments, and Clinfo.ai; protected token and initialism checks cover abbreviation endings. Do not normalize or rewrite source text.

- [ ] Step 2: Run scanner tests and existing focused tests

Run:

~~~powershell
& $py -m pytest -q tests/unit/test_m2_t03_evidence_classification_provider.py tests/contract/test_m2_t03_evidence_classification.py
~~~

Expected: all new boundary tests and existing provider/contracts pass, with no model imports or network calls.

- [ ] Step 3: Commit the scanner implementation

~~~powershell
git add app/adapters/evidence_classification.py tests/unit/test_m2_t03_evidence_classification_provider.py
git commit -m "fix: preserve dotted identifiers and decimal evidence"
~~~

---

### Task 3: Strengthen _EvidenceMatch source invariants

Files:
- Modify: app/adapters/evidence_classification.py
- Modify: tests/unit/test_m2_t03_evidence_classification_provider.py

Interfaces: _EvidenceMatch remains private but gains excerpt_start: int; _ScoredEvidenceMatch reuses evidence.excerpt_start for deterministic tie-breaking.

- [ ] Step 1: Return the bounded excerpt and its local offset together

Change _bounded_excerpt(fragment, marker_start) to return tuple[str, int], where the second value is the offset of the returned excerpt within fragment. Build each match with:

~~~python
excerpt, excerpt_local_start = _bounded_excerpt(fragment, primary_hit.start())
evidence = _EvidenceMatch(
    slot=slot,
    support_level=_select_support_level(excerpt),
    matched_marker=primary_hit.group(0),
    matched_source=source_name,
    supporting_excerpt=excerpt,
    specificity=sum(rule.specificity for rule, _ in matched),
    excerpt_start=source_offset + excerpt_local_start,
)
~~~

The selected support level must continue to receive excerpt, never the full title/abstract. Verify actual marker and excerpt are contained in the same fragment before constructing the immutable match; raise a private ValueError if an invariant is violated rather than fabricating a fallback.

- [ ] Step 2: Add direct invariant tests

For every _ScoredEvidenceMatch returned by _collect_matches(item), assert:

~~~python
source = item.title if evidence.matched_source == "title" else item.abstract
assert source is not None
assert source[
    evidence.excerpt_start : evidence.excerpt_start + len(evidence.supporting_excerpt)
] == evidence.supporting_excerpt
assert evidence.matched_marker.casefold() in evidence.supporting_excerpt.casefold()
~~~

Add a cross-sentence case where an abstract contains Cross-domain in one sentence and demonstrates in the next; assert the chosen indirect record's excerpt is only the first sentence and its support level remains INDIRECT. Retain generic-only and equal-low-specificity rejection assertions.

- [ ] Step 3: Run and commit the invariant changes

~~~powershell
& $py -m pytest -q tests/unit/test_m2_t03_evidence_classification_provider.py tests/contract/test_m2_t03_evidence_classification.py
git add app/adapters/evidence_classification.py tests/unit/test_m2_t03_evidence_classification_provider.py
git commit -m "test: enforce source-bound evidence match offsets"
~~~

---

### Task 4: Regenerate only the contract-only deterministic fake evidence

Files:
- Modify: evaluation/source-artifacts/m2-t03-evidence-classification-run.json
- Modify: evaluation/source-artifacts/m2-t03-evidence-classification-run-receipt.json
- Modify: evaluation/reports/m2-t03-evidence-classification.json

Interfaces: Use the existing runner and publication conflict protection; do not edit M2-T01/M2-T02 artifacts.

- [ ] Step 1: Record the three historical M2-T02 SHA-256 values

Require these exact results:

~~~text
0a751dbc35bfa8d07433796113939412bfbdffd4c5c13043c90390bccfc5c35b
009f4ad441b80a597b955c40ff6d31aac13476adf6574540bf3f4a790192361a
0c9ca1339a2385a9c6940f8c53bd4711c62b8faf9f3f4cf98710ff465d56ec9a
~~~

- [ ] Step 2: Run only the deterministic fake runner

~~~powershell
& $py scripts/run_m2_t03_evidence_classification.py --execute-deterministic-fake
~~~

Expected: result and receipt bind to the current execution commit; real model, human review, embedding, reranker, and download paths are not invoked.

- [ ] Step 3: Validate contract-only report fields

Require completion_scope="contract_only", scoring_eligible is false, not_run.real_model is true, not_run.human_review is true, human_judged is false, M2 IN_PROGRESS 2/5, and next task M2-T03 quality closure. Record observed distributions without changing rules to improve them.

- [ ] Step 4: Commit the refreshed evidence

~~~powershell
git add evaluation/source-artifacts/m2-t03-evidence-classification-run.json evaluation/source-artifacts/m2-t03-evidence-classification-run-receipt.json evaluation/reports/m2-t03-evidence-classification.json
git commit -m "chore: refresh contract-only M2-T03 evidence"
~~~

---

### Task 5: Run the complete local validation gate

Files: No source changes; validate Tasks 2–4.

- [ ] Step 1: Run focused and full tests

~~~powershell
& $py -m pytest -q tests/unit/test_m2_t03_evidence_classification_provider.py tests/unit/test_m2_t03_evidence_classification_runner.py tests/contract/test_m2_t03_evidence_classification.py tests/contract/test_m2_t03_evidence_classification_evidence.py tests/contract/test_project_docs_validation.py
& $py -m pytest -q -m "not packaging"
& $py -m pytest -q -m packaging
~~~

- [ ] Step 2: Run static, evidence, dependency, and diff checks

~~~powershell
& $py -m ruff check app evaluation scripts tests
& $py -m ruff check --select I app evaluation scripts tests
& $py -m mypy app evaluation scripts
& $py scripts/validate_project_docs.py
& $py scripts/validate_phase.py M0
& $py scripts/validate_m1_evidence.py
& $py scripts/validate_m2_t01_evidence.py
& $py scripts/validate_m2_t02_evidence.py
& $py scripts/validate_m2_t03_evidence.py
& $py -m pip check
git diff --check
~~~

Expected: every command exits 0; record actual focused/full/packaging totals and all new artifact/report hashes for the PR body.

---

### Task 6: Update PR #13 without merging it

Files: PR #13 body only; no public schema or status promotion.

- [ ] Step 1: Push the independent commits

~~~powershell
git push origin agent/m2-t03-evidence-slot-classification
~~~

Use a non-force push and wait for the push and pull-request docs-validation runs for the exact new head.

- [ ] Step 2: Update the PR body

Place this status block at the top:

~~~text
M2-T03 status: CONTRACT FOUNDATION READY
M2 progress: IN_PROGRESS 2/5
next task: M2-T03 quality closure
scoring eligible: false
M3: BLOCKED_BY_M2 0/5
~~~

Then report the new head, scanner behavior for Clinfo.ai, 59.70%, abbreviations, versions/URLs/DOIs, and genuine sentence splitting; result/receipt/report SHA-256 values; observed distributions; rejections and warnings; local test totals; remote CI run IDs; real_model=not_run; human_review=not_run; and M2-T04 not started. State that merging means only the M2-T03 contract/fake harness/evidence infrastructure is merged and does not complete quality or unlock M2-T04.

- [ ] Step 3: Mark PR #13 Ready only after remote gates are green

Confirm no unresolved review threads, then run gh pr ready 13. Do not merge the PR. Verify state=OPEN, isDraft=false, current head matches the pushed commit, and the local worktree is clean.

---

## Self-review checklist

- Scientific fragment cases cover dotted identifiers, decimals, abbreviations, versions, DOI/domain strings, true sentence endings, source offsets, exact substrings, and order independence.
- Scanner remains deterministic and stdlib-only; it does not disable sentence splitting globally.
- _EvidenceMatch support level and excerpt are derived from the same fragment, with absolute source offset and actual marker text.
- Fake evidence remains contract-only and scoring-ineligible; M2-T04/M2-T05/model/human paths stay untouched.
- Historical M2-T02 hashes, local gates, remote CI, review-thread state, and Draft-to-Ready transition are verified before PR state changes.
