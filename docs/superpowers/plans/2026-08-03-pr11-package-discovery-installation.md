# PR #11 Package Discovery and Installation Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the PR #11 package-discovery review finding by shipping `scripts` in wheel and editable installs, validating the installed artifact in an isolated Python 3.12 environment, and correcting the stale README state without touching M2-T02 formal model evidence.

**Architecture:** Configure the standard setuptools build backend and include `app*`, `evaluation*`, and `scripts*` packages. A packaging contract test copies the tracked source into a temporary build directory, builds a wheel there, creates a separate temporary virtual environment, installs the wheel with its declared runtime dependencies, and imports the two application packages plus both requested script modules from outside the source tree. CI runs source tests and the packaging contract as separate gates.

**Tech Stack:** Python 3.12, setuptools, wheel, `python -m build`, pytest, Ruff, mypy, GitHub Actions, and the existing M0/M1/M2-T01/M2-T02 validators.

## Global Constraints

- Work only on the existing `agent/m2-t02-reranker-provider` branch and PR #11.
- Do not start M2-T03 or modify M2-T02 model runners, candidate result, receipt, fixed evidence inputs, STATUS.md, or model files.
- Do not download, load, or run any model; the packaging test may install only the wheel's declared lightweight runtime dependencies.
- The packaging test must build in a temporary source copy and install into a fresh temporary virtual environment, never the repository environment.
- Preserve the current M2-T02 formal artifact bytes and verify their SHA-256 values before and after implementation.
- Do not squash merge, rebase merge, auto-merge, or rewrite shared history.

---

### Task 1: Reproduce and fix setuptools package discovery

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/contract/test_package_installation.py`

**Interfaces:**
- Consumes: the repository's existing setuptools package layout and runtime dependencies.
- Produces: a wheel containing `app`, `evaluation`, and `scripts`, with the requested script modules importable after installation.

- [x] **Step 1: Add the packaging contract test before changing discovery**

Create `tests/contract/test_package_installation.py` with a `packaging` pytest marker. The test copies the repository while excluding only repository-local build/cache/model directories, runs:

```text
python -m build --wheel --no-isolation --outdir <temporary-wheel-dir>
```

from the temporary source copy, creates a fresh `venv`, installs the generated wheel with `pip install <wheel>`, and runs this import probe from outside the source copy with `PYTHONPATH` removed:

```python
import app
import scripts
import scripts.run_m2_t02_candidate_reranking
import scripts.validate_m2_t02_evidence
```

The test must assert that exactly one wheel was produced, every subprocess exits zero, and the import probe resolves modules from the installed environment rather than the source copy.

- [x] **Step 2: Run the focused packaging test to capture the review failure**

Run:

```text
py -3.12 -m pytest -q tests/contract/test_package_installation.py
```

Expected: the test fails at the wheel/import assertion because the existing package discovery include list omits `scripts`.

- [x] **Step 3: Configure the standard setuptools build and package include list**

In `pyproject.toml`, add:

```toml
[build-system]
requires = ["setuptools>=68,<81", "wheel>=0.43,<1"]
build-backend = "setuptools.build_meta"
```

Change the package discovery include list to:

```toml
[tool.setuptools.packages.find]
include = ["app*", "evaluation*", "scripts*"]
```

Add `build>=1.2,<2` and `wheel>=0.43,<1` to the `dev` optional dependency set so local and CI packaging checks use declared build tooling.

- [x] **Step 4: Run the focused packaging test after the minimal fix**

Run:

```text
py -3.12 -m pytest -q tests/contract/test_package_installation.py
```

Expected: one packaging test passes, and the import probe succeeds for `app`, `scripts`, `scripts.run_m2_t02_candidate_reranking`, and `scripts.validate_m2_t02_evidence` without importing a model runtime.

### Task 2: Separate source and built-artifact CI gates

**Files:**
- Modify: `.github/workflows/docs-validation.yml`

**Interfaces:**
- Consumes: the existing validation dependency installation and the `packaging` pytest marker.
- Produces: one source-only test gate and one explicit installed-wheel gate in the same CI workflow.

- [x] **Step 1: Install the build tools in CI**

Extend the existing validation installation command with:

```text
"build>=1.2,<2"
"setuptools>=68,<81"
"wheel>=0.43,<1"
```

- [x] **Step 2: Split source tests from packaging validation**

Change the source test step to:

```yaml
- name: Run source tests
  run: python -m pytest -q -m "not packaging"
```

Add a separate step:

```yaml
- name: Build and verify installed wheel
  run: python -m pytest -q -m packaging
```

Keep Ruff, mypy, documentation validation, all phase/evidence validators, and `pip check` as independent steps.

- [x] **Step 3: Run the workflow-equivalent local commands**

Run source tests and the packaging test separately:

```text
py -3.12 -m pytest -q -m "not packaging"
py -3.12 -m pytest -q -m packaging
```

Expected: both commands pass and no build or virtual-environment residue appears in the repository.

### Task 3: Remove the stale README phase claim

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the authoritative current state in `STATUS.md`.
- Produces: a README statement that explicitly delegates phase, task, and gate status to `STATUS.md`.

- [x] **Step 1: Replace the stale M0 paragraph**

Replace the sentence claiming the project is at the M0 executable starting point with a sentence stating that current phase, task progress, and execution gates are authoritative only in `STATUS.md`. Keep the `STATUS.md` reading link and do not duplicate a hard-coded phase count in README.

- [x] **Step 2: Verify README and project documentation**

Run:

```text
py -3.12 scripts/validate_project_docs.py
```

Expected: the project documentation validator passes and README no longer contains the stale `M0` current-state claim.

### Task 4: Complete local validation and PR review closure

**Files:**
- Modify: `docs/superpowers/plans/2026-08-03-pr11-package-discovery-installation.md`
- Update remotely: PR #11 body and its existing `pyproject.toml` review thread

**Interfaces:**
- Consumes: the implementation changes and their local validation output.
- Produces: a new non-evidence fix commit, pushed PR HEAD, updated CI counts, a resolved review thread, and a requested follow-up review; no merge.

- [x] **Step 1: Run the complete required validation set without model execution**

Run:

```text
py -3.12 -m pytest -q
py -3.12 -m ruff check app evaluation scripts tests
py -3.12 -m ruff check --select I app evaluation scripts tests
py -3.12 -m mypy app evaluation scripts
py -3.12 scripts/validate_project_docs.py
py -3.12 scripts/validate_phase.py M0
py -3.12 scripts/validate_m1_evidence.py
py -3.12 scripts/validate_m2_t01_evidence.py
py -3.12 scripts/validate_m2_t02_evidence.py
py -3.12 -m pip check
```

Expected: every command exits zero; no reranker runner, preflight runner, replay, or model operation is invoked.

- [x] **Step 2: Verify scope and evidence immutability**

Run `git diff --check`, inspect `git diff --name-only`, and recompute SHA-256 for the candidate result, candidate receipt, fixed candidate snapshot/manifest, preflight evidence/receipt, selection, runtime installation, snapshot-download evidence, completion report, and candidate runner. Their hashes must equal the pre-change baseline.

- [ ] **Step 3: Commit and push only the packaging/documentation fix**

Stage only `pyproject.toml`, `tests/contract/test_package_installation.py`, `.github/workflows/docs-validation.yml`, `README.md`, and this plan. Commit with:

```text
fix: include scripts in built package validation
```

Push with a fast-forward update to `origin/agent/m2-t02-reranker-provider`; do not amend, rebase, or force-push.

- [ ] **Step 4: Verify remote CI and update the PR top**

Wait for the new HEAD's CI to finish. Update the PR body top with the new full HEAD, source-test count, packaging-test count, total CI count, all required validators, and an explicit `merge commit only` statement. Preserve the M2-T02 completion/evidence hashes and state that M2-T03 has not started.

- [ ] **Step 5: Resolve the existing review thread and request re-review**

Reply to the Copilot `pyproject.toml` thread with the implemented package-discovery and isolated-wheel validation summary, resolve thread `PRRT_kwDOTjvfqM6VxhCO`, and request a fresh review for the new HEAD. Verify the thread is resolved, the PR remains open/Ready for review, and no merge was performed.
