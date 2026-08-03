# Sanitized Download Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve a closed, non-sensitive category for a failed `hf_hub_download` call through the controlled runner, so a future authorized attempt can be diagnosed without exposing its raw exception.

**Architecture:** The preparation helper will classify the caught downloader exception using only its exception type and an integer HTTP status attribute when present; it will never inspect or retain an exception message, URL, response body, filesystem path, header, or token. `SnapshotPreparationError` will carry the optional closed category, and the runner will transfer it to its own closed error and print only `PREPARATION_FAILED` plus that category.

**Tech Stack:** Python 3.12, standard-library exception inspection, pytest, Ruff, mypy.

## Global Constraints

- Do not invoke the live downloader, make a network request, retry the failed download, load a tokenizer/model, or perform inference.
- Keep the existing public failure codes unchanged; diagnostics must be optional closed literals.
- Do not serialize raw underlying exceptions, messages, URLs, credentials, headers, response bodies, or paths.
- Retain exception chaining suppression at the controlled public boundary.

---

### Task 1: Add a sanitized downloader failure category

**Files:**
- Modify: `scripts/prepare_m2_t02_reranker_snapshot.py:19-103,716-728`
- Test: `tests/unit/test_m2_t02_reranker_snapshot.py`

**Interfaces:**
- Consumes: the exception raised by injected `download_file`.
- Produces: `SnapshotPreparationError(code="DOWNLOAD_FAILED", diagnostic=<closed literal>)`.

- [x] **Step 1: Write failing tests**

```python
with pytest.raises(module.SnapshotPreparationError) as raised:
    module.prepare_snapshot(..., download_file=downloader_that_raises)

assert raised.value.code == "DOWNLOAD_FAILED"
assert raised.value.diagnostic == "TLS_OR_CERTIFICATE_FAILURE"
assert "secret" not in str(raised.value)
```

- [x] **Step 2: Verify the new test fails**

Run: `py -3.12 -m pytest -q tests/unit/test_m2_t02_reranker_snapshot.py -k sanitized`

Expected: FAIL because `SnapshotPreparationError` has no `diagnostic` attribute.

- [x] **Step 3: Implement the closed classifier and error field**

```python
SnapshotDownloadDiagnostic = Literal[
    "HTTP_4XX", "HTTP_5XX", "NETWORK_TIMEOUT",
    "TLS_OR_CERTIFICATE_FAILURE", "NETWORK_CONNECTION_FAILED",
    "DOWNLOAD_EXCEPTION",
]

def _download_failure_diagnostic(exc: BaseException) -> SnapshotDownloadDiagnostic:
    # Use only status code and exception type name; never call str(exc).
    ...
```

Pass the classifier result only from the `download_file` exception handler to `_download_error`.

- [x] **Step 4: Run the focused preparation tests**

Run: `py -3.12 -m pytest -q tests/unit/test_m2_t02_reranker_snapshot.py`

Expected: PASS.

### Task 2: Transfer and render the sanitized category at the runner boundary

**Files:**
- Modify: `scripts/download_m2_t02_reranker_snapshot.py:24-89,385-393,444-456`
- Test: `tests/unit/test_m2_t02_reranker_snapshot_download_runner.py`

**Interfaces:**
- Consumes: `SnapshotPreparationError.diagnostic` from Task 1.
- Produces: `DownloadRunnerError(code="PREPARATION_FAILED", diagnostic=<same closed literal>)` and a stderr line containing no other details.

- [x] **Step 1: Write failing runner test**

```python
with pytest.raises(module.DownloadRunnerError) as raised:
    module.run_live_download(execute_live_download=True)

assert raised.value.code == "PREPARATION_FAILED"
assert raised.value.diagnostic == "TLS_OR_CERTIFICATE_FAILURE"
assert "secret" not in str(raised.value)
```

- [x] **Step 2: Verify the new test fails**

Run: `py -3.12 -m pytest -q tests/unit/test_m2_t02_reranker_snapshot_download_runner.py -k sanitized`

Expected: FAIL because the runner does not yet preserve the preparation diagnostic.

- [x] **Step 3: Implement the runner transfer and safe rendering**

```python
except SnapshotPreparationError as exc:
    raise DownloadRunnerError("PREPARATION_FAILED", exc.diagnostic) from None
```

Give `DownloadRunnerError` an optional matching field and render only `code` or `code:diagnostic` in `main`.

- [x] **Step 4: Run runner and contract tests**

Run: `py -3.12 -m pytest -q tests/unit/test_m2_t02_reranker_snapshot_download_runner.py tests/contract/test_m2_t02_reranker_snapshot_download.py`

Expected: PASS.

### Task 3: Validate the isolated diagnostic change

**Files:**
- Modify: `scripts/prepare_m2_t02_reranker_snapshot.py`, `scripts/download_m2_t02_reranker_snapshot.py`, `tests/unit/test_m2_t02_reranker_snapshot.py`, `tests/unit/test_m2_t02_reranker_snapshot_download_runner.py`

**Interfaces:**
- Consumes: source and test changes from Tasks 1-2.
- Produces: verified offline diagnostic handling with no live-run side effects.

- [x] **Step 1: Run the focused snapshot and runner suites**

Run: `py -3.12 -m pytest -q tests/unit/test_m2_t02_reranker_snapshot.py tests/unit/test_m2_t02_reranker_snapshot_download_runner.py tests/contract/test_m2_t02_reranker_snapshot_download.py`

Expected: PASS with no network calls.

- [x] **Step 2: Run static checks on the touched production modules**

Run: `py -3.12 -m ruff check scripts/prepare_m2_t02_reranker_snapshot.py scripts/download_m2_t02_reranker_snapshot.py tests/unit/test_m2_t02_reranker_snapshot.py tests/unit/test_m2_t02_reranker_snapshot_download_runner.py`

Expected: `All checks passed!`

Run: `py -3.12 -m mypy scripts/prepare_m2_t02_reranker_snapshot.py scripts/download_m2_t02_reranker_snapshot.py`

Expected: `Success: no issues found`.

- [x] **Step 3: Inspect the final diff and repository state**

Run: `git diff --check; git status --short`

Expected: no whitespace errors; only the plan and explicitly scoped source/test files are modified.
