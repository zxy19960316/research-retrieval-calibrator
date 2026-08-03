"""Validate the completed M2-T02 reranker evidence without model runtime imports.

The summary report is deliberately a closed, evidence-only projection.  The
validator binds every declared input to the Git blob at ``validated_commit``
before it calls the production preflight and candidate-run validators.  It
never loads a model, imports torch/transformers, accesses ``models/**``, or
performs network I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_m2_t02_candidate_reranking import (
    validate_candidate_run_report,
)
from scripts.run_m2_t02_reranker_preflight import (
    validate_preflight_evidence,
)

REPORT_PATH = Path("evaluation/reports/m2-t02-reranker.json")
REPORT_VERSION = "m2-t02-reranker.v1"
BASELINE_COMMIT = "cdadca0ad4b49098aa115a9c05b9728ccd8eb7a0"
IMPLEMENTATION_COMMIT = "5a7cf46f52b20dd0b1af92ea29a25639995ff01e"
LIVE_EXECUTION_COMMIT = "c1f95238480d8cfe69a79f705d98caadbf372220"
EVIDENCE_COMMIT = "5195a4eabfccdd64dac61f2efdc2dfc4a92242ae"
VALIDATED_COMMIT = EVIDENCE_COMMIT

RESULT_PATH = "evaluation/source-artifacts/m2-t02-reranker-candidate-run.json"
RECEIPT_PATH = (
    "evaluation/source-artifacts/m2-t02-reranker-candidate-run-receipt.json"
)
RUNNER_PATH = "scripts/run_m2_t02_candidate_reranking.py"
PREFLIGHT_PATH = "evaluation/source-artifacts/m2-t02-reranker-preflight.json"
PREFLIGHT_RECEIPT_PATH = (
    "evaluation/source-artifacts/m2-t02-reranker-preflight-run-receipt.json"
)

RESULT_SHA256 = "0a751dbc35bfa8d07433796113939412bfbdffd4c5c13043c90390bccfc5c35b"
RECEIPT_SHA256 = "0c9ca1339a2385a9c6940f8c53bd4711c62b8faf9f3f4cf98710ff465d56ec9a"
RUNNER_SHA256 = "3ad6c629b2f6b1f28c3119f9f7d810226e28e6d572535e28b0de6566ef8f63b1"
PREFLIGHT_EXECUTION_COMMIT = "a50f1b9f5467c318dbbabe8807df6ede47d79b14"

VALIDATED_INPUT_HASHES = {
    "evaluation/snapshots/m2/m1-candidates.v1.json": "4a2aec0fd0a1d22adc801fd3bc506e5da89895d1276cd572e2ac64014c162448",
    "evaluation/snapshots/m2/m1-candidates.v1.manifest.json": "b6ce7cfab2e5df6b8c84b77fb38c6f573de7429d37a9688d7b4c62e27f8546a8",
    "evaluation/source-artifacts/m2-t02-reranker-selection.json": "19e86cf2176969236e70ff9770ac086a5555d40892c0bfa913ac768eb752604b",
    "evaluation/source-artifacts/m2-t02-reranker-runtime-installation.json": "feb79ad507afb4bb7027e5aa49765b941414fe6e9a9e3ac024e7eba6d2ced831",
    "evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json": "5114ce6ef2ca637f950e1683bfc7003029daff4f0f228ca6b909d0c97fb84bf8",
    PREFLIGHT_PATH: "103a73cc82674c2c1f1139a72a7a3d1a880029a7c01d292f52de24d6dc781b06",
    PREFLIGHT_RECEIPT_PATH: "fe5ea3c239de9f1f62bdc0204c812f27984af254b08ab39a125e4ca709e05d28",
    RESULT_PATH: RESULT_SHA256,
    RECEIPT_PATH: RECEIPT_SHA256,
    RUNNER_PATH: RUNNER_SHA256,
}

STATUS_ROW = re.compile(
    r"^\|\s*(M\d+)\b[^|]*\|\s*([A-Z0-9_]+)\s*\|\s*(\d+)/(\d+)\s*\|",
    re.MULTILINE,
)
SHA1 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")

REPORT_FIELDS = {
    "report_version",
    "phase",
    "task_id",
    "baseline_commit",
    "implementation_commit",
    "live_execution_commit",
    "evidence_commit",
    "validated_commit",
    "validated_inputs",
    "automated_checks",
    "fake_evidence",
    "real_model_evidence",
    "acceptance",
    "not_run",
    "status_after_evidence",
}
RECEIPT_FIELDS = {
    "report_version",
    "phase",
    "task_id",
    "execution_commit",
    "result_path",
    "result_sha256",
    "runner_path",
    "runner_sha256",
    "candidate_snapshot_path",
    "candidate_snapshot_sha256",
    "candidate_manifest_path",
    "candidate_manifest_sha256",
    "model_selection_path",
    "model_selection_sha256",
    "runtime_installation_path",
    "runtime_installation_sha256",
    "snapshot_download_evidence_path",
    "snapshot_download_evidence_sha256",
    "preflight_evidence_path",
    "preflight_evidence_sha256",
    "preflight_receipt_path",
    "preflight_receipt_sha256",
    "runtime",
    "exit_code",
    "run_count",
    "decision_status",
    "formal_result_created",
    "full_candidate_reranking_run",
    "additional_model_rerun_performed",
    "replay_performed",
}
EXPECTED_RUNTIME = {
    "python_version": "3.12.10",
    "torch_distribution_version": "2.4.1+cpu",
    "transformers_version": "4.53.2",
    "huggingface_hub_version": "0.34.3",
    "safetensors_version": "0.5.3",
    "tokenizers_version": "0.21.2",
    "pydantic_version": "2.9.2",
    "torch_cuda_version": None,
    "torch_cuda_available": False,
}
EXPECTED_EXECUTION = {
    "provider_instance_count": 1,
    "runtime_load_count": 1,
    "provider_call_count": 17,
    "provider_scored_count": 33,
    "batch_sizes": [2] * 16 + [1],
    "cache_hits": 0,
}
EXPECTED_ACCEPTANCE = {
    "batch_partition_invariance": "passed",
    "provider_failure_fail_closed": "passed",
    "partial_scores_not_published": "passed",
    "raw_scores_preserved": "passed",
    "normalization_method": "global_min_max",
    "deterministic_tie_break": "paper_id_ascending",
    "fixed_candidate_count": 33,
    "real_model_run": "passed",
}
EXPECTED_NOT_RUN = {
    "additional_model_rerun": False,
    "b3_replay": False,
    "benchmark": False,
    "manual_relevance_judgement": False,
    "m2_t03_started": False,
}
CHECK_NAMES = {
    "focused_evidence_tests",
    "full_regression",
    "ruff",
    "ruff_import_order",
    "mypy",
    "project_docs_validation",
    "m0_evidence_validation",
    "m1_evidence_validation",
    "m2_t01_evidence_validation",
    "m2_t02_evidence_validation",
    "pip_check",
}

PREFLIGHT_RECEIPT = {
    "report_version": "m2-t02-reranker-preflight-run-receipt.v1",
    "phase": "M2",
    "task_id": "M2-T02",
    "execution_commit": PREFLIGHT_EXECUTION_COMMIT,
    "evidence_path": PREFLIGHT_PATH,
    "evidence_sha256": VALIDATED_INPUT_HASHES[PREFLIGHT_PATH],
    "runner_path": "scripts/run_m2_t02_reranker_preflight.py",
    "runner_sha256": "38ec370acd26a315fdb57bd405e9348081e37ea5f26bebb362cfcd3febd930d7",
    "selection_path": "evaluation/source-artifacts/m2-t02-reranker-selection.json",
    "selection_sha256": VALIDATED_INPUT_HASHES[
        "evaluation/source-artifacts/m2-t02-reranker-selection.json"
    ],
    "runtime_installation_path": "evaluation/source-artifacts/m2-t02-reranker-runtime-installation.json",
    "runtime_installation_sha256": VALIDATED_INPUT_HASHES[
        "evaluation/source-artifacts/m2-t02-reranker-runtime-installation.json"
    ],
    "snapshot_download_evidence_path": "evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json",
    "snapshot_download_evidence_sha256": VALIDATED_INPUT_HASHES[
        "evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json"
    ],
    "exit_code": 0,
    "run_count": 1,
    "decision_status": "cpu_float32_preflight_passed",
    "formal_evidence_created": True,
    "model_rerun_performed": False,
    "full_candidate_reranking_run": False,
}


@dataclass(frozen=True)
class EvidenceValidationResult:
    valid: bool
    errors: list[str]


def _git(*arguments: str, repository_root: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )


def _blob_bytes(commit: str, path: str, repository_root: Path) -> bytes | None:
    result = _git("show", f"{commit}:{path.replace(chr(92), '/')}", repository_root=repository_root)
    return result.stdout if result.returncode == 0 else None


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _mapping(value: object) -> dict[str, object] | None:
    return dict(value) if isinstance(value, dict) else None


def _json_blob(raw: bytes | None, label: str, errors: list[str]) -> dict[str, object] | None:
    if raw is None:
        errors.append(f"missing Git blob for {label}")
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        errors.append(f"{label} must be UTF-8 JSON")
        return None
    result = _mapping(value)
    if result is None:
        errors.append(f"{label} must be a JSON object")
    return result


def _relative_identifier(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    candidate = Path(value)
    windows_candidate = PureWindowsPath(value)
    return not (
        candidate.is_absolute()
        or windows_candidate.is_absolute()
        or ".." in candidate.parts
        or ".." in windows_candidate.parts
    )


def _is_ancestor(commit: object, repository_root: Path) -> bool:
    return (
        isinstance(commit, str)
        and SHA1.fullmatch(commit) is not None
        and _git("merge-base", "--is-ancestor", commit, "HEAD", repository_root=repository_root).returncode
        == 0
    )


def _parse_status(repository_root: Path, errors: list[str]) -> dict[str, tuple[str, int, int]]:
    try:
        text = (repository_root / "STATUS.md").read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"cannot read STATUS.md: {error}")
        return {}
    phases: dict[str, tuple[str, int, int]] = {}
    for phase, state, completed, total in STATUS_ROW.findall(text):
        if phase in phases:
            errors.append(f"STATUS.md declares duplicate current state for {phase}")
        else:
            phases[phase] = (state, int(completed), int(total))
    return phases


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def _validate_privacy(value: object, errors: list[str]) -> None:
    for text in _strings(value):
        lowered = text.lower()
        if any(
            forbidden in lowered
            for forbidden in (
                "http://",
                "https://",
                "file://",
                "authorization",
                "cookie",
                "bearer",
                "access_token",
                "api_key",
                "password",
                "hostname",
                "username",
                "/users/",
                "/home/",
            )
        ) or ":\\" in text:
            errors.append("evidence summary contains forbidden private or absolute-path text")
            return


def _validate_commits(payload: Mapping[str, object], errors: list[str], repository_root: Path) -> None:
    expected = {
        "baseline_commit": BASELINE_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "live_execution_commit": LIVE_EXECUTION_COMMIT,
        "evidence_commit": EVIDENCE_COMMIT,
        "validated_commit": VALIDATED_COMMIT,
    }
    for field, fixed in expected.items():
        value = payload.get(field)
        if value != fixed:
            errors.append(f"{field} must equal its fixed M2-T02 commit")
        if not _is_ancestor(value, repository_root):
            errors.append(f"{field} must be a HEAD ancestor")


def _validate_inputs(
    payload: Mapping[str, object], errors: list[str], repository_root: Path
) -> dict[str, bytes]:
    raw_inputs = payload.get("validated_inputs")
    if not isinstance(raw_inputs, list) or len(raw_inputs) != len(VALIDATED_INPUT_HASHES):
        errors.append("validated_inputs must contain the exact closed input set")
        return {}
    blobs: dict[str, bytes] = {}
    seen: set[str] = set()
    validated_commit = payload.get("validated_commit")
    for item in raw_inputs:
        entry = _mapping(item)
        if entry is None or set(entry) != {"path", "sha256"}:
            errors.append("each validated input must contain only path and sha256")
            continue
        path = entry["path"]
        digest = entry["sha256"]
        if not _relative_identifier(path):
            errors.append("validated input paths must be relative POSIX identifiers")
            continue
        if path in seen or path not in VALIDATED_INPUT_HASHES:
            errors.append("validated_inputs contains an unexpected or duplicate path")
            continue
        seen.add(path)
        expected_digest = VALIDATED_INPUT_HASHES[path]
        if digest != expected_digest or not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            errors.append(f"validated input hash mismatch for {path}")
        blob = _blob_bytes(str(validated_commit), path, repository_root)
        if blob is None:
            errors.append(f"validated_inputs Git blob is unavailable for {path}")
            continue
        blobs[path] = blob
        if _sha256(blob) != digest or _sha256(blob) != expected_digest:
            errors.append(f"validated_inputs Git blob hash mismatch for {path}")
        current = repository_root / path
        try:
            if not current.is_file() or current.read_bytes() != blob:
                errors.append(f"working-tree bytes differ from validated Git blob for {path}")
        except OSError:
            errors.append(f"working-tree input is unavailable for {path}")
    if seen != set(VALIDATED_INPUT_HASHES):
        errors.append("validated_inputs does not contain every required evidence input")
    return blobs


def _validate_receipt(receipt: Mapping[str, object], errors: list[str]) -> None:
    if set(receipt) != RECEIPT_FIELDS:
        errors.append("candidate execution receipt has an open schema")
        return
    if (
        receipt.get("report_version") != "m2-t02-reranker-candidate-run-receipt.v1"
        or receipt.get("phase") != "M2"
        or receipt.get("task_id") != "M2-T02"
        or receipt.get("execution_commit") != LIVE_EXECUTION_COMMIT
        or receipt.get("exit_code") != 0
        or receipt.get("run_count") != 1
        or receipt.get("decision_status") != "candidate_reranking_completed"
        or receipt.get("formal_result_created") is not True
        or receipt.get("full_candidate_reranking_run") is not True
        or receipt.get("additional_model_rerun_performed") is not False
        or receipt.get("replay_performed") is not False
    ):
        errors.append("candidate execution receipt has invalid execution identity")
    path_fields = {
        "result_path": RESULT_PATH,
        "runner_path": RUNNER_PATH,
        "candidate_snapshot_path": "evaluation/snapshots/m2/m1-candidates.v1.json",
        "candidate_manifest_path": "evaluation/snapshots/m2/m1-candidates.v1.manifest.json",
        "model_selection_path": "evaluation/source-artifacts/m2-t02-reranker-selection.json",
        "runtime_installation_path": "evaluation/source-artifacts/m2-t02-reranker-runtime-installation.json",
        "snapshot_download_evidence_path": "evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json",
        "preflight_evidence_path": PREFLIGHT_PATH,
        "preflight_receipt_path": PREFLIGHT_RECEIPT_PATH,
    }
    for field, expected in path_fields.items():
        if receipt.get(field) != expected or not _relative_identifier(receipt.get(field)):
            errors.append(f"candidate execution receipt has invalid {field}")
    hash_fields = {
        "result_sha256": RESULT_PATH,
        "runner_sha256": RUNNER_PATH,
        "candidate_snapshot_sha256": "evaluation/snapshots/m2/m1-candidates.v1.json",
        "candidate_manifest_sha256": "evaluation/snapshots/m2/m1-candidates.v1.manifest.json",
        "model_selection_sha256": "evaluation/source-artifacts/m2-t02-reranker-selection.json",
        "runtime_installation_sha256": "evaluation/source-artifacts/m2-t02-reranker-runtime-installation.json",
        "snapshot_download_evidence_sha256": "evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json",
        "preflight_evidence_sha256": PREFLIGHT_PATH,
        "preflight_receipt_sha256": PREFLIGHT_RECEIPT_PATH,
    }
    for field, path in hash_fields.items():
        if receipt.get(field) != VALIDATED_INPUT_HASHES[path]:
            errors.append(f"candidate execution receipt has invalid {field}")
    if receipt.get("runtime") != EXPECTED_RUNTIME:
        errors.append("candidate execution receipt runtime does not match the pinned runtime")
    _validate_privacy(receipt, errors)


def _validate_preflight_receipt(receipt: Mapping[str, object], errors: list[str]) -> None:
    if receipt != PREFLIGHT_RECEIPT:
        errors.append("preflight execution receipt is not the fixed B2-P receipt")


def _validate_production_artifacts(
    blobs: Mapping[str, bytes], errors: list[str], repository_root: Path
) -> None:
    preflight = _json_blob(blobs.get(PREFLIGHT_PATH), PREFLIGHT_PATH, errors)
    result = _json_blob(blobs.get(RESULT_PATH), RESULT_PATH, errors)
    receipt = _json_blob(blobs.get(RECEIPT_PATH), RECEIPT_PATH, errors)
    preflight_receipt = _json_blob(
        blobs.get(PREFLIGHT_RECEIPT_PATH), PREFLIGHT_RECEIPT_PATH, errors
    )
    if preflight is not None:
        try:
            validate_preflight_evidence(preflight)
        except Exception:  # noqa: BLE001
            errors.append("production preflight validator rejected the validated blob")
    if result is not None:
        try:
            with _repository_cwd(repository_root):
                validate_candidate_run_report(result)
        except Exception:  # noqa: BLE001
            errors.append("production candidate-run validator rejected the validated blob")
    if receipt is not None:
        _validate_receipt(receipt, errors)
    if preflight_receipt is not None:
        _validate_preflight_receipt(preflight_receipt, errors)


@contextmanager
def _repository_cwd(repository_root: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(repository_root)
    try:
        yield
    finally:
        os.chdir(previous)


def _validate_checks(payload: Mapping[str, object], errors: list[str]) -> None:
    checks = payload.get("automated_checks")
    if not isinstance(checks, list):
        errors.append("automated_checks must be a list")
        return
    names: set[str] = set()
    expected_fields = {"name", "evidence_type", "status", "exit_code", "command", "result"}
    for raw_check in checks:
        check = _mapping(raw_check)
        if check is None or set(check) != expected_fields:
            errors.append("automated checks must use the closed check schema")
            continue
        name = check.get("name")
        if not isinstance(name, str) or name in names or name not in CHECK_NAMES:
            errors.append("automated checks contain an unexpected or duplicate name")
        elif check.get("evidence_type") != "automated":
            errors.append(f"automated check {name} has invalid evidence type")
        names.add(name if isinstance(name, str) else "")
        if check.get("status") != "passed" or check.get("exit_code") != 0:
            errors.append(f"automated check {name} did not pass")
        if not isinstance(check.get("command"), str) or not isinstance(check.get("result"), str):
            errors.append(f"automated check {name} lacks a closed command/result")
    if names != CHECK_NAMES:
        errors.append("automated_checks must contain every required machine gate")


def _validate_summary_sections(payload: Mapping[str, object], errors: list[str]) -> None:
    if payload.get("fake_evidence") != {
        "classification": "deterministic_fake",
        "status": "passed",
        "contract_paths": [
            "tests/contract/test_m2_t02_candidate_reranking.py",
            "tests/contract/test_m2_t02_candidate_reranking_evidence.py",
            "tests/unit/test_m2_t02_candidate_reranking_runner.py",
            "tests/unit/test_bge_reranker_provider.py",
        ],
    }:
        errors.append("fake_evidence must remain explicitly deterministic_fake and passed")
    if payload.get("acceptance") != EXPECTED_ACCEPTANCE:
        errors.append("acceptance does not match the fixed M2-T02 gates")
    if payload.get("not_run") != EXPECTED_NOT_RUN:
        errors.append("not_run must explicitly keep replay, rerun, benchmark, judgement, and M2-T03 false")
    if payload.get("status_after_evidence") != {
        "m2": "IN_PROGRESS 2/5",
        "m3": "BLOCKED_BY_M2 0/5",
        "next_task": "M2-T03",
    }:
        errors.append("status_after_evidence must point to M2-T03 with M3 blocked")

    real = _mapping(payload.get("real_model_evidence"))
    if real is None or set(real) != {
        "classification",
        "result_path",
        "result_sha256",
        "receipt_path",
        "receipt_sha256",
        "runner_path",
        "runner_sha256",
        "model_id",
        "model_revision",
        "runtime",
        "execution",
        "duration_seconds",
    }:
        errors.append("real_model_evidence has an invalid closed schema")
        return
    if (
        real.get("classification") != "real_local_cpu_float32"
        or real.get("result_path") != RESULT_PATH
        or real.get("result_sha256") != RESULT_SHA256
        or real.get("receipt_path") != RECEIPT_PATH
        or real.get("receipt_sha256") != RECEIPT_SHA256
        or real.get("runner_path") != RUNNER_PATH
        or real.get("runner_sha256") != RUNNER_SHA256
        or real.get("model_id") != "BAAI/bge-reranker-v2-m3"
        or real.get("model_revision") != "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
        or real.get("duration_seconds") != 79.034
    ):
        errors.append("real_model_evidence does not bind the fixed real run")
    if real.get("runtime") != {
        **EXPECTED_RUNTIME,
        "device": "cpu",
        "dtype": "float32",
        "local_files_only": True,
        "trust_remote_code": False,
    }:
        errors.append("real_model_evidence runtime is not CPU float32 local-only")
    if real.get("execution") != {
        "candidate_count": 33,
        "configured_top_k": 50,
        "effective_top_k": 33,
        "batch_size": 2,
        "batch_count": 17,
        "batch_sizes": [2] * 16 + [1],
        **EXPECTED_EXECUTION,
    }:
        errors.append("real_model_evidence execution does not match the fixed 33-candidate run")
    for path in (real.get("result_path"), real.get("receipt_path"), real.get("runner_path")):
        if not _relative_identifier(path):
            errors.append("real_model_evidence contains an unsafe path")
    _validate_privacy(real, errors)


def _validate_status(payload: Mapping[str, object], errors: list[str], repository_root: Path) -> None:
    phases = _parse_status(repository_root, errors)
    if phases.get("M1") != ("COMPLETE", 4, 4):
        errors.append("M2-T02 completion must retain M1 COMPLETE 4/4")
    if phases.get("M2") != ("IN_PROGRESS", 2, 5):
        errors.append("M2-T02 completion requires M2 IN_PROGRESS 2/5")
    if phases.get("M3") != ("BLOCKED_BY_M2", 0, 5):
        errors.append("M2-T02 completion must retain M3 BLOCKED_BY_M2 0/5")
    if payload.get("status_after_evidence") != {
        "m2": "IN_PROGRESS 2/5",
        "m3": "BLOCKED_BY_M2 0/5",
        "next_task": "M2-T03",
    }:
        errors.append("report status_after_evidence does not match STATUS.md")


def validate_m2_t02_evidence(
    report_path: Path = REPORT_PATH, *, repository_root: Path = ROOT
) -> EvidenceValidationResult:
    """Validate the committed M2-T02 summary and all fixed evidence blobs."""

    errors: list[str] = []
    absolute_report = report_path if report_path.is_absolute() else repository_root / report_path
    try:
        report_raw = absolute_report.read_bytes()
    except OSError as error:
        return EvidenceValidationResult(False, [f"cannot load M2-T02 report: {error}"])
    report = _json_blob(report_raw, str(REPORT_PATH), errors)
    if report is None:
        return EvidenceValidationResult(False, errors)
    if set(report) != REPORT_FIELDS:
        errors.append("M2-T02 report must use the closed top-level schema")
    if (
        report.get("report_version") != REPORT_VERSION
        or report.get("phase") != "M2"
        or report.get("task_id") != "M2-T02"
    ):
        errors.append("M2-T02 report has invalid identity")
    _validate_commits(report, errors, repository_root)
    blobs = _validate_inputs(report, errors, repository_root)
    _validate_production_artifacts(blobs, errors, repository_root)
    _validate_checks(report, errors)
    _validate_summary_sections(report, errors)
    _validate_status(report, errors, repository_root)
    _validate_privacy(report, errors)
    return EvidenceValidationResult(not errors, errors)


def main() -> int:
    result = validate_m2_t02_evidence()
    for error in result.errors:
        print(f"ERROR: {error}")
    if not result.valid:
        return 1
    print("PASS: completed M2-T02 reranker evidence, input blobs, model result, and status gate are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
