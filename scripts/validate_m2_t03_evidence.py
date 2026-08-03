"""Validate committed M2-T03 fake evidence and its fixed input hashes."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_m2_t03_evidence_classification import (
    CANDIDATE_MANIFEST_PATH,
    CANDIDATE_SNAPSHOT_PATH,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_CANDIDATE_MANIFEST_SHA256,
    EXPECTED_CANDIDATE_SNAPSHOT_SHA256,
    EXPECTED_RERANKER_RECEIPT_SHA256,
    EXPECTED_RERANKER_RESULT_SHA256,
    FORMAL_RECEIPT_PATH,
    FORMAL_RESULT_PATH,
    RERANKER_RECEIPT_PATH,
    RERANKER_RESULT_PATH,
    validate_classification_receipt,
    validate_classification_run_report,
)

REPORT_PATH = Path("evaluation/reports/m2-t03-evidence-classification.json")
REPORT_VERSION = "m2-t03-evidence-classification.v1"
IMPLEMENTATION_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
STATUS_ROW = re.compile(
    r"^\|\s*(M\d+)\b[^|]*\|\s*([A-Z0-9_]+)\s*\|\s*(\d+)/(\d+)\s*\|",
    re.MULTILINE,
)

REPORT_FIELDS = {
    "report_version",
    "task_id",
    "implementation_commit",
    "validated_commit",
    "timestamp_utc",
    "python_version",
    "dependency_versions",
    "input_artifacts",
    "artifact",
    "record_count",
    "slot_distribution",
    "support_level_distribution",
    "title_only_count",
    "rejected_count",
    "evidence_types",
    "human_judged",
    "commands",
    "exit_codes",
    "test_totals",
    "not_run",
    "residual_risks",
    "status_after_evidence",
}
EXPECTED_INPUT_ARTIFACTS = {
    CANDIDATE_SNAPSHOT_PATH.as_posix(): EXPECTED_CANDIDATE_SNAPSHOT_SHA256,
    CANDIDATE_MANIFEST_PATH.as_posix(): EXPECTED_CANDIDATE_MANIFEST_SHA256,
    RERANKER_RESULT_PATH.as_posix(): EXPECTED_RERANKER_RESULT_SHA256,
    RERANKER_RECEIPT_PATH.as_posix(): EXPECTED_RERANKER_RECEIPT_SHA256,
}
EXPECTED_NOT_RUN = {
    "real_model": True,
    "human_review": True,
    "m2_t04": True,
    "m2_t05": True,
}
EXPECTED_STATUS = {
    "m2": "IN_PROGRESS 3/5",
    "m3": "BLOCKED_BY_M2 0/5",
    "next_task": "M2-T04",
}


@dataclass(frozen=True)
class EvidenceValidationResult:
    valid: bool
    errors: list[str]


def _git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, check=False
    )


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError
    return cast(dict[str, object], value)


def _load_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    return _as_mapping(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _validate_commit(value: object, errors: list[str], *, field: str) -> None:
    if not isinstance(value, str) or IMPLEMENTATION_COMMIT_RE.fullmatch(value) is None:
        errors.append(f"{field} is not a lowercase Git SHA-1")
        return
    if _git("merge-base", "--is-ancestor", value, "HEAD").returncode != 0:
        errors.append(f"{field} is not an ancestor of HEAD")


def _validate_privacy(value: object, errors: list[str]) -> None:
    forbidden = (
        "file://",
        "authorization",
        "cookie",
        "bearer ",
        "access_token",
        "api_key",
        "password",
        "raw exception",
        "traceback",
    )
    if isinstance(value, dict):
        for item in value.values():
            _validate_privacy(item, errors)
    elif isinstance(value, list):
        for item in value:
            _validate_privacy(item, errors)
    elif isinstance(value, str):
        if re.search(r"(?:^[A-Za-z]:[\\/]|^\\\\|/(?:Users|home|tmp)/)", value):
            errors.append("evidence contains an absolute local path")
        if any(token in value.casefold() for token in forbidden):
            errors.append("evidence contains forbidden credential or raw-error text")


def _validate_inputs(report: Mapping[str, object], errors: list[str]) -> None:
    raw_inputs = report.get("input_artifacts")
    if not isinstance(raw_inputs, list) or len(raw_inputs) != len(EXPECTED_INPUT_ARTIFACTS):
        errors.append("input_artifacts must contain the exact four fixed inputs")
        return
    seen: set[str] = set()
    for raw_item in raw_inputs:
        try:
            item = _as_mapping(raw_item)
            if set(item) != {"path", "sha256"}:
                raise ValueError
            path = item["path"]
            digest = item["sha256"]
            if not _relative_path(path) or path not in EXPECTED_INPUT_ARTIFACTS:
                raise ValueError
            if path in seen or digest != EXPECTED_INPUT_ARTIFACTS[path]:
                raise ValueError
            if _sha256(ROOT / path) != digest:
                raise ValueError
            seen.add(path)
        except (OSError, TypeError, ValueError, KeyError):
            errors.append("input_artifacts contains an invalid path or hash")
    if seen != set(EXPECTED_INPUT_ARTIFACTS):
        errors.append("input_artifacts does not cover every fixed input")


def _validate_artifacts(report: Mapping[str, object], errors: list[str]) -> None:
    try:
        artifact = _as_mapping(report["artifact"])
        if set(artifact) != {"result_path", "result_sha256", "receipt_path", "receipt_sha256"}:
            raise ValueError
        if (
            artifact["result_path"] != FORMAL_RESULT_PATH.as_posix()
            or artifact["receipt_path"] != FORMAL_RECEIPT_PATH.as_posix()
            or artifact["result_sha256"] != _sha256(ROOT / FORMAL_RESULT_PATH)
            or artifact["receipt_sha256"] != _sha256(ROOT / FORMAL_RECEIPT_PATH)
        ):
            raise ValueError
        result = _load_json(ROOT / FORMAL_RESULT_PATH)
        receipt = _load_json(ROOT / FORMAL_RECEIPT_PATH)
        validate_classification_run_report(result)
        validate_classification_receipt(receipt, report=result)
    except (OSError, TypeError, ValueError, KeyError):
        errors.append("formal M2-T03 artifact or receipt failed validation")


def _validate_commands(report: Mapping[str, object], errors: list[str]) -> None:
    commands = report.get("commands")
    exit_codes = report.get("exit_codes")
    if not isinstance(commands, list) or not isinstance(exit_codes, dict):
        errors.append("commands and exit_codes must be closed collections")
        return
    names: set[str] = set()
    for raw_command in commands:
        try:
            command = _as_mapping(raw_command)
            if set(command) != {"name", "command", "exit_code"}:
                raise ValueError
            name = command["name"]
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError
            if not isinstance(command["command"], str) or not command["command"].strip():
                raise ValueError
            if type(command["exit_code"]) is not int or command["exit_code"] != 0:
                raise ValueError
            if exit_codes.get(name) != 0:
                raise ValueError
            names.add(name)
        except (TypeError, ValueError, KeyError):
            errors.append("commands contain an invalid or non-passing entry")
    if set(exit_codes) != names or any(type(value) is not int for value in exit_codes.values()):
        errors.append("exit_codes do not match command identities")


def _validate_status(report: Mapping[str, object], errors: list[str]) -> None:
    if report.get("status_after_evidence") != EXPECTED_STATUS:
        errors.append("report status transition is not the M2-T03 handoff")
    try:
        status_text = (ROOT / "STATUS.md").read_text(encoding="utf-8")
        rows = {
            phase: (state, int(done), int(total))
            for phase, state, done, total in STATUS_ROW.findall(status_text)
        }
        if rows.get("M2") != ("IN_PROGRESS", 3, 5):
            errors.append("STATUS.md does not show M2 IN_PROGRESS 3/5")
        if rows.get("M3") != ("BLOCKED_BY_M2", 0, 5):
            errors.append("STATUS.md does not retain M3 BLOCKED_BY_M2 0/5")
        if "M2-T04" not in status_text:
            errors.append("STATUS.md does not hand off to M2-T04")
    except OSError:
        errors.append("STATUS.md is unavailable")


def validate_m2_t03_evidence(
    report_path: Path = REPORT_PATH,
    *,
    repository_root: Path = ROOT,
) -> EvidenceValidationResult:
    """Validate the summary report, formal artifact, hashes, and status gate."""

    errors: list[str] = []
    try:
        report = _load_json(report_path if report_path.is_absolute() else repository_root / report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return EvidenceValidationResult(False, ["cannot load M2-T03 evidence report"])
    try:
        if set(report) != REPORT_FIELDS:
            raise ValueError
        if report["report_version"] != REPORT_VERSION or report["task_id"] != "M2-T03":
            raise ValueError
        _validate_commit(report["implementation_commit"], errors, field="implementation_commit")
        _validate_commit(report["validated_commit"], errors, field="validated_commit")
        if not isinstance(report["timestamp_utc"], str) or TIMESTAMP_RE.fullmatch(report["timestamp_utc"]) is None:
            raise ValueError
        datetime.strptime(report["timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        if not isinstance(report["python_version"], str) or not report["python_version"].startswith("3.12."):
            raise ValueError
        dependencies = _as_mapping(report["dependency_versions"])
        if not dependencies or any(not isinstance(key, str) or not isinstance(value, str) for key, value in dependencies.items()):
            raise ValueError
        if report["record_count"] != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        slot_distribution = _as_mapping(report["slot_distribution"])
        support_distribution = _as_mapping(report["support_level_distribution"])
        if set(slot_distribution) != {
            "PROBLEM_EXISTENCE",
            "CURRENT_METHODS",
            "METHOD_TRANSFERABILITY",
            "IMPLEMENTATION_PATH",
            "EVALUATION_BASIS",
        } or set(support_distribution) != {"DIRECT", "INDIRECT", "HYPOTHETICAL"}:
            raise ValueError
        if any(type(value) is not int or value < 0 for value in [*slot_distribution.values(), *support_distribution.values()]):
            raise ValueError
        if report["title_only_count"] != 0 or report["rejected_count"] != 0:
            raise ValueError
        if report["evidence_types"] != {
            "deterministic_fake": "passed",
            "real_model": "not_run",
            "human": "not_run",
        }:
            raise ValueError
        if report["human_judged"] is not False or report["not_run"] != EXPECTED_NOT_RUN:
            raise ValueError
        test_totals = _as_mapping(report["test_totals"])
        if not test_totals or any(not isinstance(key, str) for key in test_totals):
            raise ValueError
        _validate_inputs(report, errors)
        _validate_artifacts(report, errors)
        _validate_commands(report, errors)
        _validate_status(report, errors)
        _validate_privacy(report, errors)
    except (TypeError, ValueError, KeyError):
        errors.append("M2-T03 evidence report has an invalid closed field or value")
    return EvidenceValidationResult(not errors, errors)


def main() -> int:
    result = validate_m2_t03_evidence()
    if not result.valid:
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1
    print("PASS: M2-T03 evidence classification report, artifacts, hashes, and status gate are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
