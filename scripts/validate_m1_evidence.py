"""Machine-validate the provenance-safe M1-T04R1 closure evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = Path("evaluation/reports/m1-validation.json")
REPORT_VERSION = "m1-t04-validation.v3"
SHA1 = re.compile(r"[0-9a-f]{40}")
REQUIRED_CHECKS = {
    "focused_tests",
    "full_regression",
    "ruff",
    "mypy",
    "project_docs_validation",
    "m0_evidence_validation",
    "pip_check",
}


@dataclass(frozen=True)
class EvidenceValidationResult:
    valid: bool
    errors: list[str]


def _git(*arguments: str, repository_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )


def _git_bytes(*arguments: str, repository_root: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )


def _safe_path(value: object, repository_root: Path) -> Path | None:
    if not isinstance(value, str):
        return None
    candidate = Path(value)
    windows_candidate = PureWindowsPath(value)
    if candidate.is_absolute() or windows_candidate.is_absolute() or ".." in candidate.parts:
        return None
    resolved = (repository_root / candidate).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError:
        return None
    return resolved


def _is_ancestor(commit: object, repository_root: Path) -> bool:
    return isinstance(commit, str) and bool(SHA1.fullmatch(commit)) and _git(
        "merge-base", "--is-ancestor", commit, "HEAD", repository_root=repository_root
    ).returncode == 0


def _blob_sha256(commit: str, path: str, repository_root: Path) -> str | None:
    git_path = path.replace("\\", "/")
    shown = _git_bytes("show", f"{commit}:{git_path}", repository_root=repository_root)
    if shown.returncode != 0:
        return None
    return hashlib.sha256(shown.stdout).hexdigest()


def _validate_inputs(payload: dict[str, Any], implementation_commit: str, errors: list[str], repository_root: Path) -> None:
    inputs = payload.get("validated_inputs")
    if not isinstance(inputs, list) or not inputs:
        errors.append("validated_inputs must be a non-empty list")
        return
    seen: set[str] = set()
    for item in inputs:
        if not isinstance(item, dict):
            errors.append("validated input entry must be an object")
            continue
        path = item.get("path")
        expected = item.get("sha256")
        if not isinstance(path, str) or path in seen or _safe_path(path, repository_root) is None:
            errors.append("validated input path is missing, duplicate, or unsafe")
            continue
        seen.add(path)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            errors.append(f"validated input has invalid sha256: {path}")
            continue
        actual = _blob_sha256(implementation_commit, path, repository_root)
        if actual != expected:
            errors.append(f"validated input hash mismatch: {path}")


def _validate_checks(payload: dict[str, Any], errors: list[str]) -> None:
    checks = payload.get("automated_checks")
    if not isinstance(checks, list):
        errors.append("automated_checks must be a list")
        return
    names = {item.get("name") for item in checks if isinstance(item, dict)}
    for required_name in sorted(REQUIRED_CHECKS - names):
        errors.append(f"missing automated check: {required_name}")
    for check in checks:
        if not isinstance(check, dict):
            errors.append("automated check must be an object")
            continue
        check_name = check.get("name")
        if not isinstance(check_name, str) or check.get("exit_code") != 0:
            errors.append(f"automated check is not green: {check_name}")


def _validate_real_evidence(payload: dict[str, Any], errors: list[str]) -> None:
    live = payload.get("real_external")
    replay = payload.get("real_cache_replay")
    if not isinstance(live, dict):
        errors.append("real_external must be an object")
        return
    if live.get("status") != "success" or not isinstance(live.get("network_requests"), int) or live["network_requests"] <= 0:
        errors.append("real_external must record a successful live network run")
    if live.get("cache_hits") != 0:
        errors.append("real live run must have zero cache hits")
    statuses = live.get("http_statuses")
    if not isinstance(statuses, list) or not statuses or any(status < 200 or status >= 300 for status in statuses):
        errors.append("real live HTTP statuses must all be successful")
    for key in ("real_source_id_samples", "real_url_samples"):
        if not isinstance(live.get(key), list) or not live[key]:
            errors.append(f"real live evidence lacks {key}")
    for key in ("source_id_coverage", "url_coverage"):
        if live.get(key) != 1.0:
            errors.append(f"real live {key} must equal 1.0")
    if live.get("metadata_projection_mismatch_count") != 0 or live.get("metadata_hallucination_rate") != 0.0:
        errors.append("real live provenance audit must report zero mismatches and zero rate")
    rate = live.get("rate_limit")
    if not isinstance(rate, dict) or not isinstance(rate.get("configured_min_request_interval_seconds"), (int, float)) or rate["configured_min_request_interval_seconds"] <= 0:
        errors.append("real live evidence lacks a positive configured request interval")
    elif rate.get("minimum_observed_request_start_delta_seconds") is not None and rate["minimum_observed_request_start_delta_seconds"] < rate["configured_min_request_interval_seconds"]:
        errors.append("observed request-start delta is below the configured interval")
    if not isinstance(replay, dict) or replay.get("status") != "success":
        errors.append("real_cache_replay must be successful")
    elif replay.get("transport_requests") != 0 or not isinstance(replay.get("cache_hits"), int) or replay["cache_hits"] <= 0 or replay.get("candidate_arrays_equal") is not True:
        errors.append("real cache replay must be a zero-transport, cache-hit, equal-candidate replay")


def _validate_status_and_namespaces(payload: dict[str, Any], errors: list[str], repository_root: Path) -> None:
    namespaces = payload.get("cache_namespaces")
    if not isinstance(namespaces, dict) or namespaces.get("recorded") == namespaces.get("real"):
        errors.append("recorded and real cache namespaces must differ")
    try:
        status = (repository_root / "STATUS.md").read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"cannot read STATUS.md: {error}")
        return
    if "| M1 首轮真实召回 | COMPLETE | 4/4 |" not in status:
        errors.append("STATUS.md must declare M1 COMPLETE 4/4")
    if "| M2 首轮排序与选择 | READY | 0/5 |" not in status:
        errors.append("STATUS.md must declare M2 READY")


def validate_m1_evidence(
    report_path: Path = REPORT_PATH, *, repository_root: Path = ROOT
) -> EvidenceValidationResult:
    errors: list[str] = []
    absolute_report = report_path if report_path.is_absolute() else repository_root / report_path
    try:
        payload = json.loads(absolute_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return EvidenceValidationResult(False, [f"cannot load M1 report: {error}"])
    if not isinstance(payload, dict):
        return EvidenceValidationResult(False, ["M1 report must be a JSON object"])
    if payload.get("report_version") != REPORT_VERSION:
        errors.append("report_version must be m1-t04-validation.v3")
    commits = [
        payload.get("implementation_commit_g"),
        payload.get("implementation_correction_commit_i"),
        payload.get("validated_implementation_commit"),
    ]
    if not all(_is_ancestor(commit, repository_root) for commit in commits):
        errors.append("G, I, and validated implementation commit must be HEAD ancestors")
    implementation_commit = payload.get("validated_implementation_commit")
    if isinstance(implementation_commit, str) and SHA1.fullmatch(implementation_commit):
        _validate_inputs(payload, implementation_commit, errors, repository_root)
    _validate_checks(payload, errors)
    _validate_real_evidence(payload, errors)
    _validate_status_and_namespaces(payload, errors, repository_root)
    return EvidenceValidationResult(not errors, errors)


def main() -> int:
    result = validate_m1_evidence()
    for error in result.errors:
        print(f"ERROR: {error}")
    if not result.valid:
        return 1
    print("PASS: M1 evidence report, provenance, live cache replay, and status gate are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
