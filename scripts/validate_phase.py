from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = Path("evaluation/reports/m0-validation.json")
SCHEMA_PATH = Path("evaluation/report.schema.json")
REQUIRED_AUTOMATED_CHECKS = {
    "domain_contract_tests",
    "state_machine_tests",
    "evaluation_dataset_contract_tests",
    "retrieval_metrics_tests",
    "evidence_report_contract_tests",
    "full_regression",
    "ruff",
    "mypy",
    "project_docs_validation",
    "pip_check",
}
REQUIRED_BOUNDARIES = {
    "arxiv_external_network",
    "real_model_download_or_inference",
    "platform_api_or_deployment",
    "cnki",
    "human_judgment",
    "real_source_pool_freeze",
}
REQUIRED_TASKS = {"M0-T01", "M0-T02", "M0-T03", "M0-T04"}
REQUIRED_INPUTS = {
    "PRODUCT_SPEC.md",
    "PROJECT_PLAN.md",
    "docs/data_model.md",
    "docs/evaluation.md",
    "docs/phases/M0-product-and-evaluation-contracts.md",
    "docs/superpowers/plans/2026-07-26-research-retrieval-calibrator-m0.md",
    "app/models/__init__.py",
    "app/models/enums.py",
    "app/models/feedback.py",
    "app/models/paper.py",
    "app/models/project.py",
    "app/models/query.py",
    "app/core/state_machine.py",
    "evaluation/datasets/questions.schema.json",
    "evaluation/datasets/questions.v0.1.yaml",
    "evaluation/metrics/retrieval.py",
    "tests/contract/test_domain_models.py",
    "tests/contract/test_project_docs_validation.py",
    "tests/contract/fixtures/m0_t01_public_contracts.schema.json",
    "tests/unit/test_state_machine.py",
    "tests/contract/test_evaluation_dataset.py",
    "tests/unit/test_retrieval_metrics.py",
    "tests/contract/test_evidence_report.py",
    "evaluation/report.schema.json",
    "scripts/validate_phase.py",
    ".github/workflows/docs-validation.yml",
    "evaluation/reports/m0-t01r-acceptance-fix.json",
    "evaluation/reports/m0-t02-state-machine.json",
    "evaluation/reports/m0-t03-evaluation-contracts.json",
    "evaluation/reports/m0-t03r-contract-fix.json",
}
TASK_SOURCE_IDS = {
    "M0-T01": {"M0-T01R"},
    "M0-T02": {"M0-T02"},
    "M0-T03": {"M0-T03", "M0-T03R"},
    "M0-T04": set(),
}
CURRENT_PHASE_PATTERN = re.compile(r"^- 当前阶段：`(M\d+)`$", re.MULTILINE)
CURRENT_STATUS_PATTERN = re.compile(r"^- 当前状态：`([A-Z0-9_]+)`$", re.MULTILINE)
STATUS_ROW_PATTERN = re.compile(
    r"^\|\s*(M\d+)\s+[^|]*\|\s*([A-Z0-9_]+)\s*\|\s*(\d+)/(\d+)\s*\|", re.MULTILINE
)


def run_git(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=cwd, text=True, capture_output=True, check=False
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schema_errors(payload: dict[str, Any], repo_root: Path) -> list[str]:
    schema_path = repo_root / SCHEMA_PATH
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    except (OSError, json.JSONDecodeError, jsonschema.SchemaError) as error:
        return [f"cannot load report schema: {error}"]
    return [f"schema: {error.message}" for error in validator.iter_errors(payload)]


def _safe_relative_path(value: str, repo_root: Path) -> Path | None:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    resolved = (repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return resolved


def _validate_checks(payload: dict[str, Any], errors: list[str]) -> None:
    checks = payload.get("checks", [])
    names = [check.get("name") for check in checks if isinstance(check, dict)]
    duplicates = {name for name in names if names.count(name) > 1}
    for name in sorted(duplicates, key=str):
        errors.append(f"duplicate check name: {name}")
    automated = {check.get("name"): check for check in checks if isinstance(check, dict)}
    missing = REQUIRED_AUTOMATED_CHECKS - set(automated)
    for name in sorted(missing):
        errors.append(f"missing required automated check: {name}")
    for name in REQUIRED_AUTOMATED_CHECKS & set(automated):
        check = automated[name]
        if check.get("evidence_type") != "automated" or check.get("status") != "passed":
            errors.append(f"required automated check is not passed: {name}")
        if check.get("exit_code") != 0:
            errors.append(f"required automated check has non-zero exit_code: {name}")
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("evidence_type") == "not_run" and "exit_code" in check:
            errors.append(f"not_run check must not declare exit_code: {check.get('name')}")


def _validate_boundaries(payload: dict[str, Any], errors: list[str]) -> None:
    boundaries = payload.get("external_boundaries", [])
    by_name = {boundary.get("name"): boundary for boundary in boundaries if isinstance(boundary, dict)}
    missing = REQUIRED_BOUNDARIES - set(by_name)
    for name in sorted(missing):
        errors.append(f"missing required not_run boundary: {name}")
    for name in REQUIRED_BOUNDARIES & set(by_name):
        boundary = by_name[name]
        if boundary.get("evidence_type") != "not_run" or boundary.get("status") != "not_run":
            errors.append(f"boundary is not explicitly not_run: {name}")
        if not isinstance(boundary.get("reason"), str) or not boundary["reason"].strip():
            errors.append(f"not_run boundary has no reason: {name}")
        if "exit_code" in boundary:
            errors.append(f"not_run boundary must not declare exit_code: {name}")


def _validate_task_evidence(payload: dict[str, Any], repo_root: Path, errors: list[str]) -> None:
    evidence = payload.get("task_evidence", [])
    ids = [item.get("task_id") for item in evidence if isinstance(item, dict)]
    duplicates = {task_id for task_id in ids if ids.count(task_id) > 1}
    for task_id in sorted(duplicates, key=str):
        errors.append(f"duplicate task_id: {task_id}")
    unknown = set(ids) - REQUIRED_TASKS
    for task_id in sorted(unknown, key=str):
        errors.append(f"unknown task_id: {task_id}")
    for task_id in sorted(REQUIRED_TASKS - set(ids)):
        errors.append(f"missing task evidence: {task_id}")
    for item in evidence:
        if not isinstance(item, dict) or item.get("task_id") not in REQUIRED_TASKS:
            continue
        task_id = item["task_id"]
        if item.get("status") != "passed":
            errors.append(f"task evidence is not passed: {task_id}")
        reports = item.get("source_reports", [])
        hash_entries = item.get("source_report_sha256", [])
        hashes = {
            entry["path"]: entry["sha256"]
            for entry in hash_entries
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        }
        if len(hashes) != len(hash_entries):
            errors.append(f"duplicate or malformed source report hash entry: {task_id}")
        if task_id == "M0-T04" and (reports or hashes):
            errors.append("M0-T04 must not self-reference the closing report")
        if set(reports) != set(hashes):
            errors.append(f"source report paths and hashes differ: {task_id}")
        observed_source_ids: set[str] = set()
        for report_path in reports:
            resolved = _safe_relative_path(report_path, repo_root)
            if resolved is None or not resolved.is_file():
                errors.append(f"source report is missing or unsafe: {report_path}")
                continue
            expected_hash = hashes.get(report_path)
            if sha256_file(resolved) != expected_hash:
                errors.append(f"source report hash mismatch: {report_path}")
            try:
                source_payload = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"source report is not valid JSON: {report_path}: {error}")
                continue
            if source_payload.get("phase") != "M0":
                errors.append(f"source report has non-M0 phase: {report_path}")
            source_task_id = source_payload.get("task_id")
            if isinstance(source_task_id, str):
                observed_source_ids.add(source_task_id)
        if task_id != "M0-T04" and observed_source_ids != TASK_SOURCE_IDS[task_id]:
            errors.append(f"source report task IDs do not match declaration: {task_id}")


def _validate_inputs(payload: dict[str, Any], repo_root: Path, errors: list[str]) -> set[str]:
    inputs = payload.get("validated_inputs", [])
    paths = [item.get("path") for item in inputs if isinstance(item, dict)]
    duplicates = {path for path in paths if paths.count(path) > 1}
    for path in sorted(duplicates, key=str):
        errors.append(f"duplicate validated input path: {path}")
    missing = REQUIRED_INPUTS - set(paths)
    for path in sorted(missing):
        errors.append(f"missing required validated input: {path}")
    valid_paths: set[str] = set()
    for item in inputs:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        relative_path = item["path"]
        resolved = _safe_relative_path(relative_path, repo_root)
        if resolved is None:
            errors.append(f"unsafe validated input path: {relative_path}")
            continue
        if not resolved.is_file():
            errors.append(f"validated input is not a regular file: {relative_path}")
            continue
        valid_paths.add(relative_path)
        if sha256_file(resolved) != item.get("sha256"):
            errors.append(f"validated input hash mismatch: {relative_path}")
    return valid_paths


def _status_snapshot(repo_root: Path) -> dict[str, str] | None:
    try:
        text = (repo_root / "STATUS.md").read_text(encoding="utf-8")
    except OSError:
        return None
    phases = CURRENT_PHASE_PATTERN.findall(text)
    statuses = CURRENT_STATUS_PATTERN.findall(text)
    rows = {phase: (status, f"{done}/{total}") for phase, status, done, total in STATUS_ROW_PATTERN.findall(text)}
    if len(phases) != 1 or len(statuses) != 1 or "M0" not in rows or "M1" not in rows:
        return None
    return {
        "current_phase": phases[0],
        "current_status": statuses[0],
        "m0_tasks": rows["M0"][1],
        "m1_status": rows["M1"][0],
        "m1_tasks": rows["M1"][1],
        "m2_status": rows.get("M2", ("", ""))[0],
        "m3_status": rows.get("M3", ("", ""))[0],
        "m4_status": rows.get("M4", ("", ""))[0],
        "m5_status": rows.get("M5", ("", ""))[0],
        "m6_status": rows.get("M6", ("", ""))[0],
    }


def _validate_status(payload: dict[str, Any], repo_root: Path, errors: list[str]) -> None:
    transition = payload.get("status_transition", {})
    before = {
        "current_phase": "M0", "current_status": "IN_PROGRESS", "m0_tasks": "3/4",
        "m1_status": "BLOCKED_BY_M0", "m1_tasks": "0/4",
    }
    after = {
        "current_phase": "M1", "current_status": "READY", "m0_tasks": "4/4",
        "m1_status": "READY", "m1_tasks": "0/4",
    }
    if transition.get("from") != before or transition.get("to") != after:
        errors.append("report status_transition does not declare the M0-to-M1 gate")
    status = _status_snapshot(repo_root)
    if status is None:
        errors.append("STATUS.md cannot be parsed for M0 gate")
        return
    live = {key: status[key] for key in before}
    if live not in (before, after):
        errors.append("STATUS.md is neither the pre-transition nor final M0 gate state")
    expected_later = {
        "m2_status": "BLOCKED_BY_M1", "m3_status": "BLOCKED_BY_M2", "m4_status": "BLOCKED_BY_M3",
        "m5_status": "BLOCKED_BY_M4", "m6_status": "BLOCKED_BY_M5",
    }
    for key, expected in expected_later.items():
        if status[key] != expected:
            errors.append(f"later phase changed before its gate: {key}")


def _validate_git(payload: dict[str, Any], repo_root: Path, validated_paths: set[str], errors: list[str]) -> None:
    commit = payload.get("validated_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return
    if run_git("rev-parse", "--verify", f"{commit}^{{commit}}", cwd=repo_root).returncode != 0:
        errors.append(f"validated_commit does not resolve: {commit}")
        return
    if run_git("cat-file", "-e", f"{commit}^{{commit}}", cwd=repo_root).returncode != 0:
        errors.append(f"validated_commit is not a commit object: {commit}")
    if run_git("merge-base", "--is-ancestor", commit, "HEAD", cwd=repo_root).returncode != 0:
        errors.append(f"validated_commit is not an ancestor of HEAD: {commit}")
    dirty = run_git("diff", "--name-only", "HEAD", cwd=repo_root)
    if dirty.returncode != 0:
        errors.append("cannot inspect working tree for validated inputs")
        return
    dirty_paths = {line.replace("\\", "/") for line in dirty.stdout.splitlines() if line}
    for path in sorted(dirty_paths & validated_paths):
        errors.append(f"validated input has uncommitted changes: {path}")


def validate_payload(payload: dict[str, Any], *, repo_root: Path = ROOT) -> list[str]:
    errors = _schema_errors(payload, repo_root)
    _validate_checks(payload, errors)
    _validate_boundaries(payload, errors)
    _validate_task_evidence(payload, repo_root, errors)
    validated_paths = _validate_inputs(payload, repo_root, errors)
    _validate_status(payload, repo_root, errors)
    _validate_git(payload, repo_root, validated_paths, errors)
    return errors


def validate_report_file(phase: str, *, repo_root: Path = ROOT) -> list[str]:
    if phase != "M0":
        return [f"unsupported phase: {phase}"]
    report_path = repo_root / REPORT_PATH
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot load M0 report: {error}"]
    if not isinstance(payload, dict):
        return ["M0 report must be a JSON object"]
    return validate_payload(payload, repo_root=repo_root)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/validate_phase.py M0")
        return 2
    errors = validate_report_file(sys.argv[1])
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("PASS: M0 evidence report, inputs, git ancestry, and status gate are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
