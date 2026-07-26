from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "validate_phase.py"
SCHEMA_PATH = ROOT / "evaluation" / "report.schema.json"

REQUIRED_CHECKS = (
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
)
REQUIRED_BOUNDARIES = (
    "arxiv_external_network",
    "real_model_download_or_inference",
    "platform_api_or_deployment",
    "cnki",
    "human_judgment",
    "real_source_pool_freeze",
)
REQUIRED_INPUTS = (
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
)


def _load_validator() -> Any:
    assert SCHEMA_PATH.is_file()
    assert SCRIPT_PATH.is_file()
    spec = importlib.util.spec_from_file_location("validate_phase", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(root: Path, relative_path: str, contents: str = "fixture\n") -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _status(*, final: bool = False) -> str:
    if final:
        current_phase = "M1"
        current_status = "READY"
        rows = [
            ("M0", "COMPLETE", "4/4"),
            ("M1", "READY", "0/4"),
            ("M2", "BLOCKED_BY_M1", "0/5"),
            ("M3", "BLOCKED_BY_M2", "0/5"),
            ("M4", "BLOCKED_BY_M3", "0/4"),
            ("M5", "BLOCKED_BY_M4", "0/4"),
            ("M6", "BLOCKED_BY_M5", "0/5"),
        ]
    else:
        current_phase = "M0"
        current_status = "IN_PROGRESS"
        rows = [
            ("M0", "IN_PROGRESS", "3/4"),
            ("M1", "BLOCKED_BY_M0", "0/4"),
            ("M2", "BLOCKED_BY_M1", "0/5"),
            ("M3", "BLOCKED_BY_M2", "0/5"),
            ("M4", "BLOCKED_BY_M3", "0/4"),
            ("M5", "BLOCKED_BY_M4", "0/4"),
            ("M6", "BLOCKED_BY_M5", "0/5"),
        ]
    table = "\n".join(f"| {phase} phase | {state} | {count} | evidence |" for phase, state, count in rows)
    return (
        f"- 当前阶段：`{current_phase}`\n"
        f"- 当前状态：`{current_status}`\n\n"
        "| phase | status | tasks | evidence |\n"
        "|---|---|---:|---|\n"
        f"{table}\n"
    )


def _source_report(task_id: str) -> str:
    return json.dumps({"task_id": task_id, "phase": "M0"})


def _seed_repository(root: Path, *, final_status: bool = False) -> None:
    _write(root, "STATUS.md", _status(final=final_status))
    for relative_path in REQUIRED_INPUTS:
        if relative_path.startswith("evaluation/reports/"):
            task_id = {
                "evaluation/reports/m0-t01r-acceptance-fix.json": "M0-T01R",
                "evaluation/reports/m0-t02-state-machine.json": "M0-T02",
                "evaluation/reports/m0-t03-evaluation-contracts.json": "M0-T03",
                "evaluation/reports/m0-t03r-contract-fix.json": "M0-T03R",
            }[relative_path]
            _write(root, relative_path, _source_report(task_id))
        else:
            _write(root, relative_path)
    _write(root, "evaluation/report.schema.json", SCHEMA_PATH.read_text(encoding="utf-8"))


def _report(root: Path) -> dict[str, Any]:
    sources = {
        "M0-T01": ["evaluation/reports/m0-t01r-acceptance-fix.json"],
        "M0-T02": ["evaluation/reports/m0-t02-state-machine.json"],
        "M0-T03": [
            "evaluation/reports/m0-t03-evaluation-contracts.json",
            "evaluation/reports/m0-t03r-contract-fix.json",
        ],
        "M0-T04": [],
    }
    return {
        "schema_version": "0.1",
        "phase": "M0",
        "validated_commit": "a" * 40,
        "generated_at_utc": "2026-07-26T11:00:00Z",
        "python_version": "3.12.10",
        "task_evidence": [
            {
                "task_id": task_id,
                "status": "passed",
                "source_reports": report_paths,
                "source_report_sha256": [
                    {"path": path, "sha256": _sha256(root / path)} for path in report_paths
                ],
            }
            for task_id, report_paths in sources.items()
        ],
        "checks": [
            {
                "name": name,
                "evidence_type": "automated",
                "status": "passed",
                "command": f"python -m {name}",
                "exit_code": 0,
                "observed_at_utc": "2026-07-26T11:00:00Z",
            }
            for name in REQUIRED_CHECKS
        ],
        "validated_inputs": [
            {"path": relative_path, "sha256": _sha256(root / relative_path)}
            for relative_path in REQUIRED_INPUTS
        ],
        "external_boundaries": [
            {
                "name": name,
                "evidence_type": "not_run",
                "status": "not_run",
                "reason": "Outside the M0 contract-only scope.",
            }
            for name in REQUIRED_BOUNDARIES
        ],
        "status_transition": {
            "from": {
                "current_phase": "M0",
                "current_status": "IN_PROGRESS",
                "m0_tasks": "3/4",
                "m1_status": "BLOCKED_BY_M0",
                "m1_tasks": "0/4",
            },
            "to": {
                "current_phase": "M1",
                "current_status": "READY",
                "m0_tasks": "4/4",
                "m1_status": "READY",
                "m1_tasks": "0/4",
            },
        },
    }


@pytest.fixture
def valid_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Path, dict[str, Any]]:
    validator = _load_validator()
    _seed_repository(tmp_path)

    def fake_git(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        if arguments[:2] == ("diff", "--name-only"):
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return subprocess.CompletedProcess(arguments, 0, "a" * 40 + "\n", "")

    monkeypatch.setattr(validator, "run_git", fake_git)
    return validator, tmp_path, _report(tmp_path)


def _errors(validator: Any, root: Path, payload: dict[str, Any]) -> list[str]:
    return validator.validate_payload(payload, repo_root=root)


def test_report_schema_is_valid_draft_2020_12() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)


def test_valid_pre_transition_and_final_transition_reports_pass(valid_report: tuple[Any, Path, dict[str, Any]]) -> None:
    validator, root, payload = valid_report
    assert _errors(validator, root, payload) == []

    _write(root, "STATUS.md", _status(final=True))
    assert _errors(validator, root, payload) == []


def test_schema_and_validator_reject_missing_or_malformed_commit(valid_report: tuple[Any, Path, dict[str, Any]]) -> None:
    validator, root, payload = valid_report
    for invalid in (None, "abc123", "A" * 40):
        candidate = copy.deepcopy(payload)
        if invalid is None:
            del candidate["validated_commit"]
        else:
            candidate["validated_commit"] = invalid
        assert _errors(validator, root, candidate)


def test_validator_rejects_unknown_or_nonancestor_commit(valid_report: tuple[Any, Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
    validator, root, payload = valid_report

    def missing_commit(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, 1, "", "unknown commit")

    monkeypatch.setattr(validator, "run_git", missing_commit)
    assert any("does not resolve" in error for error in _errors(validator, root, payload))

    def nonancestor(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        code = 1 if arguments[0] == "merge-base" else 0
        return subprocess.CompletedProcess(arguments, code, "", "")

    monkeypatch.setattr(validator, "run_git", nonancestor)
    assert any("not an ancestor" in error for error in _errors(validator, root, payload))


def test_validator_rejects_invalid_check_semantics(valid_report: tuple[Any, Path, dict[str, Any]]) -> None:
    validator, root, payload = valid_report
    unknown_type = copy.deepcopy(payload)
    unknown_type["checks"][0]["evidence_type"] = "combined"
    assert _errors(validator, root, unknown_type)

    duplicate = copy.deepcopy(payload)
    duplicate["checks"].append(copy.deepcopy(duplicate["checks"][0]))
    assert any("duplicate check" in error for error in _errors(validator, root, duplicate))

    missing = copy.deepcopy(payload)
    missing["checks"] = missing["checks"][1:]
    assert any("missing required automated" in error for error in _errors(validator, root, missing))

    bad_exit = copy.deepcopy(payload)
    bad_exit["checks"][0]["exit_code"] = 1
    assert _errors(validator, root, bad_exit)


def test_validator_rejects_external_and_not_run_misrepresentation(valid_report: tuple[Any, Path, dict[str, Any]]) -> None:
    validator, root, payload = valid_report
    real = copy.deepcopy(payload)
    real["checks"].append(
        {"name": "external", "evidence_type": "real_external", "status": "passed"}
    )
    assert _errors(validator, root, real)

    missing_boundary = copy.deepcopy(payload)
    missing_boundary["external_boundaries"] = missing_boundary["external_boundaries"][1:]
    assert any("missing required not_run" in error for error in _errors(validator, root, missing_boundary))

    disguised = copy.deepcopy(payload)
    disguised["external_boundaries"][0]["status"] = "passed"
    assert _errors(validator, root, disguised)


def test_validator_rejects_task_evidence_and_source_integrity_errors(valid_report: tuple[Any, Path, dict[str, Any]]) -> None:
    validator, root, payload = valid_report
    missing = copy.deepcopy(payload)
    missing["task_evidence"] = missing["task_evidence"][1:]
    assert any("missing task evidence" in error for error in _errors(validator, root, missing))

    duplicate = copy.deepcopy(payload)
    duplicate["task_evidence"].append(copy.deepcopy(duplicate["task_evidence"][0]))
    assert any("duplicate task_id" in error for error in _errors(validator, root, duplicate))

    unknown = copy.deepcopy(payload)
    unknown["task_evidence"][0]["task_id"] = "M0-T99"
    assert _errors(validator, root, unknown)

    bad_hash = copy.deepcopy(payload)
    bad_hash["task_evidence"][0]["source_report_sha256"][0]["sha256"] = "0" * 64
    assert any("source report hash mismatch" in error for error in _errors(validator, root, bad_hash))


@pytest.mark.parametrize("invalid_path", ["C:/outside.txt", "../outside.txt", "docs/../../outside.txt"])
def test_validator_rejects_unsafe_or_duplicate_validated_input_paths(
    valid_report: tuple[Any, Path, dict[str, Any]], invalid_path: str
) -> None:
    validator, root, payload = valid_report
    unsafe = copy.deepcopy(payload)
    unsafe["validated_inputs"][0]["path"] = invalid_path
    assert _errors(validator, root, unsafe)

    duplicate = copy.deepcopy(payload)
    duplicate["validated_inputs"].append(copy.deepcopy(duplicate["validated_inputs"][0]))
    assert any("duplicate validated input" in error for error in _errors(validator, root, duplicate))


def test_validator_recomputes_validated_input_hashes(valid_report: tuple[Any, Path, dict[str, Any]]) -> None:
    validator, root, payload = valid_report
    payload["validated_inputs"][0]["sha256"] = "0" * 64
    assert any("validated input hash mismatch" in error for error in _errors(validator, root, payload))


@pytest.mark.parametrize(
    ("final", "mutate"),
    [
        (False, lambda text: text.replace("| M0 phase | IN_PROGRESS | 3/4 |", "| M0 phase | COMPLETE | 4/4 |")),
        (True, lambda text: text.replace("| M1 phase | READY | 0/4 |", "| M1 phase | BLOCKED_BY_M0 | 0/4 |")),
        (True, lambda text: text.replace("| M0 phase | COMPLETE | 4/4 |", "| M0 phase | IN_PROGRESS | 3/4 |")),
        (True, lambda text: text.replace("| M2 phase | BLOCKED_BY_M1 | 0/5 |", "| M2 phase | READY | 0/5 |")),
    ],
)
def test_validator_rejects_invalid_status_transitions(
    valid_report: tuple[Any, Path, dict[str, Any]], final: bool, mutate: Any
) -> None:
    validator, root, payload = valid_report
    _write(root, "STATUS.md", mutate(_status(final=final)))
    assert _errors(validator, root, payload)
