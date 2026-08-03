"""Contract tests for alternate-root M2-T03 evidence validation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from scripts.validate_m2_t03_evidence import validate_m2_t03_evidence

_REPOSITORY_FILES = (
    Path("STATUS.md"),
    Path("evaluation/snapshots/m2/m1-candidates.v1.json"),
    Path("evaluation/snapshots/m2/m1-candidates.v1.manifest.json"),
    Path("evaluation/source-artifacts/m2-t02-reranker-candidate-run.json"),
    Path("evaluation/source-artifacts/m2-t02-reranker-candidate-run-receipt.json"),
    Path("evaluation/source-artifacts/m2-t03-evidence-classification-run.json"),
    Path("evaluation/source-artifacts/m2-t03-evidence-classification-run-receipt.json"),
    Path("evaluation/reports/m2-t03-evidence-classification.json"),
)


def _copy_repository(source_root: Path, target_root: Path) -> None:
    for relative_path in _REPOSITORY_FILES:
        target = target_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative_path, target)


def _run_git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit_repository(repository_root: Path, message: str) -> str:
    _run_git(repository_root, "add", "--all")
    subprocess.run(
        ["git", "-c", "user.name=M2-T03 test", "-c", "user.email=m2-t03@example.invalid", "commit", "--quiet", "-m", message],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return _run_git(repository_root, "rev-parse", "HEAD")


def test_validator_uses_supplied_repository_root_and_git_cwd(tmp_path: Path) -> None:
    source_root = Path(__file__).parents[2]
    temporary_root = tmp_path / "repository"
    temporary_root.mkdir()
    _copy_repository(source_root, temporary_root)
    _run_git(temporary_root, "init", "--quiet")
    _run_git(temporary_root, "config", "user.name", "M2-T03 test")
    _run_git(temporary_root, "config", "user.email", "m2-t03@example.invalid")

    report_path = temporary_root / "evaluation/reports/m2-t03-evidence-classification.json"
    first_commit = _commit_repository(temporary_root, "test: initialize alternate evidence root")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["implementation_commit"] = first_commit
    report["validated_commit"] = first_commit
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _commit_repository(temporary_root, "test: bind report to alternate root history")

    valid = validate_m2_t03_evidence(
        report_path=report_path,
        repository_root=temporary_root,
    )
    assert valid.valid, valid.errors

    source_result = source_root / "evaluation/source-artifacts/m2-t03-evidence-classification-run.json"
    original_hash = hashlib.sha256(source_result.read_bytes()).hexdigest()
    temporary_result = temporary_root / "evaluation/source-artifacts/m2-t03-evidence-classification-run.json"
    mutated_result = json.loads(temporary_result.read_text(encoding="utf-8"))
    mutated_result["execution"]["record_count"] = 32
    temporary_result.write_text(
        json.dumps(mutated_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    invalid = validate_m2_t03_evidence(
        report_path=report_path,
        repository_root=temporary_root,
    )
    assert not invalid.valid
    assert "formal M2-T03 artifact or receipt failed validation" in invalid.errors
    assert hashlib.sha256(source_result.read_bytes()).hexdigest() == original_hash
