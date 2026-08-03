"""Offline contracts for the completed M2-T02 evidence summary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import validate_m2_t02_evidence as evidence

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / evidence.REPORT_PATH

EXPECTED_REPORT_FIELDS = {
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


def _load_report() -> dict[str, object]:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_completed_m2_t02_report_passes_the_production_summary_validator() -> None:
    result = evidence.validate_m2_t02_evidence()
    assert result.valid, result.errors
    payload = _load_report()
    assert set(payload) == EXPECTED_REPORT_FIELDS
    assert payload["report_version"] == "m2-t02-reranker.v1"
    assert payload["live_execution_commit"] == evidence.LIVE_EXECUTION_COMMIT
    assert payload["evidence_commit"] == evidence.EVIDENCE_COMMIT
    assert payload["validated_commit"] == evidence.VALIDATED_COMMIT
    assert payload["acceptance"] == {
        "batch_partition_invariance": "passed",
        "provider_failure_fail_closed": "passed",
        "partial_scores_not_published": "passed",
        "raw_scores_preserved": "passed",
        "normalization_method": "global_min_max",
        "deterministic_tie_break": "paper_id_ascending",
        "fixed_candidate_count": 33,
        "real_model_run": "passed",
    }
    assert payload["not_run"] == {
        "additional_model_rerun": False,
        "b3_replay": False,
        "benchmark": False,
        "manual_relevance_judgement": False,
        "m2_t03_started": False,
    }


def test_completed_m2_t02_report_survives_the_m2_t03_status_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _load_report()
    monkeypatch.setattr(
        evidence,
        "_parse_status",
        lambda _root, _errors: {
            "M1": ("COMPLETE", 4, 4),
            "M2": evidence.POST_M2_T03_M2_STATUS,
            "M3": ("BLOCKED_BY_M2", 0, 5),
        },
    )
    errors: list[str] = []

    evidence._validate_status(payload, errors, ROOT)

    assert errors == []


def test_validated_inputs_are_bound_to_git_blobs_not_worktree_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evidence, "_blob_bytes", lambda *_args: None)
    result = evidence.validate_m2_t02_evidence()
    assert result.valid is False
    assert any("validated_inputs" in error for error in result.errors)


def test_summary_validator_rejects_closed_schema_and_not_run_drift(
    tmp_path: Path,
) -> None:
    payload = _load_report()
    payload["unexpected"] = True
    mutated = tmp_path / "m2-t02-reranker.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    result = evidence.validate_m2_t02_evidence(report_path=mutated)
    assert result.valid is False

    payload = _load_report()
    payload["not_run"] = {**payload["not_run"], "b3_replay": True}  # type: ignore[index]
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    result = evidence.validate_m2_t02_evidence(report_path=mutated)
    assert result.valid is False


def test_summary_validator_does_not_require_models_or_import_runtime_packages(
    tmp_path: Path,
) -> None:
    code = """
import sys
from scripts import validate_m2_t02_evidence as evidence
result = evidence.validate_m2_t02_evidence()
assert result.valid, result.errors
assert not any(name.split('.')[0] in {'torch', 'transformers', 'tokenizers', 'safetensors'} for name in sys.modules)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
