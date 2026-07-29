"""Red/green contract for the machine-readable M1 closure evidence."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from scripts.validate_m1_evidence import validate_current_m1_closure_status, validate_m1_evidence


def test_validator_rejects_evidence_without_live_source_id_samples(tmp_path: Path) -> None:
    report = Path("evaluation/reports/m1-validation.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["report_version"] = "m1-t04-validation.v3"
    payload.setdefault("real_external", {}).pop("real_source_id_samples", None)
    invalid_report = tmp_path / "m1-validation.json"
    invalid_report.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_m1_evidence(invalid_report, repository_root=Path("."))

    assert result.valid is False
    assert "real live evidence lacks real_source_id_samples" in result.errors


def test_current_status_with_historical_in_progress_m1_prose_is_rejected() -> None:
    status = subprocess.check_output(
        ["git", "show", "49c9a9b1b3d38144827c2c2145b6f5ae361620b3:STATUS.md"],
        text=True,
        encoding="utf-8",
    )

    errors = validate_current_m1_closure_status(status)

    assert "contradictory M1 closure prose: M1 is IN_PROGRESS" in errors
    assert "contradictory M1 closure prose: M2 remains BLOCKED_BY_M1" in errors
    assert "contradictory M1 closure prose: M1-T04 is NOT_STARTED" in errors
    assert "contradictory M1 closure prose: live-success gate remains unsatisfied" in errors


def test_m1_closure_allows_later_m2_progress_but_rejects_a_regression_to_blocked() -> None:
    status = Path("STATUS.md").read_text(encoding="utf-8")
    in_progress = re.sub(
        r"\| M2 [^\r\n]*",
        "| M2 首轮排序与选择 | IN_PROGRESS | 1/5 | M2-T01 completion evidence validated |",
        status,
        count=1,
    )
    assert validate_current_m1_closure_status(in_progress) == []

    blocked = re.sub(
        r"\| M2 [^\r\n]*",
        "| M2 首轮排序与选择 | BLOCKED_BY_M1 | 0/5 | stale |",
        status,
        count=1,
    )
    assert "STATUS.md must not declare M2 BLOCKED_BY_M1 after M1 is COMPLETE" in (
        validate_current_m1_closure_status(blocked)
    )
