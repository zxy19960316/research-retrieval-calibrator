"""Red/green contract for the machine-readable M1 closure evidence."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_m1_evidence import validate_m1_evidence


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
