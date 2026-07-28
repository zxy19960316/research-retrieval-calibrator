"""Red-first expectations for M2-T01's completed-evidence gate."""

from __future__ import annotations

from scripts.validate_m2_t01_evidence import validate_m2_t01_evidence


def test_precompletion_main_is_valid_without_a_completion_report() -> None:
    result = validate_m2_t01_evidence()
    assert result.valid is True


def test_present_completion_report_is_fail_closed_when_incomplete(tmp_path) -> None:
    report = tmp_path / "m2-t01-embedding.json"
    report.write_text('{"report_version":"m2-t01-embedding.v1"}\n', encoding="utf-8")

    result = validate_m2_t01_evidence(report_path=report)

    assert result.valid is False
    assert "report must identify phase M2 and task M2-T01" in result.errors
