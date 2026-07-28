"""Contracts for evidence-derived M1 frozen-input replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.freeze_m2_candidates import load_accepted_m1_provenance


def test_production_defaults_are_derived_from_the_accepted_m1_evidence() -> None:
    provenance = load_accepted_m1_provenance(Path("evaluation/reports/m1-validation.json"))

    assert provenance.output_sha256 == "069cd8c94d294c68bb051898d4c262f324b7e1f20eff10eabcf22d9bc435d178"
    assert provenance.candidate_array_sha256 == "e51eb84d4e772bba324a199478caa5f0983edef0b0dbf4c58fce5f9da5803209"
    assert provenance.candidate_count == 33
    assert provenance.source_id_coverage == provenance.url_coverage == 1.0


def test_provenance_rejects_recorded_or_incomplete_evidence(tmp_path: Path) -> None:
    report = tmp_path / "m1-validation.json"
    payload = {
        "baseline_commit": "a" * 40,
        "real_external": {
            "classification": "recorded_external",
            "deduplicated_candidate_count": 33,
            "source_id_coverage": 1.0,
            "url_coverage": 1.0,
            "candidate_array_sha256": "b" * 64,
            "output_hashes": {"first-round.json": "c" * 64},
        },
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="real_external"):
        load_accepted_m1_provenance(report)

    payload["real_external"]["classification"] = "real_external"
    del payload["real_external"]["output_hashes"]
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="complete accepted output provenance"):
        load_accepted_m1_provenance(report)
