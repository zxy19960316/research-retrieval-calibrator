"""Unit tests for the M2-T03 human-pending bundle boundary."""

from __future__ import annotations

import json

import pytest

import scripts.build_m2_t03_human_adjudication_bundle as builder
from app.models.m2_t03_human_adjudication import HumanAdjudicationFields


def test_human_adjudication_fields_start_empty() -> None:
    fields = HumanAdjudicationFields()

    assert fields.model_dump(mode="json") == {
        "decision": None,
        "evidence_slot": None,
        "grounded_reason": None,
        "notes": None,
        "reviewed_at_utc": None,
        "reviewer_id": None,
        "support_level": None,
        "supporting_excerpt": None,
    }


def test_bundle_builder_requires_explicit_offline_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert builder.main([]) == 2
    output = capsys.readouterr().out
    assert json.loads(output)["error_code"] == "BUNDLE_EXECUTION_NOT_AUTHORIZED"
