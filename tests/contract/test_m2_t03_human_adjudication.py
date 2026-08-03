"""Contract tests for the first-phase M2-T03 human review bundle."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import scripts.validate_m2_t03_human_adjudication_bundle as validator
from scripts.validate_m2_t03_human_adjudication_bundle import (
    BUNDLE_PATH,
    validate_bundle,
)


def _bundle_payload() -> dict[str, object]:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def test_bundle_is_closed_and_stops_at_the_human_work_point() -> None:
    payload = _bundle_payload()
    assert set(payload) == {
        "bundle_version",
        "candidate_count",
        "candidate_snapshot",
        "classification_result",
        "generated_at_utc",
        "generated_from_commit",
        "intent_context",
        "items",
        "paper_id_order",
        "phase",
        "review_policy",
        "status_boundary",
        "task_id",
    }
    assert payload["candidate_count"] == 33
    assert payload["status_boundary"] == {
        "human_review": "pending",
        "m2_progress": "IN_PROGRESS 2/5",
        "m2_t04": "not_started",
        "m3_progress": "BLOCKED_BY_M2 0/5",
        "scoring_eligible": False,
    }
    assert payload["review_policy"]["machine_labels_are_advisory"] is True  # type: ignore[index]
    assert payload["review_policy"]["human_fields_initially_empty"] is True  # type: ignore[index]

    items = payload["items"]
    assert isinstance(items, list)
    assert len(items) == 33
    assert all(
        set(item) == {"context", "machine_advisory", "human_adjudication"} for item in items
    )
    assert all(
        all(value is None for value in item["human_adjudication"].values()) for item in items
    )
    serialized = json.dumps(payload, ensure_ascii=False)
    assert all(token not in serialized.casefold() for token in ('"score"', '"rank"', '"weight"'))


def test_validator_accepts_the_committed_bundle() -> None:
    result = validate_bundle()
    assert result.valid, result.errors


def test_validator_rejects_any_prefilled_human_decision(tmp_path: Path) -> None:
    payload = copy.deepcopy(_bundle_payload())
    payload["items"][0]["human_adjudication"]["verdict"] = "SUPPORTED"  # type: ignore[index]
    mutated = tmp_path / "bundle.json"
    mutated.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = validate_bundle(mutated)

    assert not result.valid
    assert any("context or advisory" in error for error in result.errors)


def test_validator_rejects_unknown_bundle_fields(tmp_path: Path) -> None:
    payload = _bundle_payload()
    payload["unexpected"] = True
    mutated = tmp_path / "bundle.json"
    mutated.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = validate_bundle(mutated)

    assert not result.valid
    assert result.errors == ["cannot load closed M2-T03 human adjudication artifacts"]


def test_protocol_is_frozen_and_declares_blind_review_boundary() -> None:
    protocol = Path("docs/reviews/m2-t03-human-adjudication-protocol.md").read_text(
        encoding="utf-8"
    )

    for value in (
        "paper_id",
        "title",
        "abstract",
        "source",
        "source_id",
        "url",
        "ResearchIntent",
        "PROBLEM_EXISTENCE",
        "CURRENT_METHODS",
        "METHOD_TRANSFERABILITY",
        "IMPLEMENTATION_PATH",
        "EVALUATION_BASIS",
        "DIRECT",
        "INDIRECT",
        "HYPOTHETICAL",
        "CONFIRM",
        "REVISE",
        "REJECT",
    ):
        assert value in protocol


def test_current_gate_is_separate_from_pending_bundle_validation() -> None:
    assert hasattr(validator, "validate_current_m2_gate")
