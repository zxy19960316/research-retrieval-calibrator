"""Closed artifact and receipt contracts for M2-T03."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import scripts.run_m2_t03_evidence_classification as runner
from app.models.evidence_classification import EvidenceClassificationError
from tests.unit.test_m2_t03_evidence_classification_runner import _paths, _run


def test_run_report_has_exact_closed_top_level_and_record_schemas(tmp_path: Path) -> None:
    report = _run(_paths(tmp_path))
    assert set(report) == {
        "report_version",
        "phase",
        "task_id",
        "decision_status",
        "input",
        "classifier",
        "policy",
        "execution",
        "records",
    }
    assert set(report["input"]) == {  # type: ignore[index]
        "candidate_snapshot_path",
        "candidate_snapshot_sha256",
        "candidate_manifest_path",
        "candidate_manifest_sha256",
        "reranker_result_path",
        "reranker_result_sha256",
        "reranker_receipt_path",
        "reranker_receipt_sha256",
        "candidate_count",
        "paper_id_order",
    }
    records = report["records"]
    assert isinstance(records, list)
    assert all(
        set(record)
        == {
            "paper_id",
            "evidence_slot",
            "support_level",
            "reason",
            "supporting_excerpt",
            "source_text_sha256",
            "classifier_descriptor",
            "classification_version",
            "state",
        }
        for record in records
    )


@pytest.mark.parametrize("mutation", ["top-level-extra", "record-extra", "wrong-count", "wrong-hash"])
def test_run_report_validator_rejects_closed_schema_mutations(
    tmp_path: Path, mutation: str
) -> None:
    report = _run(_paths(tmp_path))
    mutated = copy.deepcopy(report)
    if mutation == "top-level-extra":
        mutated["selection_rank"] = 1
    elif mutation == "record-extra":
        mutated["records"][0]["score"] = 1.0  # type: ignore[index]
    elif mutation == "wrong-count":
        mutated["execution"]["record_count"] = 32  # type: ignore[index]
    else:
        mutated["input"]["candidate_snapshot_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(EvidenceClassificationError):
        runner.validate_classification_run_report(mutated)


def test_formal_artifacts_are_closed_json_and_have_no_absolute_paths_or_raw_errors(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _run(paths, publish_result=True)
    report = json.loads(paths.result_path.read_text(encoding="utf-8"))
    receipt = json.loads(paths.receipt_path.read_text(encoding="utf-8"))
    assert report["input"]["candidate_snapshot_path"].startswith("evaluation/")
    assert receipt["result_path"].startswith("evaluation/")
    serialized = json.dumps([report, receipt], ensure_ascii=False)
    assert "D:\\" not in serialized
    assert "provider failure must not be persisted" not in serialized
    assert "authorization" not in serialized.casefold()
    assert "cookie" not in serialized.casefold()
