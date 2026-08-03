"""Closed-schema and atomic-publication contracts for B3-I fake reports."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

import scripts.run_m2_t02_candidate_reranking as runner
from tests.unit.test_m2_t02_candidate_reranking_runner import _FakeProvider, _paths


def _report(tmp_path: Path) -> dict[str, object]:
    paths = _paths(tmp_path)
    return runner.run_candidate_reranking(
        execute_local_reranking=True,
        paths=paths,
        provider_factory=lambda _snapshot: _FakeProvider(equal_scores=True),
        model_snapshot_validator=lambda _snapshot: None,
        publish_result=False,
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("top-level-extra", "closed"),
        ("record-extra", "closed"),
        ("record-count", "count"),
        ("duplicate-id", "identity"),
        ("missing-id", "identity"),
        ("nan", "score"),
        ("infinite", "score"),
        ("wrong-rank", "rank"),
        ("out-of-range", "normalized"),
        ("normalized-mismatch", "normalized"),
        ("wrong-snapshot-hash", "snapshot"),
        ("wrong-input-hash", "input-hash"),
        ("wrong-declared-order", "candidate-truth"),
        ("wrong-order", "order"),
    ],
)
def test_candidate_report_validator_rejects_closed_contract_mutations(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    del expected
    report = _report(tmp_path)
    mutated = copy.deepcopy(report)
    records = mutated["records"]
    assert isinstance(records, list)
    if mutation == "top-level-extra":
        mutated["unexpected"] = True
    elif mutation == "record-extra":
        records[0]["title"] = "must not be persisted"
    elif mutation == "record-count":
        records.pop()
    elif mutation == "duplicate-id":
        records[1]["paper_id"] = records[0]["paper_id"]
    elif mutation == "missing-id":
        records[0]["paper_id"] = "arxiv:missing"
    elif mutation == "nan":
        records[0]["raw_score"] = float("nan")
    elif mutation == "infinite":
        records[0]["raw_score"] = float("inf")
    elif mutation == "wrong-rank":
        records[0]["rank"] = 2
    elif mutation == "out-of-range":
        records[0]["normalized_score"] = 2.0
    elif mutation == "normalized-mismatch":
        records[0]["normalized_score"] = 0.5
    elif mutation == "wrong-snapshot-hash":
        mutated["input"]["candidate_snapshot_sha256"] = "0" * 64
    elif mutation == "wrong-input-hash":
        records[0]["input_sha256"] = "0" * 64
    elif mutation == "wrong-declared-order":
        mutated["input"]["paper_id_order"] = list(reversed(mutated["input"]["paper_id_order"]))
    else:
        records[0], records[1] = records[1], records[0]

    with pytest.raises(runner.CandidateRerankingError):
        runner.validate_candidate_run_report(mutated)


def test_atomic_publish_is_byte_idempotent_and_preserves_mtime(tmp_path: Path) -> None:
    report = _report(tmp_path)
    destination = tmp_path / "candidate-run.json"

    assert runner.publish_candidate_run(report, destination=destination) == "PUBLISHED"
    first_bytes = destination.read_bytes()
    first_mtime = destination.stat().st_mtime_ns
    assert runner.publish_candidate_run(report, destination=destination) == "REUSED"
    assert destination.read_bytes() == first_bytes
    assert destination.stat().st_mtime_ns == first_mtime
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("existing", ["different", "malformed", "directory"])
def test_atomic_publish_rejects_existing_conflicts_without_replacement(
    tmp_path: Path, existing: str
) -> None:
    report = _report(tmp_path)
    destination = tmp_path / "candidate-run.json"
    if existing == "different":
        destination.write_bytes(b"different\n")
    elif existing == "malformed":
        destination.write_bytes(b"not-json\n")
    else:
        destination.mkdir()

    before = None if destination.is_dir() else destination.read_bytes()
    with pytest.raises(runner.CandidateRerankingError) as captured:
        runner.publish_candidate_run(report, destination=destination)
    assert captured.value.code == "RESULT_CONFLICT"
    if before is not None:
        assert destination.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_publish_rejects_symlink_target_when_supported(tmp_path: Path) -> None:
    report = _report(tmp_path)
    destination = tmp_path / "candidate-run.json"
    target = tmp_path / "target.json"
    target.write_bytes(b"target")
    try:
        destination.symlink_to(target)
    except OSError:
        pytest.fail("symlink creation is required for the path-safety contract")
    with pytest.raises(runner.CandidateRerankingError) as captured:
        runner.publish_candidate_run(report, destination=destination)
    assert captured.value.code == "RESULT_CONFLICT"
    assert destination.is_symlink()


def test_report_serialization_is_closed_and_deterministic(tmp_path: Path) -> None:
    report = _report(tmp_path)
    runner.validate_candidate_run_report(report)
    destination = tmp_path / "candidate-run.json"
    runner.publish_candidate_run(report, destination=destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert set(payload) == {
        "report_version",
        "phase",
        "task_id",
        "baseline_commit",
        "decision_status",
        "input",
        "model",
        "runtime",
        "policy",
        "execution",
        "records",
    }
    assert stat.S_ISREG(destination.stat().st_mode)
    assert os.path.commonpath([str(destination), str(tmp_path)]) == str(tmp_path)


def test_validator_requires_committed_candidate_snapshot_bytes_and_input_hashes(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)
    records = report["records"]
    assert isinstance(records, list)
    assert report["input"]["candidate_snapshot_sha256"] == (  # type: ignore[index]
        "4a2aec0fd0a1d22adc801fd3bc506e5da89895d1276cd572e2ac64014c162448"
    )
    expected_hash = records[0]["input_sha256"]
    records[0]["input_sha256"] = hashlib.sha256(b"not-the-fixed-candidate").hexdigest()
    assert records[0]["input_sha256"] != expected_hash
    with pytest.raises(runner.CandidateRerankingError) as captured:
        runner.validate_candidate_run_report(report)
    assert captured.value.code == "RESULT_SCHEMA_INVALID"
