"""Offline contracts for the verified M2-T02 candidate reranking evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path

import scripts.run_m2_t02_candidate_reranking as runner
from app.core.reranking import build_reranker_input
from app.models.embedding import FrozenCandidate

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULT_REL = "evaluation/source-artifacts/m2-t02-reranker-candidate-run.json"
_RECEIPT_REL = (
    "evaluation/source-artifacts/m2-t02-reranker-candidate-run-receipt.json"
)
_RUNNER_REL = "scripts/run_m2_t02_candidate_reranking.py"
_SNAPSHOT_REL = "evaluation/snapshots/m2/m1-candidates.v1.json"
_MANIFEST_REL = "evaluation/snapshots/m2/m1-candidates.v1.manifest.json"
_SELECTION_REL = "evaluation/source-artifacts/m2-t02-reranker-selection.json"
_RUNTIME_REL = "evaluation/source-artifacts/m2-t02-reranker-runtime-installation.json"
_DOWNLOAD_REL = "evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json"
_PREFLIGHT_REL = "evaluation/source-artifacts/m2-t02-reranker-preflight.json"
_PREFLIGHT_RECEIPT_REL = (
    "evaluation/source-artifacts/m2-t02-reranker-preflight-run-receipt.json"
)

_RESULT = _REPO_ROOT / _RESULT_REL
_RECEIPT = _REPO_ROOT / _RECEIPT_REL
_RUNNER = _REPO_ROOT / _RUNNER_REL
_SNAPSHOT = _REPO_ROOT / _SNAPSHOT_REL
_MANIFEST = _REPO_ROOT / _MANIFEST_REL
_SELECTION = _REPO_ROOT / _SELECTION_REL
_RUNTIME = _REPO_ROOT / _RUNTIME_REL
_DOWNLOAD = _REPO_ROOT / _DOWNLOAD_REL
_PREFLIGHT = _REPO_ROOT / _PREFLIGHT_REL
_PREFLIGHT_RECEIPT = _REPO_ROOT / _PREFLIGHT_RECEIPT_REL

_EXECUTION_COMMIT = "c1f95238480d8cfe69a79f705d98caadbf372220"
_PREFLIGHT_EXECUTION_COMMIT = "a50f1b9f5467c318dbbabe8807df6ede47d79b14"
_EXPECTED_RESULT_SHA256 = (
    "0a751dbc35bfa8d07433796113939412bfbdffd4c5c13043c90390bccfc5c35b"
)
_EXPECTED_RUNNER_SHA256 = (
    "3ad6c629b2f6b1f28c3119f9f7d810226e28e6d572535e28b0de6566ef8f63b1"
)
_EXPECTED_SNAPSHOT_SHA256 = (
    "4a2aec0fd0a1d22adc801fd3bc506e5da89895d1276cd572e2ac64014c162448"
)
_EXPECTED_MANIFEST_SHA256 = (
    "b6ce7cfab2e5df6b8c84b77fb38c6f573de7429d37a9688d7b4c62e27f8546a8"
)
_EXPECTED_PREFLIGHT_SHA256 = (
    "103a73cc82674c2c1f1139a72a7a3d1a880029a7c01d292f52de24d6dc781b06"
)
_EXPECTED_PREFLIGHT_RECEIPT_SHA256 = (
    "fe5ea3c239de9f1f62bdc0204c812f27984af254b08ab39a125e4ca709e05d28"
)
_EXPECTED_SELECTION_SHA256 = (
    "19e86cf2176969236e70ff9770ac086a5555d40892c0bfa913ac768eb752604b"
)
_EXPECTED_RUNTIME_SHA256 = (
    "feb79ad507afb4bb7027e5aa49765b941414fe6e9a9e3ac024e7eba6d2ced831"
)
_EXPECTED_DOWNLOAD_SHA256 = (
    "5114ce6ef2ca637f950e1683bfc7003029daff4f0f228ca6b909d0c97fb84bf8"
)

_EXPECTED_RESULT_FIELDS = {
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
_EXPECTED_RECORD_FIELDS = {
    "rank",
    "paper_id",
    "input_sha256",
    "raw_score",
    "normalized_score",
}
_EXPECTED_RECEIPT_FIELDS = {
    "report_version",
    "phase",
    "task_id",
    "execution_commit",
    "result_path",
    "result_sha256",
    "runner_path",
    "runner_sha256",
    "candidate_snapshot_path",
    "candidate_snapshot_sha256",
    "candidate_manifest_path",
    "candidate_manifest_sha256",
    "model_selection_path",
    "model_selection_sha256",
    "runtime_installation_path",
    "runtime_installation_sha256",
    "snapshot_download_evidence_path",
    "snapshot_download_evidence_sha256",
    "preflight_evidence_path",
    "preflight_evidence_sha256",
    "preflight_receipt_path",
    "preflight_receipt_sha256",
    "runtime",
    "exit_code",
    "run_count",
    "decision_status",
    "formal_result_created",
    "full_candidate_reranking_run",
    "additional_model_rerun_performed",
    "replay_performed",
}
_RUNTIME_MODULES = {"torch", "transformers", "tokenizers", "safetensors"}


def _read_json(path: Path) -> tuple[bytes, object]:
    raw = path.read_bytes()
    return raw, json.loads(raw.decode("utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_regular_file(path: Path) -> None:
    metadata = path.lstat()
    assert stat.S_ISREG(metadata.st_mode)
    assert not path.is_symlink()
    assert not path.is_dir()
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction):
        assert not is_junction()


def _strings(value: object):
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def _assert_privacy_safe(value: object) -> None:
    for text in _strings(value):
        lowered = text.lower()
        assert "http://" not in lowered
        assert "https://" not in lowered
        assert "file://" not in lowered
        assert "authorization" not in lowered
        assert "cookie" not in lowered
        assert "bearer" not in lowered
        assert "access_token" not in lowered
        assert "api_key" not in lowered
        assert "password" not in lowered
        assert "hostname" not in lowered
        assert "C:\\" not in text
        assert "/users/" not in lowered
        assert "/home/" not in lowered


def _assert_relative_path(value: object, expected: str) -> None:
    assert value == expected
    assert isinstance(value, str)
    path = Path(value)
    assert not path.is_absolute()
    assert "\\" not in value
    assert ".." not in path.parts


def test_candidate_result_is_closed_and_validated_from_original_bytes() -> None:
    _assert_regular_file(_RESULT)
    raw, result = _read_json(_RESULT)
    assert raw.decode("utf-8")
    assert isinstance(result, dict)

    runner.validate_candidate_run_report(result)
    assert set(result) == _EXPECTED_RESULT_FIELDS
    assert result["decision_status"] == "candidate_reranking_completed"
    assert result["input"]["candidate_snapshot_sha256"] == _EXPECTED_SNAPSHOT_SHA256
    assert result["input"]["preflight_evidence_sha256"] == _EXPECTED_PREFLIGHT_SHA256
    assert (
        result["input"]["preflight_receipt_execution_commit"]
        == _PREFLIGHT_EXECUTION_COMMIT
    )

    snapshot = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    candidates = [FrozenCandidate.model_validate(item) for item in snapshot["candidates"]]
    expected_input_hashes = {
        candidate.paper_id: build_reranker_input(candidate).input_sha256
        for candidate in candidates
    }

    records = result["records"]
    assert isinstance(records, list)
    assert len(records) == 33
    assert [record["rank"] for record in records] == list(range(1, 34))
    assert all(set(record) == _EXPECTED_RECORD_FIELDS for record in records)
    assert {record["paper_id"] for record in records} == set(manifest["paper_id_order"])
    assert {
        record["paper_id"]: record["input_sha256"] for record in records
    } == expected_input_hashes

    raw_scores = [float(record["raw_score"]) for record in records]
    assert all(math.isfinite(score) for score in raw_scores)
    ordered_pairs = [
        (score, record["paper_id"]) for score, record in zip(raw_scores, records)
    ]
    assert ordered_pairs == sorted(ordered_pairs, key=lambda item: (-item[0], item[1]))
    minimum = min(raw_scores)
    maximum = max(raw_scores)
    for record, raw_score in zip(records, raw_scores):
        expected = (
            1.0 if minimum == maximum else (raw_score - minimum) / (maximum - minimum)
        )
        assert 0.0 <= record["normalized_score"] <= 1.0
        assert record["normalized_score"] == expected
    assert records[0]["normalized_score"] == 1.0
    assert records[-1]["normalized_score"] == 0.0

    assert result["execution"] == {
        "provider_instance_count": 1,
        "runtime_load_count": 1,
        "provider_call_count": 17,
        "provider_scored_count": 33,
        "batch_sizes": [2] * 16 + [1],
        "cache_hits": 0,
    }

    serialized = raw.decode("utf-8").lower()
    for forbidden in (
        '"title"',
        '"abstract"',
        '"authors"',
        '"url"',
        '"token"',
        '"logits"',
    ):
        assert forbidden not in serialized


def test_execution_receipt_is_closed_and_hash_bound() -> None:
    _assert_regular_file(_RECEIPT)
    raw, receipt = _read_json(_RECEIPT)
    assert raw.decode("utf-8")
    assert isinstance(receipt, dict)
    assert set(receipt) == _EXPECTED_RECEIPT_FIELDS
    assert receipt["report_version"] == "m2-t02-reranker-candidate-run-receipt.v1"
    assert receipt["phase"] == "M2"
    assert receipt["task_id"] == "M2-T02"
    assert receipt["execution_commit"] == _EXECUTION_COMMIT
    assert receipt["exit_code"] == 0
    assert receipt["run_count"] == 1
    assert receipt["decision_status"] == "candidate_reranking_completed"
    assert receipt["formal_result_created"] is True
    assert receipt["full_candidate_reranking_run"] is True
    assert receipt["additional_model_rerun_performed"] is False
    assert receipt["replay_performed"] is False
    assert receipt["runtime"] == {
        "python_version": "3.12.10",
        "torch_distribution_version": "2.4.1+cpu",
        "transformers_version": "4.53.2",
        "huggingface_hub_version": "0.34.3",
        "safetensors_version": "0.5.3",
        "tokenizers_version": "0.21.2",
        "pydantic_version": "2.9.2",
        "torch_cuda_version": None,
        "torch_cuda_available": False,
    }

    relative_paths = {
        "result_path": _RESULT_REL,
        "runner_path": _RUNNER_REL,
        "candidate_snapshot_path": _SNAPSHOT_REL,
        "candidate_manifest_path": _MANIFEST_REL,
        "model_selection_path": _SELECTION_REL,
        "runtime_installation_path": _RUNTIME_REL,
        "snapshot_download_evidence_path": _DOWNLOAD_REL,
        "preflight_evidence_path": _PREFLIGHT_REL,
        "preflight_receipt_path": _PREFLIGHT_RECEIPT_REL,
    }
    for field, expected in relative_paths.items():
        _assert_relative_path(receipt[field], expected)

    hash_paths = {
        "result_sha256": _RESULT,
        "runner_sha256": _RUNNER,
        "candidate_snapshot_sha256": _SNAPSHOT,
        "candidate_manifest_sha256": _MANIFEST,
        "model_selection_sha256": _SELECTION,
        "runtime_installation_sha256": _RUNTIME,
        "snapshot_download_evidence_sha256": _DOWNLOAD,
        "preflight_evidence_sha256": _PREFLIGHT,
        "preflight_receipt_sha256": _PREFLIGHT_RECEIPT,
    }
    for field, path in hash_paths.items():
        actual = _sha256(path)
        assert receipt[field] == actual
        assert len(actual) == 64
        int(actual, 16)
    assert receipt["result_sha256"] == _EXPECTED_RESULT_SHA256
    assert receipt["runner_sha256"] == _EXPECTED_RUNNER_SHA256
    assert receipt["candidate_snapshot_sha256"] == _EXPECTED_SNAPSHOT_SHA256
    assert receipt["candidate_manifest_sha256"] == _EXPECTED_MANIFEST_SHA256
    assert receipt["preflight_evidence_sha256"] == _EXPECTED_PREFLIGHT_SHA256
    assert receipt["preflight_receipt_sha256"] == _EXPECTED_PREFLIGHT_RECEIPT_SHA256
    assert receipt["model_selection_sha256"] == _EXPECTED_SELECTION_SHA256
    assert receipt["runtime_installation_sha256"] == _EXPECTED_RUNTIME_SHA256
    assert receipt["snapshot_download_evidence_sha256"] == _EXPECTED_DOWNLOAD_SHA256

    serialized = raw.decode("utf-8").lower()
    for forbidden in (
        "absolute",
        "username",
        "hostname",
        "http://",
        "https://",
        '"title"',
        '"abstract"',
        '"score"',
        '"ranking"',
        '"token"',
        '"logits"',
    ):
        assert forbidden not in serialized
    _assert_privacy_safe(receipt)


def test_ci_contract_validates_without_a_local_model_snapshot(tmp_path: Path) -> None:
    isolated_root = tmp_path / "ci-without-models"
    isolated_root.mkdir()
    (isolated_root / "result.json").write_bytes(_RESULT.read_bytes())
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    pythonpath = [str(_REPO_ROOT)]
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    code = """
import json
import sys
from pathlib import Path

blocked = {"torch", "transformers", "tokenizers", "safetensors"}
before = {name.split(".", 1)[0] for name in sys.modules}
from scripts.run_m2_t02_candidate_reranking import validate_candidate_run_report

assert not Path("models").exists()
validate_candidate_run_report(json.loads(Path("result.json").read_text(encoding="utf-8")))
after = {name.split(".", 1)[0] for name in sys.modules}
assert not (after - before) & blocked
assert not list(Path(".").glob("m2-t02-candidate-reranking-cache-*"))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=isolated_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
