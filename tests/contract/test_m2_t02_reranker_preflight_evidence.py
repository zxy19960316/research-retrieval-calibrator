"""Offline CI contracts for the real M2-T02 preflight evidence."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE_REL = "evaluation/source-artifacts/m2-t02-reranker-preflight.json"
_RECEIPT_REL = "evaluation/source-artifacts/m2-t02-reranker-preflight-run-receipt.json"
_RUNNER_REL = "scripts/run_m2_t02_reranker_preflight.py"
_SELECTION_REL = "evaluation/source-artifacts/m2-t02-reranker-selection.json"
_RUNTIME_REL = "evaluation/source-artifacts/m2-t02-reranker-runtime-installation.json"
_DOWNLOAD_REL = "evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json"

_FORMAL_EVIDENCE = _REPO_ROOT / _EVIDENCE_REL
_RECEIPT = _REPO_ROOT / _RECEIPT_REL
_RUNNER = _REPO_ROOT / _RUNNER_REL
_SELECTION = _REPO_ROOT / _SELECTION_REL
_RUNTIME = _REPO_ROOT / _RUNTIME_REL
_DOWNLOAD = _REPO_ROOT / _DOWNLOAD_REL

_EXPECTED_EVIDENCE_FIELDS = {
    "report_version",
    "phase",
    "task_id",
    "baseline_commit",
    "decision_status",
    "platform",
    "model",
    "snapshot",
    "runtime",
    "offline_policy",
    "memory",
    "tokenizer",
    "model_load",
    "inference",
    "execution_state",
}
_EXPECTED_RECEIPT_FIELDS = {
    "report_version",
    "phase",
    "task_id",
    "execution_commit",
    "evidence_path",
    "evidence_sha256",
    "runner_path",
    "runner_sha256",
    "selection_path",
    "selection_sha256",
    "runtime_installation_path",
    "runtime_installation_sha256",
    "snapshot_download_evidence_path",
    "snapshot_download_evidence_sha256",
    "exit_code",
    "run_count",
    "decision_status",
    "formal_evidence_created",
    "model_rerun_performed",
    "full_candidate_reranking_run",
}
_EXECUTION_COMMIT = "a50f1b9f5467c318dbbabe8807df6ede47d79b14"
_RUNTIME_MODULES = {
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
}


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


def test_formal_preflight_evidence_is_closed_and_runtime_free() -> None:
    _assert_regular_file(_FORMAL_EVIDENCE)
    raw, evidence = _read_json(_FORMAL_EVIDENCE)
    assert isinstance(evidence, dict)
    assert raw.decode("utf-8")

    before = {
        name.split(".", 1)[0]
        for name in sys.modules
        if name.split(".", 1)[0] in _RUNTIME_MODULES
    }
    validator = importlib.import_module(
        "scripts.run_m2_t02_reranker_preflight"
    ).validate_preflight_evidence
    validator(evidence)
    after = {
        name.split(".", 1)[0]
        for name in sys.modules
        if name.split(".", 1)[0] in _RUNTIME_MODULES
    }
    assert after == before

    assert set(evidence) == _EXPECTED_EVIDENCE_FIELDS
    assert evidence["decision_status"] == "cpu_float32_preflight_passed"
    assert evidence["platform"] == {
        "system": "Windows",
        "machine": "AMD64",
        "python_version": "3.12.10",
    }

    snapshot = evidence["snapshot"]
    assert snapshot["preparation_status"] == "REUSED"
    assert snapshot["downloader_calls"] == 0
    assert snapshot["disk_usage_calls"] == 0

    runtime = evidence["runtime"]
    assert runtime["torch_distribution_version"] == "2.4.1+cpu"
    assert runtime["transformers_version"] == "4.53.2"
    assert runtime["huggingface_hub_version"] == "0.34.3"
    assert runtime["safetensors_version"] == "0.5.3"
    assert runtime["tokenizers_version"] == "0.21.2"
    assert runtime["torch_cuda_version"] is None
    assert runtime["torch_cuda_available"] is False

    model_load = evidence["model_load"]
    assert model_load["device"] == "cpu"
    assert model_load["torch_dtype"] == "float32"
    assert model_load["eval_mode"] is True
    assert model_load["local_files_only"] is True
    assert model_load["trust_remote_code"] is False
    assert model_load["use_safetensors"] is True

    inference = evidence["inference"]
    assert inference["batch_size"] == 2
    assert inference["max_length"] == 512
    assert inference["logits_shape"] == [2, 1]
    assert inference["logits_dtype"] == "float32"
    assert inference["logits_all_finite"] is True
    assert inference["logits_persisted"] is False

    execution = evidence["execution_state"]
    assert execution["tokenizer_loaded"] is True
    assert execution["model_loaded"] is True
    assert execution["minimal_preflight_inference_run"] is True
    assert execution["preflight_logits_generated"] is True
    assert execution["preflight_logits_persisted"] is False
    assert execution["full_candidate_reranking_run"] is False
    assert execution["real_candidate_scores_generated"] is False

    memory = evidence["memory"]
    assert set(memory) == {
        "system_total_physical_memory_bytes",
        "system_available_physical_memory_before_bytes",
        "process_working_set_before_bytes",
        "process_peak_working_set_after_load_bytes",
        "process_peak_working_set_after_inference_bytes",
    }
    assert all(type(value) is int and value >= 0 for value in memory.values())

    serialized = json.dumps(evidence, ensure_ascii=False)
    assert "raw_logits" not in serialized
    assert "candidate_id" not in serialized
    assert "人工智能" not in serialized
    _assert_privacy_safe(evidence)


def test_execution_receipt_is_closed_and_hash_bound() -> None:
    _assert_regular_file(_RECEIPT)
    _raw, receipt = _read_json(_RECEIPT)
    assert isinstance(receipt, dict)
    assert set(receipt) == _EXPECTED_RECEIPT_FIELDS
    assert receipt["report_version"] == "m2-t02-reranker-preflight-run-receipt.v1"
    assert receipt["phase"] == "M2"
    assert receipt["task_id"] == "M2-T02"
    assert receipt["execution_commit"] == _EXECUTION_COMMIT
    assert receipt["exit_code"] == 0
    assert receipt["run_count"] == 1
    assert receipt["decision_status"] == "cpu_float32_preflight_passed"
    assert receipt["formal_evidence_created"] is True
    assert receipt["model_rerun_performed"] is False
    assert receipt["full_candidate_reranking_run"] is False

    relative_paths = {
        "evidence_path": _EVIDENCE_REL,
        "runner_path": _RUNNER_REL,
        "selection_path": _SELECTION_REL,
        "runtime_installation_path": _RUNTIME_REL,
        "snapshot_download_evidence_path": _DOWNLOAD_REL,
    }
    for field, expected_path in relative_paths.items():
        assert receipt[field] == expected_path
        assert not Path(receipt[field]).is_absolute()
        assert "\\" not in receipt[field]
        assert ".." not in Path(receipt[field]).parts

    hash_paths = {
        "evidence_sha256": _FORMAL_EVIDENCE,
        "runner_sha256": _RUNNER,
        "selection_sha256": _SELECTION,
        "runtime_installation_sha256": _RUNTIME,
        "snapshot_download_evidence_sha256": _DOWNLOAD,
    }
    for field, path in hash_paths.items():
        actual = _sha256(path)
        assert receipt[field] == actual
        assert len(actual) == 64
        int(receipt[field], 16)

    assert receipt["selection_sha256"] == (
        "19e86cf2176969236e70ff9770ac086a5555d40892c0bfa913ac768eb752604b"
    )
    assert receipt["runtime_installation_sha256"] == (
        "feb79ad507afb4bb7027e5aa49765b941414fe6e9a9e3ac024e7eba6d2ced831"
    )
    _assert_privacy_safe(receipt)


def test_ci_contract_validates_without_a_local_model_snapshot(tmp_path: Path) -> None:
    isolated_root = tmp_path / "ci-without-models"
    isolated_root.mkdir()
    (isolated_root / "evidence.json").write_bytes(_FORMAL_EVIDENCE.read_bytes())
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

from scripts.run_m2_t02_reranker_preflight import validate_preflight_evidence

assert not Path("models").exists()
evidence = json.loads(Path("evidence.json").read_text(encoding="utf-8"))
validate_preflight_evidence(evidence)
runtime_modules = {"torch", "transformers", "tokenizers", "safetensors"}
assert not any(name.split(".", 1)[0] in runtime_modules for name in sys.modules)
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
