"""Offline contract for the committed M2-T02 snapshot-download evidence."""

from __future__ import annotations

import json
import re
import stat
from pathlib import Path

import pytest

from scripts import download_m2_t02_reranker_snapshot as runner
from scripts.prepare_m2_t02_reranker_snapshot import SnapshotPlan, load_snapshot_plan

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SELECTION_PATH = Path("evaluation/source-artifacts/m2-t02-reranker-selection.json")
_EVIDENCE_PATH = Path(
    "evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json"
)
_TOP_LEVEL_FIELDS = {
    "report_version",
    "phase",
    "task_id",
    "baseline_commit",
    "decision_status",
    "platform",
    "model",
    "download_policy",
    "snapshot",
    "verification",
    "execution_state",
}
_PLATFORM_FIELDS = {"system", "machine", "python_version"}
_MODEL_FIELDS = {"provider_name", "model_id", "revision"}
_DOWNLOAD_POLICY_FIELDS = {
    "client",
    "client_version",
    "repo_type",
    "token",
    "implicit_token_disabled",
    "telemetry_disabled",
    "selection_path",
    "snapshot_path",
    "network_scope",
}
_SNAPSHOT_FIELDS = {
    "preparation_status",
    "file_count",
    "total_size_bytes",
    "required_runtime_size_bytes",
    "weight_size_bytes",
    "exact_file_closure",
    "local_hub_metadata_removed",
    "git_ignored",
    "files",
}
_SNAPSHOT_FILE_FIELDS = {"path", "size_bytes", "digest", "digest_type", "verified"}
_VERIFICATION_FIELDS = {
    "disk_space_gate",
    "size_verification",
    "digest_verification",
    "atomic_publication",
    "offline_reuse_check",
    "downloader_called_during_offline_reuse",
    "git_index_contains_snapshot_files",
}
_EXECUTION_STATE_FIELDS = {
    "snapshot_present_and_verified",
    "download_performed_this_run",
    "model_loaded",
    "inference_run",
    "real_scores_generated",
}
_LOCAL_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^\\\\|/(?:Users|home|tmp)/)")
_FORBIDDEN_TEXT = re.compile(
    r"(?:https?://|file://|authorization|cookie|bearer\s|access[_-]?token|"
    r"api[_-]?key|password|username|hostname|proxy|score|benchmark|duration|"
    r"elapsed|runtime_seconds)",
    re.IGNORECASE,
)


def _as_mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _string_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [child for item in value.values() for child in _string_values(item)]
    if isinstance(value, list):
        return [child for item in value for child in _string_values(item)]
    return [value] if isinstance(value, str) else []


def _load_committed_evidence() -> tuple[dict[str, object], SnapshotPlan]:
    evidence_path = _REPO_ROOT / _EVIDENCE_PATH
    assert evidence_path.is_file()
    metadata = evidence_path.stat()
    assert stat.S_ISREG(metadata.st_mode)
    assert not evidence_path.is_symlink()
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    assert not (getattr(metadata, "st_file_attributes", 0) & reparse_point)

    evidence_text = evidence_path.read_bytes().decode("utf-8")
    evidence = json.loads(evidence_text)
    assert isinstance(evidence, dict)
    plan = load_snapshot_plan(_REPO_ROOT / _SELECTION_PATH)
    return evidence, plan


def _selection_projection(plan: SnapshotPlan) -> list[dict[str, object]]:
    return [
        {
            "path": file.path,
            "size_bytes": file.size_bytes,
            "digest": file.digest,
            "digest_type": file.digest_type,
            "verified": True,
        }
        for file in plan.files
    ]


def test_committed_evidence_is_closed_utf8_json_without_sensitive_values() -> None:
    evidence, _ = _load_committed_evidence()

    assert set(evidence) == _TOP_LEVEL_FIELDS
    for value in _string_values(evidence):
        assert not _LOCAL_ABSOLUTE_PATH.search(value)
        assert not _FORBIDDEN_TEXT.search(value)

    assert evidence["report_version"] == "m2-t02-reranker-snapshot-download.v1"
    assert evidence["phase"] == "M2"
    assert evidence["task_id"] == "M2-T02"
    assert evidence["baseline_commit"] == "47587230c9bb02307c3a68807f149fa13df51e3d"
    assert evidence["decision_status"] == "downloaded_and_verified"

    platform_data = _as_mapping(evidence["platform"])
    assert set(platform_data) == _PLATFORM_FIELDS
    assert platform_data == {
        "system": "Windows",
        "machine": "AMD64",
        "python_version": "3.12.10",
    }
    assert re.fullmatch(r"3\.12\.\d+", str(platform_data["python_version"]))

    model = _as_mapping(evidence["model"])
    assert set(model) == _MODEL_FIELDS
    assert model == {
        "provider_name": "bge_reranker_v2_m3",
        "model_id": "BAAI/bge-reranker-v2-m3",
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    }

    policy = _as_mapping(evidence["download_policy"])
    assert set(policy) == _DOWNLOAD_POLICY_FIELDS
    assert policy == {
        "client": "huggingface_hub.hf_hub_download",
        "client_version": "0.34.3",
        "repo_type": "model",
        "token": False,
        "implicit_token_disabled": True,
        "telemetry_disabled": True,
        "selection_path": _SELECTION_PATH.as_posix(),
        "snapshot_path": (
            "models/m2-t02/bge-reranker-v2-m3/"
            "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
        ),
        "network_scope": "pinned_huggingface_model_files_only",
    }


def test_committed_evidence_matches_selection_and_production_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, plan = _load_committed_evidence()
    platform_data = _as_mapping(evidence["platform"])
    system = platform_data["system"]
    machine = platform_data["machine"]
    python_version = platform_data["python_version"]
    assert isinstance(system, str)
    assert isinstance(machine, str)
    assert isinstance(python_version, str)
    monkeypatch.setattr(runner.platform, "system", lambda: system)
    monkeypatch.setattr(runner.platform, "machine", lambda: machine)
    monkeypatch.setattr(runner.platform, "python_version", lambda: python_version)

    runner.validate_download_evidence(evidence, plan)

    snapshot = _as_mapping(evidence["snapshot"])
    assert set(snapshot) == _SNAPSHOT_FIELDS
    assert snapshot["preparation_status"] == "PUBLISHED"
    assert snapshot["file_count"] == 7
    assert snapshot["total_size_bytes"] == 2_293_259_337
    assert snapshot["required_runtime_size_bytes"] == 2_293_242_108
    assert snapshot["weight_size_bytes"] == 2_271_071_852
    assert snapshot["exact_file_closure"] is True
    assert snapshot["local_hub_metadata_removed"] is True
    assert snapshot["git_ignored"] is True
    assert snapshot["files"] == _selection_projection(plan)
    for file in snapshot["files"]:  # type: ignore[union-attr]
        assert set(file) == _SNAPSHOT_FILE_FIELDS

    verification = _as_mapping(evidence["verification"])
    assert set(verification) == _VERIFICATION_FIELDS
    assert verification == {
        "disk_space_gate": "passed",
        "size_verification": "passed",
        "digest_verification": "passed",
        "atomic_publication": "passed",
        "offline_reuse_check": "passed",
        "downloader_called_during_offline_reuse": False,
        "git_index_contains_snapshot_files": False,
    }

    execution_state = _as_mapping(evidence["execution_state"])
    assert set(execution_state) == _EXECUTION_STATE_FIELDS
    assert execution_state == {
        "snapshot_present_and_verified": True,
        "download_performed_this_run": True,
        "model_loaded": False,
        "inference_run": False,
        "real_scores_generated": False,
    }
