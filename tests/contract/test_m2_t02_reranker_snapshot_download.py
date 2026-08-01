"""Closed evidence contract for the future pinned reranker snapshot download."""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_MODULE_NAME = "scripts.download_m2_t02_reranker_snapshot"
_PREPARATION_MODULE = "scripts.prepare_m2_t02_reranker_snapshot"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SELECTION_PATH = Path("evaluation/source-artifacts/m2-t02-reranker-selection.json")
_SNAPSHOT_PATH = Path(
    "models/m2-t02/bge-reranker-v2-m3/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
)
_SELECTION_ARTIFACT = _REPO_ROOT / _SELECTION_PATH
_INSTALLATION_ARTIFACT = (
    _REPO_ROOT / "evaluation/source-artifacts/m2-t02-reranker-runtime-installation.json"
)
_MODEL_ID = "BAAI/bge-reranker-v2-m3"
_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
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
    "weights_downloaded",
    "tokenizer_downloaded",
    "model_loaded",
    "inference_run",
    "real_scores_generated",
}
_LOCAL_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^\\\\|/(?:Users|home|tmp)/)")
_FORBIDDEN_TEXT = re.compile(
    r"(?:authorization|cookie|bearer\s|access[_-]?token|api[_-]?key|password|"
    r"username|hostname|benchmark|duration|elapsed|runtime_seconds)",
    re.IGNORECASE,
)


@pytest.fixture(autouse=True)
def _unload_future_runner() -> None:
    sys.modules.pop(_MODULE_NAME, None)
    yield
    sys.modules.pop(_MODULE_NAME, None)


def _download_module() -> ModuleType:
    try:
        return importlib.import_module("scripts.download_m2_t02_reranker_snapshot")
    except ModuleNotFoundError:
        pytest.fail("download_m2_t02_reranker_snapshot has not been implemented")


def _string_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [string for child in value.values() for string in _string_values(child)]
    if isinstance(value, list):
        return [string for child in value for string in _string_values(child)]
    return [value] if isinstance(value, str) else []


def _source_file_projection() -> list[dict[str, object]]:
    selection = json.loads(_SELECTION_ARTIFACT.read_text(encoding="utf-8"))
    return [
        {
            "path": entry["path"],
            "size_bytes": entry["size_bytes"],
            "digest": entry["digest"],
            "digest_type": entry["digest_type"],
            "verified": True,
        }
        for entry in selection["source_files"]
    ]


def _run_fake_live_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], ModuleType, Path]:
    def fake_preparation(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status="PUBLISHED", snapshot_dir=kwargs["snapshot_dir"])

    preparation = importlib.import_module(_PREPARATION_MODULE)
    monkeypatch.setattr(preparation, "prepare_snapshot", fake_preparation)
    module = _download_module()
    evidence_path = tmp_path / "m2-t02-reranker-snapshot-download.json"
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def fake_version(distribution: str) -> str:
        assert distribution == "huggingface-hub"
        return "0.34.3"

    def fake_download(**_kwargs: object) -> str:
        return "fake-download-result"

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> ModuleType:
        if name != "huggingface_hub":
            return real_import_module(name, package)
        assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
        assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
        assert "HF_HUB_OFFLINE" not in os.environ
        fake_hub = ModuleType("huggingface_hub")
        fake_hub.hf_hub_download = fake_download  # type: ignore[attr-defined]
        return fake_hub

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    result = module.run_download(execute_live_download=True)
    assert isinstance(result, dict)
    return result, module, evidence_path


def _mutate_evidence(evidence: dict[str, Any], mutation: str) -> None:
    if mutation == "absolute-path":
        evidence["download_policy"]["snapshot_path"] = "C:/Users/example/snapshot"
    elif mutation == "username":
        evidence["platform"]["username"] = "example-user"
    elif mutation == "machine-name":
        evidence["platform"]["machine"] = "builder-host-17"
    elif mutation == "token":
        evidence["download_policy"]["token"] = "secret"
    elif mutation in {"cookie", "authorization", "bearer"}:
        evidence["unapproved"] = f"{mutation} secret-value"
    elif mutation == "file-content":
        evidence["snapshot"]["files"][0]["content"] = "not-permitted"
    elif mutation in {"score", "benchmark", "duration"}:
        evidence[mutation] = "not-permitted"
    elif mutation == "model-loaded":
        evidence["execution_state"]["model_loaded"] = True
    elif mutation == "inference-run":
        evidence["execution_state"]["inference_run"] = True
    elif mutation == "unknown-field":
        evidence["unapproved"] = True
    elif mutation == "missing-field":
        del evidence["verification"]
    elif mutation == "duplicate-file":
        evidence["snapshot"]["files"].append(copy.deepcopy(evidence["snapshot"]["files"][0]))
    elif mutation == "extra-file":
        evidence["snapshot"]["files"].append(
            {
                "path": "unexpected.bin",
                "size_bytes": 1,
                "digest": "0" * 64,
                "digest_type": "sha256",
                "verified": True,
            }
        )
    elif mutation == "wrong-digest":
        evidence["snapshot"]["files"][0]["digest"] = "0" * 40
    elif mutation == "wrong-size":
        evidence["snapshot"]["files"][0]["size_bytes"] += 1
    elif mutation == "floating-revision":
        evidence["model"]["revision"] = "main"
    else:
        raise AssertionError(f"unknown evidence mutation: {mutation}")


def test_selection_artifact_remains_immutable_not_downloaded() -> None:
    selection = json.loads(_SELECTION_ARTIFACT.read_text(encoding="utf-8"))

    assert selection["decision_status"] == "selected_not_downloaded"
    assert selection["execution_state"] == {
        "weights_downloaded": False,
        "tokenizer_downloaded": False,
        "model_loaded": False,
        "inference_run": False,
        "real_scores_generated": False,
    }


def test_runtime_installation_evidence_still_proves_exact_clean_dependencies() -> None:
    installation = json.loads(_INSTALLATION_ARTIFACT.read_text(encoding="utf-8"))
    packages = {entry["name"]: entry for entry in installation["installed_packages"]}

    assert packages["huggingface-hub"]["installed_version"] == "0.34.3"
    assert packages["torch"]["installed_version"] == "2.4.1+cpu"
    assert installation["verification"]["pip_check"] == "passed"
    assert installation["verification"]["all_imports"] == "passed"
    assert installation["execution_state"]["model_downloaded"] is False


def test_download_evidence_is_closed_atomic_safe_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, module, evidence_path = _run_fake_live_download(tmp_path, monkeypatch)
    module.validate_download_evidence(evidence)

    assert json.loads(evidence_path.read_text(encoding="utf-8")) == evidence
    assert not list(evidence_path.parent.glob(f".{evidence_path.name}.tmp-*"))
    assert set(evidence) == _TOP_LEVEL_FIELDS
    assert evidence["report_version"] == "m2-t02-reranker-snapshot-download.v1"
    assert evidence["phase"] == "M2"
    assert evidence["task_id"] == "M2-T02"
    assert evidence["baseline_commit"] == "47587230c9bb02307c3a68807f149fa13df51e3d"
    assert evidence["decision_status"] == "downloaded_and_verified"

    platform = evidence["platform"]
    assert set(platform) == _PLATFORM_FIELDS
    assert platform == {
        "system": "Windows",
        "machine": "AMD64",
        "python_version": platform["python_version"],
    }
    assert re.fullmatch(r"3\.12\.\d+", platform["python_version"])

    model = evidence["model"]
    assert set(model) == _MODEL_FIELDS
    assert model == {
        "provider_name": "bge_reranker_v2_m3",
        "model_id": _MODEL_ID,
        "revision": _REVISION,
    }

    policy = evidence["download_policy"]
    assert set(policy) == _DOWNLOAD_POLICY_FIELDS
    assert policy == {
        "client": "huggingface_hub.hf_hub_download",
        "client_version": "0.34.3",
        "repo_type": "model",
        "token": False,
        "implicit_token_disabled": True,
        "telemetry_disabled": True,
        "selection_path": _SELECTION_PATH.as_posix(),
        "snapshot_path": _SNAPSHOT_PATH.as_posix(),
        "network_scope": "pinned_huggingface_model_files_only",
    }

    snapshot = evidence["snapshot"]
    assert set(snapshot) == _SNAPSHOT_FIELDS
    assert snapshot["preparation_status"] == "PUBLISHED"
    assert snapshot["file_count"] == 7
    assert snapshot["total_size_bytes"] == 2293259337
    assert snapshot["required_runtime_size_bytes"] == 2293242108
    assert snapshot["weight_size_bytes"] == 2271071852
    assert snapshot["exact_file_closure"] is True
    assert snapshot["local_hub_metadata_removed"] is True
    assert snapshot["git_ignored"] is True
    assert snapshot["files"] == _source_file_projection()
    for entry in snapshot["files"]:
        assert set(entry) == _SNAPSHOT_FILE_FIELDS

    verification = evidence["verification"]
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
    assert set(evidence["execution_state"]) == _EXECUTION_STATE_FIELDS
    assert evidence["execution_state"] == {
        "weights_downloaded": True,
        "tokenizer_downloaded": True,
        "model_loaded": False,
        "inference_run": False,
        "real_scores_generated": False,
    }

    for value in _string_values(evidence):
        assert not _LOCAL_ABSOLUTE_PATH.search(value)
        assert not _FORBIDDEN_TEXT.search(value)


@pytest.mark.parametrize(
    "mutation",
    (
        "absolute-path",
        "username",
        "machine-name",
        "token",
        "cookie",
        "authorization",
        "bearer",
        "file-content",
        "score",
        "benchmark",
        "duration",
        "model-loaded",
        "inference-run",
        "unknown-field",
        "missing-field",
        "duplicate-file",
        "extra-file",
        "wrong-digest",
        "wrong-size",
        "floating-revision",
    ),
)
def test_download_evidence_validator_rejects_privacy_integrity_and_scope_violations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    evidence, module, _evidence_path = _run_fake_live_download(tmp_path, monkeypatch)
    mutated = copy.deepcopy(evidence)
    _mutate_evidence(mutated, mutation)

    with pytest.raises(ValueError):
        module.validate_download_evidence(mutated)
