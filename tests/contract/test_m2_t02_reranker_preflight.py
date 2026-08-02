"""Closed schema and subprocess contracts for B2-P-I."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_MODULE_NAME = "scripts.run_m2_t02_reranker_preflight"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORMAL_EVIDENCE = _REPO_ROOT / "evaluation/source-artifacts/m2-t02-reranker-preflight.json"
_SELECTION = _REPO_ROOT / "evaluation/source-artifacts/m2-t02-reranker-selection.json"
_DOWNLOAD_EVIDENCE = _REPO_ROOT / "evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json"


@pytest.fixture(autouse=True)
def _unload_runner() -> None:
    sys.modules.pop(_MODULE_NAME, None)
    yield
    sys.modules.pop(_MODULE_NAME, None)


def _runner() -> ModuleType:
    return importlib.import_module(_MODULE_NAME)


def _run_fake_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
    unit = importlib.import_module("tests.unit.test_m2_t02_reranker_preflight_runner")
    module = _runner()
    unit._install_valid_versions(monkeypatch)
    unit._install_fake_runtime(monkeypatch)
    unit._install_fake_reuse(monkeypatch, module)
    monkeypatch.setattr(module, "PREFLIGHT_EVIDENCE_PATH", tmp_path / "preflight.json")
    return module.run_preflight(
        execute_local_preflight=True,
        memory_probe=unit._memory_probe(),
    )


def _load_temp_download_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutate: object,
) -> ModuleType:
    module = _runner()
    selection_path = tmp_path / "selection.json"
    selection_path.write_bytes(_SELECTION.read_bytes())
    evidence_path = tmp_path / "download.json"
    evidence = json.loads(_DOWNLOAD_EVIDENCE.read_text(encoding="utf-8"))
    mutate(evidence)  # type: ignore[operator]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    monkeypatch.setattr(module, "SELECTION_PATH", selection_path)
    monkeypatch.setattr(module, "DOWNLOAD_EVIDENCE_PATH", evidence_path)
    return module


def test_fixed_download_evidence_is_present_but_formal_preflight_evidence_is_absent() -> None:
    assert _DOWNLOAD_EVIDENCE.is_file()
    assert not _FORMAL_EVIDENCE.exists()


def test_download_evidence_decision_status_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_temp_download_evidence(
        monkeypatch,
        tmp_path,
        lambda value: value.update({"decision_status": "reused_and_verified"}),
    )
    with pytest.raises(RuntimeError) as raised:
        module._load_download_evidence()  # type: ignore[attr-defined]
    assert getattr(raised.value, "code", None) == "DOWNLOAD_EVIDENCE_INVALID"


def test_download_evidence_preparation_status_must_be_published(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_temp_download_evidence(
        monkeypatch,
        tmp_path,
        lambda value: value["snapshot"].update({"preparation_status": "REUSED"}),  # type: ignore[index]
    )
    with pytest.raises(RuntimeError) as raised:
        module._load_download_evidence()  # type: ignore[attr-defined]
    assert getattr(raised.value, "code", None) == "DOWNLOAD_EVIDENCE_INVALID"


@pytest.mark.parametrize(
    "projection_mutation",
    [
        lambda value: value["model"].update({"model_id": "BAAI/other"}),  # type: ignore[index]
        lambda value: value["model"].update({"revision": "0" * 40}),  # type: ignore[index]
        lambda value: value["snapshot"]["files"][0].update({"path": "other.json"}),  # type: ignore[index]
    ],
)
def test_download_evidence_model_revision_file_projection_is_pinned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    projection_mutation: object,
) -> None:
    module = _load_temp_download_evidence(monkeypatch, tmp_path, projection_mutation)
    with pytest.raises(RuntimeError) as raised:
        module._load_download_evidence()  # type: ignore[attr-defined]
    assert getattr(raised.value, "code", None) == "DOWNLOAD_EVIDENCE_INVALID"


def test_success_schema_is_closed_and_privacy_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _runner()
    evidence = _run_fake_success(monkeypatch, tmp_path)
    module.validate_preflight_evidence(evidence)
    assert set(evidence) == {
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
    assert evidence["decision_status"] == "cpu_float32_preflight_passed"
    assert evidence["offline_policy"]["local_files_only"] is True  # type: ignore[index]
    assert evidence["model_load"]["trust_remote_code"] is False  # type: ignore[index]
    assert evidence["model_load"]["use_safetensors"] is True  # type: ignore[index]
    assert evidence["execution_state"]["full_candidate_reranking_run"] is False  # type: ignore[index]
    assert evidence["execution_state"]["real_candidate_scores_generated"] is False  # type: ignore[index]
    assert evidence["inference"]["logits_persisted"] is False  # type: ignore[index]
    serialized = json.dumps(evidence, ensure_ascii=False)
    assert "raw_logits" not in serialized
    assert "candidate_id" not in serialized
    assert "人工智能" not in serialized
    assert "logits" in serialized
    assert "https://" not in serialized
    assert not any(value in serialized for value in ("C:\\", "/Users/", "hostname", "authorization"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unexpected": True}),
        lambda value: value["inference"].update({"raw_logits": [0.1, 0.2]}),
        lambda value: value["snapshot"].update({"snapshot_path": "C:\\secret"}),
        lambda value: value["inference"].update({"fixture_sha256": "https://example.invalid"}),
        lambda value: value["execution_state"].update({"real_candidate_scores_generated": True}),
    ],
)
def test_schema_rejects_extra_or_sensitive_values(
    mutation: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _runner()
    evidence = _run_fake_success(monkeypatch, tmp_path)
    mutated = json.loads(json.dumps(evidence))
    mutation(mutated)  # type: ignore[operator]
    with pytest.raises(ValueError):
        module.validate_preflight_evidence(mutated)


def test_cli_without_flag_does_not_import_runtime_or_write_formal_evidence() -> None:
    code = (
        "import pathlib,sys; "
        "import scripts.run_m2_t02_reranker_preflight as runner; "
        "assert runner.main([])==2; "
        "assert not pathlib.Path('evaluation/source-artifacts/m2-t02-reranker-preflight.json').exists(); "
        "assert not any(name.split('.')[0] in {'torch','transformers','huggingface_hub','safetensors','tokenizers','FlagEmbedding'} for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, completed.stderr


def test_atomic_publish_rejects_malformed_existing_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _runner()
    evidence = _run_fake_success(monkeypatch, tmp_path)
    target = tmp_path / "existing.json"
    target.write_text("not json\n", encoding="utf-8")
    with pytest.raises(RuntimeError) as raised:
        module.publish_preflight_evidence(evidence, target)
    assert getattr(raised.value, "code", None) == "PREFLIGHT_EVIDENCE_CONFLICT"
