"""Red contracts for the future controlled reranker snapshot-download runner."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_MODULE_NAME = "scripts.download_m2_t02_reranker_snapshot"
_PREPARATION_MODULE = "scripts.prepare_m2_t02_reranker_snapshot"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SELECTION_PATH = Path("evaluation/source-artifacts/m2-t02-reranker-selection.json")
_SNAPSHOT_DIR = Path(
    "models/m2-t02/bge-reranker-v2-m3/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
)
_EVIDENCE_PATH = Path("evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json")
_MODEL_ID = "BAAI/bge-reranker-v2-m3"
_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
_HUB_VERSION = "0.34.3"
_INSTALLATION_PATH = _REPO_ROOT / "evaluation/source-artifacts/m2-t02-reranker-runtime-installation.json"
_SELECTION_ARTIFACT = _REPO_ROOT / _SELECTION_PATH


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


def _assert_error_code(
    raised: pytest.ExceptionInfo[BaseException], expected: str
) -> None:
    assert getattr(raised.value, "code", None) == expected


def _install_fake_preparation(
    monkeypatch: pytest.MonkeyPatch,
    callback: Callable[..., object],
) -> None:
    preparation = importlib.import_module(_PREPARATION_MODULE)
    monkeypatch.setattr(preparation, "prepare_snapshot", callback)


def _configure_fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    *,
    version: object = _HUB_VERSION,
    import_error: BaseException | None = None,
    downloader: Callable[..., object] | None = None,
    events: list[str] | None = None,
) -> Callable[..., object]:
    def fake_downloader(**_kwargs: object) -> object:
        return "fake-download-result"

    hub_downloader = downloader or fake_downloader

    def fake_version(distribution: str) -> object:
        assert distribution == "huggingface-hub"
        if events is not None:
            events.append("version")
        if isinstance(version, BaseException):
            raise version
        return version

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> ModuleType:
        if name != "huggingface_hub":
            return real_import_module(name, package)
        if events is not None:
            events.append("hub-import")
        assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
        assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
        assert "HF_HUB_OFFLINE" not in os.environ
        if import_error is not None:
            raise import_error
        fake_hub = ModuleType("huggingface_hub")
        fake_hub.hf_hub_download = hub_downloader  # type: ignore[attr-defined]
        return fake_hub

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    return hub_downloader


def _success_result(status: str, snapshot_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(status=status, snapshot_dir=snapshot_dir)


def test_runner_exposes_only_fixed_download_identity_and_entry_points() -> None:
    module = _download_module()

    assert module.SELECTION_PATH == _SELECTION_PATH
    assert module.SNAPSHOT_DIR == _SNAPSHOT_DIR
    assert module.EVIDENCE_PATH == _EVIDENCE_PATH
    assert module.HUGGINGFACE_HUB_VERSION == _HUB_VERSION
    assert callable(module.run_download)
    assert callable(module.main)
    assert callable(module.validate_download_evidence)


def test_importing_runner_does_not_load_hub_or_model_runtime() -> None:
    before = set(sys.modules)

    _download_module()

    imported = set(sys.modules) - before
    forbidden = {
        "huggingface_hub",
        "torch",
        "transformers",
        "FlagEmbedding",
        "sentence_transformers",
        "app.adapters.reranking",
    }
    assert not {
        name
        for name in imported
        if name in forbidden or name.split(".")[0] in forbidden
    }


def test_run_download_refuses_false_live_switch_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation_calls: list[dict[str, object]] = []

    def fake_preparation(**kwargs: object) -> object:
        preparation_calls.append(kwargs)
        raise AssertionError("preparation must not run without explicit consent")

    _install_fake_preparation(monkeypatch, fake_preparation)
    module = _download_module()
    evidence_path = tmp_path / "evidence.json"
    snapshot_dir = tmp_path / "models" / "snapshot"
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    monkeypatch.setattr(module, "SNAPSHOT_DIR", snapshot_dir)

    def forbidden_import(_name: str, _package: str | None = None) -> ModuleType:
        raise AssertionError("Hub import must not occur without explicit consent")

    monkeypatch.setattr(importlib, "import_module", forbidden_import)

    with pytest.raises(RuntimeError) as raised:
        module.run_download(execute_live_download=False)

    _assert_error_code(raised, "LIVE_DOWNLOAD_NOT_ENABLED")
    assert preparation_calls == []
    assert not snapshot_dir.exists()
    assert not evidence_path.exists()


def test_cli_refuses_missing_live_download_flag_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation_calls: list[dict[str, object]] = []

    def fake_preparation(**kwargs: object) -> object:
        preparation_calls.append(kwargs)
        raise AssertionError("preparation must not run without explicit consent")

    _install_fake_preparation(monkeypatch, fake_preparation)
    module = _download_module()
    evidence_path = tmp_path / "evidence.json"
    snapshot_dir = tmp_path / "models" / "snapshot"
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    monkeypatch.setattr(module, "SNAPSHOT_DIR", snapshot_dir)

    assert module.main([]) != 0
    assert preparation_calls == []
    assert not snapshot_dir.exists()
    assert not evidence_path.exists()


@pytest.mark.parametrize(
    "argv",
    (
        ("--model-id", "other/model"),
        ("--revision", "main"),
        ("--selection-path", "other-selection.json"),
        ("--snapshot-path", "other-snapshot"),
        ("--evidence-path", "other-evidence.json"),
        ("--allowlist", "README.md"),
    ),
)
def test_cli_cannot_override_fixed_download_identity_or_paths(
    monkeypatch: pytest.MonkeyPatch,
    argv: tuple[str, str],
) -> None:
    module = _download_module()
    run_calls: list[dict[str, object]] = []

    def fake_run_download(**kwargs: object) -> dict[str, object]:
        run_calls.append(kwargs)
        return {}

    monkeypatch.setattr(module, "run_download", fake_run_download)

    assert module.main(["--execute-live-download", *argv]) != 0
    assert run_calls == []


@pytest.mark.parametrize(
    ("version", "expected_code"),
    (
        (importlib.metadata.PackageNotFoundError("huggingface-hub"), "HUB_VERSION_UNAVAILABLE"),
        ("", "HUB_VERSION_MISMATCH"),
        ("0.34.2", "HUB_VERSION_MISMATCH"),
        ("0.34.4", "HUB_VERSION_MISMATCH"),
        ("1.0.0", "HUB_VERSION_MISMATCH"),
        (OSError("metadata failure"), "HUB_VERSION_UNAVAILABLE"),
    ),
)
def test_exact_hub_version_is_checked_before_import_or_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: object,
    expected_code: str,
) -> None:
    preparation_calls: list[dict[str, object]] = []

    def fake_preparation(**kwargs: object) -> object:
        preparation_calls.append(kwargs)
        raise AssertionError("version failure must stop before preparation")

    _install_fake_preparation(monkeypatch, fake_preparation)
    module = _download_module()
    evidence_path = tmp_path / "evidence.json"
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    _configure_fake_hub(monkeypatch, module, version=version)

    with pytest.raises(RuntimeError) as raised:
        module.run_download(execute_live_download=True)

    _assert_error_code(raised, expected_code)
    assert preparation_calls == []
    assert not evidence_path.exists()


def test_hub_import_failure_is_closed_before_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation_calls: list[dict[str, object]] = []

    def fake_preparation(**kwargs: object) -> object:
        preparation_calls.append(kwargs)
        raise AssertionError("Hub import failure must stop before preparation")

    _install_fake_preparation(monkeypatch, fake_preparation)
    module = _download_module()
    evidence_path = tmp_path / "evidence.json"
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    _configure_fake_hub(
        monkeypatch,
        module,
        import_error=ImportError("simulated Hub import failure"),
    )

    with pytest.raises(RuntimeError) as raised:
        module.run_download(execute_live_download=True)

    _assert_error_code(raised, "HUB_IMPORT_FAILED")
    assert preparation_calls == []
    assert not evidence_path.exists()


@pytest.mark.parametrize("status", ("PUBLISHED", "REUSED"))
def test_live_download_uses_late_hub_adapter_and_existing_preparation_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    events: list[str] = []
    preparation_calls: list[dict[str, object]] = []
    download_calls: list[dict[str, object]] = []
    disk_usage = lambda _path: SimpleNamespace(free=10_000_000_000)

    def fake_hub_download(**kwargs: object) -> object:
        download_calls.append(kwargs)
        return "fake-download-result"

    def fake_preparation(**kwargs: object) -> object:
        events.append("preparation")
        preparation_calls.append(kwargs)
        kwargs["download_file"](
            repo_id=_MODEL_ID,
            filename="README.md",
            revision=_REVISION,
            repo_type="model",
            token=False,
        )
        return _success_result(status, kwargs["snapshot_dir"])

    _install_fake_preparation(monkeypatch, fake_preparation)
    module = _download_module()
    evidence_path = tmp_path / "evidence.json"
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    fake_downloader = _configure_fake_hub(
        monkeypatch,
        module,
        downloader=fake_hub_download,
        events=events,
    )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = module.os.replace

    def record_evidence_replace(source: object, target: object) -> None:
        replace_calls.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", record_evidence_replace)

    evidence = module.run_download(execute_live_download=True, disk_usage=disk_usage)

    assert events == ["version", "hub-import", "preparation"]
    assert len(preparation_calls) == 1
    assert preparation_calls[0] == {
        "selection_path": _SELECTION_PATH,
        "snapshot_dir": _SNAPSHOT_DIR,
        "download_file": fake_downloader,
        "disk_usage": disk_usage,
    }
    assert download_calls == [
        {
            "repo_id": _MODEL_ID,
            "filename": "README.md",
            "revision": _REVISION,
            "repo_type": "model",
            "token": False,
        }
    ]
    assert evidence["snapshot"]["preparation_status"] == status
    assert json.loads(evidence_path.read_text(encoding="utf-8")) == evidence
    assert len(replace_calls) == 1
    assert replace_calls[0][1] == evidence_path
    assert replace_calls[0][0].parent == evidence_path.parent
    assert replace_calls[0][0].name.startswith(f".{evidence_path.name}.tmp-")
    assert "HF_HUB_OFFLINE" not in os.environ


@pytest.mark.parametrize(
    "failure",
    ("preparation-error", "unknown-status", "wrong-snapshot-path", "evidence-write"),
)
def test_live_failure_never_claims_success_or_mutates_immutable_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    selection_before = _SELECTION_ARTIFACT.read_bytes()
    installation_before = _INSTALLATION_PATH.read_bytes()
    evidence_path = tmp_path / "m2-t02-reranker-snapshot-download.json"
    snapshot_dir = tmp_path / "models" / "snapshot"

    def fake_preparation(**kwargs: object) -> object:
        if failure == "preparation-error":
            raise RuntimeError("simulated preparation failure")
        if failure == "unknown-status":
            return _success_result("UNKNOWN", kwargs["snapshot_dir"])
        if failure == "wrong-snapshot-path":
            return _success_result("PUBLISHED", tmp_path / "wrong-snapshot")
        Path(kwargs["snapshot_dir"]).mkdir(parents=True)
        return _success_result("PUBLISHED", kwargs["snapshot_dir"])

    _install_fake_preparation(monkeypatch, fake_preparation)
    module = _download_module()
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    monkeypatch.setattr(module, "SNAPSHOT_DIR", snapshot_dir)
    _configure_fake_hub(monkeypatch, module)
    if failure == "evidence-write":
        real_replace = module.os.replace

        def fail_evidence_replace(source: object, target: object) -> None:
            assert Path(target) == evidence_path
            raise OSError("simulated evidence publication failure")

        monkeypatch.setattr(module.os, "replace", fail_evidence_replace)
    else:
        real_replace = None

    with pytest.raises(RuntimeError) as raised:
        module.run_download(execute_live_download=True)

    expected = {
        "preparation-error": "PREPARATION_FAILED",
        "unknown-status": "PREPARATION_RESULT_INVALID",
        "wrong-snapshot-path": "PREPARATION_RESULT_INVALID",
        "evidence-write": "EVIDENCE_WRITE_FAILED",
    }
    _assert_error_code(raised, expected[failure])
    assert _SELECTION_ARTIFACT.read_bytes() == selection_before
    assert _INSTALLATION_PATH.read_bytes() == installation_before
    assert not evidence_path.exists()
    assert not list(evidence_path.parent.glob(f".{evidence_path.name}.tmp-*"))
    if failure == "evidence-write":
        assert snapshot_dir.is_dir()
    else:
        assert not snapshot_dir.exists()
    assert real_replace is None or module.os.replace is not real_replace
