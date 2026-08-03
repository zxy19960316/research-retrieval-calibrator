"""Red contracts for the future controlled reranker snapshot-download runner."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import subprocess
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
    except ModuleNotFoundError as exc:
        if exc.name == _MODULE_NAME:
            pytest.fail("download_m2_t02_reranker_snapshot has not been implemented")
        raise


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


def _owned_evidence_temp_files(evidence_path: Path, sentinel: Path) -> list[Path]:
    return [
        path
        for path in evidence_path.parent.glob(f".{evidence_path.name}.tmp-*")
        if path != sentinel
    ]


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


def test_hub_offline_mode_fails_closed_before_version_import_or_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation_calls: list[dict[str, object]] = []

    def fake_preparation(**kwargs: object) -> object:
        preparation_calls.append(kwargs)
        raise AssertionError("offline mode must stop before preparation")

    _install_fake_preparation(monkeypatch, fake_preparation)
    module = _download_module()
    evidence_path = tmp_path / "evidence.json"
    snapshot_dir = tmp_path / "models" / "snapshot"
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    monkeypatch.setattr(module, "SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def forbidden_version(_distribution: str) -> str:
        raise AssertionError("offline mode must not read Hub package metadata")

    def forbidden_import(_name: str, _package: str | None = None) -> ModuleType:
        raise AssertionError("offline mode must not import the Hub client")

    monkeypatch.setattr(importlib.metadata, "version", forbidden_version)
    monkeypatch.setattr(importlib, "import_module", forbidden_import)

    with pytest.raises(RuntimeError) as raised:
        module.run_download(execute_live_download=True)

    _assert_error_code(raised, "HUB_OFFLINE_MODE_ENABLED")
    assert os.environ["HF_HUB_OFFLINE"] == "1"
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


@pytest.mark.parametrize("initial_status", ("PUBLISHED", "REUSED"))
def test_live_download_proves_offline_reuse_before_writing_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_status: str,
) -> None:
    events: list[str] = []
    preparation_calls: list[dict[str, object]] = []
    preparation_results: list[SimpleNamespace] = []
    download_calls: list[dict[str, object]] = []
    disk_usage_calls: list[object] = []
    offline_download_attempts: list[object] = []
    offline_disk_attempts: list[object] = []

    def disk_usage(_path: object) -> SimpleNamespace:
        disk_usage_calls.append(_path)
        return SimpleNamespace(free=10_000_000_000)

    def fake_hub_download(**kwargs: object) -> object:
        download_calls.append(kwargs)
        return "fake-download-result"

    def fake_preparation(**kwargs: object) -> object:
        events.append("preparation")
        preparation_calls.append(kwargs)
        if len(preparation_calls) == 1:
            if initial_status == "PUBLISHED":
                kwargs["download_file"](
                    repo_id=_MODEL_ID,
                    filename="README.md",
                    revision=_REVISION,
                    repo_type="model",
                    token=False,
                )
                kwargs["disk_usage"](kwargs["snapshot_dir"])
            result = _success_result(initial_status, kwargs["snapshot_dir"])
            preparation_results.append(result)
            return result

        assert initial_status == "PUBLISHED"
        result = _success_result("REUSED", kwargs["snapshot_dir"])
        preparation_results.append(result)
        return result

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
    sentinel = evidence_path.parent / f".{evidence_path.name}.tmp-sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = module.os.replace

    def record_evidence_replace(source: object, target: object) -> None:
        replace_calls.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", record_evidence_replace)

    evidence = module.run_download(execute_live_download=True, disk_usage=disk_usage)

    expected_events = ["version", "hub-import", "preparation"]
    if initial_status == "PUBLISHED":
        expected_events.append("preparation")
    assert events == expected_events
    expected_call_count = 2 if initial_status == "PUBLISHED" else 1
    assert len(preparation_calls) == expected_call_count
    assert preparation_calls[0]["selection_path"] == _REPO_ROOT / _SELECTION_PATH
    assert preparation_calls[0]["snapshot_dir"] == _REPO_ROOT / _SNAPSHOT_DIR
    assert preparation_calls[0]["disk_usage"] is disk_usage
    assert preparation_calls[0]["download_file"] is not fake_downloader
    assert callable(preparation_calls[0]["download_file"])
    if initial_status == "PUBLISHED":
        assert download_calls == [
            {
                "repo_id": _MODEL_ID,
                "filename": "README.md",
                "revision": _REVISION,
                "repo_type": "model",
                "token": False,
            }
        ]
        assert evidence["decision_status"] == "downloaded_and_verified"
        assert evidence["execution_state"]["download_performed_this_run"] is True
        assert len(disk_usage_calls) == 1
        assert [result.status for result in preparation_results] == ["PUBLISHED", "REUSED"]
        assert preparation_results[1].status == "REUSED"
        assert preparation_calls[1]["selection_path"] == preparation_calls[0]["selection_path"]
        assert preparation_calls[1]["snapshot_dir"] == preparation_calls[0]["snapshot_dir"]
    else:
        assert download_calls == []
        assert evidence["decision_status"] == "reused_and_verified"
        assert evidence["execution_state"]["download_performed_this_run"] is False
        assert disk_usage_calls == []
        assert [result.status for result in preparation_results] == ["REUSED"]
    assert offline_download_attempts == []
    assert offline_disk_attempts == []
    assert evidence["snapshot"]["preparation_status"] == initial_status
    assert evidence["verification"]["offline_reuse_check"] == "passed"
    assert evidence["verification"]["downloader_called_during_offline_reuse"] is False
    assert json.loads(evidence_path.read_text(encoding="utf-8")) == evidence
    assert len(replace_calls) == 1
    assert replace_calls[0][1] == evidence_path
    assert replace_calls[0][0].parent == evidence_path.parent
    assert replace_calls[0][0].name.startswith(f".{evidence_path.name}.tmp-")
    assert "HF_HUB_OFFLINE" not in os.environ
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert _owned_evidence_temp_files(evidence_path, sentinel) == []


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
    sentinel = evidence_path.parent / f".{evidence_path.name}.tmp-sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    preparation_call_count = 0

    def fake_preparation(**kwargs: object) -> object:
        nonlocal preparation_call_count
        preparation_call_count += 1
        if failure == "preparation-error":
            raise RuntimeError("simulated preparation failure")
        if failure == "unknown-status":
            return _success_result("UNKNOWN", kwargs["snapshot_dir"])
        if failure == "wrong-snapshot-path":
            return _success_result("PUBLISHED", tmp_path / "wrong-snapshot")
        Path(kwargs["snapshot_dir"]).mkdir(parents=True, exist_ok=True)
        status = "PUBLISHED" if preparation_call_count == 1 else "REUSED"
        return _success_result(status, kwargs["snapshot_dir"])

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
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert _owned_evidence_temp_files(evidence_path, sentinel) == []
    if failure == "evidence-write":
        assert preparation_call_count == 2
        assert snapshot_dir.is_dir()
    else:
        assert not snapshot_dir.exists()
    assert real_replace is None or module.os.replace is not real_replace


def test_runner_keeps_legacy_preparation_diagnostic_out_of_default_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation = importlib.import_module(_PREPARATION_MODULE)

    def fake_preparation(**_kwargs: object) -> object:
        raise preparation.SnapshotPreparationError(
            "DOWNLOAD_FAILED",
            "TLS_OR_CERTIFICATE_FAILURE",
        )

    _install_fake_preparation(monkeypatch, fake_preparation)
    module = _download_module()
    monkeypatch.setattr(module, "SNAPSHOT_DIR", tmp_path / "models" / "snapshot")
    monkeypatch.setattr(
        module,
        "EVIDENCE_PATH",
        tmp_path / "m2-t02-reranker-snapshot-download.json",
    )
    _configure_fake_hub(monkeypatch, module)

    with pytest.raises(module.DownloadRunnerError) as raised:
        module.run_download(execute_live_download=True)

    assert raised.value.code == "PREPARATION_FAILED"
    assert raised.value.diagnostic is None
    assert str(raised.value) == "PREPARATION_FAILED"

    def fail_run_download(*, execute_live_download: bool) -> dict[str, object]:
        assert execute_live_download is True
        raise raised.value

    monkeypatch.setattr(module, "run_download", fail_run_download)

    assert module.main(["--execute-live-download"]) == 1
    assert capsys.readouterr().err == "PREPARATION_FAILED\n"


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    (
        ("second-published", "OFFLINE_REUSE_FAILED"),
        ("second-unknown", "PREPARATION_RESULT_INVALID"),
        ("wrong-snapshot-path", "PREPARATION_RESULT_INVALID"),
        ("downloader-called", "OFFLINE_REUSE_FAILED"),
        ("disk-usage-called", "OFFLINE_REUSE_FAILED"),
        ("preparation-error", "OFFLINE_REUSE_FAILED"),
    ),
)
def test_offline_reuse_failures_preserve_the_published_snapshot_and_write_no_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_code: str,
) -> None:
    selection_before = _SELECTION_ARTIFACT.read_bytes()
    installation_before = _INSTALLATION_PATH.read_bytes()
    evidence_path = tmp_path / "m2-t02-reranker-snapshot-download.json"
    snapshot_dir = tmp_path / "models" / "snapshot"
    sentinel = evidence_path.parent / f".{evidence_path.name}.tmp-sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    preparation_calls: list[dict[str, object]] = []
    offline_download_attempts: list[object] = []
    offline_disk_attempts: list[object] = []

    def fake_preparation(**kwargs: object) -> object:
        preparation_calls.append(kwargs)
        if len(preparation_calls) == 1:
            Path(kwargs["snapshot_dir"]).mkdir(parents=True)
            kwargs["download_file"](
                repo_id=_MODEL_ID,
                filename="README.md",
                revision=_REVISION,
                repo_type="model",
                token=False,
            )
            kwargs["disk_usage"](kwargs["snapshot_dir"])
            return _success_result("PUBLISHED", kwargs["snapshot_dir"])

        if failure == "second-published":
            return _success_result("PUBLISHED", kwargs["snapshot_dir"])
        if failure == "second-unknown":
            return _success_result("UNKNOWN", kwargs["snapshot_dir"])
        if failure == "wrong-snapshot-path":
            return _success_result("REUSED", tmp_path / "wrong-snapshot")
        if failure == "downloader-called":
            offline_download_attempts.append(kwargs["download_file"])
            kwargs["download_file"]()
        if failure == "disk-usage-called":
            offline_disk_attempts.append(kwargs["disk_usage"])
            kwargs["disk_usage"](kwargs["snapshot_dir"])
        if failure == "preparation-error":
            raise RuntimeError("simulated offline reuse failure")
        raise AssertionError(f"unexpected offline-reuse failure case: {failure}")

    _install_fake_preparation(monkeypatch, fake_preparation)
    module = _download_module()
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    monkeypatch.setattr(module, "SNAPSHOT_DIR", snapshot_dir)
    _configure_fake_hub(monkeypatch, module)

    with pytest.raises(RuntimeError) as raised:
        module.run_download(
            execute_live_download=True,
            disk_usage=lambda _path: SimpleNamespace(free=10_000_000_000),
        )

    _assert_error_code(raised, expected_code)
    assert len(preparation_calls) == 2
    assert _SELECTION_ARTIFACT.read_bytes() == selection_before
    assert _INSTALLATION_PATH.read_bytes() == installation_before
    assert snapshot_dir.is_dir()
    assert not evidence_path.exists()
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert _owned_evidence_temp_files(evidence_path, sentinel) == []
    if failure == "downloader-called":
        assert len(offline_download_attempts) == 1
        assert offline_disk_attempts == []
    elif failure == "disk-usage-called":
        assert offline_download_attempts == []
        assert len(offline_disk_attempts) == 1
    else:
        assert offline_download_attempts == []
        assert offline_disk_attempts == []


def test_evidence_validation_happens_before_replace_and_preserves_other_temp_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "m2-t02-reranker-snapshot-download.json"
    sentinel = evidence_path.parent / f".{evidence_path.name}.tmp-sentinel"
    sentinel.write_text("preserve", encoding="utf-8")

    def fake_preparation(**kwargs: object) -> object:
        return _success_result("REUSED", kwargs["snapshot_dir"])

    _install_fake_preparation(monkeypatch, fake_preparation)
    module = _download_module()
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    _configure_fake_hub(monkeypatch, module)
    replace_calls: list[tuple[object, object]] = []

    def fail_validation(_evidence: object) -> None:
        raise ValueError("simulated validation failure")

    def record_replace(source: object, target: object) -> None:
        replace_calls.append((source, target))
        raise AssertionError("validation failure must prevent os.replace")

    monkeypatch.setattr(module, "validate_download_evidence", fail_validation)
    monkeypatch.setattr(module.os, "replace", record_replace)

    with pytest.raises(RuntimeError) as raised:
        module.run_download(execute_live_download=True)

    _assert_error_code(raised, "EVIDENCE_VALIDATION_FAILED")
    assert replace_calls == []
    assert not evidence_path.exists()
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert _owned_evidence_temp_files(evidence_path, sentinel) == []


@pytest.mark.parametrize("invocation", ("module", "script"))
def test_cli_smoke_from_non_repository_cwd_needs_no_live_flag_or_side_effects(
    tmp_path: Path,
    invocation: str,
) -> None:
    outside_cwd = tmp_path / "outside-cwd"
    outside_cwd.mkdir()
    environment = dict(os.environ)
    if invocation == "module":
        existing_python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(_REPO_ROOT)
            if not existing_python_path
            else f"{_REPO_ROOT}{os.pathsep}{existing_python_path}"
        )
        command = [sys.executable, "-m", _MODULE_NAME]
    else:
        command = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "download_m2_t02_reranker_snapshot.py"),
        ]

    completed = subprocess.run(
        command,
        cwd=outside_cwd,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 2
    assert not (outside_cwd / "models").exists()
    assert not (outside_cwd / "evaluation").exists()


@pytest.mark.parametrize(
    "offline_value",
    ("1", "true", "TRUE", "on", "ON", "yes", "YES", "  TrUe  "),
)
def test_truthy_hub_offline_values_fail_before_plan_metadata_import_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline_value: str,
) -> None:
    module = _download_module()
    evidence_path = tmp_path / "evidence.json"
    snapshot_dir = tmp_path / "models" / "snapshot"
    events: list[str] = []
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    monkeypatch.setattr(module, "SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setenv("HF_HUB_OFFLINE", offline_value)

    def forbidden_plan(_path: Path) -> object:
        events.append("plan")
        raise AssertionError("offline mode must not load a SnapshotPlan")

    def forbidden_version(_distribution: str) -> str:
        events.append("metadata")
        raise AssertionError("offline mode must not read Hub package metadata")

    def forbidden_import(_name: str, _package: str | None = None) -> ModuleType:
        events.append("hub-import")
        raise AssertionError("offline mode must not import the Hub client")

    monkeypatch.setattr(module, "load_snapshot_plan", forbidden_plan, raising=False)
    monkeypatch.setattr(importlib.metadata, "version", forbidden_version)
    monkeypatch.setattr(importlib, "import_module", forbidden_import)

    with pytest.raises(RuntimeError) as raised:
        module.run_download(execute_live_download=True)

    _assert_error_code(raised, "HUB_OFFLINE_MODE_ENABLED")
    assert events == []
    assert os.environ["HF_HUB_OFFLINE"] == offline_value
    assert not snapshot_dir.exists()
    assert not evidence_path.exists()


def test_invalid_selection_fails_before_hub_metadata_import_preparation_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _download_module()
    selection_path = tmp_path / "invalid-selection.json"
    selection_path.write_text("{", encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    snapshot_dir = tmp_path / "models" / "snapshot"
    events: list[str] = []
    preparation_calls: list[dict[str, object]] = []
    monkeypatch.setattr(module, "SELECTION_PATH", selection_path)
    monkeypatch.setattr(module, "SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def forbidden_version(_distribution: str) -> str:
        events.append("metadata")
        raise AssertionError("invalid selection must stop before package metadata")

    def forbidden_import(_name: str, _package: str | None = None) -> ModuleType:
        events.append("hub-import")
        raise AssertionError("invalid selection must stop before Hub import")

    def forbidden_preparation(**kwargs: object) -> object:
        preparation_calls.append(kwargs)
        raise AssertionError("invalid selection must stop before preparation")

    _install_fake_preparation(monkeypatch, forbidden_preparation)
    monkeypatch.setattr(importlib.metadata, "version", forbidden_version)
    monkeypatch.setattr(importlib, "import_module", forbidden_import)

    with pytest.raises(RuntimeError) as raised:
        module.run_download(execute_live_download=True)

    _assert_error_code(raised, "SELECTION_VALIDATION_FAILED")
    assert events == []
    assert preparation_calls == []
    assert not snapshot_dir.exists()
    assert not evidence_path.exists()


def test_operational_paths_are_repository_anchored_but_evidence_paths_stay_relative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside_cwd = tmp_path / "outside-cwd"
    outside_cwd.mkdir()
    evidence_path = tmp_path / "injected-evidence.json"
    preparation_calls: list[dict[str, object]] = []

    def fake_preparation(**kwargs: object) -> object:
        preparation_calls.append(kwargs)
        assert kwargs["selection_path"] == _REPO_ROOT / _SELECTION_PATH
        assert kwargs["snapshot_dir"] == _REPO_ROOT / _SNAPSHOT_DIR
        return _success_result("REUSED", kwargs["snapshot_dir"])

    _install_fake_preparation(monkeypatch, fake_preparation)
    module = _download_module()
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    _configure_fake_hub(monkeypatch, module)
    monkeypatch.chdir(outside_cwd)

    evidence = module.run_download(execute_live_download=True)

    assert module._operational_path(_SELECTION_PATH) == _REPO_ROOT / _SELECTION_PATH
    assert module._operational_path(_SNAPSHOT_DIR) == _REPO_ROOT / _SNAPSHOT_DIR
    assert module._operational_path(evidence_path) == evidence_path
    assert len(preparation_calls) == 1
    assert evidence["download_policy"]["selection_path"] == _SELECTION_PATH.as_posix()
    assert evidence["download_policy"]["snapshot_path"] == _SNAPSHOT_DIR.as_posix()
    assert evidence_path.is_file()
    assert not (outside_cwd / "models").exists()
    assert not (outside_cwd / "evaluation").exists()


def test_evidence_uses_the_verified_snapshot_plan_not_a_later_selection_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_bytes(_SELECTION_ARTIFACT.read_bytes())
    original_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    evidence_path = tmp_path / "evidence.json"
    snapshot_dir = tmp_path / "models" / "snapshot"

    def fake_preparation(**kwargs: object) -> object:
        mutated_selection = json.loads(selection_path.read_text(encoding="utf-8"))
        mutated_selection["source_files"].reverse()
        selection_path.write_text(
            json.dumps(mutated_selection, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return _success_result("REUSED", kwargs["snapshot_dir"])

    _install_fake_preparation(monkeypatch, fake_preparation)
    module = _download_module()
    monkeypatch.setattr(module, "SELECTION_PATH", selection_path)
    monkeypatch.setattr(module, "SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    _configure_fake_hub(monkeypatch, module)

    evidence = module.run_download(execute_live_download=True)

    assert evidence["snapshot"]["files"] == [
        {
            "path": entry["path"],
            "size_bytes": entry["size_bytes"],
            "digest": entry["digest"],
            "digest_type": entry["digest_type"],
            "verified": True,
        }
        for entry in original_selection["source_files"]
    ]
    assert evidence["snapshot"]["total_size_bytes"] == original_selection["selected_model"][
        "pinned_source_files_size_bytes"
    ]


@pytest.mark.parametrize(
    "existing_kind",
    ("published", "reused", "malformed", "directory", "symlink"),
)
def test_conflicting_existing_evidence_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_kind: str,
) -> None:
    module = _download_module()
    evidence_path = tmp_path / "evidence.json"
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    snapshot_sentinel = snapshot_dir / "preserve.txt"
    snapshot_sentinel.write_text("preserve", encoding="utf-8")
    selection_before = _SELECTION_ARTIFACT.read_bytes()
    installation_before = _INSTALLATION_PATH.read_bytes()
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    candidate_status = "REUSED" if existing_kind == "published" else "PUBLISHED"
    candidate = module._expected_evidence(candidate_status)

    if existing_kind == "published":
        evidence_path.write_text(
            json.dumps(module._expected_evidence("PUBLISHED"), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    elif existing_kind == "reused":
        evidence_path.write_text(
            json.dumps(module._expected_evidence("REUSED"), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    elif existing_kind == "malformed":
        evidence_path.write_text("{", encoding="utf-8")
    elif existing_kind == "directory":
        evidence_path.mkdir()
    else:
        target = tmp_path / "existing-evidence.json"
        target.write_text("preserve", encoding="utf-8")
        try:
            evidence_path.symlink_to(target)
        except OSError as exc:
            pytest.skip(f"symlinks are unavailable in this test environment: {exc}")

    before_mtime = evidence_path.lstat().st_mtime_ns
    before_bytes = None if evidence_path.is_symlink() or evidence_path.is_dir() else evidence_path.read_bytes()
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = module.os.replace

    def record_replace(source: object, target: object) -> None:
        replace_calls.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", record_replace)

    with pytest.raises(RuntimeError) as raised:
        module._publish_evidence(candidate)

    _assert_error_code(raised, "EVIDENCE_CONFLICT")
    assert evidence_path.lstat().st_mtime_ns == before_mtime
    if before_bytes is not None:
        assert evidence_path.read_bytes() == before_bytes
    assert replace_calls == []
    assert not list(evidence_path.parent.glob(f".{evidence_path.name}.tmp-*"))
    assert snapshot_sentinel.read_text(encoding="utf-8") == "preserve"
    assert _SELECTION_ARTIFACT.read_bytes() == selection_before
    assert _INSTALLATION_PATH.read_bytes() == installation_before


def test_identical_existing_evidence_is_idempotent_without_replace_or_mtime_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _download_module()
    evidence_path = tmp_path / "evidence.json"
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    candidate = module._expected_evidence("REUSED")
    candidate_bytes = json.dumps(candidate, ensure_ascii=True, indent=2).encode("utf-8") + b"\n"
    evidence_path.write_bytes(candidate_bytes)
    before_mtime = evidence_path.stat().st_mtime_ns
    replace_calls: list[tuple[object, object]] = []

    def forbidden_replace(source: object, target: object) -> None:
        replace_calls.append((source, target))
        raise AssertionError("identical evidence must not be replaced")

    monkeypatch.setattr(module.os, "replace", forbidden_replace)

    module._publish_evidence(candidate)

    assert replace_calls == []
    assert evidence_path.read_bytes() == candidate_bytes
    assert evidence_path.stat().st_mtime_ns == before_mtime
    assert not list(evidence_path.parent.glob(f".{evidence_path.name}.tmp-*"))
