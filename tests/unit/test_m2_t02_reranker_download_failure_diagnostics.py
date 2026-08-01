"""Offline contracts for sealed M2-T02 download-failure diagnostics."""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import inspect
import json
import ssl
import sys
from pathlib import Path
from types import ModuleType

import pytest

_MODULE_NAME = "scripts.download_m2_t02_reranker_snapshot"
_PREPARATION_MODULE = "scripts.prepare_m2_t02_reranker_snapshot"
_HUB_VERSION = "0.34.3"
_MODEL_ID = "BAAI/bge-reranker-v2-m3"
_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
_SERIALIZED_FIELDS = {
    "diagnostic_version",
    "failure_stage",
    "file_ordinal",
    "file_role",
    "storage_type",
    "transport_backend",
    "exception_family",
    "exception_module_family",
    "exception_type",
    "ssl_verify_code",
    "errno",
    "winerror",
    "cause_chain_types",
}


@pytest.fixture(autouse=True)
def _unload_runner() -> None:
    sys.modules.pop(_MODULE_NAME, None)
    yield
    sys.modules.pop(_MODULE_NAME, None)


def _module() -> ModuleType:
    return importlib.import_module(_MODULE_NAME)


def _identity(module: ModuleType, filename: object) -> object:
    return module.safe_identity_from_filename(filename)


def test_fixed_file_diagnostic_identities_and_unknown_fallback() -> None:
    module = _module()
    expected = (
        ("README.md", 1, "readme", "git"),
        ("config.json", 2, "config", "git"),
        ("model.safetensors", 3, "weights", "lfs"),
        ("sentencepiece.bpe.model", 4, "tokenizer_model", "lfs"),
        ("special_tokens_map.json", 5, "special_tokens", "git"),
        ("tokenizer.json", 6, "tokenizer_json", "lfs"),
        ("tokenizer_config.json", 7, "tokenizer_config", "git"),
    )

    for filename, ordinal, role, storage_type in expected:
        identity = _identity(module, filename)
        assert identity.file_ordinal == ordinal
        assert identity.file_role == role
        assert identity.storage_type == storage_type

    unknown = _identity(module, "https://example.invalid/secret.bin")
    assert unknown.file_ordinal is None
    assert unknown.file_role == "unknown"
    assert unknown.storage_type == "unknown"
    assert "secret" not in repr(unknown)


def test_certificate_diagnostic_uses_only_closed_types_and_safe_integers() -> None:
    module = _module()
    certificate_error = ssl.SSLCertVerificationError(
        1,
        "certificate message https://example.invalid token=secret",
    )
    certificate_error.verify_code = 62
    certificate_error.errno = 10054
    certificate_error.winerror = 10053
    outer = RuntimeError("outer message must not survive")
    outer.__cause__ = certificate_error

    diagnostic = module.classify_download_failure(
        outer,
        failure_stage="file_download",
        identity=_identity(module, "model.safetensors"),
    )
    payload = json.loads(module.serialize_closed_failure_diagnostic(diagnostic))

    assert diagnostic.exception_family == "certificate_verification"
    assert diagnostic.exception_module_family == "ssl"
    assert diagnostic.exception_type == "SSLCertVerificationError"
    assert diagnostic.ssl_verify_code == 62
    assert diagnostic.errno == 10054
    assert diagnostic.winerror == 10053
    assert set(payload) == _SERIALIZED_FIELDS
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "example.invalid",
        "secret",
        "outer message",
        "certificate message",
    ):
        assert forbidden not in serialized


def test_classifier_prefers_nested_certificate_failure() -> None:
    module = _module()
    certificate_error = ssl.SSLCertVerificationError(1, "opaque certificate detail")
    outer = RuntimeError("opaque outer detail")
    outer.__cause__ = certificate_error

    diagnostic = module.classify_download_failure(
        outer,
        failure_stage="file_download",
        identity=_identity(module, "README.md"),
    )

    assert diagnostic.exception_family == "certificate_verification"
    assert diagnostic.exception_type == "SSLCertVerificationError"
    assert diagnostic.cause_chain_types == ("RuntimeError", "SSLCertVerificationError")


def test_classifier_maps_xet_reset_timeouts_proxy_and_text_to_closed_values() -> None:
    module = _module()
    xet_runtime_error = type(
        "RuntimeError",
        (RuntimeError,),
        {"__module__": "hf_xet"},
    )("xet message must not survive")
    connect_timeout = type(
        "ConnectTimeout",
        (RuntimeError,),
        {"__module__": "requests.exceptions"},
    )("timeout text")
    read_timeout = type(
        "ReadTimeout",
        (RuntimeError,),
        {"__module__": "requests.exceptions"},
    )("timeout text")
    proxy_error = type(
        "ProxyError",
        (RuntimeError,),
        {"__module__": "requests.exceptions"},
    )("proxy text")

    expected = (
        (xet_runtime_error, "xet", "xet_transport", "hf_xet", "RuntimeError"),
        (ConnectionResetError("reset text"), "unknown", "connection_reset", "builtin", "ConnectionResetError"),
        (connect_timeout, "http", "connect_timeout", "requests", "ConnectTimeout"),
        (read_timeout, "http", "read_timeout", "requests", "ReadTimeout"),
        (TimeoutError("timeout text"), "unknown", "read_timeout", "builtin", "TimeoutError"),
        (proxy_error, "http", "proxy_failure", "requests", "ProxyError"),
        (RuntimeError("proxy https://example.invalid token=secret"), "unknown", "unknown_transport", "builtin", "RuntimeError"),
    )
    for exc, backend, family, module_family, type_name in expected:
        diagnostic = module.classify_download_failure(
            exc,
            failure_stage="file_download",
            identity=_identity(module, "README.md"),
        )
        assert diagnostic.transport_backend == backend
        assert diagnostic.exception_family == family
        assert diagnostic.exception_module_family == module_family
        assert diagnostic.exception_type == type_name
        assert "secret" not in module.serialize_closed_failure_diagnostic(diagnostic)


def test_classifier_does_not_render_hostile_exceptions_and_bounds_cycles() -> None:
    module = _module()

    class HostileError(RuntimeError):
        def __str__(self) -> str:
            raise AssertionError("__str__ must not be called")

        def __repr__(self) -> str:
            raise AssertionError("__repr__ must not be called")

    first = HostileError("opaque")
    second = RuntimeError("opaque")
    first.__cause__ = second
    second.__context__ = first

    diagnostic = module.classify_download_failure(
        first,
        failure_stage="file_download",
        identity=_identity(module, "README.md"),
    )

    assert diagnostic.exception_type == "RuntimeError"
    assert len(diagnostic.cause_chain_types) <= 8
    assert set(diagnostic.cause_chain_types) <= {
        "SSLCertVerificationError",
        "SSLError",
        "TimeoutError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "ConnectionError",
        "ProxyError",
        "ConnectTimeout",
        "ReadTimeout",
        "RuntimeError",
        "OSError",
        "UnknownError",
    }


def test_classifier_source_never_renders_exception_text() -> None:
    module = _module()
    tree = ast.parse(inspect.getsource(module.classify_download_failure))

    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"str", "repr"}
        for node in ast.walk(tree)
    )


def test_diagnostic_wrapper_preserves_third_file_identity_without_changing_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    preparation = importlib.import_module(_PREPARATION_MODULE)
    snapshot_dir = tmp_path / "models" / "snapshot"
    evidence_path = tmp_path / "m2-t02-reranker-snapshot-download.json"

    def hub_download(**kwargs: object) -> object:
        assert kwargs == {
            "repo_id": _MODEL_ID,
            "filename": "model.safetensors",
            "revision": _REVISION,
            "repo_type": "model",
            "token": False,
        }
        raise ssl.SSLCertVerificationError(1, "https://example.invalid token=secret")

    def fake_preparation(**kwargs: object) -> object:
        try:
            kwargs["download_file"](
                repo_id=_MODEL_ID,
                filename="model.safetensors",
                revision=_REVISION,
                repo_type="model",
                token=False,
        )
        except module.DiagnosedDownloadFailure as exc:
            assert str(exc) == "DIAGNOSED_DOWNLOAD_FAILURE"
            assert exc.__context__ is None
            assert exc.__cause__ is None
            raise preparation.SnapshotPreparationError("DOWNLOAD_FAILED") from None
        raise AssertionError("diagnostic wrapper must fail before preparation succeeds")

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> ModuleType:
        if name != "huggingface_hub":
            return real_import_module(name, package)
        fake_hub = ModuleType("huggingface_hub")
        fake_hub.hf_hub_download = hub_download  # type: ignore[attr-defined]
        return fake_hub

    monkeypatch.setattr(module, "SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setattr(module, "EVIDENCE_PATH", evidence_path)
    monkeypatch.setattr(module, "prepare_snapshot", fake_preparation)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: _HUB_VERSION)
    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    with pytest.raises(module.DownloadRunnerError) as raised:
        module.run_download(execute_live_download=True)

    assert raised.value.code == "PREPARATION_FAILED"
    assert str(raised.value) == "PREPARATION_FAILED"
    diagnostic = raised.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.file_ordinal == 3
    assert diagnostic.file_role == "weights"
    assert diagnostic.storage_type == "lfs"
    assert diagnostic.exception_family == "certificate_verification"
    assert not snapshot_dir.exists()
    assert not evidence_path.exists()


def test_cli_emits_closed_diagnostic_only_with_explicit_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    diagnostic = module.classify_download_failure(
        ssl.SSLCertVerificationError(1, "https://example.invalid token=secret"),
        failure_stage="file_download",
        identity=_identity(module, "model.safetensors"),
    )

    def fail_run_download(*, execute_live_download: bool) -> dict[str, object]:
        assert execute_live_download is True
        raise module.DownloadRunnerError("PREPARATION_FAILED", diagnostic)

    monkeypatch.setattr(module, "run_download", fail_run_download)

    assert module.main(["--execute-live-download"]) == 1
    assert capsys.readouterr().err == "PREPARATION_FAILED\n"

    assert module.main(
        ["--execute-live-download", "--emit-closed-failure-diagnostic"]
    ) == 1
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "PREPARATION_FAILED"
    assert len(lines) == 2
    assert lines[1] == module.serialize_closed_failure_diagnostic(diagnostic)
    assert json.loads(lines[1]).keys() == _SERIALIZED_FIELDS
    for forbidden in ("https://", "secret", "token", "hostname", "certificate message"):
        assert forbidden not in lines[1]
