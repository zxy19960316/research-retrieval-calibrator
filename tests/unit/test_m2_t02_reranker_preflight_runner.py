"""Offline contracts for the controlled local M2-T02 preflight runner."""

from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_MODULE_NAME = "scripts.run_m2_t02_reranker_preflight"
_PREPARATION_MODULE = "scripts.prepare_m2_t02_reranker_snapshot"
_DOWNLOAD_MODULE = "scripts.download_m2_t02_reranker_snapshot"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOWNLOAD_EVIDENCE = _REPO_ROOT / "evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json"
_SELECTION = _REPO_ROOT / "evaluation/source-artifacts/m2-t02-reranker-selection.json"
_FIXTURE_SHA256 = "8eaccc1ca1f0942a1800fc66332ccec5d80adc7c8594f9f0a440ea14d1eb288e"


@pytest.fixture(autouse=True)
def _unload_future_runner() -> None:
    sys.modules.pop(_MODULE_NAME, None)
    yield
    sys.modules.pop(_MODULE_NAME, None)


def _runner() -> ModuleType:
    return importlib.import_module(_MODULE_NAME)


def _install_fake_preparation(
    monkeypatch: pytest.MonkeyPatch,
    callback: Any,
) -> ModuleType:
    preparation = importlib.import_module(_PREPARATION_MODULE)
    monkeypatch.setattr(preparation, "prepare_snapshot", callback)
    return preparation


def _install_valid_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "torch": "2.4.1+cpu",
        "transformers": "4.53.2",
        "huggingface-hub": "0.34.3",
        "safetensors": "0.5.3",
        "tokenizers": "0.21.2",
    }

    def fake_version(name: str) -> str:
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", fake_version)


class _FakeDevice:
    type = "cpu"


class _FakeNonCpuDevice:
    type = "cuda"


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: object,
        device: object | None = None,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = _FakeDevice() if device is None else device

    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


class _FakeFiniteResult:
    def all(self) -> bool:
        return True


class _FakeInferenceContext:
    def __init__(self, torch_module: ModuleType) -> None:
        self._torch_module = torch_module

    def __enter__(self) -> None:
        self._torch_module.inference_mode_entered += 1  # type: ignore[attr-defined]

    def __exit__(self, *_args: object) -> None:
        self._torch_module.inference_mode_exited += 1  # type: ignore[attr-defined]


class _FakeModel:
    def __init__(self, torch_module: ModuleType) -> None:
        self._torch_module = torch_module
        self.to_calls: list[str] = []
        self.eval_calls = 0
        self.forward_calls = 0
        self.forward_kwargs: dict[str, object] = {}
        self.load_calls: list[tuple[object, dict[str, object]]] = []
        self.device = _FakeDevice()
        self.dtype = torch_module.float32  # type: ignore[attr-defined]
        self.training = True
        self.eval_updates_training = True
        self.preserve_device_on_to = False
        self.logits_shape = (2, 1)
        self.logits_dtype = self.dtype
        self.logits_device: object | None = None

    def to(self, device: str) -> _FakeModel:
        self.to_calls.append(device)
        if not self.preserve_device_on_to:
            self.device = _FakeDevice()
        return self

    def eval(self) -> _FakeModel:
        self.eval_calls += 1
        if self.eval_updates_training:
            self.training = False
        return self

    def parameters(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(device=self.device, dtype=self.dtype)]

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.forward_calls += 1
        self.forward_kwargs = dict(kwargs)
        return SimpleNamespace(
            logits=_FakeTensor(self.logits_shape, self.logits_dtype, self.logits_device)
        )


class _FakeTokenizer:
    def __init__(self, torch_module: ModuleType) -> None:
        self._torch_module = torch_module
        self.load_calls: list[tuple[object, dict[str, object]]] = []
        self.call_args: tuple[tuple[object, ...], dict[str, object]] | None = None
        self.return_empty = False
        self.tensor_device: object | None = None

    def __call__(self, *args: object, **kwargs: object) -> dict[str, _FakeTensor]:
        self.call_args = (args, kwargs)
        if self.return_empty:
            return {}
        return {
            "input_ids": _FakeTensor(
                (2, 3),
                self._torch_module.float32,  # type: ignore[attr-defined]
                self.tensor_device,
            ),
            "attention_mask": _FakeTensor(
                (2, 3),
                self._torch_module.float32,  # type: ignore[attr-defined]
                self.tensor_device,
            ),
        }


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    finite: bool = True,
    tokenizer_error: bool = False,
    model_error: bool = False,
    eval_updates_training: bool = True,
) -> tuple[ModuleType, _FakeTokenizer, _FakeModel]:
    torch_module = ModuleType("torch")
    torch_module.float32 = object()  # type: ignore[attr-defined]
    torch_module.inference_mode_entered = 0  # type: ignore[attr-defined]
    torch_module.inference_mode_exited = 0  # type: ignore[attr-defined]
    torch_module.version = SimpleNamespace(cuda=None)  # type: ignore[attr-defined]
    torch_module.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    torch_module.inference_mode = lambda: _FakeInferenceContext(torch_module)  # type: ignore[attr-defined]
    torch_module.isfinite = lambda _value: _FakeFiniteResult() if finite else SimpleNamespace(all=lambda: False)  # type: ignore[attr-defined]

    tokenizer = _FakeTokenizer(torch_module)
    model = _FakeModel(torch_module)
    model.eval_updates_training = eval_updates_training
    transformers_module = ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(path: object, **kwargs: object) -> _FakeTokenizer:
            tokenizer.load_calls.append((path, kwargs))
            if tokenizer_error:
                raise RuntimeError("fake tokenizer load failure")
            return tokenizer

    class _AutoModel:
        @staticmethod
        def from_pretrained(path: object, **kwargs: object) -> _FakeModel:
            model.load_calls = [(path, kwargs)]
            if model_error:
                raise RuntimeError("fake model load failure")
            return model

    transformers_module.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]
    transformers_module.AutoModelForSequenceClassification = _AutoModel  # type: ignore[attr-defined]

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> ModuleType:
        if name == "torch":
            return torch_module
        if name == "transformers":
            return transformers_module
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    return torch_module, tokenizer, model


def _install_fake_reuse(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    *,
    callback_calls: list[dict[str, object]] | None = None,
) -> None:
    def fake_preparation(**kwargs: object) -> SimpleNamespace:
        if callback_calls is not None:
            callback_calls.append(kwargs)
        return SimpleNamespace(status="REUSED", snapshot_dir=kwargs["snapshot_dir"])

    _install_fake_preparation(monkeypatch, fake_preparation)


def _memory_probe() -> Any:
    observations = iter(
        [
            {
                "system_total_physical_memory_bytes": 1000,
                "system_available_physical_memory_bytes": 700,
                "process_working_set_bytes": 100,
                "process_peak_working_set_bytes": 120,
            },
            {
                "system_total_physical_memory_bytes": 1000,
                "system_available_physical_memory_bytes": 600,
                "process_working_set_bytes": 200,
                "process_peak_working_set_bytes": 300,
            },
            {
                "system_total_physical_memory_bytes": 1000,
                "system_available_physical_memory_bytes": 500,
                "process_working_set_bytes": 250,
                "process_peak_working_set_bytes": 400,
            },
        ]
    )

    def probe() -> dict[str, int]:
        return next(observations)

    return probe


class _FakeWinFunction:
    def __init__(self, callback: Any) -> None:
        self._callback = callback
        self.argtypes: object | None = None
        self.restype: object | None = None

    def __call__(self, *args: object) -> object:
        return self._callback(*args)


def _install_fake_windows_dlls(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    *,
    global_memory_success: bool = True,
    process_memory_success: bool = True,
    process_handle: int = 0x123456789ABCDEF0,
) -> dict[str, object]:
    from ctypes import wintypes

    observations: dict[str, object] = {"process_handle": None}

    def fill_global(pointer: object) -> int:
        assert global_function.argtypes is not None
        memory_status = ctypes.cast(
            pointer,
            global_function.argtypes[0],  # type: ignore[index]
        ).contents
        memory_status.ullTotalPhys = 1000
        memory_status.ullAvailPhys = 700
        return int(global_memory_success)

    global_function = _FakeWinFunction(fill_global)

    def get_current_process() -> int:
        return process_handle

    process_function_handle = _FakeWinFunction(get_current_process)

    def fill_process(handle: object, pointer: object, _size: object) -> int:
        observations["process_handle"] = handle
        assert process_function.argtypes is not None
        counters = ctypes.cast(
            pointer,
            process_function.argtypes[1],  # type: ignore[index]
        ).contents
        counters.WorkingSetSize = 200
        counters.PeakWorkingSetSize = 300
        return int(process_memory_success)

    process_function = _FakeWinFunction(fill_process)
    kernel32 = SimpleNamespace(
        GlobalMemoryStatusEx=global_function,
        GetCurrentProcess=process_function_handle,
    )
    psapi = SimpleNamespace(GetProcessMemoryInfo=process_function)

    def fake_win_dll(name: str, *, use_last_error: bool) -> object:
        assert use_last_error is True
        return {"kernel32": kernel32, "psapi": psapi}[name]

    monkeypatch.setattr(module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(module.ctypes, "WinDLL", fake_win_dll, raising=False)
    observations["kernel32"] = kernel32
    observations["psapi"] = psapi
    observations["global_function"] = global_function
    observations["process_function"] = process_function
    observations["handle_type"] = wintypes.HANDLE
    observations["bool_type"] = wintypes.BOOL
    return observations


def _run_fake_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[ModuleType, dict[str, object], _FakeTokenizer, _FakeModel, ModuleType]:
    module = _runner()
    _install_valid_versions(monkeypatch)
    torch_module, tokenizer, model = _install_fake_runtime(monkeypatch)
    _install_fake_reuse(monkeypatch, module)
    evidence_path = tmp_path / "preflight.json"
    monkeypatch.setattr(module, "PREFLIGHT_EVIDENCE_PATH", evidence_path)
    result = module.run_preflight(
        execute_local_preflight=True,
        memory_probe=_memory_probe(),
    )
    return module, result, tokenizer, model, torch_module


def _configure_fake_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    **runtime_options: object,
) -> tuple[ModuleType, ModuleType, _FakeTokenizer, _FakeModel]:
    module = _runner()
    _install_valid_versions(monkeypatch)
    torch_module, tokenizer, model = _install_fake_runtime(
        monkeypatch,
        **runtime_options,  # type: ignore[arg-type]
    )
    _install_fake_reuse(monkeypatch, module)
    monkeypatch.setattr(module, "PREFLIGHT_EVIDENCE_PATH", tmp_path / "preflight.json")
    return module, torch_module, tokenizer, model


def test_import_does_not_load_runtime_packages() -> None:
    code = (
        "import sys; import scripts.run_m2_t02_reranker_preflight; "
        "forbidden={'torch','transformers','huggingface_hub','safetensors','tokenizers','FlagEmbedding'}; "
        "assert not any(name.split('.')[0] in forbidden for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--unknown"],
        ["--execute-local-preflight", "--execute-local-preflight"],
        ["--model", "fixed"],
        ["--revision", "fixed"],
        ["--snapshot-path", "fixed"],
        ["--device", "cpu"],
        ["--dtype", "float32"],
        ["--trust-remote-code"],
        ["--network"],
        ["--score-file", "scores.json"],
    ],
)
def test_main_rejects_every_non_exact_argument_vector(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()
    monkeypatch.setattr(
        module,
        "run_preflight",
        lambda **_kwargs: pytest.fail("invalid CLI must not execute the runner"),
    )
    assert module.main(arguments) == 2


def test_missing_explicit_flag_has_no_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _runner()
    monkeypatch.setattr(
        module,
        "_load_download_evidence",
        lambda: pytest.fail("artifact loading must wait for explicit authorization"),
    )
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=False)
    assert getattr(raised.value, "code", None) == "LOCAL_PREFLIGHT_NOT_ENABLED"


def test_fixed_identity_is_not_overrideable() -> None:
    module = _runner()
    assert module.SELECTION_PATH == Path("evaluation/source-artifacts/m2-t02-reranker-selection.json")
    assert module.DOWNLOAD_EVIDENCE_PATH == Path(
        "evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json"
    )
    assert module.SNAPSHOT_DIR == Path(
        "models/m2-t02/bge-reranker-v2-m3/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    )
    assert module.PREFLIGHT_EVIDENCE_PATH == Path(
        "evaluation/source-artifacts/m2-t02-reranker-preflight.json"
    )
    assert module.MODEL_ID == "BAAI/bge-reranker-v2-m3"
    assert module.REVISION == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def test_windows_memory_probe_declares_api_types_and_returns_strict_integers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()
    observations = _install_fake_windows_dlls(monkeypatch, module)
    result = module._windows_memory_probe()  # type: ignore[attr-defined]

    assert set(result) == {
        "system_total_physical_memory_bytes",
        "system_available_physical_memory_bytes",
        "process_working_set_bytes",
        "process_peak_working_set_bytes",
    }
    assert all(type(value) is int for value in result.values())
    global_function = observations["global_function"]
    process_function = observations["process_function"]
    process_handle_function = observations["kernel32"].GetCurrentProcess  # type: ignore[attr-defined]
    assert global_function.restype is observations["bool_type"]  # type: ignore[attr-defined]
    assert process_function.restype is observations["bool_type"]  # type: ignore[attr-defined]
    assert process_handle_function.argtypes == []
    assert process_handle_function.restype is observations["handle_type"]  # type: ignore[attr-defined]
    assert process_function.argtypes[0] is observations["handle_type"]  # type: ignore[attr-defined,index]
    assert process_function.argtypes[2] is ctypes.c_ulong  # type: ignore[attr-defined,index]


@pytest.mark.parametrize("failure", ["global", "process"])
def test_windows_memory_probe_api_failures_map_to_memory_probe_failed(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    module = _runner()
    _install_fake_windows_dlls(
        monkeypatch,
        module,
        global_memory_success=failure != "global",
        process_memory_success=failure != "process",
    )
    with pytest.raises(RuntimeError) as raised:
        module._windows_memory_probe()  # type: ignore[attr-defined]
    assert getattr(raised.value, "code", None) == "MEMORY_PROBE_FAILED"


def test_windows_memory_probe_non_windows_failure_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError) as raised:
        module._windows_memory_probe()  # type: ignore[attr-defined]
    assert getattr(raised.value, "code", None) == "MEMORY_PROBE_FAILED"


def test_windows_memory_probe_preserves_wide_process_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()
    observations = _install_fake_windows_dlls(monkeypatch, module)
    module._windows_memory_probe()  # type: ignore[attr-defined]
    assert observations["process_handle"] == 0x123456789ABCDEF0


def test_snapshot_reuse_forbids_downloader_and_disk_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()
    callback_attempts: list[str] = []

    def fake_preparation(**kwargs: object) -> SimpleNamespace:
        callback_attempts.append("prepare")
        download_file = kwargs["download_file"]
        disk_usage = kwargs["disk_usage"]
        download_file(filename="config.json")  # type: ignore[operator]
        disk_usage(Path("."))  # type: ignore[operator]
        return SimpleNamespace(status="REUSED", snapshot_dir=kwargs["snapshot_dir"])

    _install_fake_preparation(monkeypatch, fake_preparation)
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True)
    assert getattr(raised.value, "code", None) == "SNAPSHOT_REUSE_VERIFICATION_FAILED"
    assert callback_attempts == ["prepare"]


def test_success_uses_fake_runtime_exact_contract_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_values = {
        "HF_HUB_OFFLINE": "caller-hub",
        "TRANSFORMERS_OFFLINE": "caller-transformers",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "caller-token",
        "HF_HUB_DISABLE_TELEMETRY": "caller-telemetry",
    }
    for key, value in original_values.items():
        monkeypatch.setenv(key, value)
    import_events: list[dict[str, str]] = []
    real_import_module = importlib.import_module
    _install_valid_versions(monkeypatch)
    torch_module = ModuleType("torch")
    torch_module.float32 = object()  # type: ignore[attr-defined]
    torch_module.inference_mode_entered = 0  # type: ignore[attr-defined]
    torch_module.inference_mode_exited = 0  # type: ignore[attr-defined]
    torch_module.version = SimpleNamespace(cuda=None)  # type: ignore[attr-defined]
    torch_module.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    torch_module.inference_mode = lambda: _FakeInferenceContext(torch_module)  # type: ignore[attr-defined]
    torch_module.isfinite = lambda _value: SimpleNamespace(all=lambda: True)  # type: ignore[attr-defined]
    tokenizer = _FakeTokenizer(torch_module)
    model = _FakeModel(torch_module)
    transformers_module = ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(path: object, **kwargs: object) -> _FakeTokenizer:
            import_events.append(dict(os.environ))
            tokenizer.load_calls.append((path, kwargs))
            return tokenizer

    class _AutoModel:
        @staticmethod
        def from_pretrained(path: object, **kwargs: object) -> _FakeModel:
            model.load_calls = [(path, kwargs)]  # type: ignore[attr-defined]
            return model

    transformers_module.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]
    transformers_module.AutoModelForSequenceClassification = _AutoModel  # type: ignore[attr-defined]

    def fake_import_module(name: str, package: str | None = None) -> ModuleType:
        if name == "torch":
            return torch_module
        if name == "transformers":
            return transformers_module
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    module = _runner()
    _install_fake_reuse(monkeypatch, module)
    evidence_path = tmp_path / "preflight.json"
    monkeypatch.setattr(module, "PREFLIGHT_EVIDENCE_PATH", evidence_path)
    result = module.run_preflight(execute_local_preflight=True, memory_probe=_memory_probe())

    assert result["decision_status"] == "cpu_float32_preflight_passed"
    assert result["inference"]["fixture_sha256"] == _FIXTURE_SHA256  # type: ignore[index]
    assert result["inference"]["logits_shape"] == [2, 1]  # type: ignore[index]
    assert result["inference"]["logits_all_finite"] is True  # type: ignore[index]
    assert result["inference"]["logits_persisted"] is False  # type: ignore[index]
    assert tokenizer.load_calls[0][1] == {"local_files_only": True, "trust_remote_code": False}
    assert model.load_calls[0][1] == {  # type: ignore[attr-defined]
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "torch_dtype": torch_module.float32,  # type: ignore[attr-defined]
    }
    assert model.to_calls == ["cpu"]
    assert model.eval_calls == 1
    assert model.training is False
    assert model.forward_calls == 1
    assert set(model.forward_kwargs) == {"input_ids", "attention_mask"}
    assert tokenizer.call_args == (
        (
            [query for query, _passage in module.FIXTURE_PAIRS],
            [passage for _query, passage in module.FIXTURE_PAIRS],
        ),
        {
            "padding": True,
            "truncation": True,
            "max_length": 512,
            "return_tensors": "pt",
        },
    )
    assert len(tokenizer.call_args[0][0]) == 2
    assert len(tokenizer.call_args[0][0]) != 33
    assert import_events[0]["HF_HUB_OFFLINE"] == "1"
    assert import_events[0]["TRANSFORMERS_OFFLINE"] == "1"
    assert import_events[0]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert import_events[0]["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert {key: os.environ.get(key) for key in original_values} == original_values
    assert evidence_path.is_file()


def test_runtime_version_drift_fails_before_runtime_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()
    _install_fake_reuse(monkeypatch, module)
    versions = {
        "torch": "2.4.1+cpu",
        "transformers": "4.53.2",
        "huggingface-hub": "0.34.3",
        "safetensors": "0.5.3",
        "tokenizers": "0.21.3",
    }
    monkeypatch.setattr(importlib.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda *_args, **_kwargs: pytest.fail("runtime import must wait for package gates"),
    )
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True)
    assert getattr(raised.value, "code", None) == "RUNTIME_VERSION_MISMATCH"


def test_package_not_found_is_runtime_version_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()
    _install_fake_reuse(monkeypatch, module)

    def missing_package(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError("fake-runtime")

    monkeypatch.setattr(importlib.metadata, "version", missing_package)
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True)
    assert getattr(raised.value, "code", None) == "RUNTIME_VERSION_UNAVAILABLE"


@pytest.mark.parametrize(
    "package_name",
    ["torch", "transformers", "huggingface-hub", "safetensors", "tokenizers"],
)
def test_any_runtime_version_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    package_name: str,
) -> None:
    module = _runner()
    _install_fake_reuse(monkeypatch, module)
    expected = dict(module._EXPECTED_PACKAGE_VERSIONS)  # type: ignore[attr-defined]
    expected[package_name] = "drifted-version"
    monkeypatch.setattr(importlib.metadata, "version", lambda name: expected[name])
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda *_args, **_kwargs: pytest.fail("runtime imports must wait for version gates"),
    )
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True)
    assert getattr(raised.value, "code", None) == "RUNTIME_VERSION_MISMATCH"


def test_torch_cuda_version_is_rejected_before_tokenizer_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, torch_module, tokenizer, _model = _configure_fake_run(monkeypatch, tmp_path)
    torch_module.version.cuda = "12.1"  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True, memory_probe=_memory_probe())
    assert getattr(raised.value, "code", None) == "NON_CPU_TORCH_BUILD"
    assert tokenizer.load_calls == []


def test_torch_cuda_available_is_rejected_before_tokenizer_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, torch_module, tokenizer, _model = _configure_fake_run(monkeypatch, tmp_path)
    torch_module.cuda.is_available = lambda: True  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True, memory_probe=_memory_probe())
    assert getattr(raised.value, "code", None) == "NON_CPU_TORCH_BUILD"
    assert tokenizer.load_calls == []


def test_tokenizer_from_pretrained_failure_is_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _torch_module, tokenizer, _model = _configure_fake_run(
        monkeypatch,
        tmp_path,
        tokenizer_error=True,
    )
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True, memory_probe=_memory_probe())
    assert getattr(raised.value, "code", None) == "TOKENIZER_LOAD_FAILED"
    assert len(tokenizer.load_calls) == 1


def test_model_from_pretrained_failure_is_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _torch_module, tokenizer, model = _configure_fake_run(
        monkeypatch,
        tmp_path,
        model_error=True,
    )
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True, memory_probe=_memory_probe())
    assert getattr(raised.value, "code", None) == "MODEL_LOAD_FAILED"
    assert len(tokenizer.load_calls) == 1
    assert len(model.load_calls) == 1


def test_model_parameter_dtype_must_be_float32(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _torch_module, _tokenizer, model = _configure_fake_run(monkeypatch, tmp_path)
    model.dtype = object()
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True, memory_probe=_memory_probe())
    assert getattr(raised.value, "code", None) == "MODEL_NOT_FLOAT32"


def test_model_parameter_device_must_be_cpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _torch_module, _tokenizer, model = _configure_fake_run(monkeypatch, tmp_path)
    model.device = _FakeNonCpuDevice()
    model.preserve_device_on_to = True
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True, memory_probe=_memory_probe())
    assert getattr(raised.value, "code", None) == "MODEL_NOT_CPU"


def test_model_eval_must_observe_training_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _torch_module, _tokenizer, model = _configure_fake_run(
        monkeypatch,
        tmp_path,
        eval_updates_training=False,
    )
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True, memory_probe=_memory_probe())
    assert getattr(raised.value, "code", None) == "MODEL_LOAD_FAILED"
    assert model.training is True


def test_model_without_training_attribute_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _torch_module, _tokenizer, model = _configure_fake_run(monkeypatch, tmp_path)
    monkeypatch.setattr(model, "eval", lambda: model)
    del model.training
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True, memory_probe=_memory_probe())
    assert getattr(raised.value, "code", None) == "MODEL_LOAD_FAILED"


def test_empty_tokenizer_mapping_is_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _torch_module, tokenizer, _model = _configure_fake_run(monkeypatch, tmp_path)
    tokenizer.return_empty = True
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True, memory_probe=_memory_probe())
    assert getattr(raised.value, "code", None) == "TOKENIZATION_FAILED"


def test_tokenizer_tensors_must_be_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _torch_module, tokenizer, _model = _configure_fake_run(monkeypatch, tmp_path)
    tokenizer.tensor_device = _FakeNonCpuDevice()
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True, memory_probe=_memory_probe())
    assert getattr(raised.value, "code", None) == "TOKENIZATION_FAILED"


@pytest.mark.parametrize("invalid_logits", ["shape", "dtype", "device", "finite"])
def test_logits_validation_is_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid_logits: str,
) -> None:
    module, _torch_module, _tokenizer, model = _configure_fake_run(
        monkeypatch,
        tmp_path,
        finite=invalid_logits != "finite",
    )
    if invalid_logits == "shape":
        model.logits_shape = (3, 1)
    elif invalid_logits == "dtype":
        model.logits_dtype = object()
    elif invalid_logits == "device":
        model.logits_device = _FakeNonCpuDevice()
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True, memory_probe=_memory_probe())
    assert getattr(raised.value, "code", None) == "INFERENCE_RESULT_INVALID"


def test_memory_probe_failure_is_closed_before_tokenizer_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _runner()
    _install_valid_versions(monkeypatch)
    _torch_module, tokenizer, _model = _install_fake_runtime(monkeypatch)
    _install_fake_reuse(monkeypatch, module)
    monkeypatch.setattr(module, "PREFLIGHT_EVIDENCE_PATH", tmp_path / "preflight.json")

    def failing_probe() -> object:
        raise OSError("not exposed")

    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True, memory_probe=failing_probe)
    assert getattr(raised.value, "code", None) == "MEMORY_PROBE_FAILED"
    assert tokenizer.load_calls == []
    assert not (tmp_path / "preflight.json").exists()


def test_nonfinite_logits_fail_closed_without_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _runner()
    _install_valid_versions(monkeypatch)
    _install_fake_runtime(monkeypatch, finite=False)
    _install_fake_reuse(monkeypatch, module)
    evidence_path = tmp_path / "preflight.json"
    monkeypatch.setattr(module, "PREFLIGHT_EVIDENCE_PATH", evidence_path)
    with pytest.raises(RuntimeError) as raised:
        module.run_preflight(execute_local_preflight=True, memory_probe=_memory_probe())
    assert getattr(raised.value, "code", None) == "INFERENCE_RESULT_INVALID"
    assert not evidence_path.exists()


def test_atomic_publisher_is_idempotent_and_conflict_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, evidence, _tokenizer, _model, _torch = _run_fake_success(monkeypatch, tmp_path)
    evidence_path = tmp_path / "published.json"
    module.publish_preflight_evidence(evidence, evidence_path)
    first_bytes = evidence_path.read_bytes()
    first_mtime = evidence_path.stat().st_mtime_ns
    module.publish_preflight_evidence(evidence, evidence_path)
    assert evidence_path.read_bytes() == first_bytes
    assert evidence_path.stat().st_mtime_ns == first_mtime

    changed = json.loads(json.dumps(evidence))
    changed["memory"]["process_working_set_before_bytes"] = 101  # type: ignore[index]
    with pytest.raises(RuntimeError) as raised:
        module.publish_preflight_evidence(changed, evidence_path)
    assert getattr(raised.value, "code", None) == "PREFLIGHT_EVIDENCE_CONFLICT"


def test_byte_identical_evidence_skips_replace_and_preserves_mtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, evidence, _tokenizer, _model, _torch = _run_fake_success(monkeypatch, tmp_path)
    evidence_path = tmp_path / "identical.json"
    module.publish_preflight_evidence(evidence, evidence_path)
    first_mtime = evidence_path.stat().st_mtime_ns
    replace_calls: list[tuple[object, object]] = []
    real_replace = module.os.replace

    def record_replace(source: object, destination: object) -> None:
        replace_calls.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", record_replace)
    module.publish_preflight_evidence(evidence, evidence_path)
    assert replace_calls == []
    assert evidence_path.stat().st_mtime_ns == first_mtime


def test_atomic_publisher_uses_one_sibling_temp_and_one_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, evidence, _tokenizer, _model, _torch = _run_fake_success(monkeypatch, tmp_path)
    evidence_path = tmp_path / "one-replace.json"
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = module.os.replace

    def record_replace(source: object, destination: object) -> None:
        siblings = list(tmp_path.glob(f".{evidence_path.name}.tmp-*"))
        assert len(siblings) == 1
        assert Path(source) == siblings[0]
        assert Path(destination) == evidence_path
        replace_calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", record_replace)
    module.publish_preflight_evidence(evidence, evidence_path)
    assert len(replace_calls) == 1
    assert list(tmp_path.glob(f".{evidence_path.name}.tmp-*")) == []


def test_replace_failure_cleans_only_owned_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, evidence, _tokenizer, _model, _torch = _run_fake_success(monkeypatch, tmp_path)
    evidence_path = tmp_path / "replace-fails.json"
    foreign_temp = tmp_path / f".{evidence_path.name}.tmp-foreign"
    foreign_temp.write_bytes(b"not owned")

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("fake replace failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(RuntimeError) as raised:
        module.publish_preflight_evidence(evidence, evidence_path)
    assert getattr(raised.value, "code", None) == "PREFLIGHT_EVIDENCE_WRITE_FAILED"
    assert not evidence_path.exists()
    assert list(tmp_path.glob(f".{evidence_path.name}.tmp-*")) == [foreign_temp]


def test_atomic_publisher_rejects_symlink_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, evidence, _tokenizer, _model, _torch = _run_fake_success(monkeypatch, tmp_path)
    target = tmp_path / "symlink-target.json"
    target.write_bytes(b"target")
    evidence_path = tmp_path / "symlink-evidence.json"
    os.symlink(target, evidence_path)
    with pytest.raises(RuntimeError) as raised:
        module.publish_preflight_evidence(evidence, evidence_path)
    assert getattr(raised.value, "code", None) == "PREFLIGHT_EVIDENCE_CONFLICT"
    assert evidence_path.is_symlink()


def test_atomic_publisher_rejects_junction_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("junctions are Windows-only")
    module, evidence, _tokenizer, _model, _torch = _run_fake_success(monkeypatch, tmp_path)
    target = tmp_path / "junction-target"
    target.mkdir()
    evidence_path = tmp_path / "junction-evidence.json"
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(evidence_path), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not evidence_path.is_junction():
        pytest.skip("junction creation is unavailable")
    with pytest.raises(RuntimeError) as raised:
        module.publish_preflight_evidence(evidence, evidence_path)
    assert getattr(raised.value, "code", None) == "PREFLIGHT_EVIDENCE_CONFLICT"
    assert evidence_path.is_junction()


def test_atomic_publisher_rejects_directory_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, evidence, _tokenizer, _model, _torch = _run_fake_success(monkeypatch, tmp_path)
    evidence_path = tmp_path / "directory-evidence.json"
    evidence_path.mkdir()
    with pytest.raises(RuntimeError) as raised:
        module.publish_preflight_evidence(evidence, evidence_path)
    assert getattr(raised.value, "code", None) == "PREFLIGHT_EVIDENCE_CONFLICT"
    assert evidence_path.is_dir()
