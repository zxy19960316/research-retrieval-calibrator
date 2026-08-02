"""Run the explicitly authorized, local-only M2-T02 reranker preflight."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if __package__ in {None, ""}:
    repository_root_text = str(_REPOSITORY_ROOT)
    if repository_root_text not in sys.path:
        sys.path.insert(0, repository_root_text)

from scripts import download_m2_t02_reranker_snapshot as download_runner
from scripts import prepare_m2_t02_reranker_snapshot as snapshot_preparation

SELECTION_PATH = Path("evaluation/source-artifacts/m2-t02-reranker-selection.json")
DOWNLOAD_EVIDENCE_PATH = Path(
    "evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json"
)
SNAPSHOT_DIR = Path(
    "models/m2-t02/bge-reranker-v2-m3/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
)
PREFLIGHT_EVIDENCE_PATH = Path(
    "evaluation/source-artifacts/m2-t02-reranker-preflight.json"
)
MODEL_ID = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
BASELINE_COMMIT = "7a234ff2684f24030dde91b3df6fe8483aa92688"

_EXPECTED_PACKAGE_VERSIONS = {
    "torch": "2.4.1+cpu",
    "transformers": "4.53.2",
    "huggingface-hub": "0.34.3",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.2",
}
_PRIVACY_ENVIRONMENT_KEYS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    "HF_HUB_DISABLE_TELEMETRY",
)
_FIXED_PAIRS: tuple[tuple[str, str], ...] = (
    (
        "人工智能如何帮助研究人员筛选相关文献？",
        "重排序模型根据查询与文献内容计算相关性。",
    ),
    (
        "人工智能如何帮助研究人员筛选相关文献？",
        "植物细胞壁主要由纤维素等成分构成。",
    ),
)
_FIXTURE_SHA256 = "8eaccc1ca1f0942a1800fc66332ccec5d80adc7c8594f9f0a440ea14d1eb288e"
FIXTURE_PAIRS = _FIXED_PAIRS
FIXTURE_SHA256 = _FIXTURE_SHA256
_PREFLIGHT_TOP_LEVEL_FIELDS = {
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
_LOCAL_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^\\\\|/(?:Users|home|tmp)/)")
_FORBIDDEN_TEXT = re.compile(
    r"(?:https?://|file://|authorization|cookie|bearer\s|access[_-]?token|"
    r"api[_-]?key|password|username|hostname|raw[_-]?logit|candidate[_-]?id|"
    r"/root/)",
    re.IGNORECASE,
)

PreflightRunnerErrorCode = Literal[
    "LOCAL_PREFLIGHT_NOT_ENABLED",
    "DOWNLOAD_EVIDENCE_INVALID",
    "SNAPSHOT_REUSE_VERIFICATION_FAILED",
    "RUNTIME_VERSION_UNAVAILABLE",
    "RUNTIME_VERSION_MISMATCH",
    "RUNTIME_IMPORT_FAILED",
    "NON_CPU_TORCH_BUILD",
    "MEMORY_PROBE_FAILED",
    "TOKENIZER_LOAD_FAILED",
    "MODEL_LOAD_FAILED",
    "MODEL_NOT_FLOAT32",
    "MODEL_NOT_CPU",
    "TOKENIZATION_FAILED",
    "INFERENCE_FAILED",
    "INFERENCE_RESULT_INVALID",
    "PREFLIGHT_EVIDENCE_VALIDATION_FAILED",
    "PREFLIGHT_EVIDENCE_CONFLICT",
    "PREFLIGHT_EVIDENCE_WRITE_FAILED",
]


class PreflightRunnerError(RuntimeError):
    """Closed, non-sensitive failure reported by the preflight runner."""

    code: PreflightRunnerErrorCode

    def __init__(self, code: PreflightRunnerErrorCode) -> None:
        self.code = code
        super().__init__(code)


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError
        result[key] = item
    return result


def _require_keys(value: object, expected: set[str]) -> dict[str, object]:
    actual = _as_mapping(value)
    if set(actual) != expected:
        raise ValueError
    return actual


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _strict_non_negative_int(value: object) -> int:
    if not _is_strict_int(value) or cast(int, value) < 0:
        raise ValueError
    return cast(int, value)


def _string_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [text for item in value.values() for text in _string_values(item)]
    if isinstance(value, list):
        return [text for item in value for text in _string_values(item)]
    return [value] if isinstance(value, str) else []


def _operational_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return _REPOSITORY_ROOT / path


def _is_link_or_junction(path: Path, metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    try:
        return path.is_junction()
    except OSError:
        return True


def _read_json_file(path: Path, code: PreflightRunnerErrorCode) -> object:
    try:
        metadata = path.lstat()
        if _is_link_or_junction(path, metadata) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError
        with path.open("rb") as handle:
            opened_metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened_metadata.st_mode):
                raise ValueError
            if (opened_metadata.st_dev, opened_metadata.st_ino) != (
                metadata.st_dev,
                metadata.st_ino,
            ):
                raise ValueError
            raw_payload = handle.read()
        return json.loads(raw_payload.decode("utf-8"))
    except PreflightRunnerError:
        raise
    except Exception:  # noqa: BLE001
        raise PreflightRunnerError(code) from None


def _load_snapshot_plan() -> snapshot_preparation.SnapshotPlan:
    try:
        return snapshot_preparation.load_snapshot_plan(_operational_path(SELECTION_PATH))
    except Exception:  # noqa: BLE001
        raise PreflightRunnerError("DOWNLOAD_EVIDENCE_INVALID") from None


def _recorded_platform(evidence: object) -> tuple[str, str, str]:
    actual = _as_mapping(evidence)
    platform_data = _as_mapping(actual["platform"])
    system = platform_data["system"]
    machine = platform_data["machine"]
    python_version = platform_data["python_version"]
    if not all(isinstance(value, str) and value.strip() for value in (system, machine, python_version)):
        raise ValueError
    return cast(str, system), cast(str, machine), cast(str, python_version)


def _validate_download_evidence(
    evidence: object,
    snapshot_plan: snapshot_preparation.SnapshotPlan,
) -> None:
    try:
        actual = _as_mapping(evidence)
        if actual.get("decision_status") != "downloaded_and_verified":
            raise ValueError
        snapshot_data = _as_mapping(actual["snapshot"])
        if snapshot_data.get("preparation_status") != "PUBLISHED":
            raise ValueError
        system, machine, python_version = _recorded_platform(evidence)
    except Exception:  # noqa: BLE001
        raise PreflightRunnerError("DOWNLOAD_EVIDENCE_INVALID") from None

    download_any: Any = download_runner
    recorded_platform: Any = download_any.platform
    original = (
        recorded_platform.system,
        recorded_platform.machine,
        recorded_platform.python_version,
    )
    try:
        setattr(recorded_platform, "system", lambda value=system: value)  # noqa: B010
        setattr(recorded_platform, "machine", lambda value=machine: value)  # noqa: B010
        setattr(recorded_platform, "python_version", lambda value=python_version: value)  # noqa: B010
        download_runner.validate_download_evidence(evidence, snapshot_plan)
    except Exception:  # noqa: BLE001
        raise PreflightRunnerError("DOWNLOAD_EVIDENCE_INVALID") from None
    finally:
        (
            recorded_platform.system,
            recorded_platform.machine,
            recorded_platform.python_version,
        ) = original


def _load_download_evidence() -> dict[str, object]:
    snapshot_plan = _load_snapshot_plan()
    evidence = _read_json_file(
        _operational_path(DOWNLOAD_EVIDENCE_PATH),
        "DOWNLOAD_EVIDENCE_INVALID",
    )
    _validate_download_evidence(evidence, snapshot_plan)
    return _as_mapping(evidence)


def _forbidden_downloader(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("offline downloader callback")


def _forbidden_disk_usage(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("offline disk callback")


def _reuse_snapshot(snapshot_plan: snapshot_preparation.SnapshotPlan) -> None:
    del snapshot_plan
    selection_path = _operational_path(SELECTION_PATH)
    snapshot_path = _operational_path(SNAPSHOT_DIR)
    try:
        result = snapshot_preparation.prepare_snapshot(
            selection_path=selection_path,
            snapshot_dir=snapshot_path,
            download_file=_forbidden_downloader,
            disk_usage=_forbidden_disk_usage,
        )
        if getattr(result, "status", None) != "REUSED":
            raise ValueError
        if getattr(result, "snapshot_dir", None) != snapshot_path:
            raise ValueError
    except Exception:  # noqa: BLE001
        raise PreflightRunnerError("SNAPSHOT_REUSE_VERIFICATION_FAILED") from None


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        for package_name in _EXPECTED_PACKAGE_VERSIONS:
            version = importlib.metadata.version(package_name)
            if not isinstance(version, str):
                raise TypeError
            versions[package_name] = version
    except Exception:  # noqa: BLE001
        raise PreflightRunnerError("RUNTIME_VERSION_UNAVAILABLE") from None
    if versions != _EXPECTED_PACKAGE_VERSIONS:
        raise PreflightRunnerError("RUNTIME_VERSION_MISMATCH")
    return versions


def _restore_environment(previous: Mapping[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@contextmanager
def _privacy_environment() -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in _PRIVACY_ENVIRONMENT_KEYS}
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        yield
    finally:
        _restore_environment(previous)


def _import_runtime() -> tuple[ModuleType, ModuleType]:
    try:
        torch_module = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")
    except Exception:  # noqa: BLE001
        raise PreflightRunnerError("RUNTIME_IMPORT_FAILED") from None
    return torch_module, transformers_module


def _verify_cpu_torch(torch_module: ModuleType) -> None:
    try:
        torch_any: Any = torch_module
        torch_version = torch_any.version
        cuda_version = torch_version.cuda
        cuda_module = torch_any.cuda
        cuda_available = cuda_module.is_available()
        if cuda_version is not None or cuda_available is not False:
            raise ValueError
    except PreflightRunnerError:
        raise
    except Exception:  # noqa: BLE001
        raise PreflightRunnerError("NON_CPU_TORCH_BUILD") from None


def _windows_memory_probe() -> Mapping[str, object]:
    """Read memory through Windows APIs without importing a runtime package."""

    if platform.system() != "Windows":
        raise OSError

    try:
        win_dll_factory = cast(Any, ctypes).WinDLL
        from ctypes import wintypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        class _ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        kernel32: Any = win_dll_factory("kernel32", use_last_error=True)
        psapi: Any = win_dll_factory("psapi", use_last_error=True)
        system_status = _MemoryStatusEx()
        system_status.dwLength = ctypes.sizeof(system_status)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(system_status)):
            raise OSError
        process_counters = _ProcessMemoryCountersEx()
        process_counters.cb = ctypes.sizeof(process_counters)
        process_handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(
            process_handle,
            ctypes.byref(process_counters),
            ctypes.sizeof(process_counters),
        ):
            raise OSError
        return {
            "system_total_physical_memory_bytes": int(system_status.ullTotalPhys),
            "system_available_physical_memory_bytes": int(system_status.ullAvailPhys),
            "process_working_set_bytes": int(process_counters.WorkingSetSize),
            "process_peak_working_set_bytes": int(process_counters.PeakWorkingSetSize),
        }
    except Exception:  # noqa: BLE001
        raise OSError from None


_MEMORY_FIELDS = {
    "system_total_physical_memory_bytes",
    "system_available_physical_memory_bytes",
    "process_working_set_bytes",
    "process_peak_working_set_bytes",
}


def _capture_memory(probe: Callable[[], object]) -> dict[str, int]:
    try:
        raw = _as_mapping(probe())
        if set(raw) != _MEMORY_FIELDS:
            raise ValueError
        return {key: _strict_non_negative_int(raw[key]) for key in _MEMORY_FIELDS}
    except Exception:  # noqa: BLE001
        raise PreflightRunnerError("MEMORY_PROBE_FAILED") from None


def _is_cpu_device(value: object) -> bool:
    if isinstance(value, str):
        return value == "cpu"
    return getattr(value, "type", None) == "cpu"


def _model_parameters(model: object) -> list[object]:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return []
    try:
        return list(parameters())
    except Exception:  # noqa: BLE001
        raise PreflightRunnerError("MODEL_LOAD_FAILED") from None


def _validate_model_state(model: object, torch_module: ModuleType) -> None:
    torch_any: Any = torch_module
    parameters = _model_parameters(model)
    observed_dtype = False
    observed_device = False
    for parameter in parameters:
        dtype = getattr(parameter, "dtype", None)
        if dtype is not None:
            observed_dtype = True
            if dtype != torch_any.float32:
                raise PreflightRunnerError("MODEL_NOT_FLOAT32")
        device = getattr(parameter, "device", None)
        if device is not None:
            observed_device = True
            if not _is_cpu_device(device):
                raise PreflightRunnerError("MODEL_NOT_CPU")

    model_dtype = getattr(model, "dtype", None)
    if model_dtype is not None:
        observed_dtype = True
        if model_dtype != torch_any.float32:
            raise PreflightRunnerError("MODEL_NOT_FLOAT32")
    model_device = getattr(model, "device", None)
    if model_device is not None:
        observed_device = True
        if not _is_cpu_device(model_device):
            raise PreflightRunnerError("MODEL_NOT_CPU")

    if not observed_dtype:
        raise PreflightRunnerError("MODEL_NOT_FLOAT32")
    if not observed_device:
        raise PreflightRunnerError("MODEL_NOT_CPU")


def _validate_tokenized_inputs(encoded: object) -> Mapping[str, object]:
    if not isinstance(encoded, Mapping) or not encoded:
        raise PreflightRunnerError("TOKENIZATION_FAILED")
    for value in encoded.values():
        if not _is_cpu_device(getattr(value, "device", None)):
            raise PreflightRunnerError("TOKENIZATION_FAILED")
    return cast(Mapping[str, object], encoded)


def _extract_logits(outputs: object) -> object:
    if isinstance(outputs, Mapping) and "logits" in outputs:
        return outputs["logits"]
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise PreflightRunnerError("INFERENCE_RESULT_INVALID")
    return logits


def _validate_logits(
    torch_module: ModuleType,
    logits: object,
) -> list[int]:
    try:
        logits_any: Any = logits
        torch_any: Any = torch_module
        shape_value = logits_any.shape
        if not isinstance(shape_value, (tuple, list)) or not shape_value:
            raise ValueError
        shape = [_strict_non_negative_int(value) for value in shape_value]
        if shape[0] != len(_FIXED_PAIRS):
            raise ValueError
        numel_method = getattr(logits, "numel", None)
        if callable(numel_method):
            total = numel_method()
        else:
            total = 1
            for dimension in shape:
                total *= dimension
        if not _is_strict_int(total) or total <= 0 or total % len(_FIXED_PAIRS) != 0:
            raise ValueError
        if logits_any.dtype != torch_any.float32:
            raise ValueError
        if not _is_cpu_device(getattr(logits, "device", None)):
            raise ValueError
        finite_result = torch_module.isfinite(logits)
        all_result = finite_result.all() if callable(getattr(finite_result, "all", None)) else finite_result
        item_method = getattr(all_result, "item", None)
        if callable(item_method):
            all_result = item_method()
        if all_result is not True:
            raise ValueError
        return shape
    except PreflightRunnerError:
        raise
    except Exception:  # noqa: BLE001
        raise PreflightRunnerError("INFERENCE_RESULT_INVALID") from None


def _serialize_fixture() -> bytes:
    pairs = [[query, passage] for query, passage in _FIXED_PAIRS]
    return json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _build_evidence(
    versions: Mapping[str, str],
    memory_before: Mapping[str, int],
    memory_after_load: Mapping[str, int],
    memory_after_inference: Mapping[str, int],
    logits_shape: list[int],
) -> dict[str, object]:
    return {
        "report_version": "m2-t02-reranker-preflight.v1",
        "phase": "M2",
        "task_id": "M2-T02",
        "baseline_commit": BASELINE_COMMIT,
        "decision_status": "cpu_float32_preflight_passed",
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "model": {
            "provider_name": "bge_reranker_v2_m3",
            "model_id": MODEL_ID,
            "revision": REVISION,
        },
        "snapshot": {
            "selection_path": SELECTION_PATH.as_posix(),
            "download_evidence_path": DOWNLOAD_EVIDENCE_PATH.as_posix(),
            "snapshot_path": SNAPSHOT_DIR.as_posix(),
            "preparation_status": "REUSED",
            "download_evidence_status": "downloaded_and_verified",
            "file_count": 7,
            "exact_file_closure": True,
            "digest_verification": "passed",
            "downloader_calls": 0,
            "disk_usage_calls": 0,
        },
        "runtime": {
            "torch_distribution_version": versions["torch"],
            "torch_base_version": "2.4.1",
            "transformers_version": versions["transformers"],
            "huggingface_hub_version": versions["huggingface-hub"],
            "safetensors_version": versions["safetensors"],
            "tokenizers_version": versions["tokenizers"],
            "torch_cuda_version": None,
            "torch_cuda_available": False,
        },
        "offline_policy": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "local_files_only": True,
            "network_forbidden": True,
        },
        "memory": {
            "system_total_physical_memory_bytes": memory_before[
                "system_total_physical_memory_bytes"
            ],
            "system_available_physical_memory_before_bytes": memory_before[
                "system_available_physical_memory_bytes"
            ],
            "process_working_set_before_bytes": memory_before["process_working_set_bytes"],
            "process_peak_working_set_after_load_bytes": memory_after_load[
                "process_peak_working_set_bytes"
            ],
            "process_peak_working_set_after_inference_bytes": memory_after_inference[
                "process_peak_working_set_bytes"
            ],
        },
        "tokenizer": {
            "local_files_only": True,
            "trust_remote_code": False,
            "padding": True,
            "truncation": True,
            "max_length": 512,
            "return_tensors": "pt",
            "batch_size": 2,
        },
        "model_load": {
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
            "torch_dtype": "float32",
            "device": "cpu",
            "eval_mode": True,
        },
        "inference": {
            "torch_inference_mode": True,
            "batch_size": 2,
            "max_length": 512,
            "fixture_sha256": _FIXTURE_SHA256,
            "logits_shape": logits_shape,
            "logits_dtype": "float32",
            "logits_all_finite": True,
            "logits_persisted": False,
            "model_device": "cpu",
            "inputs_device": "cpu",
        },
        "execution_state": {
            "snapshot_reused": True,
            "tokenizer_loaded": True,
            "model_loaded": True,
            "minimal_preflight_inference_run": True,
            "preflight_logits_generated": True,
            "preflight_logits_persisted": False,
            "full_candidate_reranking_run": False,
            "real_candidate_scores_generated": False,
        },
    }


def _validate_hex_digest(value: object) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError


def validate_preflight_evidence(evidence: object) -> None:
    """Validate the closed, privacy-safe preflight evidence schema."""

    actual = _require_keys(evidence, _PREFLIGHT_TOP_LEVEL_FIELDS)
    for text in _string_values(actual):
        if _LOCAL_ABSOLUTE_PATH.search(text) or _FORBIDDEN_TEXT.search(text):
            raise ValueError
    if actual["report_version"] != "m2-t02-reranker-preflight.v1":
        raise ValueError
    if actual["phase"] != "M2" or actual["task_id"] != "M2-T02":
        raise ValueError
    if actual["baseline_commit"] != BASELINE_COMMIT:
        raise ValueError
    if actual["decision_status"] != "cpu_float32_preflight_passed":
        raise ValueError

    platform_data = _require_keys(actual["platform"], {"system", "machine", "python_version"})
    for key in platform_data:
        if not isinstance(platform_data[key], str) or not cast(str, platform_data[key]).strip():
            raise ValueError
    if re.fullmatch(r"3\.12\.\d+", cast(str, platform_data["python_version"])) is None:
        raise ValueError

    model_data = _require_keys(actual["model"], {"provider_name", "model_id", "revision"})
    if model_data != {
        "provider_name": "bge_reranker_v2_m3",
        "model_id": MODEL_ID,
        "revision": REVISION,
    }:
        raise ValueError

    snapshot_data = _require_keys(
        actual["snapshot"],
        {
            "selection_path",
            "download_evidence_path",
            "snapshot_path",
            "preparation_status",
            "download_evidence_status",
            "file_count",
            "exact_file_closure",
            "digest_verification",
            "downloader_calls",
            "disk_usage_calls",
        },
    )
    if snapshot_data != {
        "selection_path": SELECTION_PATH.as_posix(),
        "download_evidence_path": DOWNLOAD_EVIDENCE_PATH.as_posix(),
        "snapshot_path": SNAPSHOT_DIR.as_posix(),
        "preparation_status": "REUSED",
        "download_evidence_status": "downloaded_and_verified",
        "file_count": 7,
        "exact_file_closure": True,
        "digest_verification": "passed",
        "downloader_calls": 0,
        "disk_usage_calls": 0,
    }:
        raise ValueError

    runtime_data = _require_keys(
        actual["runtime"],
        {
            "torch_distribution_version",
            "torch_base_version",
            "transformers_version",
            "huggingface_hub_version",
            "safetensors_version",
            "tokenizers_version",
            "torch_cuda_version",
            "torch_cuda_available",
        },
    )
    if runtime_data != {
        "torch_distribution_version": "2.4.1+cpu",
        "torch_base_version": "2.4.1",
        "transformers_version": "4.53.2",
        "huggingface_hub_version": "0.34.3",
        "safetensors_version": "0.5.3",
        "tokenizers_version": "0.21.2",
        "torch_cuda_version": None,
        "torch_cuda_available": False,
    }:
        raise ValueError

    offline_data = _require_keys(
        actual["offline_policy"],
        {
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN",
            "HF_HUB_DISABLE_TELEMETRY",
            "local_files_only",
            "network_forbidden",
        },
    )
    if offline_data != {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "local_files_only": True,
        "network_forbidden": True,
    }:
        raise ValueError

    memory_data = _require_keys(
        actual["memory"],
        {
            "system_total_physical_memory_bytes",
            "system_available_physical_memory_before_bytes",
            "process_working_set_before_bytes",
            "process_peak_working_set_after_load_bytes",
            "process_peak_working_set_after_inference_bytes",
        },
    )
    for value in memory_data.values():
        _strict_non_negative_int(value)

    tokenizer_data = _require_keys(
        actual["tokenizer"],
        {
            "local_files_only",
            "trust_remote_code",
            "padding",
            "truncation",
            "max_length",
            "return_tensors",
            "batch_size",
        },
    )
    if tokenizer_data != {
        "local_files_only": True,
        "trust_remote_code": False,
        "padding": True,
        "truncation": True,
        "max_length": 512,
        "return_tensors": "pt",
        "batch_size": 2,
    }:
        raise ValueError

    model_load_data = _require_keys(
        actual["model_load"],
        {
            "local_files_only",
            "trust_remote_code",
            "use_safetensors",
            "torch_dtype",
            "device",
            "eval_mode",
        },
    )
    if model_load_data != {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "torch_dtype": "float32",
        "device": "cpu",
        "eval_mode": True,
    }:
        raise ValueError

    inference_data = _require_keys(
        actual["inference"],
        {
            "torch_inference_mode",
            "batch_size",
            "max_length",
            "fixture_sha256",
            "logits_shape",
            "logits_dtype",
            "logits_all_finite",
            "logits_persisted",
            "model_device",
            "inputs_device",
        },
    )
    if inference_data["torch_inference_mode"] is not True:
        raise ValueError
    if inference_data["batch_size"] != 2 or inference_data["max_length"] != 512:
        raise ValueError
    if inference_data["logits_dtype"] != "float32":
        raise ValueError
    if inference_data["logits_all_finite"] is not True:
        raise ValueError
    if inference_data["logits_persisted"] is not False:
        raise ValueError
    if inference_data["model_device"] != "cpu" or inference_data["inputs_device"] != "cpu":
        raise ValueError
    if inference_data["fixture_sha256"] != _FIXTURE_SHA256:
        raise ValueError
    _validate_hex_digest(inference_data["fixture_sha256"])
    shape = inference_data["logits_shape"]
    if not isinstance(shape, list) or not shape or any(not _is_strict_int(value) for value in shape):
        raise ValueError
    if shape[0] != len(_FIXED_PAIRS):
        raise ValueError
    total = 1
    for dimension in shape:
        if cast(int, dimension) < 0:
            raise ValueError
        total *= cast(int, dimension)
    if total <= 0 or total % len(_FIXED_PAIRS) != 0:
        raise ValueError

    execution_data = _require_keys(
        actual["execution_state"],
        {
            "snapshot_reused",
            "tokenizer_loaded",
            "model_loaded",
            "minimal_preflight_inference_run",
            "preflight_logits_generated",
            "preflight_logits_persisted",
            "full_candidate_reranking_run",
            "real_candidate_scores_generated",
        },
    )
    if execution_data != {
        "snapshot_reused": True,
        "tokenizer_loaded": True,
        "model_loaded": True,
        "minimal_preflight_inference_run": True,
        "preflight_logits_generated": True,
        "preflight_logits_persisted": False,
        "full_candidate_reranking_run": False,
        "real_candidate_scores_generated": False,
    }:
        raise ValueError


def _serialized_evidence(evidence: dict[str, object]) -> bytes:
    return (json.dumps(evidence, ensure_ascii=True, indent=2) + "\n").encode("utf-8")


def _existing_preflight_bytes(path: Path) -> bytes | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise PreflightRunnerError("PREFLIGHT_EVIDENCE_CONFLICT") from None
    if _is_link_or_junction(path, metadata) or not stat.S_ISREG(metadata.st_mode):
        raise PreflightRunnerError("PREFLIGHT_EVIDENCE_CONFLICT")
    try:
        return path.read_bytes()
    except OSError:
        raise PreflightRunnerError("PREFLIGHT_EVIDENCE_CONFLICT") from None


def publish_preflight_evidence(
    evidence: dict[str, object],
    evidence_path: Path | None = None,
) -> None:
    """Validate and atomically publish only a closed preflight evidence file."""

    try:
        validate_preflight_evidence(evidence)
    except Exception:  # noqa: BLE001
        raise PreflightRunnerError("PREFLIGHT_EVIDENCE_VALIDATION_FAILED") from None

    final_path = (
        _operational_path(PREFLIGHT_EVIDENCE_PATH)
        if evidence_path is None
        else Path(evidence_path)
    )
    serialized = _serialized_evidence(evidence)
    existing = _existing_preflight_bytes(final_path)
    if existing is not None:
        if existing == serialized:
            return
        raise PreflightRunnerError("PREFLIGHT_EVIDENCE_CONFLICT")

    temporary_path: Path | None = None
    try:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = final_path.parent / f".{final_path.name}.tmp-{uuid.uuid4().hex}"
        temporary_path.write_bytes(serialized)
        if _existing_preflight_bytes(final_path) is not None:
            raise PreflightRunnerError("PREFLIGHT_EVIDENCE_CONFLICT")
        os.replace(temporary_path, final_path)
        temporary_path = None
    except PreflightRunnerError:
        raise
    except Exception:  # noqa: BLE001
        raise PreflightRunnerError("PREFLIGHT_EVIDENCE_WRITE_FAILED") from None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _run_runtime_preflight(
    versions: Mapping[str, str],
    memory_probe: Callable[[], object],
) -> dict[str, object]:
    if hashlib.sha256(_serialize_fixture()).hexdigest() != _FIXTURE_SHA256:
        raise PreflightRunnerError("INFERENCE_RESULT_INVALID")
    with _privacy_environment():
        torch_module, transformers_module = _import_runtime()
        torch_any: Any = torch_module
        transformers_any: Any = transformers_module
        _verify_cpu_torch(torch_module)
        memory_before = _capture_memory(memory_probe)

        try:
            auto_tokenizer = transformers_any.AutoTokenizer
            tokenizer = auto_tokenizer.from_pretrained(
                _operational_path(SNAPSHOT_DIR),
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception:  # noqa: BLE001
            raise PreflightRunnerError("TOKENIZER_LOAD_FAILED") from None

        try:
            auto_model = transformers_any.AutoModelForSequenceClassification
            model = auto_model.from_pretrained(
                _operational_path(SNAPSHOT_DIR),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                torch_dtype=torch_any.float32,
            )
            moved_model = model.to("cpu")
            if moved_model is not None:
                model = moved_model
            evaluated_model = model.eval()
            if evaluated_model is not None:
                model = evaluated_model
            _validate_model_state(model, torch_module)
        except PreflightRunnerError:
            raise
        except Exception:  # noqa: BLE001
            raise PreflightRunnerError("MODEL_LOAD_FAILED") from None
        memory_after_load = _capture_memory(memory_probe)

        queries = [query for query, _passage in _FIXED_PAIRS]
        passages = [passage for _query, passage in _FIXED_PAIRS]
        try:
            encoded = tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded_inputs = _validate_tokenized_inputs(encoded)
        except PreflightRunnerError:
            raise
        except Exception:  # noqa: BLE001
            raise PreflightRunnerError("TOKENIZATION_FAILED") from None

        try:
            with torch_module.inference_mode():
                outputs = model(**encoded_inputs)
                logits = _extract_logits(outputs)
        except PreflightRunnerError:
            raise
        except Exception:  # noqa: BLE001
            raise PreflightRunnerError("INFERENCE_FAILED") from None
        logits_shape = _validate_logits(torch_module, logits)
        memory_after_inference = _capture_memory(memory_probe)

        return _build_evidence(
            versions,
            memory_before,
            memory_after_load,
            memory_after_inference,
            logits_shape,
        )


def run_preflight(
    *,
    execute_local_preflight: bool = False,
    memory_probe: Callable[[], object] | None = None,
) -> dict[str, object]:
    """Run the fixed local preflight only after explicit caller authorization."""

    if execute_local_preflight is not True:
        raise PreflightRunnerError("LOCAL_PREFLIGHT_NOT_ENABLED")

    snapshot_plan = _load_snapshot_plan()
    _load_download_evidence()
    _reuse_snapshot(snapshot_plan)
    versions = _runtime_versions()
    selected_memory_probe = _windows_memory_probe if memory_probe is None else memory_probe
    evidence = _run_runtime_preflight(versions, selected_memory_probe)
    publish_preflight_evidence(evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    """Run only with the one explicit local-preflight switch."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments != ["--execute-local-preflight"]:
        return 2
    try:
        run_preflight(execute_local_preflight=True)
    except PreflightRunnerError as exc:
        print(exc.code, file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001
        print("RUNTIME_IMPORT_FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
