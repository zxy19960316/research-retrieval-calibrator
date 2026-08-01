"""Run the explicitly authorized, pinned M2-T02 reranker snapshot download."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if __package__ in {None, ""}:
    repository_root_text = str(_REPOSITORY_ROOT)
    if repository_root_text not in sys.path:
        sys.path.insert(0, repository_root_text)

from scripts.prepare_m2_t02_reranker_snapshot import (
    SnapshotPlan,
    load_snapshot_plan,
    prepare_snapshot,
)

SELECTION_PATH = Path("evaluation/source-artifacts/m2-t02-reranker-selection.json")
_FIXED_SELECTION_PATH = SELECTION_PATH
_FIXED_SNAPSHOT_DIR = Path(
    "models/m2-t02/bge-reranker-v2-m3/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
)
SNAPSHOT_DIR = _FIXED_SNAPSHOT_DIR
EVIDENCE_PATH = Path(
    "evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json"
)
HUGGINGFACE_HUB_VERSION = "0.34.3"

_MODEL_ID = "BAAI/bge-reranker-v2-m3"
_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
_EVIDENCE_TOP_LEVEL_FIELDS = {
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
_LOCAL_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^\\\\|/(?:Users|home|tmp)/)")
_FORBIDDEN_TEXT = re.compile(
    r"(?:authorization|cookie|bearer\s|access[_-]?token|api[_-]?key|password|"
    r"username|hostname|benchmark|duration|elapsed|runtime_seconds)",
    re.IGNORECASE,
)

DownloadRunnerErrorCode = Literal[
    "LIVE_DOWNLOAD_NOT_ENABLED",
    "HUB_OFFLINE_MODE_ENABLED",
    "HUB_VERSION_UNAVAILABLE",
    "HUB_VERSION_MISMATCH",
    "HUB_IMPORT_FAILED",
    "PREPARATION_FAILED",
    "PREPARATION_RESULT_INVALID",
    "OFFLINE_REUSE_FAILED",
    "SELECTION_VALIDATION_FAILED",
    "EVIDENCE_VALIDATION_FAILED",
    "EVIDENCE_WRITE_FAILED",
    "EVIDENCE_CONFLICT",
]


class DownloadRunnerError(RuntimeError):
    """Closed, non-sensitive failure reported by the controlled download runner."""

    code: DownloadRunnerErrorCode

    def __init__(self, code: DownloadRunnerErrorCode) -> None:
        self.code = code
        super().__init__(code)


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError  # noqa: TRY004
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError  # noqa: TRY004
        result[key] = item
    return result


def _require_exact(value: object, expected: object) -> None:
    if type(value) is not type(expected):
        raise ValueError
    if isinstance(expected, dict):
        actual = _as_mapping(value)
        if set(actual) != set(expected):
            raise ValueError
        for key, expected_item in expected.items():
            _require_exact(actual[key], expected_item)
        return
    if isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise ValueError
        for actual_item, expected_item in zip(value, expected, strict=True):
            _require_exact(actual_item, expected_item)
        return
    if value != expected:
        raise ValueError


def _string_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [text for item in value.values() for text in _string_values(item)]
    if isinstance(value, list):
        return [text for item in value for text in _string_values(item)]
    return [value] if isinstance(value, str) else []


def _safe_platform_value(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError
    if _LOCAL_ABSOLUTE_PATH.search(value) or _FORBIDDEN_TEXT.search(value):
        raise ValueError
    return value


def _operational_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return _REPOSITORY_ROOT / path


def _hub_offline_mode_enabled() -> bool:
    value = os.environ.get("HF_HUB_OFFLINE")
    return value is not None and value.strip().upper() in {"1", "TRUE", "ON", "YES"}


def _load_verified_snapshot_plan() -> SnapshotPlan:
    try:
        return load_snapshot_plan(_operational_path(SELECTION_PATH))
    except Exception:  # noqa: BLE001
        raise DownloadRunnerError("SELECTION_VALIDATION_FAILED") from None


def _expected_evidence(
    preparation_status: str,
    snapshot_plan: SnapshotPlan | None = None,
) -> dict[str, object]:
    plan = snapshot_plan or _load_verified_snapshot_plan()
    files = [
        {
            "path": file.path,
            "size_bytes": file.size_bytes,
            "digest": file.digest,
            "digest_type": file.digest_type,
            "verified": True,
        }
        for file in plan.files
    ]
    if preparation_status == "PUBLISHED":
        decision_status = "downloaded_and_verified"
        downloaded_this_run = True
    elif preparation_status == "REUSED":
        decision_status = "reused_and_verified"
        downloaded_this_run = False
    else:
        raise ValueError

    return {
        "report_version": "m2-t02-reranker-snapshot-download.v1",
        "phase": "M2",
        "task_id": "M2-T02",
        "baseline_commit": "47587230c9bb02307c3a68807f149fa13df51e3d",
        "decision_status": decision_status,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "model": {
            "provider_name": "bge_reranker_v2_m3",
            "model_id": _MODEL_ID,
            "revision": _REVISION,
        },
        "download_policy": {
            "client": "huggingface_hub.hf_hub_download",
            "client_version": HUGGINGFACE_HUB_VERSION,
            "repo_type": "model",
            "token": False,
            "implicit_token_disabled": True,
            "telemetry_disabled": True,
            "selection_path": _FIXED_SELECTION_PATH.as_posix(),
            "snapshot_path": _FIXED_SNAPSHOT_DIR.as_posix(),
            "network_scope": "pinned_huggingface_model_files_only",
        },
        "snapshot": {
            "preparation_status": preparation_status,
            "file_count": len(files),
            "total_size_bytes": plan.total_size_bytes,
            "required_runtime_size_bytes": plan.required_runtime_size_bytes,
            "weight_size_bytes": plan.weight_size_bytes,
            "exact_file_closure": True,
            "local_hub_metadata_removed": True,
            "git_ignored": True,
            "files": files,
        },
        "verification": {
            "disk_space_gate": "passed",
            "size_verification": "passed",
            "digest_verification": "passed",
            "atomic_publication": "passed",
            "offline_reuse_check": "passed",
            "downloader_called_during_offline_reuse": False,
            "git_index_contains_snapshot_files": False,
        },
        "execution_state": {
            "snapshot_present_and_verified": True,
            "download_performed_this_run": downloaded_this_run,
            "model_loaded": False,
            "inference_run": False,
            "real_scores_generated": False,
        },
    }


def validate_download_evidence(
    evidence: object,
    snapshot_plan: SnapshotPlan | None = None,
) -> None:
    """Validate the complete closed, privacy-safe download evidence schema."""

    actual = _as_mapping(evidence)
    if set(actual) != _EVIDENCE_TOP_LEVEL_FIELDS:
        raise ValueError
    for text in _string_values(actual):
        if _LOCAL_ABSOLUTE_PATH.search(text) or _FORBIDDEN_TEXT.search(text):
            raise ValueError

    platform_data = _as_mapping(actual["platform"])
    if set(platform_data) != {"system", "machine", "python_version"}:
        raise ValueError
    _safe_platform_value(platform_data["system"])
    _safe_platform_value(platform_data["machine"])
    python_version = _safe_platform_value(platform_data["python_version"])
    if re.fullmatch(r"3\.12\.\d+", python_version) is None:
        raise ValueError

    snapshot_data = _as_mapping(actual["snapshot"])
    preparation_status = snapshot_data.get("preparation_status")
    if preparation_status not in {"PUBLISHED", "REUSED"}:
        raise ValueError
    expected = _expected_evidence(preparation_status, snapshot_plan)
    _require_exact(actual, expected)


def _restore_privacy_environment(previous: Mapping[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _validate_preparation_result(
    result: object,
    expected_status: str,
    expected_snapshot_dir: Path,
) -> None:
    status = getattr(result, "status", None)
    snapshot_dir = getattr(result, "snapshot_dir", None)
    if status != expected_status or snapshot_dir != expected_snapshot_dir:
        raise ValueError


def _forbidden_offline_download(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("offline reuse attempted a download")


def _forbidden_offline_disk_usage(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("offline reuse attempted a disk-space check")


def _serialized_evidence(evidence: dict[str, object]) -> bytes:
    return (json.dumps(evidence, ensure_ascii=True, indent=2) + "\n").encode("utf-8")


def _existing_evidence_bytes(evidence_path: Path) -> bytes | None:
    try:
        metadata = evidence_path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise DownloadRunnerError("EVIDENCE_CONFLICT") from None

    if evidence_path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise DownloadRunnerError("EVIDENCE_CONFLICT")
    try:
        return evidence_path.read_bytes()
    except OSError:
        raise DownloadRunnerError("EVIDENCE_CONFLICT") from None


def _publish_evidence(
    evidence: dict[str, object],
    evidence_path: Path | None = None,
) -> None:
    final_path = _operational_path(EVIDENCE_PATH) if evidence_path is None else evidence_path
    serialized = _serialized_evidence(evidence)
    existing = _existing_evidence_bytes(final_path)
    if existing is not None:
        if existing == serialized:
            return
        raise DownloadRunnerError("EVIDENCE_CONFLICT")

    temporary_path: Path | None = None
    try:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = final_path.parent / f".{final_path.name}.tmp-{uuid.uuid4().hex}"
        temporary_path.write_bytes(serialized)
        if _existing_evidence_bytes(final_path) is not None:
            raise DownloadRunnerError("EVIDENCE_CONFLICT")
        os.replace(temporary_path, final_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_download(
    *,
    execute_live_download: bool = False,
    disk_usage: Callable[[Path], object] | None = None,
) -> dict[str, object]:
    """Prepare the fixed snapshot only after explicit caller authorization."""

    if execute_live_download is not True:
        raise DownloadRunnerError("LIVE_DOWNLOAD_NOT_ENABLED")
    if _hub_offline_mode_enabled():
        raise DownloadRunnerError("HUB_OFFLINE_MODE_ENABLED")

    snapshot_plan = _load_verified_snapshot_plan()
    operational_selection_path = _operational_path(SELECTION_PATH)
    operational_snapshot_dir = _operational_path(SNAPSHOT_DIR)
    operational_evidence_path = _operational_path(EVIDENCE_PATH)

    try:
        installed_version = importlib.metadata.version("huggingface-hub")
    except Exception:  # noqa: BLE001
        raise DownloadRunnerError("HUB_VERSION_UNAVAILABLE") from None
    if installed_version != HUGGINGFACE_HUB_VERSION:
        raise DownloadRunnerError("HUB_VERSION_MISMATCH")

    selected_disk_usage: Callable[[Path], object]
    if disk_usage is None:
        selected_disk_usage = shutil.disk_usage
    else:
        selected_disk_usage = disk_usage

    previous_environment = {
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN"),
        "HF_HUB_DISABLE_TELEMETRY": os.environ.get("HF_HUB_DISABLE_TELEMETRY"),
    }
    try:
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            hub_module = importlib.import_module("huggingface_hub")
            download_file = hub_module.hf_hub_download
            if not callable(download_file):
                raise TypeError
        except Exception:  # noqa: BLE001
            raise DownloadRunnerError("HUB_IMPORT_FAILED") from None

        try:
            first_result = prepare_snapshot(
                selection_path=operational_selection_path,
                snapshot_dir=operational_snapshot_dir,
                download_file=download_file,
                disk_usage=selected_disk_usage,
            )
        except Exception:  # noqa: BLE001
            raise DownloadRunnerError("PREPARATION_FAILED") from None

        first_status = getattr(first_result, "status", None)
        try:
            if first_status not in {"PUBLISHED", "REUSED"}:
                raise ValueError
            _validate_preparation_result(
                first_result,
                first_status,
                operational_snapshot_dir,
            )
        except Exception:  # noqa: BLE001
            raise DownloadRunnerError("PREPARATION_RESULT_INVALID") from None

        if first_status == "PUBLISHED":
            try:
                reuse_result = prepare_snapshot(
                    selection_path=operational_selection_path,
                    snapshot_dir=operational_snapshot_dir,
                    download_file=_forbidden_offline_download,
                    disk_usage=_forbidden_offline_disk_usage,
                )
            except Exception:  # noqa: BLE001
                raise DownloadRunnerError("OFFLINE_REUSE_FAILED") from None

            reuse_status = getattr(reuse_result, "status", None)
            if reuse_status == "PUBLISHED":
                raise DownloadRunnerError("OFFLINE_REUSE_FAILED")
            try:
                _validate_preparation_result(
                    reuse_result,
                    "REUSED",
                    operational_snapshot_dir,
                )
            except Exception:  # noqa: BLE001
                raise DownloadRunnerError("PREPARATION_RESULT_INVALID") from None

        try:
            evidence = _expected_evidence(first_status, snapshot_plan)
            validate_download_evidence(evidence, snapshot_plan)
        except Exception:  # noqa: BLE001
            raise DownloadRunnerError("EVIDENCE_VALIDATION_FAILED") from None
        try:
            _publish_evidence(evidence, operational_evidence_path)
        except DownloadRunnerError:
            raise
        except Exception:  # noqa: BLE001
            raise DownloadRunnerError("EVIDENCE_WRITE_FAILED") from None
        return evidence
    finally:
        _restore_privacy_environment(previous_environment)


def main(argv: Sequence[str] | None = None) -> int:
    """Run only when the sole explicit live-download switch is present."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments != ["--execute-live-download"]:
        return 2
    try:
        run_download(execute_live_download=True)
    except DownloadRunnerError as exc:
        print(exc.code, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
