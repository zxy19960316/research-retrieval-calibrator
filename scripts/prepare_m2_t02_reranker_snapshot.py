"""Prepare the pinned M2-T02 reranker snapshot without runtime model imports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Protocol, TypeGuard, cast

DOWNLOAD_HEADROOM_BYTES = 1_073_741_824

SnapshotPreparationErrorCode = Literal[
    "INSUFFICIENT_DISK_SPACE",
    "DOWNLOAD_FAILED",
    "INTEGRITY_CHECK_FAILED",
]
SnapshotDownloadDiagnostic = Literal[
    "HTTP_4XX",
    "HTTP_5XX",
    "NETWORK_TIMEOUT",
    "TLS_OR_CERTIFICATE_FAILURE",
    "NETWORK_CONNECTION_FAILED",
    "DOWNLOAD_EXCEPTION",
]
DigestType = Literal["git_blob_sha1", "sha256"]
StorageType = Literal["git", "lfs"]
SnapshotPreparationStatus = Literal["PUBLISHED", "REUSED"]
_PathIdentity = tuple[int, int]
_ParentChainState = tuple[tuple[Path, _PathIdentity | None], ...]

_MODEL_ID = "BAAI/bge-reranker-v2-m3"
_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
_HASH_CHUNK_SIZE = 1024 * 1024
_EXECUTION_STATE_KEYS = (
    "weights_downloaded",
    "tokenizer_downloaded",
    "model_loaded",
    "inference_run",
    "real_scores_generated",
)
_FILE_SPECS: tuple[tuple[str, DigestType, StorageType, bool], ...] = (
    ("README.md", "git_blob_sha1", "git", False),
    ("config.json", "git_blob_sha1", "git", True),
    ("model.safetensors", "sha256", "lfs", True),
    ("sentencepiece.bpe.model", "sha256", "lfs", True),
    ("special_tokens_map.json", "git_blob_sha1", "git", True),
    ("tokenizer.json", "sha256", "lfs", True),
    ("tokenizer_config.json", "git_blob_sha1", "git", True),
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _DiskUsageResult(Protocol):
    free: object


class SnapshotPreparationError(RuntimeError):
    """Closed public failure for snapshot preparation."""

    code: SnapshotPreparationErrorCode
    diagnostic: SnapshotDownloadDiagnostic | None

    def __init__(
        self,
        code: SnapshotPreparationErrorCode,
        diagnostic: SnapshotDownloadDiagnostic | None = None,
    ) -> None:
        self.code = code
        self.diagnostic = diagnostic
        super().__init__(code)


@dataclass(frozen=True)
class SnapshotFilePlan:
    """One immutable file entry in the pinned snapshot plan."""

    path: str
    size_bytes: int
    digest: str
    digest_type: DigestType
    storage_type: StorageType
    required_for_runtime: bool


@dataclass(frozen=True)
class SnapshotPlan:
    """Validated immutable plan for the complete pinned snapshot."""

    model_id: str
    revision: str
    files: tuple[SnapshotFilePlan, ...]
    total_size_bytes: int
    required_runtime_size_bytes: int
    weight_size_bytes: int


@dataclass(frozen=True)
class SnapshotPreparationResult:
    """Result of publishing or reusing a complete pinned snapshot."""

    status: SnapshotPreparationStatus
    snapshot_dir: Path


def _execution_file_order(
    files: tuple[SnapshotFilePlan, ...],
) -> tuple[SnapshotFilePlan, ...]:
    """Return supporting files first without changing the canonical plan."""

    expected_paths = tuple(spec[0] for spec in _FILE_SPECS)
    paths = tuple(file_plan.path for file_plan in files)
    if (
        len(paths) != len(expected_paths)
        or len(set(paths)) != len(paths)
        or set(paths) != set(expected_paths)
    ):
        raise ValueError

    weights = tuple(
        file_plan for file_plan in files if file_plan.path == "model.safetensors"
    )
    if len(weights) != 1:
        raise ValueError

    return tuple(
        file_plan for file_plan in files if file_plan.path != "model.safetensors"
    ) + weights


def _integrity_error() -> SnapshotPreparationError:
    return SnapshotPreparationError("INTEGRITY_CHECK_FAILED")


def _download_error(
    diagnostic: SnapshotDownloadDiagnostic | None = None,
) -> SnapshotPreparationError:
    return SnapshotPreparationError("DOWNLOAD_FAILED", diagnostic)


def _is_strict_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _download_failure_diagnostic(exc: BaseException) -> SnapshotDownloadDiagnostic:
    """Classify a downloader exception without reading its text or payload."""

    try:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if not _is_strict_int(status_code):
            status_code = getattr(exc, "status_code", None)
    except Exception:  # noqa: BLE001
        status_code = None
    if _is_strict_int(status_code):
        if 400 <= status_code < 500:
            return "HTTP_4XX"
        if 500 <= status_code < 600:
            return "HTTP_5XX"

    type_names = tuple(error_type.__name__.casefold() for error_type in type(exc).__mro__)
    if any("timeout" in name for name in type_names):
        return "NETWORK_TIMEOUT"
    if any(
        marker in name
        for name in type_names
        for marker in ("ssl", "tls", "certificate")
    ):
        return "TLS_OR_CERTIFICATE_FAILURE"
    if any(
        marker in name
        for name in type_names
        for marker in ("connection", "connect", "proxy")
    ):
        return "NETWORK_CONNECTION_FAILED"
    return "DOWNLOAD_EXCEPTION"


def _is_path_input(value: object) -> TypeGuard[str | os.PathLike[str]]:
    return isinstance(value, (str, os.PathLike))


def _as_object_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError
    if not all(isinstance(key, str) for key in value):
        raise TypeError
    return cast(Mapping[str, object], value)


def _is_link_or_junction(path: Path, metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or path.is_junction()


def _ordinary_directory_identity(path: Path) -> _PathIdentity:
    metadata = path.lstat()
    if _is_link_or_junction(path, metadata):
        raise ValueError
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError
    return metadata.st_dev, metadata.st_ino


def _require_directory_identity(path: Path, expected: _PathIdentity) -> None:
    if _ordinary_directory_identity(path) != expected:
        raise ValueError


def _require_directory_identity_for_integrity(
    path: Path,
    expected: _PathIdentity,
) -> None:
    try:
        _require_directory_identity(path, expected)
    except Exception:  # noqa: BLE001
        raise _integrity_error() from None


def _capture_parent_chain_state(destination: Path) -> _ParentChainState:
    try:
        state: list[tuple[Path, _PathIdentity | None]] = []
        for parent in reversed(destination.parents):
            try:
                metadata = parent.lstat()
            except FileNotFoundError:
                state.append((parent, None))
                continue
            if _is_link_or_junction(parent, metadata):
                raise ValueError
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError
            state.append((parent, (metadata.st_dev, metadata.st_ino)))
        return tuple(state)
    except SnapshotPreparationError:
        raise
    except Exception:  # noqa: BLE001
        raise _integrity_error() from None


def _require_parent_chain_state(expected: _ParentChainState) -> None:
    try:
        for parent, expected_identity in expected:
            try:
                metadata = parent.lstat()
            except FileNotFoundError:
                if expected_identity is None:
                    continue
                raise ValueError from None
            if expected_identity is None:
                raise ValueError
            if _is_link_or_junction(parent, metadata):
                raise ValueError
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError
            if (metadata.st_dev, metadata.st_ino) != expected_identity:
                raise ValueError
    except SnapshotPreparationError:
        raise
    except Exception:  # noqa: BLE001
        raise _integrity_error() from None


def _is_safe_single_filename(value: object, expected: str) -> bool:
    if not isinstance(value, str) or value == "" or value in {".", ".."}:
        return False
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        return False
    if posix_path.parts != (value,) or windows_path.parts != (value,):
        return False
    if "/" in value or "\\" in value:
        return False
    return value == expected


def _validate_summary_value(value: object, expected: int) -> None:
    if not _is_strict_int(value) or value != expected:
        raise ValueError


def _parse_snapshot_plan(payload: object) -> SnapshotPlan:
    root = _as_object_mapping(payload)
    if root.get("decision_status") != "selected_not_downloaded":
        raise ValueError
    if root.get("phase") != "M2" or root.get("task_id") != "M2-T02":
        raise ValueError

    execution_state = _as_object_mapping(root.get("execution_state"))
    if set(execution_state) != set(_EXECUTION_STATE_KEYS):
        raise ValueError
    if any(execution_state.get(key) is not False for key in _EXECUTION_STATE_KEYS):
        raise ValueError

    selected_model = _as_object_mapping(root.get("selected_model"))
    official_metadata = _as_object_mapping(
        selected_model.get("official_metadata_source")
    )
    if selected_model.get("model_id") != _MODEL_ID:
        raise ValueError
    if selected_model.get("model_revision") != _REVISION:
        raise ValueError
    if official_metadata.get("model_id") != _MODEL_ID:
        raise ValueError
    if official_metadata.get("resolved_revision") != _REVISION:
        raise ValueError

    source_files_value = root.get("source_files")
    if not isinstance(source_files_value, list):
        raise TypeError
    source_files = cast(list[object], source_files_value)
    if len(source_files) != len(_FILE_SPECS):
        raise ValueError

    files: list[SnapshotFilePlan] = []
    for raw_entry, spec in zip(source_files, _FILE_SPECS, strict=True):
        entry = _as_object_mapping(raw_entry)
        expected_path, expected_digest_type, expected_storage_type, expected_runtime = (
            spec
        )
        if not _is_safe_single_filename(entry.get("path"), expected_path):
            raise ValueError

        size_bytes = entry.get("size_bytes")
        if not _is_strict_int(size_bytes) or size_bytes <= 0:
            raise ValueError

        digest = entry.get("digest")
        if not isinstance(digest, str):
            raise TypeError
        expected_digest_length = 40 if expected_digest_type == "git_blob_sha1" else 64
        if re.fullmatch(rf"[0-9a-f]{{{expected_digest_length}}}", digest) is None:
            raise ValueError

        if entry.get("digest_type") != expected_digest_type:
            raise ValueError
        if entry.get("storage_type") != expected_storage_type:
            raise ValueError
        runtime_value = entry.get("required_for_runtime")
        if type(runtime_value) is not bool or runtime_value is not expected_runtime:
            raise ValueError

        files.append(
            SnapshotFilePlan(
                path=expected_path,
                size_bytes=size_bytes,
                digest=digest,
                digest_type=expected_digest_type,
                storage_type=expected_storage_type,
                required_for_runtime=expected_runtime,
            )
        )

    total_size_bytes = sum(file_plan.size_bytes for file_plan in files)
    required_runtime_size_bytes = sum(
        file_plan.size_bytes for file_plan in files if file_plan.required_for_runtime
    )
    weight_size_bytes = next(
        file_plan.size_bytes
        for file_plan in files
        if file_plan.path == "model.safetensors"
    )
    _validate_summary_value(
        selected_model.get("pinned_source_files_size_bytes"),
        total_size_bytes,
    )
    _validate_summary_value(
        selected_model.get("required_runtime_files_size_bytes"),
        required_runtime_size_bytes,
    )
    _validate_summary_value(
        selected_model.get("weight_size_bytes"),
        weight_size_bytes,
    )

    return SnapshotPlan(
        model_id=_MODEL_ID,
        revision=_REVISION,
        files=tuple(files),
        total_size_bytes=total_size_bytes,
        required_runtime_size_bytes=required_runtime_size_bytes,
        weight_size_bytes=weight_size_bytes,
    )


def load_snapshot_plan(selection_path: Path | str) -> SnapshotPlan:
    """Load and fully validate the closed M2-T02 selection artifact."""

    try:
        path = Path(selection_path)
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
        decoded_payload = raw_payload.decode("utf-8")
        payload: object = json.loads(decoded_payload)
        return _parse_snapshot_plan(payload)
    except SnapshotPreparationError:
        raise
    except Exception:  # noqa: BLE001
        raise _integrity_error() from None


def _stream_digest(
    path: Path,
    digest_type: DigestType,
    expected_size: int,
) -> str:
    try:
        metadata = path.lstat()
        if _is_link_or_junction(path, metadata) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError
        if metadata.st_size != expected_size:
            raise ValueError

        if digest_type == "git_blob_sha1":
            hasher = hashlib.sha1(usedforsecurity=False)
            hasher.update(f"blob {metadata.st_size}\0".encode("ascii"))
        else:
            hasher = hashlib.sha256()

        bytes_read = 0
        with path.open("rb") as handle:
            opened_metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened_metadata.st_mode):
                raise ValueError
            if (opened_metadata.st_dev, opened_metadata.st_ino) != (
                metadata.st_dev,
                metadata.st_ino,
            ):
                raise ValueError
            if opened_metadata.st_size != expected_size:
                raise ValueError
            while True:
                chunk = handle.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_read += len(chunk)
                hasher.update(chunk)
            final_metadata = os.fstat(handle.fileno())
            if (final_metadata.st_dev, final_metadata.st_ino) != (
                metadata.st_dev,
                metadata.st_ino,
            ):
                raise ValueError
            if final_metadata.st_size != expected_size:
                raise ValueError
        if bytes_read != metadata.st_size:
            raise ValueError
        return hasher.hexdigest()
    except SnapshotPreparationError:
        raise
    except Exception:  # noqa: BLE001
        raise _integrity_error() from None


def _verify_snapshot(snapshot_dir: Path, plan: SnapshotPlan) -> None:
    try:
        directory_metadata = snapshot_dir.lstat()
        if _is_link_or_junction(snapshot_dir, directory_metadata):
            raise ValueError
        if not stat.S_ISDIR(directory_metadata.st_mode):
            raise ValueError

        with os.scandir(snapshot_dir) as iterator:
            entries = list(iterator)
        expected_names = {file_plan.path for file_plan in plan.files}
        if len(entries) != len(plan.files):
            raise ValueError
        if {entry.name for entry in entries} != expected_names:
            raise ValueError

        for file_plan in plan.files:
            file_path = snapshot_dir / file_plan.path
            file_metadata = file_path.lstat()
            if _is_link_or_junction(file_path, file_metadata):
                raise ValueError
            if not stat.S_ISREG(file_metadata.st_mode):
                raise ValueError
            if file_metadata.st_size != file_plan.size_bytes:
                raise ValueError
            if (
                _stream_digest(
                    file_path,
                    file_plan.digest_type,
                    file_plan.size_bytes,
                )
                != file_plan.digest
            ):
                raise ValueError
    except SnapshotPreparationError:
        raise
    except Exception:  # noqa: BLE001
        raise _integrity_error() from None


def _contains_unresolved_parent(path: Path) -> bool:
    return any(part == ".." for part in re.split(r"[\\/]+", os.fspath(path)))


def _find_current_repository_root() -> Path | None:
    current = Path.cwd()
    for candidate in (current, *current.parents):
        try:
            (candidate / ".git").lstat()
        except FileNotFoundError:
            continue
        except OSError:
            continue
        return candidate.resolve(strict=False)
    return None


def _inspect_snapshot_target(
    requested_snapshot_dir: Path,
    selection_path: Path,
) -> tuple[bool, Path]:
    try:
        if _contains_unresolved_parent(requested_snapshot_dir):
            raise ValueError
        if requested_snapshot_dir.name in {"", ".", ".."}:
            raise ValueError

        resolved_target = requested_snapshot_dir.resolve(strict=False)
        resolved_selection = selection_path.resolve(strict=False)
        if resolved_target == resolved_selection:
            raise ValueError

        repository_roots = {_REPOSITORY_ROOT.resolve(strict=False)}
        current_repository_root = _find_current_repository_root()
        if current_repository_root is not None:
            repository_roots.add(current_repository_root)
        if resolved_target in repository_roots:
            raise ValueError

        absolute_target = Path(os.path.abspath(requested_snapshot_dir))
        nearest_existing_parent: Path | None = None
        for parent in reversed(absolute_target.parents):
            try:
                parent_metadata = parent.lstat()
            except FileNotFoundError:
                continue
            if _is_link_or_junction(parent, parent_metadata):
                raise ValueError
            if not stat.S_ISDIR(parent_metadata.st_mode):
                raise ValueError
            nearest_existing_parent = parent
        if nearest_existing_parent is None:
            raise ValueError

        try:
            target_metadata = requested_snapshot_dir.lstat()
        except FileNotFoundError:
            target_metadata = None
        if target_metadata is not None:
            if _is_link_or_junction(requested_snapshot_dir, target_metadata):
                raise ValueError
            if not stat.S_ISDIR(target_metadata.st_mode):
                raise ValueError
            return True, requested_snapshot_dir
        return False, nearest_existing_parent
    except SnapshotPreparationError:
        raise
    except Exception:  # noqa: BLE001
        raise _integrity_error() from None


def _check_disk_space(
    probe_path: Path,
    required_free_bytes: int,
    disk_usage: Callable[[Path], object],
) -> None:
    try:
        usage = cast(_DiskUsageResult, disk_usage(probe_path))
        free = usage.free
    except Exception:  # noqa: BLE001
        raise _download_error() from None
    if not _is_strict_int(free) or free < 0:
        raise _download_error() from None
    if free < required_free_bytes:
        raise SnapshotPreparationError("INSUFFICIENT_DISK_SPACE")


def _remove_huggingface_metadata(
    staging_dir: Path,
    staging_identity: _PathIdentity,
) -> None:
    try:
        _require_directory_identity(staging_dir, staging_identity)
        cache_dir = staging_dir / ".cache"
        try:
            cache_metadata = cache_dir.lstat()
        except FileNotFoundError:
            return
        if _is_link_or_junction(cache_dir, cache_metadata):
            raise ValueError
        if not stat.S_ISDIR(cache_metadata.st_mode):
            return

        huggingface_dir = cache_dir / "huggingface"
        try:
            huggingface_metadata = huggingface_dir.lstat()
        except FileNotFoundError:
            return
        if _is_link_or_junction(huggingface_dir, huggingface_metadata):
            raise ValueError
        if not stat.S_ISDIR(huggingface_metadata.st_mode):
            raise ValueError

        staging_resolved = staging_dir.resolve(strict=True)
        metadata_resolved = huggingface_dir.resolve(strict=True)
        if not metadata_resolved.is_relative_to(staging_resolved):
            raise ValueError
        shutil.rmtree(huggingface_dir)

        with os.scandir(cache_dir) as iterator:
            cache_is_empty = next(iterator, None) is None
        if cache_is_empty:
            cache_dir.rmdir()
        _require_directory_identity(staging_dir, staging_identity)
    except SnapshotPreparationError:
        raise
    except Exception:  # noqa: BLE001
        raise _download_error() from None


def _cleanup_owned_path(
    path: Path,
    *,
    expected_identity: _PathIdentity,
    remove_replaced_link: bool = False,
) -> None:
    for _attempt in range(2):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        except Exception:  # noqa: BLE001, S112
            continue

        try:
            is_link = stat.S_ISLNK(metadata.st_mode)
            is_junction = path.is_junction()
            if is_link:
                if not remove_replaced_link:
                    return
                path.unlink()
            elif is_junction:
                if not remove_replaced_link:
                    return
                path.rmdir()
            elif stat.S_ISDIR(metadata.st_mode):
                if (metadata.st_dev, metadata.st_ino) != expected_identity:
                    return
                shutil.rmtree(path)
            else:
                return
        except Exception:  # noqa: BLE001, S112
            continue


def _validate_download_result(
    result: object,
    staging_dir: Path,
    staging_identity: _PathIdentity,
    parent_chain_state: _ParentChainState,
    file_plan: SnapshotFilePlan,
) -> None:
    try:
        _require_parent_chain_state(parent_chain_state)
        _require_directory_identity(staging_dir, staging_identity)
        if not _is_path_input(result):
            raise ValueError
        returned_path = Path(result).resolve(strict=False)
        expected_path = (staging_dir / file_plan.path).resolve(strict=False)
        if returned_path != expected_path:
            raise ValueError
    except SnapshotPreparationError:
        raise
    except Exception:  # noqa: BLE001
        raise _integrity_error() from None


def _destination_is_absent(destination: Path) -> bool:
    try:
        destination.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        raise _integrity_error() from None
    return False


def prepare_snapshot(
    selection_path: Path | str,
    snapshot_dir: Path | str,
    download_file: Callable[..., object],
    disk_usage: Callable[[Path], object],
) -> SnapshotPreparationResult:
    """Prepare, verify, and atomically publish the pinned reranker snapshot."""

    try:
        requested_snapshot_dir = Path(snapshot_dir)
        requested_selection_path = Path(selection_path)
        if _contains_unresolved_parent(requested_snapshot_dir):
            raise ValueError
        working_snapshot_dir = Path(os.path.abspath(requested_snapshot_dir))
        working_selection_path = Path(os.path.abspath(requested_selection_path))
    except Exception:  # noqa: BLE001
        raise _integrity_error() from None

    plan = load_snapshot_plan(working_selection_path)
    target_exists, disk_probe_path = _inspect_snapshot_target(
        working_snapshot_dir,
        working_selection_path,
    )
    parent_chain_before_probe = _capture_parent_chain_state(working_snapshot_dir)
    if target_exists:
        _verify_snapshot(working_snapshot_dir, plan)
        return SnapshotPreparationResult(
            status="REUSED",
            snapshot_dir=requested_snapshot_dir,
        )

    required_free_bytes = plan.total_size_bytes + DOWNLOAD_HEADROOM_BYTES
    _check_disk_space(disk_probe_path, required_free_bytes, disk_usage)
    _require_parent_chain_state(parent_chain_before_probe)

    target_exists_after_probe, _probe_path_after_probe = _inspect_snapshot_target(
        working_snapshot_dir,
        working_selection_path,
    )
    if target_exists_after_probe:
        raise _integrity_error() from None

    try:
        working_snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        raise _download_error() from None
    target_exists_after_mkdir, _probe_path_after_mkdir = _inspect_snapshot_target(
        working_snapshot_dir,
        working_selection_path,
    )
    if target_exists_after_mkdir:
        raise _integrity_error() from None
    parent_chain_after_mkdir = _capture_parent_chain_state(working_snapshot_dir)

    staging_dir: Path | None = None
    staging_identity: _PathIdentity | None = None
    staging_prefix = f".{working_snapshot_dir.name}.staging-"
    try:
        try:
            _require_parent_chain_state(parent_chain_after_mkdir)
            staging_value = tempfile.mkdtemp(
                prefix=staging_prefix,
                dir=working_snapshot_dir.parent,
            )
            staging_candidate = Path(staging_value)
            if not staging_candidate.name.startswith(staging_prefix):
                raise ValueError
            if staging_candidate.name == staging_prefix:
                raise ValueError
            if (
                staging_candidate.parent.resolve(strict=False)
                != working_snapshot_dir.parent.resolve(strict=False)
            ):
                raise ValueError
            staging_dir = staging_candidate
            staging_identity = _ordinary_directory_identity(staging_dir)
            _require_parent_chain_state(parent_chain_after_mkdir)
        except Exception:  # noqa: BLE001
            raise _download_error() from None

        if staging_identity is None:
            raise _download_error() from None
        for file_plan in _execution_file_order(plan.files):
            try:
                result = download_file(
                    repo_id=plan.model_id,
                    filename=file_plan.path,
                    revision=plan.revision,
                    repo_type="model",
                    token=False,
                    local_dir=staging_dir,
                    local_dir_use_symlinks=False,
                )
            except Exception as exc:  # noqa: BLE001
                raise _download_error(_download_failure_diagnostic(exc)) from None
            _validate_download_result(
                result,
                staging_dir,
                staging_identity,
                parent_chain_after_mkdir,
                file_plan,
            )

        _require_parent_chain_state(parent_chain_after_mkdir)
        _remove_huggingface_metadata(staging_dir, staging_identity)
        _require_parent_chain_state(parent_chain_after_mkdir)
        _require_directory_identity_for_integrity(staging_dir, staging_identity)
        _verify_snapshot(staging_dir, plan)
        _require_parent_chain_state(parent_chain_after_mkdir)
        _require_directory_identity_for_integrity(staging_dir, staging_identity)
        if not _destination_is_absent(working_snapshot_dir):
            raise _integrity_error() from None

        try:
            os.replace(staging_dir, working_snapshot_dir)
        except Exception:  # noqa: BLE001
            _cleanup_owned_path(
                working_snapshot_dir,
                expected_identity=staging_identity,
            )
            raise _download_error() from None
        staging_dir = None

        try:
            _require_parent_chain_state(parent_chain_after_mkdir)
            _require_directory_identity_for_integrity(
                working_snapshot_dir,
                staging_identity,
            )
            _verify_snapshot(working_snapshot_dir, plan)
            _require_parent_chain_state(parent_chain_after_mkdir)
            _require_directory_identity_for_integrity(
                working_snapshot_dir,
                staging_identity,
            )
        except Exception:  # noqa: BLE001
            _cleanup_owned_path(
                working_snapshot_dir,
                expected_identity=staging_identity,
            )
            raise _integrity_error() from None

        return SnapshotPreparationResult(
            status="PUBLISHED",
            snapshot_dir=requested_snapshot_dir,
        )
    finally:
        if staging_dir is not None and staging_identity is not None:
            _cleanup_owned_path(
                staging_dir,
                expected_identity=staging_identity,
                remove_replaced_link=True,
            )
