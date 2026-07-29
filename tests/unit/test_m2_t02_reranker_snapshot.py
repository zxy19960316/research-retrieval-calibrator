"""Offline red contracts for the future pinned reranker snapshot preparer."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SELECTION_ARTIFACT = (
    _REPO_ROOT / "evaluation/source-artifacts/m2-t02-reranker-selection.json"
)
_GITIGNORE = _REPO_ROOT / ".gitignore"
_MODULE_NAME = "scripts.prepare_m2_t02_reranker_snapshot"
_MODEL_ID = "BAAI/bge-reranker-v2-m3"
_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
_HEADROOM_BYTES = 1_073_741_824
_COMMITTED_MINIMUM_REQUIRED_BYTES = 3_367_001_161
_EXPECTED_FILES = (
    ("README.md", "git_blob_sha1", "git", False),
    ("config.json", "git_blob_sha1", "git", True),
    ("model.safetensors", "sha256", "lfs", True),
    ("sentencepiece.bpe.model", "sha256", "lfs", True),
    ("special_tokens_map.json", "git_blob_sha1", "git", True),
    ("tokenizer.json", "sha256", "lfs", True),
    ("tokenizer_config.json", "git_blob_sha1", "git", True),
)
_EXECUTION_STATE = {
    "weights_downloaded": False,
    "tokenizer_downloaded": False,
    "model_loaded": False,
    "inference_run": False,
    "real_scores_generated": False,
}
_INVALID_SELECTION_MUTATIONS = (
    "wrong-model-id",
    "main-revision",
    "latest-revision",
    "short-revision",
    "metadata-revision-mismatch",
    "wrong-decision-status",
    "weights-downloaded",
    "tokenizer-downloaded",
    "model-loaded",
    "inference-run",
    "real-scores-generated",
    "missing-source-file",
    "duplicate-source-path",
    "unknown-source-file",
    "empty-source-path",
    "absolute-source-path",
    "parent-traversal",
    "normalized-parent-traversal",
    "zero-size",
    "negative-size",
    "unknown-digest-type",
    "unknown-storage-type",
    "git-storage-with-sha256",
    "lfs-storage-with-git-blob-sha1",
    "non-bool-runtime-flag",
    "wrong-digest-length",
    "non-hex-digest",
    "wrong-pinned-size-total",
    "wrong-runtime-size-total",
    "wrong-weight-size",
)
def _snapshot_module() -> ModuleType:
    try:
        return importlib.import_module(_MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name != _MODULE_NAME:
            raise
        pytest.fail("prepare_m2_t02_reranker_snapshot has not been implemented")


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()


def _digest(payload: bytes, digest_type: str) -> str:
    if digest_type == "git_blob_sha1":
        return _git_blob_sha1(payload)
    if digest_type == "sha256":
        return hashlib.sha256(payload).hexdigest()
    raise AssertionError(f"unsupported fixture digest type: {digest_type}")


def _same_length_corruption(payload: bytes) -> bytes:
    assert payload
    corrupted = bytes([payload[0] ^ 0x01]) + payload[1:]
    assert len(corrupted) == len(payload)
    assert corrupted != payload
    return corrupted


def _small_selection() -> tuple[dict[str, Any], dict[str, bytes]]:
    artifact = json.loads(_SELECTION_ARTIFACT.read_text(encoding="utf-8"))
    payloads = {
        path: f"offline-payload::{path}".encode()
        for path, _digest_type, _storage_type, _required in _EXPECTED_FILES
    }
    for entry in artifact["source_files"]:
        payload = payloads[entry["path"]]
        entry["size_bytes"] = len(payload)
        entry["digest"] = _digest(payload, entry["digest_type"])
    selected = artifact["selected_model"]
    selected["pinned_source_files_size_bytes"] = sum(
        entry["size_bytes"] for entry in artifact["source_files"]
    )
    selected["required_runtime_files_size_bytes"] = sum(
        entry["size_bytes"]
        for entry in artifact["source_files"]
        if entry["required_for_runtime"]
    )
    selected["weight_size_bytes"] = next(
        entry["size_bytes"]
        for entry in artifact["source_files"]
        if entry["path"] == "model.safetensors"
    )
    return artifact, payloads


def _write_small_selection(
    tmp_path: Path, *, mutation: str | None = None
) -> tuple[Path, dict[str, bytes]]:
    artifact, payloads = _small_selection()
    if mutation is not None:
        _mutate_selection(artifact, mutation)
    path = tmp_path / f"selection-{mutation or 'valid'}.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path, payloads


def _source_entry(artifact: dict[str, Any], path: str) -> dict[str, Any]:
    return next(entry for entry in artifact["source_files"] if entry["path"] == path)


def _mutate_selection(artifact: dict[str, Any], mutation: str) -> None:
    selected = artifact["selected_model"]
    source_files = artifact["source_files"]
    if mutation == "wrong-model-id":
        selected["model_id"] = "wrong/model"
    elif mutation == "main-revision":
        selected["model_revision"] = "main"
    elif mutation == "latest-revision":
        selected["model_revision"] = "latest"
    elif mutation == "short-revision":
        selected["model_revision"] = _REVISION[:12]
    elif mutation == "metadata-revision-mismatch":
        selected["official_metadata_source"]["resolved_revision"] = "0" * 40
    elif mutation == "wrong-decision-status":
        artifact["decision_status"] = "downloaded"
    elif mutation in {
        "weights-downloaded",
        "tokenizer-downloaded",
        "model-loaded",
        "inference-run",
        "real-scores-generated",
    }:
        key = mutation.replace("-", "_")
        artifact["execution_state"][key] = True
    elif mutation == "missing-source-file":
        source_files.pop()
    elif mutation == "duplicate-source-path":
        source_files[-1]["path"] = source_files[0]["path"]
    elif mutation == "unknown-source-file":
        source_files[0]["path"] = "unknown.bin"
    elif mutation == "empty-source-path":
        source_files[0]["path"] = ""
    elif mutation == "absolute-source-path":
        source_files[0]["path"] = "C:/models/README.md"
    elif mutation == "parent-traversal":
        source_files[0]["path"] = "../README.md"
    elif mutation == "normalized-parent-traversal":
        source_files[0]["path"] = "nested/../README.md"
    elif mutation == "zero-size":
        source_files[0]["size_bytes"] = 0
    elif mutation == "negative-size":
        source_files[0]["size_bytes"] = -1
    elif mutation == "unknown-digest-type":
        source_files[0]["digest_type"] = "md5"
    elif mutation == "unknown-storage-type":
        source_files[0]["storage_type"] = "unknown"
    elif mutation == "git-storage-with-sha256":
        _source_entry(artifact, "tokenizer.json")["storage_type"] = "git"
    elif mutation == "lfs-storage-with-git-blob-sha1":
        source_files[0]["storage_type"] = "lfs"
    elif mutation == "non-bool-runtime-flag":
        source_files[0]["required_for_runtime"] = "false"
    elif mutation == "wrong-digest-length":
        source_files[0]["digest"] = "0" * 39
    elif mutation == "non-hex-digest":
        source_files[0]["digest"] = "g" * 40
    elif mutation == "wrong-pinned-size-total":
        selected["pinned_source_files_size_bytes"] += 1
    elif mutation == "wrong-runtime-size-total":
        selected["required_runtime_files_size_bytes"] += 1
    elif mutation == "wrong-weight-size":
        selected["weight_size_bytes"] += 1
    else:
        raise AssertionError(f"unknown selection mutation: {mutation}")


def _write_downloaded_file(
    kwargs: dict[str, object], payloads: dict[str, bytes]
) -> Path:
    filename = str(kwargs["filename"])
    destination = Path(str(kwargs["local_dir"])) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payloads[filename])
    return destination


def _materialize_snapshot(snapshot_dir: Path, payloads: dict[str, bytes]) -> None:
    snapshot_dir.mkdir()
    for relative_path, payload in payloads.items():
        path = snapshot_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _assert_no_staging(snapshot_dir: Path) -> None:
    if not snapshot_dir.parent.is_dir():
        return
    pattern = f".{snapshot_dir.name}.staging-*"
    assert not list(snapshot_dir.parent.glob(pattern))


def _tree_state(root: Path) -> dict[str, tuple[object, ...]]:
    if not root.exists() and not root.is_symlink():
        return {}
    if root.is_symlink():
        return {".": ("symlink", os.readlink(root))}
    if root.is_file():
        stat = root.stat()
        return {".": ("file", root.read_bytes(), stat.st_mtime_ns)}
    state: dict[str, tuple[object, ...]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            state[relative] = ("symlink", os.readlink(path))
        elif path.is_dir():
            state[relative] = ("dir",)
        else:
            stat = path.stat()
            state[relative] = ("file", path.read_bytes(), stat.st_mtime_ns)
    return state


def _assert_exact_snapshot(snapshot_dir: Path, payloads: dict[str, bytes]) -> None:
    assert snapshot_dir.is_dir()
    assert {
        path.relative_to(snapshot_dir).as_posix()
        for path in snapshot_dir.rglob("*")
        if path.is_file()
    } == set(payloads)
    assert not [path for path in snapshot_dir.rglob("*") if path.is_symlink()]
    assert not [path for path in snapshot_dir.rglob("*") if path.is_dir()]
    for relative_path, payload in payloads.items():
        assert (snapshot_dir / relative_path).read_bytes() == payload


def _ample_disk(_path: Path) -> SimpleNamespace:
    return SimpleNamespace(free=10_000_000_000)


def test_gitignore_protects_model_snapshots_and_safetensors() -> None:
    entries = {
        line.strip()
        for line in _GITIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "models/" in entries
    assert "*.safetensors" in entries


def test_committed_selection_artifact_is_still_selected_not_downloaded() -> None:
    artifact = json.loads(_SELECTION_ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["decision_status"] == "selected_not_downloaded"
    assert artifact["execution_state"] == _EXECUTION_STATE


def test_git_blob_sha1_is_not_plain_file_sha1() -> None:
    payload = b"reranker-snapshot\n"
    git_blob_digest = _git_blob_sha1(payload)
    assert git_blob_digest == "f62d0c3a99a7648eaced175f791324b04930b7a0"
    assert git_blob_digest != hashlib.sha1(payload).hexdigest()


def test_load_snapshot_plan_reads_the_exact_selected_model_and_seven_files() -> None:
    module = _snapshot_module()

    plan = module.load_snapshot_plan(_SELECTION_ARTIFACT)

    assert plan.model_id == _MODEL_ID
    assert plan.revision == _REVISION
    assert [
        (
            entry.path,
            entry.digest_type,
            entry.storage_type,
            entry.required_for_runtime,
        )
        for entry in plan.files
    ] == list(_EXPECTED_FILES)
    assert len(plan.files) == 7


@pytest.mark.parametrize("mutation", _INVALID_SELECTION_MUTATIONS)
def test_load_snapshot_plan_rejects_closed_selection_mutations(
    tmp_path: Path, mutation: str
) -> None:
    module = _snapshot_module()
    selection_path, _payloads = _write_small_selection(tmp_path, mutation=mutation)
    selection_before = selection_path.read_bytes()

    with pytest.raises(module.SnapshotPreparationError) as raised:
        module.load_snapshot_plan(selection_path)

    assert raised.value.code == "INTEGRITY_CHECK_FAILED"
    assert selection_path.read_bytes() == selection_before


def test_committed_plan_exposes_exact_download_headroom_and_minimum_space() -> None:
    module = _snapshot_module()

    plan = module.load_snapshot_plan(_SELECTION_ARTIFACT)

    assert module.DOWNLOAD_HEADROOM_BYTES == _HEADROOM_BYTES
    assert plan.total_size_bytes == 2_293_259_337
    assert plan.total_size_bytes + module.DOWNLOAD_HEADROOM_BYTES == (
        _COMMITTED_MINIMUM_REQUIRED_BYTES
    )


def test_disk_boundary_allows_download_when_free_equals_required(tmp_path: Path) -> None:
    module = _snapshot_module()
    selection_path, payloads = _write_small_selection(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    plan = module.load_snapshot_plan(selection_path)
    required = plan.total_size_bytes + module.DOWNLOAD_HEADROOM_BYTES
    calls: list[str] = []

    def downloader(**kwargs: object) -> str:
        calls.append(str(kwargs["filename"]))
        return str(_write_downloaded_file(kwargs, payloads))

    result = module.prepare_snapshot(
        selection_path=selection_path,
        snapshot_dir=snapshot_dir,
        download_file=downloader,
        disk_usage=lambda _path: SimpleNamespace(free=required),
    )

    assert result.status == "PUBLISHED"
    assert calls == [entry[0] for entry in _EXPECTED_FILES]
    _assert_exact_snapshot(snapshot_dir, payloads)
    _assert_no_staging(snapshot_dir)


def test_disk_boundary_rejects_one_byte_less_before_downloader(tmp_path: Path) -> None:
    module = _snapshot_module()
    selection_path, _payloads = _write_small_selection(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    plan = module.load_snapshot_plan(selection_path)
    required = plan.total_size_bytes + module.DOWNLOAD_HEADROOM_BYTES
    calls: list[dict[str, object]] = []

    def downloader(**kwargs: object) -> str:
        calls.append(kwargs)
        raise AssertionError("insufficient disk space must stop before download")

    with pytest.raises(module.SnapshotPreparationError) as raised:
        module.prepare_snapshot(
            selection_path=selection_path,
            snapshot_dir=snapshot_dir,
            download_file=downloader,
            disk_usage=lambda _path: SimpleNamespace(free=required - 1),
        )

    assert raised.value.code == "INSUFFICIENT_DISK_SPACE"
    assert calls == []
    assert not snapshot_dir.exists()
    _assert_no_staging(snapshot_dir)


def test_disk_usage_oserror_maps_to_download_failed_without_partial_state(
    tmp_path: Path,
) -> None:
    module = _snapshot_module()
    selection_path, _payloads = _write_small_selection(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    calls: list[dict[str, object]] = []

    def disk_usage(_path: Path) -> object:
        raise OSError("simulated disk probe failure")

    def downloader(**kwargs: object) -> str:
        calls.append(kwargs)
        raise AssertionError("disk probe failure must stop before download")

    with pytest.raises(module.SnapshotPreparationError) as raised:
        module.prepare_snapshot(
            selection_path=selection_path,
            snapshot_dir=snapshot_dir,
            download_file=downloader,
            disk_usage=disk_usage,
        )

    assert raised.value.code == "DOWNLOAD_FAILED"
    assert "simulated disk probe failure" not in str(raised.value)
    assert calls == []
    assert not snapshot_dir.exists()
    _assert_no_staging(snapshot_dir)


@pytest.mark.parametrize(
    "target_case",
    (
        "existing-file",
        "symlink",
        "selection-path",
        "repository-root",
        "unresolved-parent",
        "file-parent",
    ),
)
def test_unsafe_snapshot_targets_fail_before_downloader_without_modification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_case: str
) -> None:
    module = _snapshot_module()
    selection_path, _payloads = _write_small_selection(tmp_path)
    external_marker = tmp_path / "external-marker"
    external_marker.write_bytes(b"preserve-me")
    if target_case == "existing-file":
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.write_bytes(b"existing-file")
    elif target_case == "symlink":
        real_target = tmp_path / "real-target"
        real_target.mkdir()
        (real_target / "marker").write_bytes(b"preserve-real-target")
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.symlink_to(real_target, target_is_directory=True)
    elif target_case == "selection-path":
        snapshot_dir = selection_path
    elif target_case == "repository-root":
        snapshot_dir = tmp_path / "synthetic-repository"
        snapshot_dir.mkdir()
        (snapshot_dir / ".git").mkdir()
        (snapshot_dir / "marker").write_bytes(b"preserve-repository")
        monkeypatch.chdir(snapshot_dir)
    elif target_case == "unresolved-parent":
        base = tmp_path / "base"
        base.mkdir()
        snapshot_dir = base / ".." / "snapshot"
    elif target_case == "file-parent":
        parent = tmp_path / "parent-file"
        parent.write_bytes(b"preserve-parent")
        snapshot_dir = parent / "snapshot"
    else:
        raise AssertionError(target_case)
    selection_before = selection_path.read_bytes()
    marker_before = external_marker.read_bytes()
    target_before = _tree_state(snapshot_dir)
    calls: list[dict[str, object]] = []

    def downloader(**kwargs: object) -> str:
        calls.append(kwargs)
        raise AssertionError("unsafe target must stop before download")

    with pytest.raises(module.SnapshotPreparationError) as raised:
        module.prepare_snapshot(
            selection_path=selection_path,
            snapshot_dir=snapshot_dir,
            download_file=downloader,
            disk_usage=_ample_disk,
        )

    assert raised.value.code == "INTEGRITY_CHECK_FAILED"
    assert calls == []
    assert selection_path.read_bytes() == selection_before
    assert external_marker.read_bytes() == marker_before
    target_after = _tree_state(snapshot_dir)
    assert target_after == target_before
    _assert_no_staging(snapshot_dir)


def test_prepare_snapshot_uses_safe_parameters_and_one_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _snapshot_module()
    selection_path, payloads = _write_small_selection(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    download_calls: list[dict[str, object]] = []
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def downloader(**kwargs: object) -> str:
        download_calls.append(kwargs)
        assert not snapshot_dir.exists()
        return str(_write_downloaded_file(kwargs, payloads))

    def atomic_replace(source: object, target: object) -> None:
        source_path = Path(source)
        target_path = Path(target)
        replace_calls.append((source_path, target_path))
        assert source_path.parent == target_path.parent
        assert target_path == snapshot_dir
        assert not target_path.exists()
        assert re.fullmatch(
            rf"\.{re.escape(snapshot_dir.name)}\.staging-.+",
            source_path.name,
        )
        _assert_exact_snapshot(source_path, payloads)
        real_replace(source_path, target_path)

    monkeypatch.setattr(module.os, "replace", atomic_replace)

    result = module.prepare_snapshot(
        selection_path=selection_path,
        snapshot_dir=snapshot_dir,
        download_file=downloader,
        disk_usage=_ample_disk,
    )

    assert result.status == "PUBLISHED"
    assert result.snapshot_dir == snapshot_dir
    assert len(replace_calls) == 1
    assert [call["filename"] for call in download_calls] == [
        entry[0] for entry in _EXPECTED_FILES
    ]
    staging_dirs = {Path(str(call["local_dir"])) for call in download_calls}
    assert len(staging_dirs) == 1
    assert staging_dirs == {replace_calls[0][0]}
    for call in download_calls:
        assert set(call) == {
            "repo_id",
            "filename",
            "revision",
            "repo_type",
            "token",
            "local_dir",
            "local_dir_use_symlinks",
        }
        assert call["repo_id"] == _MODEL_ID
        assert call["revision"] == _REVISION
        assert call["repo_type"] == "model"
        assert call["token"] is False
        assert call["local_dir_use_symlinks"] is False
    _assert_exact_snapshot(snapshot_dir, payloads)
    _assert_no_staging(snapshot_dir)


@pytest.mark.parametrize(
    ("corrupt_path", "digest_type"),
    (("config.json", "git_blob_sha1"), ("tokenizer.json", "sha256")),
)
def test_prepare_snapshot_rejects_same_length_digest_mismatches(
    tmp_path: Path, corrupt_path: str, digest_type: str
) -> None:
    module = _snapshot_module()
    selection_path, payloads = _write_small_selection(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    staging_sentinel = tmp_path / ".snapshot.staging-do-not-touch"
    staging_sentinel.mkdir()
    (staging_sentinel / "marker").write_bytes(b"preserve-staging-sentinel")
    sentinel_before = _tree_state(staging_sentinel)
    artifact = json.loads(selection_path.read_text(encoding="utf-8"))
    expected_digest = _source_entry(artifact, corrupt_path)["digest"]

    def downloader(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        payload = payloads[filename]
        if filename == corrupt_path:
            corrupted = _same_length_corruption(payload)
            assert len(corrupted) == len(payload)
            assert _digest(corrupted, digest_type) != expected_digest
            payload = corrupted
        destination = Path(str(kwargs["local_dir"])) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return str(destination)

    with pytest.raises(module.SnapshotPreparationError) as raised:
        module.prepare_snapshot(
            selection_path=selection_path,
            snapshot_dir=snapshot_dir,
            download_file=downloader,
            disk_usage=_ample_disk,
        )

    assert raised.value.code == "INTEGRITY_CHECK_FAILED"
    assert not snapshot_dir.exists()
    assert _tree_state(staging_sentinel) == sentinel_before
    assert set(snapshot_dir.parent.glob(".snapshot.staging-*")) == {staging_sentinel}


def test_existing_complete_snapshot_is_reused_without_touching_disk_or_files(
    tmp_path: Path,
) -> None:
    module = _snapshot_module()
    selection_path, payloads = _write_small_selection(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    _materialize_snapshot(snapshot_dir, payloads)
    before = _tree_state(snapshot_dir)
    disk_calls: list[Path] = []
    download_calls: list[dict[str, object]] = []

    def disk_usage(path: Path) -> object:
        disk_calls.append(path)
        raise AssertionError("verified snapshot reuse must not probe disk")

    def downloader(**kwargs: object) -> str:
        download_calls.append(kwargs)
        raise AssertionError("verified snapshot reuse must not download")

    result = module.prepare_snapshot(
        selection_path=selection_path,
        snapshot_dir=snapshot_dir,
        download_file=downloader,
        disk_usage=disk_usage,
    )

    assert result.status == "REUSED"
    assert result.snapshot_dir == snapshot_dir
    assert disk_calls == []
    assert download_calls == []
    assert _tree_state(snapshot_dir) == before
    _assert_no_staging(snapshot_dir)


@pytest.mark.parametrize(
    "corruption",
    ("missing-file", "same-length-digest", "extra-file", "extra-directory", "symlink-file"),
)
def test_existing_corrupt_snapshot_fails_closed_without_repair_or_overwrite(
    tmp_path: Path, corruption: str
) -> None:
    module = _snapshot_module()
    selection_path, payloads = _write_small_selection(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    _materialize_snapshot(snapshot_dir, payloads)
    external = tmp_path / "external"
    external.write_bytes(b"external-preserve")
    if corruption == "missing-file":
        (snapshot_dir / "README.md").unlink()
    elif corruption == "same-length-digest":
        target = snapshot_dir / "config.json"
        target.write_bytes(_same_length_corruption(target.read_bytes()))
    elif corruption == "extra-file":
        (snapshot_dir / "unexpected.bin").write_bytes(b"unexpected")
    elif corruption == "extra-directory":
        (snapshot_dir / "unexpected-directory").mkdir()
    elif corruption == "symlink-file":
        target = snapshot_dir / "config.json"
        target.unlink()
        target.symlink_to(external)
    else:
        raise AssertionError(corruption)
    before = _tree_state(snapshot_dir)
    external_before = external.read_bytes()
    disk_calls: list[Path] = []
    download_calls: list[dict[str, object]] = []

    def disk_usage(path: Path) -> object:
        disk_calls.append(path)
        raise AssertionError("corrupt existing snapshots must fail before disk probe")

    def downloader(**kwargs: object) -> str:
        download_calls.append(kwargs)
        raise AssertionError("corrupt existing snapshots must not be repaired")

    with pytest.raises(module.SnapshotPreparationError) as raised:
        module.prepare_snapshot(
            selection_path=selection_path,
            snapshot_dir=snapshot_dir,
            download_file=downloader,
            disk_usage=disk_usage,
        )

    assert raised.value.code == "INTEGRITY_CHECK_FAILED"
    assert disk_calls == []
    assert download_calls == []
    assert _tree_state(snapshot_dir) == before
    assert external.read_bytes() == external_before
    _assert_no_staging(snapshot_dir)


@pytest.mark.parametrize(
    "returned_path_case",
    ("outside-file", "final-destination", "other-staging", "relative-traversal"),
)
def test_downloader_return_path_must_resolve_inside_current_staging(
    tmp_path: Path, returned_path_case: str
) -> None:
    module = _snapshot_module()
    selection_path, payloads = _write_small_selection(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    external = tmp_path / "outside.bin"
    external.write_bytes(b"outside-preserve")
    other_staging = tmp_path / ".other.staging-existing"
    other_staging.mkdir()
    other_file = other_staging / "README.md"
    other_file.write_bytes(b"other-preserve")
    external_before = external.read_bytes()
    other_before = _tree_state(other_staging)

    def downloader(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        _write_downloaded_file(kwargs, payloads)
        if returned_path_case == "outside-file":
            return str(external)
        if returned_path_case == "final-destination":
            return str(snapshot_dir / filename)
        if returned_path_case == "other-staging":
            return str(other_file)
        if returned_path_case == "relative-traversal":
            return "../outside.bin"
        raise AssertionError(returned_path_case)

    with pytest.raises(module.SnapshotPreparationError) as raised:
        module.prepare_snapshot(
            selection_path=selection_path,
            snapshot_dir=snapshot_dir,
            download_file=downloader,
            disk_usage=_ample_disk,
        )

    assert raised.value.code == "INTEGRITY_CHECK_FAILED"
    assert not snapshot_dir.exists()
    assert external.read_bytes() == external_before
    assert _tree_state(other_staging) == other_before
    _assert_no_staging(snapshot_dir)


@pytest.mark.parametrize(
    "staging_mutation",
    (
        "extra-file",
        "extra-directory",
        "huggingface-cache",
        "symlink-file",
        "nested-declared-file",
    ),
)
def test_staging_content_is_closed_before_publication(
    tmp_path: Path, staging_mutation: str
) -> None:
    module = _snapshot_module()
    selection_path, payloads = _write_small_selection(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    external = tmp_path / "symlink-target"
    external.write_bytes(payloads["README.md"])
    external_before = external.read_bytes()
    external_cache = tmp_path / "external-cache/huggingface"
    external_cache.mkdir(parents=True)
    (external_cache / "marker").write_bytes(b"preserve-external-cache")
    external_cache_before = _tree_state(external_cache.parent)

    def downloader(**kwargs: object) -> str:
        destination = _write_downloaded_file(kwargs, payloads)
        if str(kwargs["filename"]) == _EXPECTED_FILES[-1][0]:
            staging = Path(str(kwargs["local_dir"]))
            if staging_mutation == "extra-file":
                (staging / "unexpected.bin").write_bytes(b"unexpected")
            elif staging_mutation == "extra-directory":
                (staging / "unexpected-directory").mkdir()
            elif staging_mutation == "huggingface-cache":
                cache = staging / ".cache/huggingface"
                cache.mkdir(parents=True)
                (cache / "metadata.json").write_bytes(b"cache")
            elif staging_mutation == "symlink-file":
                declared = staging / "README.md"
                declared.unlink()
                declared.symlink_to(external)
            elif staging_mutation == "nested-declared-file":
                nested = staging / "nested"
                nested.mkdir()
                (staging / "README.md").replace(nested / "README.md")
            else:
                raise AssertionError(staging_mutation)
        return str(destination)

    if staging_mutation == "huggingface-cache":
        try:
            result = module.prepare_snapshot(
                selection_path=selection_path,
                snapshot_dir=snapshot_dir,
                download_file=downloader,
                disk_usage=_ample_disk,
            )
        except module.SnapshotPreparationError as exc:
            assert exc.code == "INTEGRITY_CHECK_FAILED"
            assert not snapshot_dir.exists()
        else:
            assert result.status == "PUBLISHED"
            _assert_exact_snapshot(snapshot_dir, payloads)
    else:
        with pytest.raises(module.SnapshotPreparationError) as raised:
            module.prepare_snapshot(
                selection_path=selection_path,
                snapshot_dir=snapshot_dir,
                download_file=downloader,
                disk_usage=_ample_disk,
            )
        assert raised.value.code == "INTEGRITY_CHECK_FAILED"
        assert not snapshot_dir.exists()
    assert external.read_bytes() == external_before
    assert _tree_state(external_cache.parent) == external_cache_before
    _assert_no_staging(snapshot_dir)


@pytest.mark.parametrize(
    ("failure_index", "write_before_raise"),
    ((0, False), (2, False), (6, False), (2, True)),
)
def test_download_failures_at_every_stage_clean_all_partial_state(
    tmp_path: Path, failure_index: int, write_before_raise: bool
) -> None:
    module = _snapshot_module()
    selection_path, payloads = _write_small_selection(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "marker").write_bytes(b"preserve-unrelated")
    unrelated_before = _tree_state(unrelated)
    staging_sentinel = tmp_path / ".snapshot.staging-do-not-touch"
    staging_sentinel.mkdir()
    (staging_sentinel / "marker").write_bytes(b"preserve-staging-sentinel")
    sentinel_before = _tree_state(staging_sentinel)
    attempts: list[str] = []
    partial_paths: list[Path] = []

    def downloader(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        attempts.append(filename)
        if len(attempts) - 1 == failure_index:
            if write_before_raise:
                partial_paths.append(_write_downloaded_file(kwargs, payloads))
            raise OSError(f"simulated download failure at {failure_index}")
        return str(_write_downloaded_file(kwargs, payloads))

    with pytest.raises(module.SnapshotPreparationError) as raised:
        module.prepare_snapshot(
            selection_path=selection_path,
            snapshot_dir=snapshot_dir,
            download_file=downloader,
            disk_usage=_ample_disk,
        )

    assert raised.value.code == "DOWNLOAD_FAILED"
    assert len(attempts) == failure_index + 1
    assert not snapshot_dir.exists()
    assert all(not path.exists() for path in partial_paths)
    assert _tree_state(unrelated) == unrelated_before
    assert _tree_state(staging_sentinel) == sentinel_before
    assert set(snapshot_dir.parent.glob(".snapshot.staging-*")) == {staging_sentinel}


def test_staging_creation_failure_is_mapped_and_leaves_no_partial_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _snapshot_module()
    selection_path, _payloads = _write_small_selection(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    staging_sentinel = tmp_path / ".snapshot.staging-do-not-touch"
    staging_sentinel.mkdir()
    (staging_sentinel / "marker").write_bytes(b"preserve-staging-sentinel")
    sentinel_before = _tree_state(staging_sentinel)
    calls: list[dict[str, object]] = []

    def fail_mkdtemp(*_args: object, **_kwargs: object) -> str:
        raise OSError("simulated staging mkdir failure")

    def downloader(**kwargs: object) -> str:
        calls.append(kwargs)
        raise AssertionError("staging creation failure must stop before download")

    monkeypatch.setattr(module.tempfile, "mkdtemp", fail_mkdtemp)

    with pytest.raises(module.SnapshotPreparationError) as raised:
        module.prepare_snapshot(
            selection_path=selection_path,
            snapshot_dir=snapshot_dir,
            download_file=downloader,
            disk_usage=_ample_disk,
        )

    assert raised.value.code == "DOWNLOAD_FAILED"
    assert calls == []
    assert not snapshot_dir.exists()
    assert _tree_state(staging_sentinel) == sentinel_before
    assert set(snapshot_dir.parent.glob(".snapshot.staging-*")) == {staging_sentinel}


def test_atomic_replace_failure_removes_downloaded_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _snapshot_module()
    selection_path, payloads = _write_small_selection(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    staging_sentinel = tmp_path / ".snapshot.staging-do-not-touch"
    staging_sentinel.mkdir()
    (staging_sentinel / "marker").write_bytes(b"preserve-staging-sentinel")
    sentinel_before = _tree_state(staging_sentinel)
    replace_calls: list[tuple[Path, Path]] = []
    downloaded_paths: list[Path] = []

    def downloader(**kwargs: object) -> str:
        downloaded = _write_downloaded_file(kwargs, payloads)
        downloaded_paths.append(downloaded)
        return str(downloaded)

    def fail_replace(source: object, target: object) -> None:
        replace_calls.append((Path(source), Path(target)))
        raise OSError("simulated atomic publication failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(module.SnapshotPreparationError) as raised:
        module.prepare_snapshot(
            selection_path=selection_path,
            snapshot_dir=snapshot_dir,
            download_file=downloader,
            disk_usage=_ample_disk,
        )

    assert raised.value.code == "DOWNLOAD_FAILED"
    assert len(replace_calls) == 1
    assert replace_calls[0][1] == snapshot_dir
    assert not snapshot_dir.exists()
    assert all(not path.exists() for path in downloaded_paths)
    assert _tree_state(staging_sentinel) == sentinel_before
    assert set(snapshot_dir.parent.glob(".snapshot.staging-*")) == {staging_sentinel}


def test_independent_failures_use_unique_sibling_staging_directories(
    tmp_path: Path,
) -> None:
    module = _snapshot_module()
    selection_path, _payloads = _write_small_selection(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "marker").write_bytes(b"preserve")
    unrelated_before = _tree_state(unrelated)
    staging_sentinel = tmp_path / ".snapshot.staging-do-not-touch"
    staging_sentinel.mkdir()
    (staging_sentinel / "marker").write_bytes(b"preserve-staging-sentinel")
    sentinel_before = _tree_state(staging_sentinel)
    staging_paths: list[Path] = []

    def downloader(**kwargs: object) -> str:
        staging_paths.append(Path(str(kwargs["local_dir"])))
        raise OSError("simulated independent failure")

    for _attempt in range(2):
        with pytest.raises(module.SnapshotPreparationError) as raised:
            module.prepare_snapshot(
                selection_path=selection_path,
                snapshot_dir=snapshot_dir,
                download_file=downloader,
                disk_usage=_ample_disk,
            )
        assert raised.value.code == "DOWNLOAD_FAILED"

    assert len(staging_paths) == 2
    assert staging_paths[0] != staging_paths[1]
    assert all(path.parent == snapshot_dir.parent for path in staging_paths)
    assert all(
        re.fullmatch(
            rf"\.{re.escape(snapshot_dir.name)}\.staging-.+",
            path.name,
        )
        for path in staging_paths
    )
    assert all(not path.exists() for path in staging_paths)
    assert _tree_state(unrelated) == unrelated_before
    assert _tree_state(staging_sentinel) == sentinel_before
    assert not snapshot_dir.exists()
    assert set(snapshot_dir.parent.glob(".snapshot.staging-*")) == {staging_sentinel}


def test_import_and_fake_preparation_do_not_load_model_runtime_packages(
    tmp_path: Path,
) -> None:
    _snapshot_module()
    selection_path, _payloads = _write_small_selection(tmp_path)
    snapshot_dir = tmp_path / "subprocess-snapshot"
    script = r"""
import pathlib
import sys
import types
from types import SimpleNamespace

class ForbiddenProvider:
    def __init__(self, *args, **kwargs):
        raise AssertionError("snapshot preparation must not construct BgeRerankerProvider")

fake_adapter = types.ModuleType("app.adapters.reranking")
fake_adapter.BgeRerankerProvider = ForbiddenProvider
sys.modules["app.adapters.reranking"] = fake_adapter

import scripts.prepare_m2_t02_reranker_snapshot as module

def downloader(**kwargs):
    filename = kwargs["filename"]
    payload = f"offline-payload::{filename}".encode()
    destination = pathlib.Path(kwargs["local_dir"]) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return str(destination)

module.prepare_snapshot(
    selection_path=pathlib.Path(sys.argv[1]),
    snapshot_dir=pathlib.Path(sys.argv[2]),
    download_file=downloader,
    disk_usage=lambda _path: SimpleNamespace(free=10_000_000_000),
)
forbidden = {
    "torch",
    "transformers",
    "huggingface_hub",
    "FlagEmbedding",
    "sentence_transformers",
}
loaded = {name.split(".")[0] for name in sys.modules}
raise SystemExit(bool(forbidden & loaded))
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(selection_path), str(snapshot_dir)],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    _assert_no_staging(snapshot_dir)
