"""Offline red contracts for the future pinned reranker snapshot preparer."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_SELECTION_ARTIFACT = Path("evaluation/source-artifacts/m2-t02-reranker-selection.json")
_MODEL_ID = "BAAI/bge-reranker-v2-m3"
_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
_EXPECTED_FILES = (
    ("README.md", "git_blob_sha1"),
    ("config.json", "git_blob_sha1"),
    ("model.safetensors", "sha256"),
    ("sentencepiece.bpe.model", "sha256"),
    ("special_tokens_map.json", "git_blob_sha1"),
    ("tokenizer.json", "sha256"),
    ("tokenizer_config.json", "git_blob_sha1"),
)


def _snapshot_module() -> ModuleType:
    try:
        return importlib.import_module("scripts.prepare_m2_t02_reranker_snapshot")
    except ModuleNotFoundError:
        pytest.fail("prepare_m2_t02_reranker_snapshot has not been implemented")


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def _small_selection_artifact(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    artifact = json.loads(_SELECTION_ARTIFACT.read_text(encoding="utf-8"))
    payloads = {
        path: f"offline-payload::{path}".encode()
        for path, _digest_type in _EXPECTED_FILES
    }
    for entry in artifact["source_files"]:
        payload = payloads[entry["path"]]
        entry["size_bytes"] = len(payload)
        entry["digest"] = (
            _git_blob_sha1(payload)
            if entry["digest_type"] == "git_blob_sha1"
            else hashlib.sha256(payload).hexdigest()
        )
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
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path, payloads


def _assert_no_staging(snapshot_dir: Path) -> None:
    assert not list(snapshot_dir.parent.glob(f".{snapshot_dir.name}.staging-*"))


def test_load_snapshot_plan_reads_the_exact_selected_model_and_seven_files() -> None:
    module = _snapshot_module()

    plan = module.load_snapshot_plan(_SELECTION_ARTIFACT)

    assert plan.model_id == _MODEL_ID
    assert plan.revision == _REVISION
    assert [(entry.path, entry.digest_type) for entry in plan.files] == _EXPECTED_FILES
    assert len(plan.files) == 7


def test_prepare_snapshot_fails_disk_preflight_before_a_download(tmp_path: Path) -> None:
    module = _snapshot_module()
    selection_path, _payloads = _small_selection_artifact(tmp_path)
    calls: list[dict[str, object]] = []

    def downloader(**kwargs: object) -> str:
        calls.append(kwargs)
        raise AssertionError("disk preflight must run before a download")

    with pytest.raises(module.SnapshotPreparationError) as raised:
        module.prepare_snapshot(
            selection_path=selection_path,
            snapshot_dir=tmp_path / "snapshot",
            download_file=downloader,
            disk_usage=lambda _path: SimpleNamespace(free=0),
        )

    assert raised.value.code == "INSUFFICIENT_DISK_SPACE"
    assert calls == []
    assert not (tmp_path / "snapshot").exists()
    _assert_no_staging(tmp_path / "snapshot")


def test_prepare_snapshot_uses_pinned_unauthenticated_hugging_face_parameters(
    tmp_path: Path,
) -> None:
    module = _snapshot_module()
    selection_path, payloads = _small_selection_artifact(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    calls: list[dict[str, object]] = []

    def downloader(**kwargs: object) -> str:
        calls.append(kwargs)
        assert not snapshot_dir.exists(), "the final directory must stay unpublished"
        destination = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[str(kwargs["filename"])])
        return str(destination)

    result = module.prepare_snapshot(
        selection_path=selection_path,
        snapshot_dir=snapshot_dir,
        download_file=downloader,
        disk_usage=lambda _path: SimpleNamespace(free=10_000_000),
    )

    assert result.status == "PUBLISHED"
    assert result.snapshot_dir == snapshot_dir
    assert [call["filename"] for call in calls] == [path for path, _kind in _EXPECTED_FILES]
    for call in calls:
        assert call["repo_id"] == _MODEL_ID
        assert call["revision"] == _REVISION
        assert call["repo_type"] == "model"
        assert call["token"] is False
        assert call["local_dir_use_symlinks"] is False
        assert Path(str(call["local_dir"])) != snapshot_dir
    assert {path.relative_to(snapshot_dir).as_posix() for path in snapshot_dir.rglob("*") if path.is_file()} == {
        path for path, _kind in _EXPECTED_FILES
    }
    _assert_no_staging(snapshot_dir)


@pytest.mark.parametrize("corrupt_path", ["config.json", "tokenizer.json"])
def test_prepare_snapshot_rejects_git_blob_and_sha256_digest_mismatches(
    tmp_path: Path, corrupt_path: str
) -> None:
    module = _snapshot_module()
    selection_path, payloads = _small_selection_artifact(tmp_path)
    snapshot_dir = tmp_path / "snapshot"

    def downloader(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        destination = Path(str(kwargs["local_dir"])) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"corrupted" if filename == corrupt_path else payloads[filename])
        return str(destination)

    with pytest.raises(module.SnapshotPreparationError) as raised:
        module.prepare_snapshot(
            selection_path=selection_path,
            snapshot_dir=snapshot_dir,
            download_file=downloader,
            disk_usage=lambda _path: SimpleNamespace(free=10_000_000),
        )

    assert raised.value.code == "INTEGRITY_CHECK_FAILED"
    assert not snapshot_dir.exists()
    _assert_no_staging(snapshot_dir)


def test_prepare_snapshot_cleans_staging_and_never_publishes_a_partial_download(
    tmp_path: Path,
) -> None:
    module = _snapshot_module()
    selection_path, payloads = _small_selection_artifact(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    attempted: list[str] = []

    def downloader(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        attempted.append(filename)
        if filename == "model.safetensors":
            raise OSError("simulated interrupted download")
        destination = Path(str(kwargs["local_dir"])) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[filename])
        return str(destination)

    with pytest.raises(module.SnapshotPreparationError) as raised:
        module.prepare_snapshot(
            selection_path=selection_path,
            snapshot_dir=snapshot_dir,
            download_file=downloader,
            disk_usage=lambda _path: SimpleNamespace(free=10_000_000),
        )

    assert raised.value.code == "DOWNLOAD_FAILED"
    assert attempted == ["README.md", "config.json", "model.safetensors"]
    assert not snapshot_dir.exists()
    _assert_no_staging(snapshot_dir)


def test_prepare_snapshot_atomically_publishes_then_reuses_a_verified_snapshot(
    tmp_path: Path,
) -> None:
    module = _snapshot_module()
    selection_path, payloads = _small_selection_artifact(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    first_calls: list[str] = []

    def first_downloader(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        first_calls.append(filename)
        destination = Path(str(kwargs["local_dir"])) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[filename])
        return str(destination)

    first = module.prepare_snapshot(
        selection_path=selection_path,
        snapshot_dir=snapshot_dir,
        download_file=first_downloader,
        disk_usage=lambda _path: SimpleNamespace(free=10_000_000),
    )
    retained_bytes = {path: (snapshot_dir / path).read_bytes() for path in payloads}

    def reused_downloader(**_kwargs: object) -> str:
        raise AssertionError("a verified final snapshot must be reused without downloading")

    reused = module.prepare_snapshot(
        selection_path=selection_path,
        snapshot_dir=snapshot_dir,
        download_file=reused_downloader,
        disk_usage=lambda _path: SimpleNamespace(free=0),
    )

    assert first.status == "PUBLISHED"
    assert first_calls == [path for path, _kind in _EXPECTED_FILES]
    assert reused.status == "REUSED"
    assert reused.snapshot_dir == snapshot_dir
    assert {path: (snapshot_dir / path).read_bytes() for path in payloads} == retained_bytes
    _assert_no_staging(snapshot_dir)
