"""Red-first expectations for fake and pinned BGE-M3 provider boundaries."""

from __future__ import annotations

import math
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import ModuleType

import pytest

from app.adapters import embedding as adapter
from app.adapters.embedding import (
    BGE_M3_REQUIRED_FILES,
    BgeM3DenseProvider,
    DeterministicFakeEmbeddingProvider,
)
from app.models.embedding import (
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    EmbeddingModelDescriptor,
    EmbeddingTaskError,
)
from scripts.preflight_bge_m3 import run_preflight


@pytest.fixture
def fake_descriptor() -> EmbeddingModelDescriptor:
    return EmbeddingModelDescriptor(
        provider_name="deterministic_fake",
        model_id="sha256-vector",
        model_revision="fake-v1",
        provider_library="stdlib",
        provider_library_version="3.12",
        embedding_mode="dense",
        input_format_version="m2-title-abstract-v1",
        normalized=True,
        dimension=16,
        cache_namespace="embedding:fake",
    )


@pytest.fixture
def runtime_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "FlagEmbedding": "1.3.5",
        "torch": "2.4.1",
        "transformers": "4.45.2",
        "huggingface-hub": "0.25.2",
        "numpy": "2.1.1",
    }
    monkeypatch.setattr(adapter, "version", versions.__getitem__)


def test_fake_provider_is_deterministic_finite_and_local(
    fake_descriptor: EmbeddingModelDescriptor,
) -> None:
    provider = DeterministicFakeEmbeddingProvider(fake_descriptor)
    vectors = provider.embed(["title:\nSource-backed record", "title:\nTitle-only record"], batch_size=2)
    assert vectors == provider.embed(["title:\nSource-backed record", "title:\nTitle-only record"], batch_size=1)
    assert len(vectors) == 2 and all(len(vector) == 16 for vector in vectors)


@pytest.mark.parametrize(
    "revision",
    ["main", "master", "latest", "head", "5617a9f", "f" * 39, "f" * 41, "F" * 40, "g" * 40, "v1.3.5"],
)
def test_bge_rejects_every_non_commit_revision_without_substituting_fake(revision: str) -> None:
    with pytest.raises(EmbeddingTaskError, match="MODEL_REVISION_UNPINNED"):
        BgeM3DenseProvider(
            model_id=BGE_M3_MODEL_ID,
            model_revision=revision,
            cache_namespace="embedding:bge-m3",
            model_cache_dir=".runtime/bge-m3",
        )


def test_bge_reads_all_runtime_versions_from_metadata(
    tmp_path: Path, runtime_versions: None
) -> None:
    provider = BgeM3DenseProvider(
        model_id=BGE_M3_MODEL_ID,
        model_revision=BGE_M3_MODEL_REVISION,
        cache_namespace="embedding:bge-m3",
        model_cache_dir=tmp_path / "model-cache",
        device="cpu",
    )

    assert provider.descriptor.provider_library_version == "1.3.5"
    assert provider.runtime == {
        "python_version": "3.12.x",
        "flagembedding_version": "1.3.5",
        "torch_version": "2.4.1",
        "transformers_version": "4.45.2",
        "huggingface_hub_version": "0.25.2",
        "numpy_version": "2.1.1",
        "device_request": "cpu",
        "use_fp16": False,
        "model_revision": BGE_M3_MODEL_REVISION,
    }


def test_bge_missing_runtime_package_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(adapter, "version", missing)
    with pytest.raises(EmbeddingTaskError, match="EMBEDDING_PROVIDER_UNAVAILABLE"):
        BgeM3DenseProvider(
            model_id=BGE_M3_MODEL_ID,
            model_revision=BGE_M3_MODEL_REVISION,
            cache_namespace="embedding:bge-m3",
            model_cache_dir=tmp_path / "model-cache",
        )


def test_bge_rejects_a_present_but_unpinned_flagembedding_version(
    tmp_path: Path, runtime_versions: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    versions = {
        "FlagEmbedding": "1.3.4",
        "torch": "2.4.1",
        "transformers": "4.45.2",
        "huggingface-hub": "0.25.2",
        "numpy": "2.1.1",
    }
    monkeypatch.setattr(adapter, "version", versions.__getitem__)

    with pytest.raises(EmbeddingTaskError, match="EMBEDDING_PROVIDER_UNAVAILABLE"):
        BgeM3DenseProvider(
            model_id=BGE_M3_MODEL_ID,
            model_revision=BGE_M3_MODEL_REVISION,
            cache_namespace="embedding:bge-m3",
            model_cache_dir=tmp_path / "model-cache",
        )


@pytest.mark.parametrize(
    ("device", "expected_device_kwargs"),
    [("cpu", {"devices": "cpu"}), ("cuda:0", {"devices": "cuda:0"}), (None, {})],
)
def test_bge_uses_pinned_snapshot_and_supported_wrapper_arguments(
    tmp_path: Path,
    runtime_versions: None,
    monkeypatch: pytest.MonkeyPatch,
    device: str | None,
    expected_device_kwargs: dict[str, str],
) -> None:
    model_cache_dir = tmp_path / "model-cache"
    snapshot_calls: list[dict[str, object]] = []
    wrapper_calls: list[tuple[object, dict[str, object]]] = []

    class InferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_: object) -> None:
            return None

    torch = ModuleType("torch")
    torch.inference_mode = InferenceMode  # type: ignore[attr-defined]

    class Wrapper:
        def __init__(self, model_path: object, **kwargs: object) -> None:
            wrapper_calls.append((model_path, kwargs))

        def encode(self, texts: list[str], **kwargs: object) -> dict[str, list[list[float]]]:
            assert kwargs == {
                "batch_size": 8,
                "return_dense": True,
                "return_sparse": False,
                "return_colbert_vecs": False,
            }
            return {"dense_vecs": [[1.0 / math.sqrt(1024)] * 1024 for _ in texts]}

    flag_embedding = ModuleType("FlagEmbedding")
    flag_embedding.BGEM3FlagModel = Wrapper  # type: ignore[attr-defined]
    hub = ModuleType("huggingface_hub")
    snapshot_path = model_cache_dir / "snapshots" / BGE_M3_MODEL_REVISION

    def snapshot_download(**kwargs: object) -> str:
        snapshot_calls.append(kwargs)
        for relative_path in BGE_M3_REQUIRED_FILES:
            required_file = snapshot_path / relative_path
            required_file.parent.mkdir(parents=True, exist_ok=True)
            required_file.write_bytes(b"required model file")
        return str(snapshot_path)

    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "FlagEmbedding", flag_embedding)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    provider = BgeM3DenseProvider(
        model_id=BGE_M3_MODEL_ID,
        model_revision=BGE_M3_MODEL_REVISION,
        cache_namespace="embedding:bge-m3",
        model_cache_dir=model_cache_dir,
        device=device,
    )
    vectors = provider.embed(["a"] * 34, batch_size=8)

    assert len(vectors) == 34 and all(len(vector) == 1024 for vector in vectors)
    assert snapshot_calls == [
        {
            "repo_id": BGE_M3_MODEL_ID,
            "revision": BGE_M3_MODEL_REVISION,
            "cache_dir": str(model_cache_dir),
            "allow_patterns": list(BGE_M3_REQUIRED_FILES),
            "max_workers": 1,
        }
    ]
    assert "onnx/model.onnx_data" not in BGE_M3_REQUIRED_FILES
    assert "long.jpg" not in BGE_M3_REQUIRED_FILES
    assert not any(path.startswith(("onnx/", "imgs/")) for path in BGE_M3_REQUIRED_FILES)
    assert wrapper_calls == [
        (
            str(model_cache_dir / "snapshots" / BGE_M3_MODEL_REVISION),
            {
                "normalize_embeddings": True,
                "use_fp16": False,
                "cache_dir": str(model_cache_dir),
                **expected_device_kwargs,
            },
        )
    ]


@pytest.mark.parametrize(
    ("required_path", "file_state"),
    [
        ("pytorch_model.bin", "missing"),
        ("pytorch_model.bin", "empty"),
        ("tokenizer.json", "missing"),
        ("config.json", "directory"),
    ],
)
def test_bge_refuses_incomplete_or_non_regular_required_files_before_constructing_wrapper(
    tmp_path: Path,
    runtime_versions: None,
    monkeypatch: pytest.MonkeyPatch,
    required_path: str,
    file_state: str,
) -> None:
    model_cache_dir = tmp_path / "model-cache"
    snapshot_path = model_cache_dir / "snapshots" / BGE_M3_MODEL_REVISION
    wrapper_calls: list[tuple[object, dict[str, object]]] = []

    torch = ModuleType("torch")
    flag_embedding = ModuleType("FlagEmbedding")
    hub = ModuleType("huggingface_hub")

    class Wrapper:
        def __init__(self, model_path: object, **kwargs: object) -> None:
            wrapper_calls.append((model_path, kwargs))

    def snapshot_download(**_kwargs: object) -> str:
        for relative_path in BGE_M3_REQUIRED_FILES:
            required_file = snapshot_path / relative_path
            required_file.parent.mkdir(parents=True, exist_ok=True)
            required_file.write_bytes(b"required model file")
        target = snapshot_path / required_path
        if file_state == "missing":
            target.unlink()
        elif file_state == "empty":
            target.write_bytes(b"")
        else:
            target.unlink()
            target.mkdir()
        return str(snapshot_path)

    flag_embedding.BGEM3FlagModel = Wrapper  # type: ignore[attr-defined]
    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "FlagEmbedding", flag_embedding)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    provider = BgeM3DenseProvider(
        model_id=BGE_M3_MODEL_ID,
        model_revision=BGE_M3_MODEL_REVISION,
        cache_namespace="embedding:bge-m3",
        model_cache_dir=model_cache_dir,
    )

    with pytest.raises(EmbeddingTaskError, match="EMBEDDING_PROVIDER_UNAVAILABLE"):
        provider._load_model_if_needed()
    assert wrapper_calls == []


@pytest.mark.parametrize(
    ("encoded", "error_code"),
    [
        ({}, "EMBEDDING_PROVIDER_FAILED"),
        ({"dense_vecs": [[0.0] * 1024] * 33}, "EMBEDDING_COUNT_MISMATCH"),
        ({"dense_vecs": [[0.0] * 1023]}, "EMBEDDING_DIMENSION_MISMATCH"),
        ({"dense_vecs": [[0.0] * 1025]}, "EMBEDDING_DIMENSION_MISMATCH"),
        ({"dense_vecs": [[float("nan")] * 1024]}, "EMBEDDING_NON_FINITE"),
    ],
)
def test_bge_rejects_non_dense_count_dimension_and_finiteness_failures(
    tmp_path: Path,
    runtime_versions: None,
    encoded: dict[str, list[list[float]]],
    error_code: str,
) -> None:
    provider = BgeM3DenseProvider(
        model_id=BGE_M3_MODEL_ID,
        model_revision=BGE_M3_MODEL_REVISION,
        cache_namespace="embedding:bge-m3",
        model_cache_dir=tmp_path / "model-cache",
    )
    provider._model = type("Wrapper", (), {"encode": lambda *_args, **_kwargs: encoded})()
    provider._torch = type("Torch", (), {"inference_mode": lambda self: _NoOpContext()})()

    with pytest.raises(EmbeddingTaskError, match=error_code):
        provider.embed(["input"], batch_size=1)


class _NoOpContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def test_preflight_contract_accepts_one_normalized_dense_vector_without_writing_artifacts() -> None:
    vector = [1.0 / math.sqrt(1024)] * 1024
    provider = type(
        "StubBgeProvider",
        (),
        {
            "descriptor": type("Descriptor", (), {"provider_library_version": "1.3.5"})(),
            "runtime": {
                "python_version": "3.12.x",
                "flagembedding_version": "1.3.5",
                "torch_version": "2.4.1",
                "transformers_version": "4.45.2",
                "huggingface_hub_version": "0.25.2",
                "numpy_version": "2.1.1",
                "device_request": "cpu",
                "use_fp16": False,
                "model_revision": BGE_M3_MODEL_REVISION,
            },
            "embed": lambda _self, _texts, *, batch_size: [vector],
        },
    )()

    assert run_preflight(provider, device="cpu")["status"] == "success"
