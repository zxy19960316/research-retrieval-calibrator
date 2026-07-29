"""Replaceable local embedding providers for M2-T01."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from app.models.embedding import (
    BGE_M3_FLAGEMBEDDING_VERSION,
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    EmbeddingModelDescriptor,
    EmbeddingTaskError,
)

_FAKE_DESCRIPTOR = EmbeddingModelDescriptor(
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
_BGE_M3_DIMENSION = 1024
_RUNTIME_PACKAGES = {
    "flagembedding_version": "FlagEmbedding",
    "torch_version": "torch",
    "transformers_version": "transformers",
    "huggingface_hub_version": "huggingface-hub",
    "numpy_version": "numpy",
}
_ALLOWED_DEVICES = frozenset({"cpu", "cuda", "cuda:0", "mps"})


class EmbeddingProvider(Protocol):
    """The provider boundary used by later cache and batch orchestration."""

    @property
    def descriptor(self) -> EmbeddingModelDescriptor: ...

    def embed(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]: ...


class DeterministicFakeEmbeddingProvider:
    """A dependency-free test provider, never a substitute for a BGE run."""

    def __init__(self, descriptor: EmbeddingModelDescriptor = _FAKE_DESCRIPTOR) -> None:
        if descriptor != _FAKE_DESCRIPTOR:
            raise ValueError("Deterministic fake provider requires the fixed fake descriptor")
        self._descriptor = descriptor
        self._descriptor_manifest = json.dumps(
            descriptor.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @property
    def descriptor(self) -> EmbeddingModelDescriptor:
        return self._descriptor

    def embed(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        _validate_embed_request(texts, batch_size)
        return [self._vector_for(text) for text in texts]

    def _vector_for(self, text: str) -> list[float]:
        payload = text.encode("utf-8") + self._descriptor_manifest
        values: list[float] = []
        block_number = 0
        while len(values) < self._descriptor.dimension:
            block = hashlib.sha256(payload + block_number.to_bytes(8, "big")).digest()
            for offset in range(0, len(block), 4):
                values.append(int.from_bytes(block[offset : offset + 4], "big") / 2**32)
                if len(values) == self._descriptor.dimension:
                    break
            block_number += 1
        return values


class BgeM3DenseProvider:
    """Lazy optional BGE-M3 provider that exposes dense vectors only."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        cache_namespace: str,
        model_cache_dir: Path | str,
        device: str | None = None,
    ) -> None:
        if model_revision != BGE_M3_MODEL_REVISION:
            raise EmbeddingTaskError("MODEL_REVISION_UNPINNED")
        if model_id != BGE_M3_MODEL_ID:
            raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT")
        if not cache_namespace.strip() or cache_namespace == "embedding:fake":
            raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT")
        if device is not None and device not in _ALLOWED_DEVICES:
            raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT")
        try:
            runtime_versions = {field: version(package) for field, package in _RUNTIME_PACKAGES.items()}
        except PackageNotFoundError as error:
            raise EmbeddingTaskError("EMBEDDING_PROVIDER_UNAVAILABLE") from error
        if runtime_versions["flagembedding_version"] != BGE_M3_FLAGEMBEDDING_VERSION:
            raise EmbeddingTaskError("EMBEDDING_PROVIDER_UNAVAILABLE")
        self._model_id = model_id
        self._model_revision = model_revision
        self._device = device
        self._model_cache_dir = Path(model_cache_dir)
        self._runtime = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.x",
            **runtime_versions,
            "device_request": device,
            "use_fp16": False,
            "model_revision": model_revision,
        }
        self._descriptor = EmbeddingModelDescriptor(
            provider_name="bge_m3",
            model_id=model_id,
            model_revision=model_revision,
            provider_library="FlagEmbedding",
            provider_library_version=runtime_versions["flagembedding_version"],
            embedding_mode="dense",
            input_format_version="m2-title-abstract-v1",
            normalized=True,
            dimension=_BGE_M3_DIMENSION,
            cache_namespace=cache_namespace,
        )
        self._model: object | None = None
        self._torch: object | None = None

    @property
    def descriptor(self) -> EmbeddingModelDescriptor:
        return self._descriptor

    @property
    def runtime(self) -> dict[str, object]:
        """Return the environment facts that must accompany real vector output."""

        return dict(self._runtime)

    def embed(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        _validate_embed_request(texts, batch_size)
        if not texts:
            return []
        self._load_model_if_needed()
        assert self._model is not None
        assert self._torch is not None
        try:
            model: Any = self._model
            torch: Any = self._torch
            with torch.inference_mode():
                encoded = model.encode(
                    list(texts),
                    batch_size=batch_size,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
            if not isinstance(encoded, dict) or "dense_vecs" not in encoded:
                raise EmbeddingTaskError("EMBEDDING_PROVIDER_FAILED")
            dense_vectors = encoded["dense_vecs"]
            vectors = [[float(value) for value in vector] for vector in dense_vectors]
        except EmbeddingTaskError:
            raise
        except Exception as error:
            raise EmbeddingTaskError("EMBEDDING_PROVIDER_FAILED") from error
        if len(vectors) != len(texts):
            raise EmbeddingTaskError("EMBEDDING_COUNT_MISMATCH")
        if any(len(vector) != self._descriptor.dimension for vector in vectors):
            raise EmbeddingTaskError("EMBEDDING_DIMENSION_MISMATCH")
        if any(not all(math.isfinite(value) for value in vector) for vector in vectors):
            raise EmbeddingTaskError("EMBEDDING_NON_FINITE")
        return vectors

    def _load_model_if_needed(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-not-found]
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        except (ImportError, OSError) as error:
            raise EmbeddingTaskError("EMBEDDING_PROVIDER_UNAVAILABLE") from error
        try:
            snapshot_path = snapshot_download(
                repo_id=self._model_id,
                revision=self._model_revision,
                cache_dir=str(self._model_cache_dir),
            )
            model_kwargs: dict[str, object] = {
                "normalize_embeddings": True,
                "use_fp16": False,
                "cache_dir": str(self._model_cache_dir),
            }
            if self._device is not None:
                model_kwargs["devices"] = self._device
            self._model = BGEM3FlagModel(
                snapshot_path,
                **model_kwargs,
            )
            self._torch = torch
        except Exception as error:
            self._model = None
            self._torch = None
            raise EmbeddingTaskError("EMBEDDING_PROVIDER_UNAVAILABLE") from error


def _validate_embed_request(texts: Sequence[str], batch_size: int) -> None:
    if batch_size < 1 or any(not isinstance(text, str) for text in texts):
        raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT")
