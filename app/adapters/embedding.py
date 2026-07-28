"""Replaceable local embedding providers for M2-T01."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any, Protocol

from app.models.embedding import EmbeddingModelDescriptor, EmbeddingTaskError

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
_FLOATING_REVISIONS = frozenset({"main", "master", "latest", "head"})


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
        device: str | None = None,
        provider_library_version: str = "optional",
    ) -> None:
        if not model_revision.strip() or model_revision.casefold() in _FLOATING_REVISIONS:
            raise EmbeddingTaskError("MODEL_REVISION_UNPINNED")
        if not model_id.strip() or not cache_namespace.strip() or not provider_library_version.strip():
            raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT")
        self._model_id = model_id
        self._model_revision = model_revision
        self._device = device
        self._descriptor = EmbeddingModelDescriptor(
            provider_name="bge_m3",
            model_id=model_id,
            model_revision=model_revision,
            provider_library="FlagEmbedding",
            provider_library_version=provider_library_version,
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
            eval_method = model.eval
            if callable(eval_method):
                eval_method()
            with torch.inference_mode():
                encoded = model.encode(
                    list(texts),
                    batch_size=batch_size,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
            dense_vectors = encoded["dense_vecs"]
            vectors = [[float(value) for value in vector] for vector in dense_vectors]
        except Exception as error:
            raise EmbeddingTaskError("EMBEDDING_PROVIDER_FAILED") from error
        if len(vectors) != len(texts) or any(
            len(vector) != self._descriptor.dimension
            or not all(math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise EmbeddingTaskError("EMBEDDING_PROVIDER_FAILED")
        return vectors

    def _load_model_if_needed(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-not-found]
        except (ImportError, OSError) as error:
            raise EmbeddingTaskError("EMBEDDING_PROVIDER_UNAVAILABLE") from error
        try:
            model_kwargs: dict[str, object] = {"use_fp16": False}
            if self._device is not None:
                model_kwargs["device"] = self._device
            self._model = BGEM3FlagModel(
                self._model_id,
                revision=self._model_revision,
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
