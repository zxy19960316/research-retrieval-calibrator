"""Red-first expectations for fake and lazy BGE-M3 provider boundaries."""

from __future__ import annotations

import pytest
from app.adapters.embedding import BgeM3DenseProvider, DeterministicFakeEmbeddingProvider

from app.models.embedding import EmbeddingModelDescriptor, EmbeddingTaskError


@pytest.fixture
def fake_descriptor() -> EmbeddingModelDescriptor:
    return EmbeddingModelDescriptor(provider_name="deterministic_fake", model_id="sha256-vector", model_revision="fake-v1", provider_library="stdlib", provider_library_version="3.12", embedding_mode="dense", input_format_version="m2-title-abstract-v1", normalized=True, dimension=16, cache_namespace="embedding:fake")


def test_fake_provider_is_deterministic_finite_and_local(fake_descriptor: EmbeddingModelDescriptor) -> None:
    provider = DeterministicFakeEmbeddingProvider(fake_descriptor)
    vectors = provider.embed(["title:\nSource-backed record", "title:\nTitle-only record"], batch_size=2)
    assert vectors == provider.embed(["title:\nSource-backed record", "title:\nTitle-only record"], batch_size=1)
    assert len(vectors) == 2 and all(len(vector) == 16 for vector in vectors)


def test_bge_rejects_unpinned_revision_without_substituting_fake() -> None:
    with pytest.raises(EmbeddingTaskError, match="MODEL_REVISION_UNPINNED"):
        BgeM3DenseProvider(model_id="BAAI/bge-m3", model_revision="main", cache_namespace="embedding:real")
