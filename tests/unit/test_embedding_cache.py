"""Red-first expectations for the M2 per-input embedding cache."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from app.adapters.embedding import DeterministicFakeEmbeddingProvider
from app.core.embedding import embed_inputs

from app.models.embedding import EmbeddingInput, EmbeddingModelDescriptor, EmbeddingTaskError


@pytest.fixture
def fake_descriptor() -> EmbeddingModelDescriptor:
    return EmbeddingModelDescriptor(provider_name="deterministic_fake", model_id="sha256-vector", model_revision="fake-v1", provider_library="stdlib", provider_library_version="3.12", embedding_mode="dense", input_format_version="m2-title-abstract-v1", normalized=True, dimension=16, cache_namespace="embedding:fake")


@pytest.fixture
def inputs() -> list[EmbeddingInput]:
    texts = ["title:\nFirst source-backed record", "title:\nSecond title-only record"]
    return [EmbeddingInput(input_id=f"arxiv:2401.0000{index + 1}", input_kind="paper", paper_id=f"arxiv:2401.0000{index + 1}", query_id=None, input_format_version="m2-title-abstract-v1", text=text, text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), source_snapshot_sha256="a" * 64) for index, text in enumerate(texts)]


@pytest.mark.parametrize("batch_size", [1, 2, 3, 10])
def test_cache_replay_is_order_stable_and_batch_size_independent(tmp_path: Path, fake_descriptor: EmbeddingModelDescriptor, inputs: list[EmbeddingInput], batch_size: int) -> None:
    provider = DeterministicFakeEmbeddingProvider(fake_descriptor)
    first, first_stats = embed_inputs(inputs, provider, tmp_path, batch_size=batch_size)
    replay, replay_stats = embed_inputs(inputs, provider, tmp_path, batch_size=batch_size)
    assert [record.input_id for record in first] == [item.input_id for item in inputs]
    assert [record.vector for record in replay] == [record.vector for record in first]
    assert first_stats.cache_misses == len(inputs)
    assert replay_stats.cache_hits == len(inputs) and replay_stats.provider_call_count == 0


def test_corrupt_json_is_recorded_as_a_miss(tmp_path: Path, fake_descriptor: EmbeddingModelDescriptor, inputs: list[EmbeddingInput]) -> None:
    provider = DeterministicFakeEmbeddingProvider(fake_descriptor)
    _, _ = embed_inputs(inputs, provider, tmp_path, batch_size=2)
    next(tmp_path.glob("*.json")).write_text("{not json", encoding="utf-8")
    _, stats = embed_inputs(inputs, provider, tmp_path, batch_size=2)
    assert stats.cache_corrupt_count == 1


def test_duplicate_input_id_is_a_stable_error(tmp_path: Path, fake_descriptor: EmbeddingModelDescriptor, inputs: list[EmbeddingInput]) -> None:
    provider = DeterministicFakeEmbeddingProvider(fake_descriptor)
    with pytest.raises(EmbeddingTaskError, match="DUPLICATE_EMBEDDING_ID"):
        embed_inputs([inputs[0], inputs[0]], provider, tmp_path, batch_size=1)
