"""Red-first expectations for deterministic M2 embedding text construction."""

from __future__ import annotations

import hashlib

from app.core.embedding import build_embedding_text, build_query_embedding_input

from app.models.dedup import SourceIdentity
from app.models.embedding import EmbeddingModelDescriptor, FrozenCandidate


def _descriptor() -> EmbeddingModelDescriptor:
    return EmbeddingModelDescriptor(
        provider_name="deterministic_fake", model_id="sha256-vector", model_revision="fake-v1",
        provider_library="stdlib", provider_library_version="3.12", embedding_mode="dense",
        input_format_version="m2-title-abstract-v1", normalized=True, dimension=16,
        cache_namespace="embedding:fake",
    )


def _candidates() -> list[FrozenCandidate]:
    return [
        FrozenCandidate(paper_id="arxiv:2401.00001", source="arxiv", source_id="2401.00001", title="  Calibrated Retrieval\r\nMethods  ", abstract="  Preserved abstract\r\ntext.  ", authors=["Ada"], year=2024, doi=None, url="https://arxiv.org/abs/2401.00001", language="en", categories=["cs.IR"], retrieval_paths=["Q1"], cluster_id="cluster:1", member_source_identities=[SourceIdentity(source="arxiv", source_id="2401.00001", url="https://arxiv.org/abs/2401.00001")]),
        FrozenCandidate(paper_id="arxiv:2401.00002", source="arxiv", source_id="2401.00002", title="Title only", abstract=None, authors=["Bea"], year=2024, doi=None, url="https://arxiv.org/abs/2401.00002", language="en", categories=[], retrieval_paths=["Q2"], cluster_id="cluster:2", member_source_identities=[SourceIdentity(source="arxiv", source_id="2401.00002", url="https://arxiv.org/abs/2401.00002")]),
    ]


def test_title_abstract_protocol_preserves_internal_text_and_hashes_utf8() -> None:
    _descriptor()
    record = _candidates()[0]
    result = build_embedding_text(record, "a" * 64)
    assert result.text == "title:\nCalibrated Retrieval\nMethods\n\nabstract:\nPreserved abstract\ntext."
    assert result.text_sha256 == hashlib.sha256(result.text.encode("utf-8")).hexdigest()


def test_title_only_and_query_inputs_have_one_source_identity() -> None:
    _descriptor()
    candidate = build_embedding_text(_candidates()[1], "a" * 64)
    query = build_query_embedding_input("Which methods calibrate retrieval?", "a" * 64)
    assert candidate.text == "title:\nTitle only"
    assert candidate.paper_id == "arxiv:2401.00002" and candidate.query_id is None
    assert query.query_id is not None and query.paper_id is None
