"""Closed contracts for M2-T01 frozen embedding inputs and vectors."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models.dedup import SourceIdentity
from app.models.embedding import (
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    EmbeddingCacheEntry,
    EmbeddingInput,
    EmbeddingModelDescriptor,
    EmbeddingTaskError,
    EmbeddingVectorRecord,
    FrozenCandidate,
    FrozenCandidateSnapshot,
)
from scripts.freeze_m2_candidates import (
    FROZEN_INPUT_MISSING,
    _canonical_json_sha256,
    _render_json_bytes,
    freeze_candidates,
    validate_frozen_snapshot_bytes,
)


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
def source_backed_candidates() -> list[FrozenCandidate]:
    return [
        FrozenCandidate(
            paper_id="arxiv:2401.00001",
            source="arxiv",
            source_id="2401.00001",
            title="Retrieval Calibration for Radiation Shielding",
            abstract="A source-backed study of retrieval calibration.",
            authors=["Ada Author"],
            year=2024,
            doi=None,
            url="https://arxiv.org/abs/2401.00001",
            language="en",
            categories=["cs.IR"],
            retrieval_paths=["Q1"],
            cluster_id="cluster:2401.00001",
            member_source_identities=[
                SourceIdentity(
                    source="arxiv",
                    source_id="2401.00001",
                    url="https://arxiv.org/abs/2401.00001",
                )
            ],
        ),
        FrozenCandidate(
            paper_id="arxiv:2401.00002",
            source="arxiv",
            source_id="2401.00002",
            title="A Title-Only Source-Backed Candidate",
            abstract=None,
            authors=["Bea Author"],
            year=2024,
            doi="10.1000/example.2",
            url="https://arxiv.org/abs/2401.00002",
            language="en",
            categories=[],
            retrieval_paths=["Q2"],
            cluster_id="cluster:2401.00002",
            member_source_identities=[
                SourceIdentity(
                    source="arxiv",
                    source_id="2401.00002",
                    url="https://arxiv.org/abs/2401.00002",
                )
            ],
        ),
    ]


def test_descriptor_is_frozen_closed_and_uses_the_test_fake(
    fake_descriptor: EmbeddingModelDescriptor,
) -> None:
    assert fake_descriptor.model_dump()["dimension"] == 16
    with pytest.raises(ValidationError):
        EmbeddingModelDescriptor.model_validate(
            {**fake_descriptor.model_dump(), "unexpected": True}
        )
    with pytest.raises(ValidationError):
        fake_descriptor.dimension = 32  # type: ignore[misc]


def test_frozen_candidate_rejects_placeholder_abstracts_and_non_http_urls(
    source_backed_candidates: list[FrozenCandidate],
) -> None:
    payload = source_backed_candidates[0].model_dump()
    payload["abstract"] = "  No Abstract Available  "
    with pytest.raises(ValidationError, match="placeholder"):
        FrozenCandidate.model_validate(payload)

    payload = source_backed_candidates[0].model_dump()
    payload["url"] = "ftp://arxiv.org/abs/2401.00001"
    with pytest.raises(ValidationError, match="HTTP\\(S\\)"):
        FrozenCandidate.model_validate(payload)


def test_snapshot_rejects_duplicate_paper_and_source_identities(
    source_backed_candidates: list[FrozenCandidate],
) -> None:
    duplicate_paper = source_backed_candidates[1].model_copy(
        update={"paper_id": source_backed_candidates[0].paper_id}
    )
    with pytest.raises(ValidationError, match="paper IDs"):
        FrozenCandidateSnapshot(question="Which retrieval methods calibrate results?", candidates=[source_backed_candidates[0], duplicate_paper])

    duplicate_source = source_backed_candidates[1].model_copy(
        update={"source_id": source_backed_candidates[0].source_id}
    )
    with pytest.raises(ValidationError, match="source identities"):
        FrozenCandidateSnapshot(question="Which retrieval methods calibrate results?", candidates=[source_backed_candidates[0], duplicate_source])


def test_real_descriptors_require_immutable_revisions_and_distinct_namespaces(
    fake_descriptor: EmbeddingModelDescriptor,
) -> None:
    real = {
        **fake_descriptor.model_dump(),
        "provider_name": "bge_m3",
        "model_id": BGE_M3_MODEL_ID,
        "model_revision": BGE_M3_MODEL_REVISION,
        "provider_library": "FlagEmbedding",
        "provider_library_version": "1.3.5",
        "dimension": 1024,
        "cache_namespace": "embedding:bge-m3",
    }
    for revision in ("", "main", "latest"):
        with pytest.raises(ValidationError, match="40-character"):
            EmbeddingModelDescriptor.model_validate({**real, "model_revision": revision})
    with pytest.raises(ValidationError, match="namespace"):
        EmbeddingModelDescriptor.model_validate({**real, "cache_namespace": "embedding:fake"})
    with pytest.raises(ValidationError, match="fake"):
        EmbeddingModelDescriptor.model_validate({**fake_descriptor.model_dump(), "cache_namespace": "embedding:real"})


def test_bge_m3_descriptor_requires_the_fixed_full_commit_identity(
    fake_descriptor: EmbeddingModelDescriptor,
) -> None:
    real = {
        **fake_descriptor.model_dump(),
        "provider_name": "bge_m3",
        "model_id": BGE_M3_MODEL_ID,
        "model_revision": BGE_M3_MODEL_REVISION,
        "provider_library": "FlagEmbedding",
        "provider_library_version": "1.3.5",
        "dimension": 1024,
        "cache_namespace": "embedding:bge-m3",
    }
    assert EmbeddingModelDescriptor.model_validate(real).model_revision == BGE_M3_MODEL_REVISION
    for invalid in ("main", "5617a9f", BGE_M3_MODEL_REVISION.upper(), "f" * 40):
        with pytest.raises(ValidationError, match="40-character|fixed BGE-M3"):
            EmbeddingModelDescriptor.model_validate({**real, "model_revision": invalid})
    with pytest.raises(ValidationError, match="provider library version"):
        EmbeddingModelDescriptor.model_validate({**real, "provider_library_version": "optional"})


def test_embedding_input_vector_and_cache_require_matching_finite_dimensions(
    fake_descriptor: EmbeddingModelDescriptor,
) -> None:
    text = "title:\nA source-backed paper"
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    input_record = EmbeddingInput(
        input_id="arxiv:2401.00001",
        input_kind="paper",
        paper_id="arxiv:2401.00001",
        query_id=None,
        input_format_version="m2-title-abstract-v1",
        text=text,
        text_sha256=text_sha256,
        source_snapshot_sha256="a" * 64,
    )
    with pytest.raises(ValidationError, match="exactly one"):
        EmbeddingInput.model_validate({**input_record.model_dump(), "query_id": "Q1"})

    vector = EmbeddingVectorRecord(
        input_id=input_record.input_id,
        text_sha256=text_sha256,
        descriptor=fake_descriptor,
        dimension=16,
        vector=[0.0] * 16,
    )
    with pytest.raises(ValidationError, match="dimension"):
        EmbeddingVectorRecord.model_validate({**vector.model_dump(), "dimension": 15})
    with pytest.raises(ValidationError, match="finite"):
        EmbeddingVectorRecord.model_validate({**vector.model_dump(), "vector": [float("nan")] * 16})
    with pytest.raises(ValidationError, match="dimension"):
        EmbeddingCacheEntry(
            cache_key="b" * 64,
            text_sha256=text_sha256,
            descriptor=fake_descriptor,
            dimension=16,
            vector=[0.0] * 15,
        )


def test_embedding_task_errors_have_only_the_declared_codes() -> None:
    assert EmbeddingTaskError("EMBEDDING_CACHE_CORRUPT").code == "EMBEDDING_CACHE_CORRUPT"
    with pytest.raises(ValueError, match="Unknown embedding task error code"):
        EmbeddingTaskError("NOT_A_DECLARED_CODE")


def test_freeze_gate_blocks_absent_accepted_m1_artifacts_without_creating_snapshots(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "snapshots"
    result = freeze_candidates(
        m1_output=tmp_path / "missing-first-round.json",
        m1_cache_dir=tmp_path / "missing-real-cache",
        output_dir=output_dir,
    )

    assert result.status == "blocked"
    assert result.error_code == FROZEN_INPUT_MISSING
    assert not output_dir.exists()


def test_frozen_snapshot_helper_accepts_a_synthetic_closed_33_candidate_snapshot(
    source_backed_candidates: list[FrozenCandidate],
) -> None:
    candidates: list[dict[str, object]] = []
    for index in range(33):
        source_id = f"2401.{index:05d}"
        source = source_backed_candidates[0].model_dump(mode="json")
        source.update(
            {
                "paper_id": f"arxiv:{source_id}",
                "source_id": source_id,
                "title": f"Source-backed candidate {index:02d}",
                "url": f"https://arxiv.org/abs/{source_id}",
                "cluster_id": f"cluster:{source_id}",
                "member_source_identities": [
                    {
                        "source": "arxiv",
                        "source_id": source_id,
                        "url": f"https://arxiv.org/abs/{source_id}",
                    }
                ],
            }
        )
        candidates.append(source)
    snapshot: dict[str, object] = {
        "candidates": candidates,
        "count": 33,
        "question": "How can graph-based retrieval support scientific literature discovery?",
        "snapshot_version": "m2-candidates.v1",
        "source_evidence_report": "evaluation/reports/m1-validation.json",
        "source_evidence_report_sha256": "a" * 64,
        "source_evidence_baseline_commit": "b" * 40,
        "source_candidate_array_sha256": "c" * 64,
        "source_merge_commit": "d" * 40,
        "source_output_sha256": "e" * 64,
        "validated_implementation_commit": "f" * 40,
    }
    manifest: dict[str, object] = {
        "candidate_count": 33,
        "implementation_ancestry": ["f" * 40],
        "m1_completion_merge_commit": "d" * 40,
        "m1_evidence_baseline_commit": "b" * 40,
        "metadata_mismatch_count": 0,
        "source_evidence_report": "evaluation/reports/m1-validation.json",
        "source_evidence_report_sha256": "a" * 64,
        "source_id_coverage": 1.0,
        "url_coverage": 1.0,
        "validated_implementation_commit": "f" * 40,
        "candidate_identity_sha256": _canonical_json_sha256(
            [(item["paper_id"], item["source"], item["source_id"]) for item in candidates]
        ),
        "source_identity_set_sha256": _canonical_json_sha256(
            sorted((item["source"], item["source_id"]) for item in candidates)
        ),
        "zero_transport_replay": {
            "cache_hits": 12,
            "empty_cache_entry_count": 0,
            "query_count": 12,
            "transport_requests": 0,
        },
        "artifact_source_classification": "EXACT_HISTORICAL_ARTIFACTS_RECOVERED",
        "source_bundle": None,
    }
    snapshot_bytes = _render_json_bytes(snapshot)
    manifest["snapshot_sha256"] = hashlib.sha256(snapshot_bytes).hexdigest()

    assert validate_frozen_snapshot_bytes(
        snapshot_bytes, manifest, expected_candidate_count=33
    ) is None

    snapshot["count"] = 32
    assert validate_frozen_snapshot_bytes(
        _render_json_bytes(snapshot), manifest, expected_candidate_count=33
    ) == (
        "frozen snapshot must contain exactly 33 candidates"
    )
