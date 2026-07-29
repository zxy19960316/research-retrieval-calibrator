"""Red-first expectations for M2-T01's completed-evidence gate."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import validate_m2_t01_evidence as evidence
from scripts.validate_m2_t01_evidence import validate_m2_t01_evidence

M1_REPORT_PATH = "evaluation/reports/m1-rebaseline-2026-07-28.json"
M1_BUNDLE_PATH = "evaluation/source-artifacts/m1-rebaseline-2026-07-28"
CANDIDATE_ARRAY_SHA256 = "e51eb84d4e772bba324a199478caa5f0983edef0b0dbf4c58fce5f9da5803209"


def _json_bytes(payload: object, *, indent: int | None = 2) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True) + "\n").encode("utf-8")


def _candidate_snapshot() -> dict[str, object]:
    return {
        "snapshot_version": "m2-candidates.v1",
        "count": 33,
        "question": "Which verified question binds the query embedding?",
        "source_phase": "M1",
        "source_merge_commit": evidence.BASELINE_COMMIT,
        "source_evidence_report": M1_REPORT_PATH,
        "source_candidate_array_sha256": CANDIDATE_ARRAY_SHA256,
        "candidates": [
            {
                "paper_id": f"arxiv:{index:04d}.00001",
                "source": "arxiv",
                "source_id": f"{index:04d}.00001",
                "url": f"https://arxiv.org/abs/{index:04d}.00001",
                "title": f"Verified title {index}.",
                "abstract": "Verified abstract.",
            }
            for index in range(33)
        ],
    }


def _candidate_manifest(snapshot_sha256: str) -> dict[str, object]:
    return {
        "snapshot_sha256": snapshot_sha256,
        "candidate_count": 33,
        "source_id_coverage": 1.0,
        "url_coverage": 1.0,
        "metadata_mismatch_count": 0,
        "artifact_source_classification": "M1_REBASELINE_SOURCE_BUNDLE",
        "source_bundle": M1_BUNDLE_PATH,
        "zero_transport_replay": {
            "transport_requests": 0,
            "cache_hits": 12,
            "query_count": 12,
            "empty_cache_entry_count": 1,
        },
    }


def _candidate_validation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    snapshot: dict[str, object] | None = None,
    manifest: dict[str, object] | None = None,
    snapshot_sha256: str | None = None,
    snapshot_indent: int | None = 2,
) -> tuple[str | None, list[str]]:
    candidate_snapshot = deepcopy(snapshot or _candidate_snapshot())
    raw_snapshot = _json_bytes(candidate_snapshot, indent=snapshot_indent)
    exact_snapshot_sha256 = hashlib.sha256(raw_snapshot).hexdigest()
    candidate_manifest = deepcopy(manifest or _candidate_manifest(exact_snapshot_sha256))
    raw_manifest = _json_bytes(candidate_manifest)
    report_snapshot_sha256 = snapshot_sha256 or exact_snapshot_sha256
    payload = {
        "candidate_snapshot": {
            "path": "snapshots/candidates.json",
            "blob_sha256": exact_snapshot_sha256,
            "manifest_path": "snapshots/candidates.manifest.json",
            "manifest_blob_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "snapshot_sha256": report_snapshot_sha256,
        }
    }
    blobs = {
        "snapshots/candidates.json": raw_snapshot,
        "snapshots/candidates.manifest.json": raw_manifest,
    }
    monkeypatch.setattr(evidence, "_blob_bytes", lambda _commit, path, _root: blobs.get(path))
    monkeypatch.setattr(evidence, "B1_CANDIDATE_SNAPSHOT_SHA256", exact_snapshot_sha256)
    errors: list[str] = []
    result = evidence._validate_candidate_snapshot(payload, "a" * 40, errors, Path("."))
    return result, errors


def _real_provider() -> dict[str, object]:
    return {
        "provider_name": "bge_m3", "model_id": "BAAI/bge-m3",
        "model_revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "provider_library": "FlagEmbedding", "provider_library_version": "1.3.5",
        "embedding_mode": "dense", "normalized": True, "dimension": 1024,
        "cache_namespace": "embedding:bge-m3",
    }


def _real_runtime() -> dict[str, object]:
    return {
        "python_version": "3.12.x", "flagembedding_version": "1.3.5", "torch_version": "2.4.1",
        "transformers_version": "4.45.2", "huggingface_hub_version": "0.25.2", "numpy_version": "2.1.1",
        "device_request": "cpu", "use_fp16": False,
        "model_revision": "5617a9f61b028005a4858fdac845db406aefb181",
    }


def _vector_validation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dimension: int = 1024,
    vector_scale: float = 1.0,
    record_descriptor: dict[str, object] | None = None,
    manifest_provider: dict[str, object] | None = None,
    input_id_override: str | None = None,
    text_sha256_override: str | None = None,
) -> list[str]:
    candidate_sha256 = evidence.B1_CANDIDATE_SNAPSHOT_SHA256
    expected_ids = {*(f"candidate:{index}" for index in range(33)), "query:main"}
    context = evidence.CandidateEvidenceContext(
        snapshot_sha256=candidate_sha256,
        expected_input_ids=frozenset(expected_ids),
        expected_text_sha256_by_input_id={input_id: "a" * 64 for input_id in expected_ids},
    )
    provider = _real_provider()
    runtime = _real_runtime()
    vector = [vector_scale / (dimension**0.5)] * dimension
    snapshot = {
        "snapshot_version": "m2-embedding-v1",
        "candidate_snapshot_sha256": candidate_sha256,
        "provider": provider,
        "records": [
            {"input_id": input_id_override if index == 0 and input_id_override else f"candidate:{index}", "text_sha256": text_sha256_override or "a" * 64, "descriptor": record_descriptor or provider, "dimension": dimension, "vector": vector}
            for index in range(33)
        ]
        + [{"input_id": "query:main", "text_sha256": "a" * 64, "descriptor": record_descriptor or provider, "dimension": dimension, "vector": vector}],
    }
    raw_snapshot = _json_bytes(snapshot)
    canonical_sha256 = evidence._canonical_sha256(snapshot)
    manifest = {
        "vector_snapshot_sha256": canonical_sha256, "candidate_snapshot_sha256": candidate_sha256,
        "candidate_count": 33, "query_count": 1, "evidence_type": "real", "provider": manifest_provider or provider,
        "runtime": runtime, "stats": {},
        "vector_norm_audit": {"checked_count": 34, "non_unit_norm_count": 0, "relative_tolerance": 0.001, "absolute_tolerance": 0.001},
    }
    raw_manifest = _json_bytes(manifest)
    payload = {
        "vector_snapshot": {
            "path": "snapshots/vectors.json",
            "blob_sha256": hashlib.sha256(raw_snapshot).hexdigest(),
            "manifest_path": "snapshots/vectors.manifest.json",
            "manifest_blob_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "canonical_sha256": canonical_sha256,
        }
    }
    blobs = {"snapshots/vectors.json": raw_snapshot, "snapshots/vectors.manifest.json": raw_manifest}
    monkeypatch.setattr(evidence, "_blob_bytes", lambda _commit, path, _root: blobs.get(path))
    errors: list[str] = []
    evidence._validate_vector_snapshot(payload, "a" * 40, context, errors, Path("."))
    return errors


def test_precompletion_main_is_valid_without_a_completion_report() -> None:
    result = validate_m2_t01_evidence()
    assert result.valid is True


def test_b2_run_audit_contract_is_pinned() -> None:
    assert evidence.BGE_RUN_AUDIT_PATH == "evaluation/reports/m2-t01-bge-run-2026-07-29.json"
    assert evidence.BGE_RUN_AUDIT_VERSION == "m2-t01-bge-run.v1"
    assert evidence.M2_IMPLEMENTATION_COMMIT == "84cd61261c4f496e6b8255ad44c3849ed98841c2"
    assert evidence.M2_SNAPSHOT_COMMIT == "438e7ad79a8d9e8085ea7eccefce68f5be85abec"


def test_present_completion_report_is_fail_closed_when_incomplete(tmp_path: Path) -> None:
    report = tmp_path / "m2-t01-embedding.json"
    report.write_text('{"report_version":"m2-t01-embedding.v1"}\n', encoding="utf-8")

    result = validate_m2_t01_evidence(report_path=report)

    assert result.valid is False
    assert "report must identify phase M2 and task M2-T01" in result.errors


def test_exact_candidate_snapshot_bytes_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_sha256, errors = _candidate_validation(monkeypatch)

    assert candidate_sha256 is not None
    assert errors == []


def test_candidate_formatting_only_mutation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _candidate_snapshot()
    original_sha256 = hashlib.sha256(_json_bytes(original, indent=2)).hexdigest()
    manifest = _candidate_manifest(original_sha256)

    _, errors = _candidate_validation(
        monkeypatch,
        snapshot=original,
        manifest=manifest,
        snapshot_sha256=original_sha256,
        snapshot_indent=4,
    )

    assert "candidate snapshot exact SHA-256 does not match manifest/report" in errors


@pytest.mark.parametrize(
    ("snapshot_override", "manifest_override", "report_sha256", "expected_error"),
    [
        ({}, {"snapshot_sha256": "0" * 64}, None, "candidate snapshot exact SHA-256 does not match manifest/report"),
        ({}, {}, "0" * 64, "candidate snapshot exact SHA-256 does not match manifest/report"),
        ({"source_evidence_report": "evaluation/reports/wrong.json"}, {}, None, "candidate snapshot has invalid M1 rebaseline provenance"),
        ({"source_candidate_array_sha256": "0" * 64}, {}, None, "candidate snapshot has invalid M1 rebaseline provenance"),
        ({}, {"zero_transport_replay": {"transport_requests": 0, "cache_hits": 11, "query_count": 12, "empty_cache_entry_count": 1}}, None, "candidate snapshot manifest has invalid zero-transport replay audit"),
        ({}, {"zero_transport_replay": {"transport_requests": 0, "cache_hits": 12, "query_count": 11, "empty_cache_entry_count": 1}}, None, "candidate snapshot manifest has invalid zero-transport replay audit"),
        ({}, {"zero_transport_replay": {"transport_requests": 0, "cache_hits": 12, "query_count": 12, "empty_cache_entry_count": 0}}, None, "candidate snapshot manifest has invalid zero-transport replay audit"),
        ({}, {"artifact_source_classification": "EXACT_HISTORICAL_ARTIFACTS_RECOVERED"}, None, "candidate snapshot has invalid M1 rebaseline provenance"),
        ({}, {"source_bundle": "evaluation/source-artifacts/wrong"}, None, "candidate snapshot has invalid M1 rebaseline provenance"),
    ],
)
def test_candidate_snapshot_rejects_invalid_exact_hash_or_provenance(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_override: dict[str, object],
    manifest_override: dict[str, object],
    report_sha256: str | None,
    expected_error: str,
) -> None:
    snapshot = _candidate_snapshot()
    snapshot.update(snapshot_override)
    raw_snapshot = _json_bytes(snapshot)
    manifest = _candidate_manifest(hashlib.sha256(raw_snapshot).hexdigest())
    manifest.update(manifest_override)

    _, errors = _candidate_validation(
        monkeypatch,
        snapshot=snapshot,
        manifest=manifest,
        snapshot_sha256=report_sha256,
    )

    assert expected_error in errors


def test_vector_snapshot_retains_canonical_hash_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _vector_validation(monkeypatch) == []


@pytest.mark.parametrize("dimension", [2, 1023])
def test_vector_snapshot_rejects_non_1024_dimensions(
    monkeypatch: pytest.MonkeyPatch, dimension: int
) -> None:
    assert "vector snapshot records must be exactly 1024-dimensional" in _vector_validation(
        monkeypatch, dimension=dimension
    )


def test_vector_snapshot_rejects_non_unit_vectors_and_cross_artifact_provider_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "vector snapshot contains non-unit vectors" in _vector_validation(monkeypatch, vector_scale=2.0)
    assert "vector snapshot record descriptor must match snapshot provider" in _vector_validation(
        monkeypatch, record_descriptor={**_real_provider(), "cache_namespace": "embedding:other"}
    )
    assert "vector snapshot and manifest providers must match exactly" in _vector_validation(
        monkeypatch, manifest_provider={**_real_provider(), "cache_namespace": "embedding:other"}
    )


def test_vector_snapshot_rejects_wrong_b1_input_id_and_text_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "vector snapshot input IDs must exactly match the B1-derived input set" in _vector_validation(
        monkeypatch, input_id_override="query:substituted"
    )
    assert "vector snapshot record text SHA-256 does not bind to the B1 input" in _vector_validation(
        monkeypatch, text_sha256_override="b" * 64
    )


def test_final_evidence_rejects_forged_real_provider_runtime_identity() -> None:
    provider, runtime = _real_provider(), _real_runtime()
    errors: list[str] = []
    evidence._validate_real_provider_contract(provider, runtime, errors)
    assert errors == []
    for field, value in (("provider_library_version", "optional"), ("model_revision", "main"), ("dimension", 16), ("normalized", False), ("model_id", "wrong"), ("cache_namespace", "embedding:fake")):
        invalid = deepcopy(provider)
        invalid[field] = value
        errors = []
        evidence._validate_real_provider_contract(invalid, runtime, errors)
        assert errors
    for invalid_runtime in (
        {**runtime, "use_fp16": "false"},
        {**runtime, "use_fp16": True},
        {**runtime, "flagembedding_version": "1.3.4"},
        {**runtime, "device_request": "cuda:1"},
    ):
        errors = []
        evidence._validate_real_provider_contract(provider, invalid_runtime, errors)
        assert errors


def test_final_report_cannot_describe_a_different_vector_provider_or_runtime() -> None:
    provider, runtime = _real_provider(), _real_runtime()
    vector_context = evidence.VectorEvidenceContext(
        canonical_sha256="a" * 64, provider=provider, runtime=runtime, stats={}
    )
    payload = {
        "fake_evidence": {"classification": "deterministic_fake", "provider": {"provider_name": "deterministic_fake", "cache_namespace": "embedding:fake"}},
        "real_model_evidence": {
            "classification": "real", "provider": provider, "runtime": runtime,
            "candidate_vector_count": 33, "query_vector_count": 1, "non_finite_count": 0,
            "non_unit_norm_count": 0,
            "live_cache": {"cache_hits": 0, "provider_call_count": 1},
            "replay_cache": {"provider_call_count": 0, "cache_misses": 0, "vector_arrays_equal": True},
        },
    }
    errors: list[str] = []
    audit_context = evidence.BgeRunAuditContext(
        live={"cache_hits": 0, "provider_call_count": 1},
        replay={"provider_call_count": 0, "cache_misses": 0},
        canonical_sha256="a" * 64,
    )
    payload["real_model_evidence"]["replay_cache"]["exact_bytes_equal"] = True
    evidence._validate_model_evidence(payload, vector_context, audit_context, errors)
    assert errors == []
    payload["real_model_evidence"] = {**payload["real_model_evidence"], "provider": {**provider, "cache_namespace": "embedding:other"}}
    errors = []
    evidence._validate_model_evidence(payload, vector_context, audit_context, errors)
    assert "real model evidence provider must match vector manifest" in errors
