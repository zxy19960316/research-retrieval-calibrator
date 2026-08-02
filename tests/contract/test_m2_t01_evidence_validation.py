"""Red-first expectations for M2-T01's completed-evidence gate."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

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


def _valid_vector_context() -> evidence.VectorEvidenceContext:
    return evidence.VectorEvidenceContext(
        canonical_sha256="a" * 64,
        provider=_real_provider(), runtime=_real_runtime(),
        stats={"cache_corrupt_count": 0, "cache_hits": 0, "cache_misses": 34, "provider_call_count": 17, "provider_input_count": 34},
    )


def _valid_run_audit() -> dict[str, object]:
    preflight = {"status": "success", "model_id": "BAAI/bge-m3", "model_revision": evidence.BGE_M3_MODEL_REVISION, "dimension": 1024, "provider_library_version": "1.3.5", "device": "cpu"}
    return {"report_version": evidence.BGE_RUN_AUDIT_VERSION, "phase": "M2", "task_id": "M2-T01", "a6_2_commit": evidence.M2_IMPLEMENTATION_COMMIT, "evidence_type": "real", "download_policy": {"allowlist": "BGE_M3_REQUIRED_FILES", "max_workers": 1}, "xet_disabled": True, "onnx_file_count": 0, "pytorch_model_sha256": evidence.BGE_M3_WEIGHT_SHA256, "pytorch_model_size_bytes": evidence.BGE_M3_WEIGHT_SIZE_BYTES, "online_preflight": preflight, "offline_preflight": {**preflight, "offline": True}, "live": _valid_vector_context().stats, "replay": {"cache_corrupt_count": 0, "cache_hits": 34, "cache_misses": 0, "provider_call_count": 0, "provider_input_count": 0, "empty_model_cache_file_count": 0}, "vector_arrays_equal": True, "exact_bytes_equal": True, "vector_snapshot_canonical_sha256": "a" * 64}


def test_valid_bge_run_audit_returns_exact_context() -> None:
    errors: list[str] = []
    context = evidence._validate_bge_run_audit(_valid_run_audit(), _valid_vector_context(), errors)
    assert errors == []
    assert context is not None and context.live == _valid_vector_context().stats
    assert context.replay["cache_hits"] == 34 and context.canonical_sha256 == "a" * 64


@pytest.mark.parametrize("path,value", [
    ("report_version", "wrong"), ("phase", "M1"), ("task_id", "wrong"), ("a6_2_commit", "0" * 40), ("evidence_type", "fake"),
    ("download_policy.allowlist", "wrong"), ("download_policy.max_workers", 2), ("xet_disabled", False), ("onnx_file_count", 1),
    ("pytorch_model_sha256", "0" * 64), ("pytorch_model_size_bytes", 1), ("online_preflight.status", "failed"),
    ("online_preflight.model_id", "wrong"), ("online_preflight.model_revision", "0" * 40), ("online_preflight.dimension", 16),
    ("online_preflight.provider_library_version", "1.0"), ("online_preflight.device", "cuda"), ("offline_preflight.offline", False),
    ("live.cache_hits", 1), ("live.cache_misses", 0), ("live.provider_call_count", 0), ("replay.cache_hits", 0),
    ("replay.empty_model_cache_file_count", 1), ("vector_arrays_equal", False), ("exact_bytes_equal", False),
    ("vector_snapshot_canonical_sha256", "b" * 64),
])
def test_bge_run_audit_rejects_each_security_critical_mutation(path: str, value: object) -> None:
    audit = deepcopy(_valid_run_audit())
    target: dict[str, object] = audit
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]  # type: ignore[assignment,index]
    target[parts[-1]] = value
    errors: list[str] = []
    evidence._validate_bge_run_audit(audit, _valid_vector_context(), errors)
    assert errors


def _valid_inputs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    blobs = {path: b"fixed:" + path.encode() for path in evidence.REQUIRED_VALIDATED_INPUT_PATHS}
    monkeypatch.setattr(evidence, "_blob_bytes", lambda _commit, path, _root: blobs.get(path))
    return [{"path": path, "sha256": hashlib.sha256(raw).hexdigest()} for path, raw in blobs.items()]


def test_validated_inputs_accepts_exact_required_set(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []
    inputs = _valid_inputs(monkeypatch)
    assert evidence._validate_inputs({"validated_inputs": inputs}, "a" * 40, errors, Path(".")) == evidence.REQUIRED_VALIDATED_INPUT_PATHS
    assert errors == []


@pytest.mark.parametrize("removed", sorted(evidence.REQUIRED_VALIDATED_INPUT_PATHS))
def test_validated_inputs_rejects_each_missing_required_path(monkeypatch: pytest.MonkeyPatch, removed: str) -> None:
    errors: list[str] = []
    inputs = [item for item in _valid_inputs(monkeypatch) if item["path"] != removed]
    evidence._validate_inputs({"validated_inputs": inputs}, "a" * 40, errors, Path("."))
    assert f"missing required validated input: {removed}" in errors


@pytest.mark.parametrize("path,expected", [("C:/x.json", "validated input path is missing, duplicate, or unsafe"), ("/x.json", "validated input path is missing, duplicate, or unsafe"), ("../x.json", "validated input path is missing, duplicate, or unsafe"), ("evaluation/x.json", "validated input hash mismatch: evaluation/x.json")])
def test_validated_inputs_rejects_unsafe_or_unavailable_entries(monkeypatch: pytest.MonkeyPatch, path: str, expected: str) -> None:
    errors: list[str] = []
    inputs = _valid_inputs(monkeypatch) + [{"path": path, "sha256": "a" * 64}]
    evidence._validate_inputs({"validated_inputs": inputs}, "a" * 40, errors, Path("."))
    assert expected in errors


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda inputs: inputs.append(deepcopy(inputs[0])), "validated input path is missing, duplicate, or unsafe"),
        (lambda inputs: inputs.append({"path": r"C:\\outside.json", "sha256": "a" * 64}), "validated input path is missing, duplicate, or unsafe"),
        (lambda inputs: inputs.append({"path": "/outside.json", "sha256": "a" * 64}), "validated input path is missing, duplicate, or unsafe"),
        (lambda inputs: inputs.append({"path": "evaluation/../outside.json", "sha256": "a" * 64}), "validated input path is missing, duplicate, or unsafe"),
        (lambda inputs: inputs.__setitem__(0, {**inputs[0], "sha256": "not-a-sha"}), "validated input has invalid sha256"),
        (lambda inputs: inputs.__setitem__(0, {**inputs[0], "sha256": "0" * 64}), "validated input hash mismatch"),
        (lambda inputs: inputs.__setitem__(0, {**inputs[0], "path": "evaluation/missing.json", "sha256": "a" * 64}), "validated input hash mismatch: evaluation/missing.json"),
        (lambda inputs: inputs.__setitem__(0, "not-an-object"), "validated input entry must be an object"),
    ],
)
def test_validated_inputs_rejects_each_remaining_invalid_entry(
    monkeypatch: pytest.MonkeyPatch,
    mutate: object,
    expected: str,
) -> None:
    errors: list[str] = []
    inputs: list[object] = _valid_inputs(monkeypatch)
    mutate(inputs)  # type: ignore[operator]
    evidence._validate_inputs({"validated_inputs": inputs}, "a" * 40, errors, Path("."))
    assert any(error.startswith(expected) for error in errors)


@pytest.mark.parametrize("inputs", [None, [], "not-a-list"])
def test_validated_inputs_rejects_missing_or_non_list_collection(inputs: object) -> None:
    errors: list[str] = []
    assert evidence._validate_inputs({"validated_inputs": inputs}, "a" * 40, errors, Path(".")) == set()
    assert errors == ["validated_inputs must be a non-empty list"]


@pytest.mark.parametrize("field", ["baseline_commit", "implementation_commit_a", "snapshot_commit_b"])
def test_provenance_rejects_each_fixed_commit_mutation(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    monkeypatch.setattr(evidence, "_is_ancestor", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(evidence, "_git", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    payload = {"baseline_commit": evidence.BASELINE_COMMIT, "implementation_commit_a": evidence.M2_IMPLEMENTATION_COMMIT, "snapshot_commit_b": evidence.M2_SNAPSHOT_COMMIT, "validated_commit": "a" * 40}
    payload[field] = "b" * 40
    errors: list[str] = []
    evidence._validate_provenance(payload, errors, Path("."))
    assert errors


def _provenance_payload() -> dict[str, str]:
    return {
        "baseline_commit": evidence.BASELINE_COMMIT,
        "implementation_commit_a": evidence.M2_IMPLEMENTATION_COMMIT,
        "snapshot_commit_b": evidence.M2_SNAPSHOT_COMMIT,
        "validated_commit": "a" * 40,
    }


def test_provenance_accepts_the_fixed_ordered_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evidence, "_git", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    errors: list[str] = []
    assert evidence._validate_provenance(_provenance_payload(), errors, Path(".")) == "a" * 40
    assert errors == []


@pytest.mark.parametrize(
    ("mutate", "git_returncode", "expected"),
    [
        (lambda payload: payload.__setitem__("baseline_commit", "b" * 40), 0, "baseline_commit must equal the M1 merge baseline"),
        (lambda payload: payload.__setitem__("implementation_commit_a", "b" * 40), 0, "implementation_commit_a must equal the fixed A6.2 implementation commit"),
        (lambda payload: payload.__setitem__("snapshot_commit_b", "b" * 40), 0, "snapshot_commit_b must equal the fixed B2 snapshot commit"),
        (lambda payload: payload.__setitem__("validated_commit", "not-a-sha"), 0, "baseline, implementation, snapshot, and validated commits must be HEAD ancestors"),
        (lambda _payload: None, 1, "baseline, implementation, snapshot, and validated commits must be HEAD ancestors"),
    ],
)
def test_provenance_rejects_invalid_identity_or_non_ancestor(
    monkeypatch: pytest.MonkeyPatch,
    mutate: object,
    git_returncode: int,
    expected: str,
) -> None:
    mutate(_provenance_payload())  # type: ignore[operator]
    payload = _provenance_payload()
    mutate(payload)  # type: ignore[operator]
    monkeypatch.setattr(evidence, "_git", lambda *_args, **_kwargs: SimpleNamespace(returncode=git_returncode))
    errors: list[str] = []
    evidence._validate_provenance(payload, errors, Path("."))
    assert expected in errors


def test_provenance_rejects_validated_ancestry_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def git(*_args: str, **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=1 if calls == 6 else 0)

    monkeypatch.setattr(evidence, "_git", git)
    errors: list[str] = []
    evidence._validate_provenance(_provenance_payload(), errors, Path("."))
    assert errors == ["implementation/data/validated commits are not in required ancestry order"]


@pytest.mark.parametrize("case", ["missing_input", "validated_missing", "frozen_missing", "bytes_differ", "invalid_json", "bad_weight", "bad_canonical"])
def test_run_audit_loader_rejects_each_invalid_source(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    audit = _valid_run_audit()
    if case == "bad_weight":
        audit["pytorch_model_sha256"] = "0" * 64
    if case == "bad_canonical":
        audit["vector_snapshot_canonical_sha256"] = "b" * 64
    raw = b"not-json" if case == "invalid_json" else json.dumps(audit).encode()
    inputs = {evidence.BGE_RUN_AUDIT_PATH}
    if case == "missing_input":
        inputs = set()
    def blob(commit: str | None, path: str, _root: Path) -> bytes | None:
        if path != evidence.BGE_RUN_AUDIT_PATH or case == "validated_missing" and commit == "a" * 40 or case == "frozen_missing" and commit == evidence.M2_SNAPSHOT_COMMIT:
            return None
        return raw + (b"x" if case == "bytes_differ" and commit == evidence.M2_SNAPSHOT_COMMIT else b"")
    monkeypatch.setattr(evidence, "_blob_bytes", blob)
    errors: list[str] = []
    context = evidence._load_validated_bge_run_audit("a" * 40, inputs, _valid_vector_context(), errors, Path("."))
    assert context is None
    assert errors


def test_run_audit_loader_returns_a_validated_context(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = json.dumps(_valid_run_audit()).encode()
    monkeypatch.setattr(evidence, "_blob_bytes", lambda *_args: raw)
    errors: list[str] = []
    context = evidence._load_validated_bge_run_audit(
        "a" * 40,
        {evidence.BGE_RUN_AUDIT_PATH},
        _valid_vector_context(),
        errors,
        Path("."),
    )
    assert errors == []
    assert context is not None
    assert context.live == _valid_vector_context().stats
    assert context.replay["cache_hits"] == 34
    assert context.canonical_sha256 == "a" * 64


def _complete_model_payload() -> dict[str, object]:
    context = evidence.BgeRunAuditContext(
        live=_valid_vector_context().stats,
        replay={
            "cache_corrupt_count": 0,
            "cache_hits": 34,
            "cache_misses": 0,
            "provider_call_count": 0,
            "provider_input_count": 0,
            "empty_model_cache_file_count": 0,
        },
        canonical_sha256="a" * 64,
    )
    return {
        "fake_evidence": {
            "classification": "deterministic_fake",
            "provider": {
                "provider_name": "deterministic_fake",
                "cache_namespace": "embedding:fake",
            },
        },
        "real_model_evidence": {
            "classification": "real",
            "provider": _real_provider(),
            "runtime": _real_runtime(),
            "candidate_vector_count": 33,
            "query_vector_count": 1,
            "non_finite_count": 0,
            "non_unit_norm_count": 0,
            "live_cache": context.live,
            "replay_cache": {
                **context.replay,
                "vector_arrays_equal": True,
                "exact_bytes_equal": True,
            },
        },
    }


def _complete_run_context() -> evidence.BgeRunAuditContext:
    payload = _complete_model_payload()
    real = payload["real_model_evidence"]
    assert isinstance(real, dict)
    replay = real["replay_cache"]
    assert isinstance(replay, dict)
    return evidence.BgeRunAuditContext(
        live=_valid_vector_context().stats,
        replay={key: value for key, value in replay.items() if key not in {"vector_arrays_equal", "exact_bytes_equal"}},
        canonical_sha256="a" * 64,
    )


def test_model_evidence_accepts_the_complete_b2_live_and_replay_fixture() -> None:
    errors: list[str] = []
    evidence._validate_model_evidence(
        _complete_model_payload(), _valid_vector_context(), _complete_run_context(), errors
    )
    assert errors == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda _payload: None, "completed evidence requires validated B2 run audit context"),
        (lambda payload: payload["real_model_evidence"]["live_cache"].pop("cache_hits"), "real live cache evidence must exactly match B2 run audit"),
        (lambda payload: payload["real_model_evidence"]["live_cache"].__setitem__("cache_hits", 1), "real live cache evidence must exactly match B2 run audit"),
        (lambda payload: payload["real_model_evidence"]["replay_cache"].pop("cache_hits"), "real replay cache evidence must exactly match B2 run audit"),
        (lambda payload: payload["real_model_evidence"]["replay_cache"].__setitem__("cache_hits", 0), "real replay cache evidence must exactly match B2 run audit"),
        (lambda payload: payload["real_model_evidence"]["replay_cache"].__setitem__("vector_arrays_equal", False), "real replay cache evidence must exactly match B2 run audit"),
        (lambda payload: payload["real_model_evidence"]["replay_cache"].__setitem__("exact_bytes_equal", False), "real replay cache evidence must exactly match B2 run audit"),
        (lambda payload: payload["real_model_evidence"].__setitem__("provider", {**_real_provider(), "cache_namespace": "embedding:other"}), "real model evidence provider must match vector manifest"),
        (lambda payload: payload["real_model_evidence"].__setitem__("runtime", {**_real_runtime(), "device_request": "cuda:9"}), "real model evidence runtime must match vector manifest"),
        (lambda payload: payload["real_model_evidence"].__setitem__("candidate_vector_count", 32), "real model evidence must record 33 candidate vectors and one query vector"),
        (lambda payload: payload["real_model_evidence"].__setitem__("query_vector_count", 2), "real model evidence must record 33 candidate vectors and one query vector"),
        (lambda payload: payload["real_model_evidence"].__setitem__("non_finite_count", 1), "real model evidence must record zero non-finite vectors"),
        (lambda payload: payload["real_model_evidence"].__setitem__("non_unit_norm_count", 1), "real model evidence must record zero non-unit vectors"),
    ],
)
def test_model_evidence_rejects_each_completion_gate_mutation(
    mutate: object, expected: str
) -> None:
    payload = _complete_model_payload()
    mutate(payload)  # type: ignore[operator]
    errors: list[str] = []
    audit = None if expected == "completed evidence requires validated B2 run audit context" else _complete_run_context()
    evidence._validate_model_evidence(payload, _valid_vector_context(), audit, errors)
    assert expected in errors


def _run_synthetic_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_audit_context: evidence.BgeRunAuditContext | None,
) -> tuple[evidence.EvidenceValidationResult, dict[str, object]]:
    report = tmp_path / "m2-t01-completion.json"
    report.write_text(
        json.dumps(
            {
                "report_version": evidence.REPORT_VERSION,
                "phase": "M2",
                "task_id": "M2-T01",
                "m1_provenance": {"metadata_projection_mismatch_count": 0},
            }
        ),
        encoding="utf-8",
    )
    calls: dict[str, object] = {}
    candidate = evidence.CandidateEvidenceContext("b" * 64, frozenset(), {})
    vector = _valid_vector_context()
    inputs = set(evidence.REQUIRED_VALIDATED_INPUT_PATHS)
    monkeypatch.setattr(evidence, "_validate_provenance", lambda *_args: "a" * 40)
    monkeypatch.setattr(evidence, "_validate_inputs", lambda *_args: inputs)
    monkeypatch.setattr(evidence, "_validate_candidate_snapshot", lambda *_args: candidate)
    monkeypatch.setattr(evidence, "_validate_vector_snapshot", lambda *_args: vector)
    monkeypatch.setattr(evidence, "_validate_checks", lambda *_args: None)

    def load(commit: str, seen: set[str], context: object, *_args: object) -> evidence.BgeRunAuditContext | None:
        calls["loader"] = (commit, seen, context)
        return run_audit_context

    def model(_payload: object, context: object, audit: object, errors: list[str]) -> None:
        calls["model"] = (context, audit)
        if audit is None:
            errors.append("completed evidence requires validated B2 run audit context")

    monkeypatch.setattr(evidence, "_load_validated_bge_run_audit", load)
    monkeypatch.setattr(evidence, "_validate_model_evidence", model)
    monkeypatch.setattr(evidence, "_validate_status", lambda *_args: None)
    monkeypatch.setattr(
        evidence,
        "_validate_committed_b2_precompletion",
        lambda *_args: pytest.fail("completion must not use the precompletion validator"),
    )
    return evidence.validate_m2_t01_evidence(report_path=report, repository_root=tmp_path), calls


def test_completion_wiring_passes_the_loader_context_to_model_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audit = _complete_run_context()
    result, calls = _run_synthetic_completion(monkeypatch, tmp_path, audit)
    assert result.valid is True
    assert calls["loader"] == ("a" * 40, evidence.REQUIRED_VALIDATED_INPUT_PATHS, _valid_vector_context())
    assert calls["model"] == (_valid_vector_context(), audit)


def test_completion_wiring_fails_closed_when_loader_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, calls = _run_synthetic_completion(monkeypatch, tmp_path, None)
    assert result.valid is False
    assert result.errors == ["completed evidence requires validated B2 run audit context"]
    assert calls["model"] == (_valid_vector_context(), None)


@pytest.mark.parametrize(
    ("state", "completed", "total", "valid"),
    [
        ("IN_PROGRESS", 1, 5, True),
        ("IN_PROGRESS", 2, 5, True),
        ("IN_PROGRESS", 4, 5, True),
        ("COMPLETE", 5, 5, True),
        ("IN_PROGRESS", 0, 5, False),
        ("IN_PROGRESS", 5, 5, False),
        ("COMPLETE", 4, 5, False),
        ("READY", 0, 5, False),
    ],
)
def test_completed_m2_t01_status_accepts_later_m2_progress_without_claiming_completion(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    completed: int,
    total: int,
    valid: bool,
) -> None:
    monkeypatch.setattr(
        evidence,
        "_parse_status",
        lambda *_args: {
            "M1": ("COMPLETE", 4, 4),
            "M2": (state, completed, total),
            "M3": ("BLOCKED_BY_M2", 0, 5),
        },
    )
    errors: list[str] = []
    evidence._validate_status(
        {
            "status_after_evidence": {
                "m2": f"{state} {completed}/{total}",
                "m3": "BLOCKED_BY_M2",
            }
        },
        errors,
        Path("."),
    )
    assert bool(errors) is (not valid)
