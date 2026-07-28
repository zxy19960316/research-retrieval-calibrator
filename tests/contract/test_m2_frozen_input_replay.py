"""Contracts for the M2-T01R2 frozen M1 replay and paired publication path."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.adapters.arxiv import ArxivAdapter, ArxivAdapterConfig, ArxivResponse
from app.core.first_round import run_first_round
from app.core.paper_dedup import deduplicate_papers
from app.models.embedding import EmbeddingTaskError
from app.models.first_round import FirstRoundConfig
from app.models.paper import PaperRecord
from scripts import freeze_m2_candidates as freeze_module
from scripts.embed_frozen_candidates import load_validated_frozen_inputs
from scripts.freeze_m2_candidates import (
    M1_COMPLETION_MERGE_COMMIT,
    REAL_CACHE_INCOMPLETE,
    REAL_CACHE_PROVENANCE_MISMATCH,
    SNAPSHOT_PUBLICATION_FAILED,
    ZERO_TRANSPORT_REPLAY_FAILED,
    AcceptedM1Provenance,
    FreezeGateError,
    FreezeResult,
    _canonical_json_sha256,
    _metadata_mismatch_field,
    _normalize_evidence_report_label,
    _normalize_repository_relative_label,
    _project_abstract,
    _publish_snapshot_pair,
    _render_json_bytes,
    _replay_and_project,
    _repository_relative_posix_path,
    freeze_candidates,
    load_accepted_m1_provenance,
    validate_frozen_snapshot_bytes,
)

QUESTION = "How can graph-based retrieval support scientific literature discovery?"
NOW = datetime(2026, 7, 28, 0, 0, tzinfo=UTC)
SYNTHETIC_REPORT_LABEL = "evaluation/reports/synthetic-m1-validation.json"


class RecordedTransport:
    """Test-only cache seeder; the freeze replay itself always uses no transport."""

    def __init__(self, responses: list[ArxivResponse]) -> None:
        self.responses = list(responses)
        self.request_count = 0

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> ArxivResponse:
        del url, headers, timeout_seconds
        self.request_count += 1
        return self.responses.pop(0)


def _atom_response(index: int) -> ArxivResponse:
    source_id = f"2401.{index:05d}"
    body = f'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
    <entry><id>https://arxiv.org/abs/{source_id}</id>
    <title>Graph retrieval paper {index:02d}</title>
    <summary>Canonical abstract {index:02d}.</summary>
    <published>2024-01-01</published><author><name>Ada Author</name></author></entry>
    </feed>'''.encode()
    return ArxivResponse(200, body, {})


def _seed_synthetic_real_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Use the real M1 pipeline to create a 12-query synthetic real cache."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "m1-cache"
    config = FirstRoundConfig(
        cache_dir=cache_dir,
        mode="real",
        cache_namespace="first-round:real",
        adapter_schema_version="m1-t04.v2",
        max_results_per_query=1,
        max_total_candidates=60,
        max_total_attempts=20,
        timeout_seconds=1.0,
        min_request_interval_seconds=1.0,
    )
    transport = RecordedTransport([_atom_response(index) for index in range(12)])
    adapter = ArxivAdapter(
        ArxivAdapterConfig(
            user_agent="m2-t01r2-synthetic-cache-seed",
            timeout_seconds=1.0,
            page_size=1,
            min_request_interval_seconds=1.0,
            max_attempts=1,
            max_total_results=1,
            max_total_attempts=20,
            initial_backoff_seconds=0.0,
            cache_dir=cache_dir,
            cache_schema_version="m1-t04.v2",
            cache_namespace="first-round:real",
        ),
        transport=transport,
        monotonic=lambda: 10.0,
        sleeper=lambda _: None,
        utc_now=lambda: NOW,
    )
    run = run_first_round(
        QUESTION,
        config=config,
        adapter=adapter,
        now=lambda: NOW,
        monotonic=lambda: 10.0,
    )
    assert run.status.value == "success"
    assert len(run.query_plan.queries if run.query_plan else []) == 12
    assert transport.request_count == 12
    raw_output = _render_json_bytes(run.model_dump(mode="json"))
    output_path = tmp_path / "first-round.json"
    output_path.write_bytes(raw_output)
    report_payload: dict[str, object] = {
        "baseline_commit": "f0f167766589e3321821b0caf7793b00c8ff7291",
        "validated_implementation_commit": "7b19d437bd30f29e9ec2debaf022920bc47c3f0d",
        "implementation_ancestry": ["7b19d437bd30f29e9ec2debaf022920bc47c3f0d"],
        "real_external": {
            "classification": "real_external",
            "deduplicated_candidate_count": 12,
            "source_id_coverage": 1.0,
            "url_coverage": 1.0,
            "candidate_array_sha256": _canonical_json_sha256(run.model_dump(mode="json")["candidates"]),
            "output_hashes": {"first-round.json": hashlib.sha256(raw_output).hexdigest()},
        },
    }
    report_path = tmp_path / "m1-validation.json"
    report_path.write_bytes(_render_json_bytes(report_payload))
    return output_path, cache_dir, report_path


@pytest.fixture
def synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    return _seed_synthetic_real_run(tmp_path)


def _synthetic_provenance(report_path: Path) -> AcceptedM1Provenance:
    return load_accepted_m1_provenance(report_path, required_candidate_count=12)


def _snapshot_pair(
    output_path: Path, cache_dir: Path, report_path: Path
) -> tuple[dict[str, object], dict[str, object], bytes, bytes]:
    provenance = _synthetic_provenance(report_path)
    snapshot, manifest = _replay_and_project(
        output_path.read_bytes(),
        m1_cache_dir=cache_dir,
        provenance=provenance,
        evidence_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        source_evidence_report=SYNTHETIC_REPORT_LABEL,
    )
    snapshot_bytes = _render_json_bytes(snapshot)
    manifest["snapshot_sha256"] = hashlib.sha256(snapshot_bytes).hexdigest()
    manifest_bytes = _render_json_bytes(manifest)
    return snapshot, manifest, snapshot_bytes, manifest_bytes


def _freeze_synthetic(**kwargs: object) -> FreezeResult:
    return freeze_candidates(
        source_evidence_report_label=SYNTHETIC_REPORT_LABEL,
        **kwargs,
    )


def _publish_synthetic_pair(**kwargs: object) -> None:
    _publish_snapshot_pair(expected_candidate_count=12, **kwargs)


def test_production_evidence_provenance_uses_completion_merge_not_evidence_baseline() -> None:
    provenance = load_accepted_m1_provenance(Path("evaluation/reports/m1-validation.json"))

    assert provenance.output_sha256 == "069cd8c94d294c68bb051898d4c262f324b7e1f20eff10eabcf22d9bc435d178"
    assert provenance.candidate_array_sha256 == "e51eb84d4e772bba324a199478caa5f0983edef0b0dbf4c58fce5f9da5803209"
    assert provenance.m1_completion_merge_commit == M1_COMPLETION_MERGE_COMMIT
    assert provenance.m1_evidence_baseline_commit == "f0f167766589e3321821b0caf7793b00c8ff7291"
    assert provenance.validated_implementation_commit in provenance.implementation_ancestry


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("real_external", "output_hashes", "first-round.json"), None),
        (("real_external", "candidate_array_sha256"), None),
        (("real_external", "output_hashes", "first-round.json"), "g" * 64),
        (("real_external", "deduplicated_candidate_count"), 32),
        (("real_external", "source_id_coverage"), 0.99),
    ],
)
def test_provenance_rejects_missing_malformed_or_unaccepted_evidence(
    tmp_path: Path, field_path: tuple[str, ...], value: object
) -> None:
    output_hashes: dict[str, object] = {"first-round.json": "c" * 64}
    real: dict[str, object] = {
        "classification": "real_external",
        "deduplicated_candidate_count": 33,
        "source_id_coverage": 1.0,
        "url_coverage": 1.0,
        "candidate_array_sha256": "b" * 64,
        "output_hashes": output_hashes,
    }
    payload: dict[str, object] = {
        "baseline_commit": "a" * 40,
        "validated_implementation_commit": "d" * 40,
        "implementation_ancestry": ["d" * 40],
        "real_external": real,
    }
    target: dict[str, object] = payload
    for key in field_path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    if value is None:
        del target[field_path[-1]]
    else:
        target[field_path[-1]] = value
    report = tmp_path / "m1-validation.json"
    report.write_bytes(_render_json_bytes(payload))

    with pytest.raises(ValueError):
        load_accepted_m1_provenance(report)


def test_synthetic_twelve_query_real_cache_replays_without_transport_and_publishes(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    output_dir = tmp_path / "snapshot"

    result = _freeze_synthetic(
        m1_output=output_path,
        m1_cache_dir=cache_dir,
        output_dir=output_dir,
        m1_evidence_report=report_path,
        expected_candidate_count=12,
    )

    assert result.status == "success"
    assert result.candidate_count == 12
    snapshot_bytes = (output_dir / "m1-candidates.v1.json").read_bytes()
    manifest = json.loads((output_dir / "m1-candidates.v1.manifest.json").read_text(encoding="utf-8"))
    assert manifest["zero_transport_replay"] == {
        "cache_hits": 12,
        "query_count": 12,
        "transport_requests": 0,
    }
    assert manifest["m1_completion_merge_commit"] == M1_COMPLETION_MERGE_COMMIT
    assert manifest["m1_evidence_baseline_commit"] == "f0f167766589e3321821b0caf7793b00c8ff7291"
    assert manifest["candidate_count"] == 12
    assert manifest["source_identity_set_sha256"] is not None
    assert json.loads(snapshot_bytes)["source_merge_commit"] == M1_COMPLETION_MERGE_COMMIT
    assert validate_frozen_snapshot_bytes(
        snapshot_bytes, manifest, expected_candidate_count=12
    ) is None


def test_missing_cache_and_corrupt_or_wrong_manifest_have_stable_codes(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    cache_file = next(cache_dir.rglob("*.json"))
    cache_file.unlink()
    missing = _freeze_synthetic(
        m1_output=output_path,
        m1_cache_dir=cache_dir,
        output_dir=tmp_path / "missing",
        m1_evidence_report=report_path,
        expected_candidate_count=12,
    )
    assert missing.error_code == REAL_CACHE_INCOMPLETE

    output_path, cache_dir, report_path = _seed_synthetic_real_run(tmp_path / "corrupt")
    cache_file = next(cache_dir.rglob("*.json"))
    cache_file.write_text("{not json", encoding="utf-8")
    corrupt = _freeze_synthetic(
        m1_output=output_path,
        m1_cache_dir=cache_dir,
        output_dir=tmp_path / "corrupt-out",
        m1_evidence_report=report_path,
        expected_candidate_count=12,
    )
    assert corrupt.error_code == REAL_CACHE_PROVENANCE_MISMATCH

    for field_name, value in (
        ("endpoint", "https://wrong.invalid/api/query"),
        ("cache_namespace", "first-round:recorded"),
        ("adapter_schema_version", "m1-t04.v1"),
    ):
        output_path, cache_dir, report_path = _seed_synthetic_real_run(tmp_path / field_name)
        cache_file = next(cache_dir.rglob("*.json"))
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        payload["manifest"][field_name] = value
        cache_file.write_bytes(_render_json_bytes(payload))
        mismatch = _freeze_synthetic(
            m1_output=output_path,
            m1_cache_dir=cache_dir,
            output_dir=tmp_path / f"{field_name}-out",
            m1_evidence_report=report_path,
            expected_candidate_count=12,
        )
        assert mismatch.error_code == REAL_CACHE_PROVENANCE_MISMATCH


def test_raw_and_deduplicated_cache_count_mismatches_fail_closed(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    raw_payload = json.loads(output_path.read_text(encoding="utf-8"))
    raw_payload["raw_candidate_count"] = 13
    raw_payload["metrics"]["raw_candidate_count"] = 13
    changed_output = _render_json_bytes(raw_payload)
    output_path.write_bytes(changed_output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["real_external"]["output_hashes"]["first-round.json"] = hashlib.sha256(changed_output).hexdigest()
    report_path.write_bytes(_render_json_bytes(report))
    raw_mismatch = _freeze_synthetic(
        m1_output=output_path,
        m1_cache_dir=cache_dir,
        output_dir=tmp_path / "raw-out",
        m1_evidence_report=report_path,
        expected_candidate_count=12,
    )
    assert raw_mismatch.error_code == REAL_CACHE_INCOMPLETE

    output_path, cache_dir, report_path = _seed_synthetic_real_run(tmp_path / "dedup")
    cache_files = list(cache_dir.rglob("*.json"))
    first = json.loads(cache_files[0].read_text(encoding="utf-8"))
    duplicate = json.loads(cache_files[-1].read_text(encoding="utf-8"))
    duplicate["records"] = first["records"]
    cache_files[-1].write_bytes(_render_json_bytes(duplicate))
    dedup_mismatch = _freeze_synthetic(
        m1_output=output_path,
        m1_cache_dir=cache_dir,
        output_dir=tmp_path / "dedup-out",
        m1_evidence_report=report_path,
        expected_candidate_count=12,
    )
    assert dedup_mismatch.error_code == REAL_CACHE_INCOMPLETE


def test_transport_invocation_maps_to_zero_transport_code(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    monkeypatch.setattr(freeze_module, "_validate_cache_entries", lambda *args: None)
    for cache_file in cache_dir.rglob("*.json"):
        cache_file.unlink()

    result = _freeze_synthetic(
        m1_output=output_path,
        m1_cache_dir=cache_dir,
        output_dir=tmp_path / "out",
        m1_evidence_report=report_path,
        expected_candidate_count=12,
    )

    assert result.error_code == ZERO_TRANSPORT_REPLAY_FAILED


@pytest.mark.parametrize(
    "field_name",
    [
        "paper_id",
        "source",
        "source_id",
        "title",
        "authors",
        "year",
        "doi",
        "url",
        "retrieval_paths",
        "cluster_id",
        "member_source_identities",
        "merge_reasons",
    ],
)
def test_every_legacy_metadata_field_has_a_dedicated_mismatch_detection(
    field_name: str,
) -> None:
    record = PaperRecord(
        paper_id="arxiv:2401.00001",
        source="arxiv",
        source_id="2401.00001",
        title="Canonical title",
        abstract="Canonical abstract",
        authors=["Ada Author"],
        year=2024,
        doi=None,
        url="https://arxiv.org/abs/2401.00001",
        language="en",
        retrieval_paths=["Q1"],
    )
    cluster = deduplicate_papers([record]).clusters[0]
    candidate = run_first_round  # preserves a direct reference to the real M1 producer
    del candidate
    from app.models.first_round import CandidateOutput

    output = CandidateOutput(
        paper_id=record.paper_id,
        source=record.source,
        source_id=record.source_id,
        title=record.title,
        abstract=record.abstract,
        authors=record.authors,
        year=record.year,
        doi=record.doi,
        url=record.url,
        retrieval_paths=cluster.retrieval_paths,
        cluster_id=cluster.cluster_id,
        member_source_identities=cluster.source_identities,
        merge_reasons=cluster.merge_reasons,
    )
    altered: dict[str, object] = {
        "paper_id": "arxiv:2401.99999",
        "source": "other",
        "source_id": "2401.99999",
        "title": "Changed title",
        "authors": ["Grace Author"],
        "year": 2025,
        "doi": "10.1000/example",
        "url": "https://arxiv.org/abs/2401.99999",
        "retrieval_paths": ["Q2"],
        "cluster_id": "cluster:changed",
        "member_source_identities": [],
        "merge_reasons": [object()],
    }
    changed = output.model_copy(update={field_name: altered[field_name]})

    assert _metadata_mismatch_field(changed, cluster) == field_name


def test_abstract_projection_is_direct_and_categories_are_empty(
    synthetic_inputs: tuple[Path, Path, Path]
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    snapshot, _, _, _ = _snapshot_pair(output_path, cache_dir, report_path)

    assert _project_abstract(None) is None
    assert _project_abstract("   ") is None
    assert _project_abstract("Exact abstract") == "Exact abstract"
    assert all(candidate["categories"] == [] for candidate in snapshot["candidates"])  # type: ignore[index]


def test_exact_byte_snapshot_hash_rejects_alternate_formatting(
    synthetic_inputs: tuple[Path, Path, Path]
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    _, manifest, snapshot_bytes, _ = _snapshot_pair(output_path, cache_dir, report_path)

    assert validate_frozen_snapshot_bytes(
        snapshot_bytes, manifest, expected_candidate_count=12
    ) is None
    assert validate_frozen_snapshot_bytes(
        snapshot_bytes + b" ", manifest, expected_candidate_count=12
    ) == (
        "frozen snapshot manifest SHA-256 does not match exact snapshot bytes"
    )


def test_paired_publication_rolls_back_manifest_failure_and_leaves_no_temp_residue(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    _, _, snapshot_bytes, manifest_bytes = _snapshot_pair(output_path, cache_dir, report_path)
    snapshot_path = tmp_path / "out" / "m1-candidates.v1.json"
    manifest_path = tmp_path / "out" / "m1-candidates.v1.manifest.json"

    def fail_manifest_replace(source: Path, target: Path) -> None:
        if target == manifest_path:
            raise OSError("injected manifest publication failure")
        os.replace(source, target)

    with pytest.raises(FreezeGateError, match="paired snapshot publication failed") as failure:
        _publish_synthetic_pair(
            snapshot_path=snapshot_path,
            snapshot_bytes=snapshot_bytes,
            manifest_path=manifest_path,
            manifest_bytes=manifest_bytes,
            replace=fail_manifest_replace,
        )

    assert failure.value.code == SNAPSHOT_PUBLICATION_FAILED
    assert not snapshot_path.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.rglob(".m2-freeze-*"))


@pytest.mark.parametrize("failing_name", ["m1-candidates.v1.json", "m1-candidates.v1.manifest.json"])
def test_paired_publication_cleans_staging_when_a_temp_write_fails(
    synthetic_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_name: str,
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    _, _, snapshot_bytes, manifest_bytes = _snapshot_pair(output_path, cache_dir, report_path)
    snapshot_path = tmp_path / "out" / "m1-candidates.v1.json"
    manifest_path = tmp_path / "out" / "m1-candidates.v1.manifest.json"
    original_write = Path.write_bytes

    def fail_staged_write(path: Path, contents: bytes) -> int:
        if path.name == failing_name and path.parent.name.startswith(".m2-freeze-"):
            raise OSError("injected temporary write failure")
        return original_write(path, contents)

    monkeypatch.setattr(Path, "write_bytes", fail_staged_write)
    with pytest.raises(FreezeGateError) as failure:
        _publish_synthetic_pair(
            snapshot_path=snapshot_path,
            snapshot_bytes=snapshot_bytes,
            manifest_path=manifest_path,
            manifest_bytes=manifest_bytes,
        )
    assert failure.value.code == SNAPSHOT_PUBLICATION_FAILED
    assert not snapshot_path.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.rglob(".m2-freeze-*"))


def test_paired_publication_is_idempotent_rejects_conflicts_and_is_byte_identical(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    _, _, snapshot_bytes, manifest_bytes = _snapshot_pair(output_path, cache_dir, report_path)
    snapshot_path = tmp_path / "out" / "m1-candidates.v1.json"
    manifest_path = tmp_path / "out" / "m1-candidates.v1.manifest.json"

    _publish_synthetic_pair(
        snapshot_path=snapshot_path,
        snapshot_bytes=snapshot_bytes,
        manifest_path=manifest_path,
        manifest_bytes=manifest_bytes,
    )
    first = (snapshot_path.read_bytes(), manifest_path.read_bytes())
    _publish_synthetic_pair(
        snapshot_path=snapshot_path,
        snapshot_bytes=snapshot_bytes,
        manifest_path=manifest_path,
        manifest_bytes=manifest_bytes,
    )
    assert first == (snapshot_path.read_bytes(), manifest_path.read_bytes())

    snapshot_path.write_bytes(b"conflict")
    with pytest.raises(FreezeGateError) as failure:
        _publish_synthetic_pair(
            snapshot_path=snapshot_path,
            snapshot_bytes=snapshot_bytes,
            manifest_path=manifest_path,
            manifest_bytes=manifest_bytes,
        )
    assert failure.value.code == SNAPSHOT_PUBLICATION_FAILED


def test_paired_publication_creates_missing_snapshot_when_manifest_already_matches(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    _, _, snapshot_bytes, manifest_bytes = _snapshot_pair(output_path, cache_dir, report_path)
    snapshot_path = tmp_path / "out" / "m1-candidates.v1.json"
    manifest_path = tmp_path / "out" / "m1-candidates.v1.manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_bytes(manifest_bytes)

    _publish_synthetic_pair(
        snapshot_path=snapshot_path,
        snapshot_bytes=snapshot_bytes,
        manifest_path=manifest_path,
        manifest_bytes=manifest_bytes,
    )

    assert snapshot_path.read_bytes() == snapshot_bytes
    assert manifest_path.read_bytes() == manifest_bytes


def test_paired_publication_rejects_cross_directory_targets(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    _, _, snapshot_bytes, manifest_bytes = _snapshot_pair(output_path, cache_dir, report_path)

    with pytest.raises(FreezeGateError) as failure:
        _publish_synthetic_pair(
            snapshot_path=tmp_path / "snapshot" / "m1-candidates.v1.json",
            snapshot_bytes=snapshot_bytes,
            manifest_path=tmp_path / "manifest" / "m1-candidates.v1.manifest.json",
            manifest_bytes=manifest_bytes,
        )
    assert failure.value.code == SNAPSHOT_PUBLICATION_FAILED


def test_snapshot_validator_requires_an_explicit_synthetic_or_production_count(
    synthetic_inputs: tuple[Path, Path, Path]
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    _, manifest, snapshot_bytes, _ = _snapshot_pair(output_path, cache_dir, report_path)

    assert validate_frozen_snapshot_bytes(
        snapshot_bytes, manifest, expected_candidate_count=12
    ) is None
    assert validate_frozen_snapshot_bytes(
        snapshot_bytes, manifest, expected_candidate_count=33
    ) == "frozen snapshot must contain exactly 33 candidates"
    assert validate_frozen_snapshot_bytes(
        snapshot_bytes, manifest, expected_candidate_count=0
    ) == "expected candidate count must be positive"


def test_synthetic_freeze_records_the_explicit_evidence_report_label(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    output_dir = tmp_path / "snapshot"

    result = freeze_candidates(
        m1_output=output_path,
        m1_cache_dir=cache_dir,
        output_dir=output_dir,
        m1_evidence_report=report_path,
        expected_candidate_count=12,
        source_evidence_report_label="evaluation/reports/synthetic-m1-validation.json",
    )

    assert result.status == "success"
    snapshot = json.loads((output_dir / "m1-candidates.v1.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "m1-candidates.v1.manifest.json").read_text(encoding="utf-8"))
    assert snapshot["source_evidence_report"] == "evaluation/reports/synthetic-m1-validation.json"
    assert manifest["source_evidence_report"] == snapshot["source_evidence_report"]

@pytest.mark.parametrize(
    ("snapshot_exists", "manifest_exists"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_paired_publication_handles_every_legal_initial_state(
    synthetic_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
    snapshot_exists: bool,
    manifest_exists: bool,
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    _, _, snapshot_bytes, manifest_bytes = _snapshot_pair(output_path, cache_dir, report_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    snapshot_path = output_dir / "m1-candidates.v1.json"
    manifest_path = output_dir / "m1-candidates.v1.manifest.json"
    if snapshot_exists:
        snapshot_path.write_bytes(snapshot_bytes)
    if manifest_exists:
        manifest_path.write_bytes(manifest_bytes)
    _publish_synthetic_pair(snapshot_path=snapshot_path, snapshot_bytes=snapshot_bytes, manifest_path=manifest_path, manifest_bytes=manifest_bytes)
    assert snapshot_path.read_bytes() == snapshot_bytes
    assert manifest_path.read_bytes() == manifest_bytes
    assert not list(output_dir.glob(".m2-freeze-*"))


def test_paired_publication_rejects_conflicting_manifest_without_touching_snapshot(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    _, _, snapshot_bytes, manifest_bytes = _snapshot_pair(output_path, cache_dir, report_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    snapshot_path = output_dir / "m1-candidates.v1.json"
    manifest_path = output_dir / "m1-candidates.v1.manifest.json"
    snapshot_path.write_bytes(snapshot_bytes)
    manifest_path.write_bytes(b"conflict")
    with pytest.raises(FreezeGateError) as failure:
        _publish_synthetic_pair(snapshot_path=snapshot_path, snapshot_bytes=snapshot_bytes, manifest_path=manifest_path, manifest_bytes=manifest_bytes)
    assert failure.value.code == SNAPSHOT_PUBLICATION_FAILED
    assert snapshot_path.read_bytes() == snapshot_bytes
    assert manifest_path.read_bytes() == b"conflict"


def test_publication_failure_preserves_preexisting_side_and_removes_only_new_side(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    _, _, snapshot_bytes, manifest_bytes = _snapshot_pair(output_path, cache_dir, report_path)
    output_dir = tmp_path / "snapshot-existing"
    output_dir.mkdir()
    snapshot_path = output_dir / "m1-candidates.v1.json"
    manifest_path = output_dir / "m1-candidates.v1.manifest.json"
    snapshot_path.write_bytes(snapshot_bytes)

    def fail_manifest(source: Path, target: Path) -> None:
        if target == manifest_path:
            raise OSError("injected manifest failure")
        os.replace(source, target)

    with pytest.raises(FreezeGateError) as failure:
        _publish_synthetic_pair(snapshot_path=snapshot_path, snapshot_bytes=snapshot_bytes, manifest_path=manifest_path, manifest_bytes=manifest_bytes, replace=fail_manifest)
    assert failure.value.code == SNAPSHOT_PUBLICATION_FAILED
    assert snapshot_path.read_bytes() == snapshot_bytes
    assert not manifest_path.exists()

    output_dir = tmp_path / "manifest-existing"
    output_dir.mkdir()
    snapshot_path = output_dir / "m1-candidates.v1.json"
    manifest_path = output_dir / "m1-candidates.v1.manifest.json"
    manifest_path.write_bytes(manifest_bytes)

    def fail_snapshot(source: Path, target: Path) -> None:
        if target == snapshot_path:
            raise OSError("injected snapshot failure")
        os.replace(source, target)

    with pytest.raises(FreezeGateError) as failure:
        _publish_synthetic_pair(snapshot_path=snapshot_path, snapshot_bytes=snapshot_bytes, manifest_path=manifest_path, manifest_bytes=manifest_bytes, replace=fail_snapshot)
    assert failure.value.code == SNAPSHOT_PUBLICATION_FAILED
    assert not snapshot_path.exists()
    assert manifest_path.read_bytes() == manifest_bytes


@pytest.mark.parametrize(
    "mutation",
    [
        ("source_merge_commit", None, "0" * 40),
        ("source_evidence_baseline_commit", None, "1" * 40),
        ("validated_implementation_commit", None, "2" * 40),
        ("source_evidence_report", None, "evaluation/reports/other.json"),
        ("source_evidence_report_sha256", None, "3" * 64),
        ("source_output_sha256", None, "z" * 64),
        ("source_candidate_array_sha256", None, "g" * 64),
        (None, "source_identity_set_sha256", "6" * 64),
        (None, "candidate_identity_sha256", "7" * 64),
        (None, "source_id_coverage", 0.5),
        (None, "url_coverage", 0.5),
        (None, "metadata_mismatch_count", 1),
        (None, "zero_transport_replay", {"transport_requests": 0, "cache_hits": 11, "query_count": 12}),
        (None, "implementation_ancestry", []),
    ],
)
def test_snapshot_manifest_provenance_mutations_are_rejected(
    synthetic_inputs: tuple[Path, Path, Path],
    mutation: tuple[str | None, str | None, object],
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    snapshot, manifest, _, _ = _snapshot_pair(output_path, cache_dir, report_path)
    snapshot_field, manifest_field, value = mutation
    if snapshot_field is not None:
        snapshot[snapshot_field] = value
    if manifest_field is not None:
        manifest[manifest_field] = value
    snapshot_bytes = _render_json_bytes(snapshot)
    manifest["snapshot_sha256"] = hashlib.sha256(snapshot_bytes).hexdigest()
    assert validate_frozen_snapshot_bytes(snapshot_bytes, manifest, expected_candidate_count=12) is not None


def test_production_embedding_loader_rejects_synthetic_twelve_candidate_snapshot(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    _, _, snapshot_bytes, manifest_bytes = _snapshot_pair(output_path, cache_dir, report_path)
    snapshot_path = tmp_path / "m1-candidates.v1.json"
    manifest_path = tmp_path / "m1-candidates.v1.manifest.json"
    snapshot_path.write_bytes(snapshot_bytes)
    manifest_path.write_bytes(manifest_bytes)
    with pytest.raises(EmbeddingTaskError, match="FROZEN_SNAPSHOT_HASH_MISMATCH"):
        load_validated_frozen_inputs(snapshot_path, manifest_path)


def test_evidence_report_labels_are_posix_and_external_production_paths_are_rejected(
    tmp_path: Path,
) -> None:
    assert _repository_relative_posix_path(Path("evaluation\\reports\\m1-validation.json")) == "evaluation/reports/m1-validation.json"
    with pytest.raises(FreezeGateError):
        _repository_relative_posix_path(tmp_path / "external-m1-validation.json")


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        ("evaluation/reports/m1-validation.json", "evaluation/reports/m1-validation.json"),
        (r"evaluation\reports\m1-validation.json", "evaluation/reports/m1-validation.json"),
        (
            r"evaluation\reports\m1-rebaseline-2026-07-28.json",
            "evaluation/reports/m1-rebaseline-2026-07-28.json",
        ),
    ],
)
def test_evidence_report_labels_normalize_consistently_across_interfaces(
    input_value: str, expected: str
) -> None:
    assert _normalize_repository_relative_label(input_value) == expected
    assert _normalize_evidence_report_label(input_value) == expected
    assert _repository_relative_posix_path(Path(input_value)) == expected


@pytest.mark.parametrize(
    "input_value",
    [
        r"C:\outside\report.json",
        "C:/outside/report.json",
        r"\\server\share\report.json",
        "//server/share/report.json",
        "/tmp/report.json",
        "../report.json",
        "evaluation/../report.json",
        ".",
        "",
        "docs/report.json",
        "evaluation/report.json",
        "evaluation/reports",
        "evaluation/reports/report.txt",
    ],
)
def test_evidence_report_label_contract_rejects_external_or_nonreport_paths(input_value: str) -> None:
    with pytest.raises(FreezeGateError):
        _normalize_repository_relative_label(input_value)
    with pytest.raises(FreezeGateError):
        _normalize_evidence_report_label(input_value)
    with pytest.raises(FreezeGateError):
        _repository_relative_posix_path(Path(input_value))


def test_snapshot_validator_rejects_noncanonical_evidence_report_label(
    synthetic_inputs: tuple[Path, Path, Path],
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    snapshot, manifest, _, _ = _snapshot_pair(output_path, cache_dir, report_path)
    snapshot["source_evidence_report"] = r"evaluation\reports\synthetic-m1-validation.json"
    manifest["source_evidence_report"] = snapshot["source_evidence_report"]
    snapshot_bytes = _render_json_bytes(snapshot)
    manifest["snapshot_sha256"] = hashlib.sha256(snapshot_bytes).hexdigest()
    assert validate_frozen_snapshot_bytes(snapshot_bytes, manifest, expected_candidate_count=12) == (
        "frozen snapshot evidence report path is invalid"
    )


@pytest.mark.parametrize(
    "invalid_label",
    [
        r"C:\outside\report.json",
        "C:/outside/report.json",
        r"\\server\share\report.json",
        "//server/share/report.json",
        "/tmp/report.json",
        "../report.json",
        "evaluation/../report.json",
        ".",
        "",
        "docs/report.json",
        "evaluation/report.json",
        "evaluation/reports",
        "evaluation/reports/report.txt",
    ],
)
def test_snapshot_validator_reuses_the_evidence_report_label_contract(
    synthetic_inputs: tuple[Path, Path, Path], invalid_label: str
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    snapshot, manifest, _, _ = _snapshot_pair(output_path, cache_dir, report_path)
    snapshot["source_evidence_report"] = invalid_label
    manifest["source_evidence_report"] = invalid_label
    snapshot_bytes = _render_json_bytes(snapshot)
    manifest["snapshot_sha256"] = hashlib.sha256(snapshot_bytes).hexdigest()
    assert validate_frozen_snapshot_bytes(snapshot_bytes, manifest, expected_candidate_count=12) == (
        "frozen snapshot evidence report path is invalid"
    )


def test_fully_idempotent_publication_skips_staging_and_preserves_mtimes(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    _, _, snapshot_bytes, manifest_bytes = _snapshot_pair(output_path, cache_dir, report_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    snapshot_path = output_dir / "m1-candidates.v1.json"
    manifest_path = output_dir / "m1-candidates.v1.manifest.json"
    snapshot_path.write_bytes(snapshot_bytes)
    manifest_path.write_bytes(manifest_bytes)
    mtimes = (snapshot_path.stat().st_mtime_ns, manifest_path.stat().st_mtime_ns)

    def fail_stage(*_: object, **__: object) -> str:
        raise AssertionError("fully idempotent publication must not stage")

    monkeypatch.setattr(freeze_module.tempfile, "mkdtemp", fail_stage)
    _publish_synthetic_pair(
        snapshot_path=snapshot_path,
        snapshot_bytes=snapshot_bytes,
        manifest_path=manifest_path,
        manifest_bytes=manifest_bytes,
    )
    assert (snapshot_path.read_bytes(), manifest_path.read_bytes()) == (snapshot_bytes, manifest_bytes)
    assert (snapshot_path.stat().st_mtime_ns, manifest_path.stat().st_mtime_ns) == mtimes
    assert not list(output_dir.glob(".m2-freeze-*"))


def test_paired_publication_rejects_staging_creation_and_readback_validation_failures(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    _, _, snapshot_bytes, manifest_bytes = _snapshot_pair(output_path, cache_dir, report_path)
    snapshot_path = tmp_path / "out" / "m1-candidates.v1.json"
    manifest_path = tmp_path / "out" / "m1-candidates.v1.manifest.json"

    def fail_stage(*_: object, **__: object) -> str:
        raise OSError("injected staging creation failure")

    monkeypatch.setattr(freeze_module.tempfile, "mkdtemp", fail_stage)
    with pytest.raises(FreezeGateError) as failure:
        _publish_synthetic_pair(snapshot_path=snapshot_path, snapshot_bytes=snapshot_bytes, manifest_path=manifest_path, manifest_bytes=manifest_bytes)
    assert failure.value.code == SNAPSHOT_PUBLICATION_FAILED
    assert not snapshot_path.exists() and not manifest_path.exists()

    monkeypatch.undo()
    with pytest.raises(FreezeGateError) as failure:
        _publish_synthetic_pair(snapshot_path=snapshot_path, snapshot_bytes=snapshot_bytes, manifest_path=manifest_path, manifest_bytes=b"{}")
    assert failure.value.code == SNAPSHOT_PUBLICATION_FAILED
    assert not snapshot_path.exists() and not manifest_path.exists()
    assert not list(tmp_path.rglob(".m2-freeze-*"))


def test_paired_publication_rejects_existing_directory_targets(
    synthetic_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output_path, cache_dir, report_path = synthetic_inputs
    _, _, snapshot_bytes, manifest_bytes = _snapshot_pair(output_path, cache_dir, report_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    snapshot_path = output_dir / "m1-candidates.v1.json"
    snapshot_path.mkdir()
    manifest_path = output_dir / "m1-candidates.v1.manifest.json"

    with pytest.raises(FreezeGateError) as failure:
        _publish_synthetic_pair(snapshot_path=snapshot_path, snapshot_bytes=snapshot_bytes, manifest_path=manifest_path, manifest_bytes=manifest_bytes)
    assert failure.value.code == SNAPSHOT_PUBLICATION_FAILED
