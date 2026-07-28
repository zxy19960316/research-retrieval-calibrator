"""Fail-closed M2-T01 freeze gate for recovered M1 real-source artifacts.

The command stays blocked without the historical M1 output/cache pair.  When
they are available, it replays the accepted real cache without transport,
checks canonical M1 metadata, and publishes an exact-byte snapshot/manifest
pair.  This task creates no historical artifact by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, TypeGuard

if TYPE_CHECKING:
    from app.adapters.arxiv import ArxivResponse
    from app.models.first_round import CandidateOutput
    from app.models.paper import PaperRecord

M1_COMPLETION_MERGE_COMMIT = "0eb45fc22d10adb72cb66aa45494333057080bc1"
ROOT = Path(__file__).resolve().parents[1]

FROZEN_INPUT_MISSING = "M2_T01_FROZEN_INPUT_MISSING"
FROZEN_INPUT_INVALID = "M2_T01_FROZEN_INPUT_INVALID"
ZERO_TRANSPORT_REPLAY_FAILED = "M2_T01_ZERO_TRANSPORT_REPLAY_FAILED"
REAL_CACHE_INCOMPLETE = "M2_T01_REAL_CACHE_INCOMPLETE"
REAL_CACHE_PROVENANCE_MISMATCH = "M2_T01_REAL_CACHE_PROVENANCE_MISMATCH"
M1_METADATA_MISMATCH = "M2_T01_M1_METADATA_MISMATCH"
SNAPSHOT_PUBLICATION_FAILED = "M2_T01_SNAPSHOT_PUBLICATION_FAILED"

FREEZE_GATE_CODES = frozenset(
    {
        FROZEN_INPUT_MISSING,
        FROZEN_INPUT_INVALID,
        ZERO_TRANSPORT_REPLAY_FAILED,
        REAL_CACHE_INCOMPLETE,
        REAL_CACHE_PROVENANCE_MISMATCH,
        M1_METADATA_MISMATCH,
        SNAPSHOT_PUBLICATION_FAILED,
    }
)


class FreezeGateError(RuntimeError):
    """A stable, explicit M2-T01 freeze-gate classification."""

    def __init__(self, code: str, reason: str) -> None:
        if code not in FREEZE_GATE_CODES:
            raise ValueError(f"Unknown M2-T01 freeze gate code: {code}")
        self.code = code
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class AcceptedM1Provenance:
    """Immutable provenance accepted by the completed M1 evidence report."""

    output_sha256: str
    candidate_array_sha256: str
    candidate_count: int
    source_id_coverage: float
    url_coverage: float
    m1_completion_merge_commit: str
    m1_evidence_baseline_commit: str
    validated_implementation_commit: str
    implementation_ancestry: tuple[str, ...]


@dataclass(frozen=True)
class FreezeResult:
    """Structured result for the freeze gate and its paired publication."""

    status: str
    error_code: str | None = None
    reason: str | None = None
    snapshot_path: str | None = None
    manifest_path: str | None = None
    snapshot_sha256: str | None = None
    candidate_count: int | None = None

    def as_dict(self) -> dict[str, object | None]:
        result: dict[str, object | None] = {
            "candidate_count": self.candidate_count,
            "error_code": self.error_code,
            "manifest_path": self.manifest_path,
            "reason": self.reason,
            "snapshot_path": self.snapshot_path,
            "snapshot_sha256": self.snapshot_sha256,
            "status": self.status,
        }
        return {key: value for key, value in result.items() if value is not None}


@dataclass(frozen=True)
class ExistingPublicationState:
    """Existing publication bytes and the exact targets this call may create."""

    snapshot_bytes: bytes | None
    manifest_bytes: bytes | None
    snapshot_needs_publish: bool
    manifest_needs_publish: bool


def _is_hex_sha(value: object, *, length: int) -> TypeGuard[str]:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def load_accepted_m1_provenance(
    report_path: Path, *, required_candidate_count: int = 33
) -> AcceptedM1Provenance:
    """Load accepted evidence facts without treating its baseline as M1 completion."""

    if required_candidate_count < 1:
        raise ValueError("required candidate count must be positive")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("M1 evidence report is unavailable or invalid") from error
    if not isinstance(payload, Mapping):
        raise TypeError("M1 evidence report must be an object")
    real = payload.get("real_external")
    if not isinstance(real, Mapping) or real.get("classification") != "real_external":
        raise ValueError("M1 evidence report must contain accepted real_external provenance")
    hashes = real.get("output_hashes")
    output_sha256 = hashes.get("first-round.json") if isinstance(hashes, Mapping) else None
    candidate_array_sha256 = real.get("candidate_array_sha256")
    candidate_count = real.get("deduplicated_candidate_count")
    source_id_coverage = real.get("source_id_coverage")
    url_coverage = real.get("url_coverage")
    evidence_baseline = payload.get("baseline_commit")
    validated_implementation = payload.get("validated_implementation_commit")
    ancestry = payload.get("implementation_ancestry")

    if not _is_hex_sha(M1_COMPLETION_MERGE_COMMIT, length=40):
        raise ValueError("configured M1 completion merge commit must be a 40-character SHA")
    if not _is_hex_sha(output_sha256, length=64) or not _is_hex_sha(
        candidate_array_sha256, length=64
    ):
        raise ValueError("M1 evidence report lacks valid accepted SHA-256 output provenance")
    if not _is_hex_sha(evidence_baseline, length=40) or not _is_hex_sha(
        validated_implementation, length=40
    ):
        raise ValueError("M1 evidence report lacks valid completion provenance")
    if (
        not isinstance(ancestry, list)
        or not all(_is_hex_sha(commit, length=40) for commit in ancestry)
        or validated_implementation not in ancestry
    ):
        raise ValueError("M1 evidence report lacks validated implementation ancestry")
    if (
        not isinstance(candidate_count, int)
        or isinstance(candidate_count, bool)
        or not isinstance(source_id_coverage, float)
        or not isinstance(url_coverage, float)
    ):
        raise TypeError("M1 evidence report lacks complete accepted output provenance")
    if (
        candidate_count != required_candidate_count
        or source_id_coverage != 1.0
        or url_coverage != 1.0
    ):
        raise ValueError(
            f"M1 evidence report does not satisfy accepted {required_candidate_count}-candidate coverage"
        )
    return AcceptedM1Provenance(
        output_sha256=output_sha256,
        candidate_array_sha256=candidate_array_sha256,
        candidate_count=candidate_count,
        source_id_coverage=source_id_coverage,
        url_coverage=url_coverage,
        m1_completion_merge_commit=M1_COMPLETION_MERGE_COMMIT,
        m1_evidence_baseline_commit=evidence_baseline,
        validated_implementation_commit=validated_implementation,
        implementation_ancestry=tuple(ancestry),
    )


class NetworkForbiddenTransport:
    """Transport boundary that turns a cache miss into a stable freeze failure."""

    def __init__(self) -> None:
        self.request_count = 0

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> ArxivResponse:
        del url, headers, timeout_seconds
        self.request_count += 1
        raise FreezeGateError(
            ZERO_TRANSPORT_REPLAY_FAILED,
            "network transport was invoked during cache-only replay",
        )


def _canonical_json_sha256(value: object) -> str:
    """Hash semantic JSON only for M1 evidence-derived identity fields."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _render_json_bytes(value: object) -> bytes:
    """Render the deterministic file representation covered by snapshot SHA-256."""

    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _parse_snapshot_bytes(snapshot_bytes: bytes) -> dict[str, object] | None:
    try:
        snapshot = json.loads(snapshot_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return snapshot if isinstance(snapshot, dict) else None


def validate_frozen_snapshot_bytes(
    snapshot_bytes: bytes,
    manifest: Mapping[str, object],
    *,
    expected_candidate_count: int,
) -> str | None:
    """Validate the exact file bytes and all closed snapshot/manifest invariants."""

    from app.models.embedding import FrozenCandidate

    if expected_candidate_count < 1:
        return "expected candidate count must be positive"
    snapshot = _parse_snapshot_bytes(snapshot_bytes)
    if snapshot is None:
        return "frozen snapshot file is not a JSON object"
    candidates = snapshot.get("candidates")
    if snapshot.get("snapshot_version") != "m2-candidates.v1" or not isinstance(
        candidates, list
    ):
        return "frozen snapshot has an invalid version or candidate collection"
    try:
        frozen = [FrozenCandidate.model_validate(candidate) for candidate in candidates]
    except (TypeError, ValueError):
        return "frozen snapshot contains an invalid source-backed candidate"
    if (
        len(frozen) != expected_candidate_count
        or snapshot.get("count") != expected_candidate_count
        or manifest.get("candidate_count") != expected_candidate_count
    ):
        return f"frozen snapshot must contain exactly {expected_candidate_count} candidates"
    paper_ids = [candidate.paper_id for candidate in frozen]
    if paper_ids != sorted(paper_ids):
        return "frozen snapshot candidates must be ordered by paper_id"
    if len(paper_ids) != len(set(paper_ids)):
        return "frozen snapshot has duplicate paper IDs"
    source_identities = [(candidate.source, candidate.source_id) for candidate in frozen]
    if len(source_identities) != len(set(source_identities)):
        return "frozen snapshot has duplicate source identities"
    if not _is_hex_sha(manifest.get("snapshot_sha256"), length=64):
        return "frozen snapshot manifest has an invalid snapshot SHA-256"
    if manifest.get("snapshot_sha256") != hashlib.sha256(snapshot_bytes).hexdigest():
        return "frozen snapshot manifest SHA-256 does not match exact snapshot bytes"
    provenance_fields = (
        ("source_merge_commit", "m1_completion_merge_commit", 40),
        ("source_evidence_baseline_commit", "m1_evidence_baseline_commit", 40),
        ("validated_implementation_commit", "validated_implementation_commit", 40),
        ("source_evidence_report_sha256", "source_evidence_report_sha256", 64),
    )
    for snapshot_key, manifest_key, length in provenance_fields:
        snapshot_value = snapshot.get(snapshot_key)
        manifest_value = manifest.get(manifest_key)
        if not _is_hex_sha(snapshot_value, length=length) or not _is_hex_sha(
            manifest_value, length=length
        ):
            return f"frozen snapshot has invalid {snapshot_key} provenance"
        if snapshot_value != manifest_value:
            return f"frozen snapshot and manifest {snapshot_key} provenance do not match"
    source_report = snapshot.get("source_evidence_report")
    manifest_report = manifest.get("source_evidence_report")
    if (
        not isinstance(source_report, str)
        or not source_report
        or "\\" in source_report
        or PurePosixPath(source_report).is_absolute()
        or ".." in PurePosixPath(source_report).parts
        or source_report != manifest_report
    ):
        return "frozen snapshot and manifest evidence report path do not match"
    source_hashes = (
        "source_evidence_report_sha256",
        "source_output_sha256",
        "source_candidate_array_sha256",
    )
    if not all(_is_hex_sha(snapshot.get(field), length=64) for field in source_hashes):
        return "frozen snapshot has invalid source SHA-256 provenance"
    ancestry = manifest.get("implementation_ancestry")
    if (
        not isinstance(ancestry, list)
        or not ancestry
        or not all(_is_hex_sha(commit, length=40) for commit in ancestry)
        or manifest.get("validated_implementation_commit") not in ancestry
    ):
        return "frozen snapshot manifest has invalid implementation ancestry"
    if (
        manifest.get("source_id_coverage") != 1.0
        or manifest.get("url_coverage") != 1.0
        or manifest.get("metadata_mismatch_count") != 0
    ):
        return "frozen snapshot manifest has invalid coverage or metadata audit"
    replay = manifest.get("zero_transport_replay")
    if (
        not isinstance(replay, Mapping)
        or replay.get("transport_requests") != 0
        or replay.get("cache_hits") != replay.get("query_count")
        or replay.get("query_count") != 12
    ):
        return "frozen snapshot manifest lacks a complete zero-transport replay audit"
    identities = [(item.source, item.source_id) for item in frozen]
    expected_source_identity = _canonical_json_sha256(sorted(identities))
    expected_candidate_identity = _canonical_json_sha256(
        [(item.paper_id, item.source, item.source_id) for item in frozen]
    )
    if not _is_hex_sha(manifest.get("source_identity_set_sha256"), length=64):
        return "frozen snapshot manifest has an invalid source identity SHA-256"
    if manifest.get("source_identity_set_sha256") != expected_source_identity:
        return "frozen snapshot manifest source identity hash does not match candidates"
    if not _is_hex_sha(manifest.get("candidate_identity_sha256"), length=64):
        return "frozen snapshot manifest has an invalid candidate identity SHA-256"
    if manifest.get("candidate_identity_sha256") != expected_candidate_identity:
        return "frozen snapshot manifest candidate identity hash does not match candidates"
    return None


def _validate_cache_entries(
    adapter: Any, run: Any, m1_cache_dir: Path
) -> None:
    """Classify every expected persistent-cache entry before invoking the adapter."""

    from app.models.paper import PaperRecord

    for query in run.query_plan.queries:
        path = adapter._persistent_cache_path(query.query_text, run.config.max_results_per_query)
        if path is None or not path.is_file():
            raise FreezeGateError(
                REAL_CACHE_INCOMPLETE, f"cache entry missing for query {query.query_id}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FreezeGateError(
                REAL_CACHE_PROVENANCE_MISMATCH,
                f"cache entry is unreadable for query {query.query_id}",
            ) from error
        if not isinstance(payload, Mapping) or payload.get("manifest") != adapter._cache_manifest(
            query.query_text, run.config.max_results_per_query
        ):
            raise FreezeGateError(
                REAL_CACHE_PROVENANCE_MISMATCH,
                f"cache manifest provenance mismatch for query {query.query_id}",
            )
        records = payload.get("records")
        if not isinstance(records, list):
            raise FreezeGateError(
                REAL_CACHE_PROVENANCE_MISMATCH,
                f"cache records are invalid for query {query.query_id}",
            )
        try:
            [PaperRecord.model_validate(record) for record in records]
        except (TypeError, ValueError) as error:
            raise FreezeGateError(
                REAL_CACHE_PROVENANCE_MISMATCH,
                f"cache record provenance mismatch for query {query.query_id}",
            ) from error
        if not records:
            raise FreezeGateError(
                REAL_CACHE_INCOMPLETE, f"cache entry has no records for query {query.query_id}"
            )


def _metadata_mismatch_field(candidate: CandidateOutput, cluster: Any) -> str | None:
    """Return the first failed legacy M1 metadata projection field, if any."""

    record: PaperRecord = cluster.canonical_record
    expected: dict[str, object] = {
        "paper_id": record.paper_id,
        "source": record.source,
        "source_id": record.source_id,
        "title": record.title,
        "authors": record.authors,
        "year": record.year,
        "doi": record.doi,
        "url": record.url,
        "retrieval_paths": cluster.retrieval_paths,
        "cluster_id": cluster.cluster_id,
        "member_source_identities": cluster.source_identities,
        "merge_reasons": cluster.merge_reasons,
    }
    for field_name, expected_value in expected.items():
        if getattr(candidate, field_name) != expected_value:
            return field_name
    return None


def _project_abstract(abstract: str | None) -> str | None:
    """Copy canonical source abstract without placeholders, translation, or summary."""

    return abstract if abstract is not None and abstract.strip() else None


def _replay_and_project(
    raw_output: bytes,
    *,
    m1_cache_dir: Path,
    provenance: AcceptedM1Provenance,
    evidence_report_sha256: str,
    source_evidence_report: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Replay twelve real-cache queries and project only canonical source fields."""

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app.adapters.arxiv import ArxivAdapter, ArxivAdapterConfig, ArxivAdapterError
    from app.core.paper_dedup import deduplicate_papers
    from app.models.embedding import FrozenCandidate
    from app.models.first_round import FirstRoundRun, FirstRoundStatus

    try:
        run = FirstRoundRun.model_validate(json.loads(raw_output))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise FreezeGateError(FROZEN_INPUT_INVALID, "M1 output is not a valid first-round run") from error
    if run.status is not FirstRoundStatus.SUCCESS or run.config.mode != "real":
        raise FreezeGateError(FROZEN_INPUT_INVALID, "M1 output is not a successful real run")
    if run.query_plan is None or len(run.query_plan.queries) != 12:
        raise FreezeGateError(FROZEN_INPUT_INVALID, "M1 output must contain exactly 12 planned queries")
    if (
        run.config.cache_namespace != "first-round:real"
        or run.config.adapter_schema_version != "m1-t04.v2"
    ):
        raise FreezeGateError(
            REAL_CACHE_PROVENANCE_MISMATCH,
            "M1 output cache namespace or adapter schema does not match accepted real cache",
        )
    transport = NetworkForbiddenTransport()
    adapter = ArxivAdapter(
        ArxivAdapterConfig(
            user_agent="research-retrieval-calibrator/0.1 (M2-T01R2 freeze replay)",
            timeout_seconds=run.config.timeout_seconds,
            page_size=run.config.max_results_per_query,
            min_request_interval_seconds=run.config.min_request_interval_seconds,
            max_attempts=1,
            max_total_results=run.config.max_results_per_query,
            max_total_attempts=run.config.max_total_attempts,
            initial_backoff_seconds=0.0,
            cache_dir=m1_cache_dir,
            cache_schema_version=run.config.adapter_schema_version,
            cache_namespace=run.config.cache_namespace,
        ),
        transport=transport,
        sleeper=lambda _: None,
    )
    _validate_cache_entries(adapter, run, m1_cache_dir)
    records = []
    cache_hits = 0
    try:
        for query in run.query_plan.queries:
            records.extend(adapter.search(query, max_results=run.config.max_results_per_query))
            cache_hits += int(adapter.last_observation.cache_hit)
    except FreezeGateError:
        raise
    except ArxivAdapterError as error:
        code = (
            ZERO_TRANSPORT_REPLAY_FAILED
            if transport.request_count > 0
            else REAL_CACHE_PROVENANCE_MISMATCH
        )
        raise FreezeGateError(code, f"cache-only replay failed: {error.code}") from error
    if transport.request_count != 0:
        raise FreezeGateError(
            ZERO_TRANSPORT_REPLAY_FAILED,
            "network transport was invoked during cache-only replay",
        )
    if cache_hits != len(run.query_plan.queries):
        raise FreezeGateError(
            REAL_CACHE_INCOMPLETE,
            f"cache replay produced {cache_hits} hits for {len(run.query_plan.queries)} queries",
        )
    try:
        deduplicated = deduplicate_papers(records)
    except ValueError as error:
        raise FreezeGateError(
            REAL_CACHE_PROVENANCE_MISMATCH, "cached records cannot be deduplicated safely"
        ) from error
    if len(records) != run.raw_candidate_count:
        raise FreezeGateError(
            REAL_CACHE_INCOMPLETE,
            f"cached raw count {len(records)} does not match M1 output {run.raw_candidate_count}",
        )
    if len(deduplicated.clusters) != provenance.candidate_count:
        raise FreezeGateError(
            REAL_CACHE_INCOMPLETE,
            "cached deduplicated count does not match accepted M1 evidence",
        )
    historical = {
        (candidate.paper_id, candidate.source, candidate.source_id): candidate
        for candidate in run.candidates
    }
    frozen: list[FrozenCandidate] = []
    for cluster in deduplicated.clusters:
        record = cluster.canonical_record
        key = (record.paper_id, record.source, record.source_id)
        candidate = historical.get(key)
        mismatch = None if candidate is None else _metadata_mismatch_field(candidate, cluster)
        if candidate is None or mismatch is not None:
            field = "candidate identity" if candidate is None else mismatch
            raise FreezeGateError(
                M1_METADATA_MISMATCH, f"M1 metadata mismatch for {field} on {record.paper_id}"
            )
        frozen.append(
            FrozenCandidate(
                paper_id=record.paper_id,
                source=record.source,
                source_id=record.source_id,
                title=record.title,
                abstract=_project_abstract(record.abstract),
                authors=record.authors,
                year=record.year,
                doi=record.doi,
                url=record.url,
                language=record.language,
                categories=[],
                retrieval_paths=cluster.retrieval_paths,
                cluster_id=cluster.cluster_id,
                member_source_identities=cluster.source_identities,
            )
        )
    if len(historical) != len(frozen):
        raise FreezeGateError(M1_METADATA_MISMATCH, "M1 candidate identities do not match replay")
    candidates = [
        candidate.model_dump(mode="json") for candidate in sorted(frozen, key=lambda item: item.paper_id)
    ]
    snapshot = {
        "snapshot_version": "m2-candidates.v1",
        "source_phase": "M1",
        "source_merge_commit": provenance.m1_completion_merge_commit,
        "source_evidence_baseline_commit": provenance.m1_evidence_baseline_commit,
        "validated_implementation_commit": provenance.validated_implementation_commit,
        "source_evidence_report": source_evidence_report,
        "source_evidence_report_sha256": evidence_report_sha256,
        "source_output_sha256": provenance.output_sha256,
        "source_candidate_array_sha256": provenance.candidate_array_sha256,
        "question": run.question,
        "count": len(candidates),
        "candidates": candidates,
    }
    manifest = {
        "candidate_count": len(candidates),
        "source_id_coverage": provenance.source_id_coverage,
        "url_coverage": provenance.url_coverage,
        "source_evidence_report": source_evidence_report,
        "source_evidence_report_sha256": evidence_report_sha256,
        "m1_completion_merge_commit": provenance.m1_completion_merge_commit,
        "m1_evidence_baseline_commit": provenance.m1_evidence_baseline_commit,
        "validated_implementation_commit": provenance.validated_implementation_commit,
        "implementation_ancestry": list(provenance.implementation_ancestry),
        "paper_id_order": [candidate["paper_id"] for candidate in candidates],
        "source_identity_set_sha256": _canonical_json_sha256(
            sorted((item["source"], item["source_id"]) for item in candidates)
        ),
        "candidate_identity_sha256": _canonical_json_sha256(
            [(item["paper_id"], item["source"], item["source_id"]) for item in candidates]
        ),
        "abstract_present_count": sum(item["abstract"] is not None for item in candidates),
        "abstract_missing_count": sum(item["abstract"] is None for item in candidates),
        "metadata_mismatch_count": 0,
        "zero_transport_replay": {
            "transport_requests": transport.request_count,
            "cache_hits": cache_hits,
            "query_count": len(run.query_plan.queries),
        },
        "artifact_recovery_classification": "EXACT_HISTORICAL_ARTIFACTS_RECOVERED",
    }
    return snapshot, manifest


def validate_m1_freeze_input(
    raw_output: bytes,
    *,
    expected_output_sha256: str,
    expected_candidate_array_sha256: str,
    expected_candidate_count: int,
) -> str | None:
    """Validate explicit evidence-derived M1 hashes and count without any I/O."""

    if (
        not _is_hex_sha(expected_output_sha256, length=64)
        or not _is_hex_sha(expected_candidate_array_sha256, length=64)
    ):
        return "accepted M1 evidence contains an invalid SHA-256 value"
    if expected_candidate_count < 1:
        return "accepted M1 evidence contains an invalid candidate count"
    if hashlib.sha256(raw_output).hexdigest() != expected_output_sha256:
        return "M1 first-round output SHA-256 does not match accepted evidence"
    try:
        payload = json.loads(raw_output)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return "M1 first-round output is not valid JSON"
    if not isinstance(payload, dict):
        return "M1 first-round output must be a JSON object"
    config = payload.get("config")
    candidates = payload.get("candidates")
    if not isinstance(config, dict) or config.get("mode") != "real":
        return "M1 output is not from real mode"
    if not isinstance(candidates, list) or len(candidates) != expected_candidate_count:
        return f"M1 output must contain exactly {expected_candidate_count} candidates"
    if _canonical_json_sha256(candidates) != expected_candidate_array_sha256:
        return "M1 candidate-array SHA-256 does not match accepted evidence"
    return None


def _read_existing_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes() if path.exists() else None
    except OSError as error:
        raise FreezeGateError(
            SNAPSHOT_PUBLICATION_FAILED, f"cannot read existing publication target {path.name}"
        ) from error


def _existing_publication_state(
    snapshot_path: Path,
    snapshot_bytes: bytes,
    manifest_path: Path,
    manifest_bytes: bytes,
) -> ExistingPublicationState:
    snapshot_existing = _read_existing_bytes(snapshot_path)
    manifest_existing = _read_existing_bytes(manifest_path)
    if snapshot_existing not in {None, snapshot_bytes}:
        raise FreezeGateError(SNAPSHOT_PUBLICATION_FAILED, "snapshot target already has different bytes")
    if manifest_existing not in {None, manifest_bytes}:
        raise FreezeGateError(SNAPSHOT_PUBLICATION_FAILED, "manifest target already has different bytes")
    return ExistingPublicationState(
        snapshot_bytes=snapshot_existing,
        manifest_bytes=manifest_existing,
        snapshot_needs_publish=snapshot_existing is None,
        manifest_needs_publish=manifest_existing is None,
    )


def _publish_snapshot_pair(
    *,
    snapshot_path: Path,
    snapshot_bytes: bytes,
    manifest_path: Path,
    manifest_bytes: bytes,
    expected_candidate_count: int,
    replace: Callable[[Path, Path], None] = os.replace,
) -> None:
    """Publish verified snapshot/manifest bytes together or restore caller-visible state."""

    if snapshot_path.resolve() == manifest_path.resolve():
        raise FreezeGateError(SNAPSHOT_PUBLICATION_FAILED, "snapshot and manifest paths must differ")
    if snapshot_path.parent.resolve() != manifest_path.parent.resolve():
        raise FreezeGateError(
            SNAPSHOT_PUBLICATION_FAILED,
            "snapshot and manifest must share one publication directory",
        )
    state = _existing_publication_state(
        snapshot_path, snapshot_bytes, manifest_path, manifest_bytes
    )
    try:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".m2-freeze-", dir=snapshot_path.parent))
    except OSError as error:
        raise FreezeGateError(SNAPSHOT_PUBLICATION_FAILED, "cannot create publication staging") from error
    staged_snapshot = stage / snapshot_path.name
    staged_manifest = stage / manifest_path.name
    snapshot_published_this_run = False
    manifest_published_this_run = False
    completed = False
    try:
        staged_snapshot.write_bytes(snapshot_bytes)
        staged_manifest.write_bytes(manifest_bytes)
        staged_manifest_payload = json.loads(staged_manifest.read_text(encoding="utf-8"))
        if not isinstance(staged_manifest_payload, Mapping):
            raise FreezeGateError(SNAPSHOT_PUBLICATION_FAILED, "staged manifest is not an object")
        validation_error = validate_frozen_snapshot_bytes(
            staged_snapshot.read_bytes(),
            staged_manifest_payload,
            expected_candidate_count=expected_candidate_count,
        )
        if validation_error is not None:
            raise FreezeGateError(SNAPSHOT_PUBLICATION_FAILED, validation_error)
        if state.snapshot_needs_publish:
            replace(staged_snapshot, snapshot_path)
            snapshot_published_this_run = True
        if state.manifest_needs_publish:
            replace(staged_manifest, manifest_path)
            manifest_published_this_run = True
        if snapshot_path.read_bytes() != snapshot_bytes or manifest_path.read_bytes() != manifest_bytes:
            raise FreezeGateError(
                SNAPSHOT_PUBLICATION_FAILED,
                "published snapshot pair does not match staged bytes",
            )
        completed = True
    except FreezeGateError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreezeGateError(SNAPSHOT_PUBLICATION_FAILED, "paired snapshot publication failed") from error
    finally:
        if not completed and manifest_published_this_run:
            try:
                manifest_path.unlink(missing_ok=True)
            except OSError:
                pass
        if not completed and snapshot_published_this_run:
            try:
                snapshot_path.unlink(missing_ok=True)
            except OSError:
                pass
        shutil.rmtree(stage, ignore_errors=True)


def _repository_relative_posix_path(path: Path) -> str:
    """Return a repository-relative POSIX evidence path or reject external input."""

    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise FreezeGateError(
            FROZEN_INPUT_INVALID,
            "M1 evidence report must be inside the repository",
        ) from error


def _normalize_evidence_report_label(label: str) -> str:
    normalized = label.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise FreezeGateError(FROZEN_INPUT_INVALID, "M1 evidence report label must be relative")
    return path.as_posix()


def freeze_candidates(
    *,
    m1_output: Path,
    m1_cache_dir: Path,
    output_dir: Path,
    m1_evidence_report: Path = Path("evaluation/reports/m1-validation.json"),
    expected_candidate_count: int | None = None,
    source_evidence_report_label: str | None = None,
) -> FreezeResult:
    """Freeze only an accepted, cache-replayable M1 source; otherwise fail closed."""

    required_count = 33 if expected_candidate_count is None else expected_candidate_count
    try:
        provenance = load_accepted_m1_provenance(
            m1_evidence_report, required_candidate_count=required_count
        )
    except (TypeError, ValueError) as error:
        return FreezeResult(status="blocked", error_code=FROZEN_INPUT_INVALID, reason=str(error))
    if not m1_output.is_file() or not m1_cache_dir.is_dir():
        return FreezeResult(
            status="blocked",
            error_code=FROZEN_INPUT_MISSING,
            reason="accepted M1 output and first-round:real cache must both be present",
        )
    try:
        raw_output = m1_output.read_bytes()
    except OSError:
        return FreezeResult(
            status="blocked",
            error_code=FROZEN_INPUT_MISSING,
            reason="accepted M1 first-round output is unavailable",
        )
    validation_error = validate_m1_freeze_input(
        raw_output,
        expected_output_sha256=provenance.output_sha256,
        expected_candidate_array_sha256=provenance.candidate_array_sha256,
        expected_candidate_count=provenance.candidate_count,
    )
    if validation_error is not None:
        return FreezeResult(status="blocked", error_code=FROZEN_INPUT_INVALID, reason=validation_error)
    try:
        source_evidence_report = (
            _repository_relative_posix_path(m1_evidence_report)
            if source_evidence_report_label is None
            else _normalize_evidence_report_label(source_evidence_report_label)
        )
        snapshot, manifest = _replay_and_project(
            raw_output,
            m1_cache_dir=m1_cache_dir,
            provenance=provenance,
            evidence_report_sha256=hashlib.sha256(m1_evidence_report.read_bytes()).hexdigest(),
            source_evidence_report=source_evidence_report,
        )
        snapshot_bytes = _render_json_bytes(snapshot)
        snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
        manifest["snapshot_sha256"] = snapshot_sha256
        manifest_bytes = _render_json_bytes(manifest)
        snapshot_path = output_dir / "m1-candidates.v1.json"
        manifest_path = output_dir / "m1-candidates.v1.manifest.json"
        _publish_snapshot_pair(
            snapshot_path=snapshot_path,
            snapshot_bytes=snapshot_bytes,
            manifest_path=manifest_path,
            manifest_bytes=manifest_bytes,
            expected_candidate_count=provenance.candidate_count,
        )
    except FreezeGateError as error:
        return FreezeResult(status="blocked", error_code=error.code, reason=error.reason)
    except OSError as error:
        return FreezeResult(
            status="blocked", error_code=SNAPSHOT_PUBLICATION_FAILED, reason=str(error)
        )
    return FreezeResult(
        status="success",
        snapshot_path=snapshot_path.as_posix(),
        manifest_path=manifest_path.as_posix(),
        snapshot_sha256=snapshot_sha256,
        candidate_count=provenance.candidate_count,
    )


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-output", type=Path, required=True)
    parser.add_argument("--m1-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--m1-evidence-report",
        type=Path,
        default=Path("evaluation/reports/m1-validation.json"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    result = freeze_candidates(
        m1_output=arguments.m1_output,
        m1_cache_dir=arguments.m1_cache_dir,
        output_dir=arguments.output_dir,
        m1_evidence_report=arguments.m1_evidence_report,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0 if result.status == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
