"""Fail-closed M2-T01 gate for the historical M1 real-source artifacts.

This initial gate deliberately does *not* replay the M1 cache or write an M2
snapshot.  It gives the missing historical source a stable, machine-readable
outcome and exposes a pure provenance validator for the later replay step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.adapters.arxiv import ArxivResponse

M1_OUTPUT_SHA256 = "069cd8c94d294c68bb051898d4e262f324b7e1f20eff10eabcf22d9bc435d178"
M1_CANDIDATE_ARRAY_SHA256 = "e51eb84d4e772bba324a199478caa5f0983edef0b0dbf4c58fce5f9da5803209"
FROZEN_INPUT_MISSING = "M2_T01_FROZEN_INPUT_MISSING"
FROZEN_INPUT_INVALID = "M2_T01_FROZEN_INPUT_INVALID"


@dataclass(frozen=True)
class AcceptedM1Provenance:
    """The immutable M1 facts accepted by the completed M1 evidence report."""

    output_sha256: str
    candidate_array_sha256: str
    candidate_count: int
    source_id_coverage: float
    url_coverage: float
    source_merge_commit: str


def load_accepted_m1_provenance(report_path: Path) -> AcceptedM1Provenance:
    """Load only the accepted real M1 output provenance from an evidence report."""

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
    source_merge_commit = payload.get("baseline_commit")
    if (
        not isinstance(output_sha256, str)
        or len(output_sha256) != 64
        or not isinstance(candidate_array_sha256, str)
        or len(candidate_array_sha256) != 64
        or not isinstance(candidate_count, int)
        or isinstance(candidate_count, bool)
        or not isinstance(source_id_coverage, float)
        or not isinstance(url_coverage, float)
        or not isinstance(source_merge_commit, str)
        or len(source_merge_commit) != 40
    ):
        raise ValueError("M1 evidence report lacks complete accepted output provenance")
    if candidate_count != 33 or source_id_coverage != 1.0 or url_coverage != 1.0:
        raise ValueError("M1 evidence report does not satisfy accepted 33-candidate coverage")
    return AcceptedM1Provenance(
        output_sha256=output_sha256,
        candidate_array_sha256=candidate_array_sha256,
        candidate_count=candidate_count,
        source_id_coverage=source_id_coverage,
        url_coverage=url_coverage,
        source_merge_commit=source_merge_commit,
    )


@dataclass(frozen=True)
class FreezeResult:
    """Structured result for the no-write precondition gate."""

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


class NetworkForbiddenTransport:
    """A replay transport that proves no cache miss can reach the network."""

    def __init__(self) -> None:
        self.request_count = 0

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> ArxivResponse:
        del url, headers, timeout_seconds
        self.request_count += 1
        raise AssertionError("network transport forbidden during M2 freeze replay")


def _canonical_json_sha256(value: object) -> str:
    """Hash the canonical JSON representation used by the M1 evidence."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _render_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, contents: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(contents)
    temporary.replace(path)


def validate_frozen_snapshot_payload(
    snapshot: dict[str, object], manifest: dict[str, object]
) -> str | None:
    """Purely validate a generated snapshot and manifest before any write.

    This compatibility boundary deliberately validates supplied values only;
    it neither reads the M1 cache nor projects candidates from it.
    """

    from app.models.embedding import FrozenCandidate

    candidates = snapshot.get("candidates")
    if snapshot.get("snapshot_version") != "m2-candidates.v1" or not isinstance(
        candidates, list
    ):
        return "frozen snapshot has an invalid version or candidate collection"
    try:
        frozen = [FrozenCandidate.model_validate(candidate) for candidate in candidates]
    except (TypeError, ValueError):
        return "frozen snapshot contains an invalid source-backed candidate"
    if len(frozen) != 33 or snapshot.get("count") != 33:
        return "frozen snapshot must contain exactly 33 candidates"
    paper_ids = [candidate.paper_id for candidate in frozen]
    if paper_ids != sorted(paper_ids):
        return "frozen snapshot candidates must be ordered by paper_id"
    if len(paper_ids) != len(set(paper_ids)):
        return "frozen snapshot has duplicate paper IDs"
    source_identities = [(candidate.source, candidate.source_id) for candidate in frozen]
    if len(source_identities) != len(set(source_identities)):
        return "frozen snapshot has duplicate source identities"
    snapshot_hashes = {
        _canonical_json_sha256(snapshot),
        hashlib.sha256(_render_json_bytes(snapshot)).hexdigest(),
    }
    if manifest.get("snapshot_sha256") not in snapshot_hashes:
        return "frozen snapshot manifest SHA-256 does not match snapshot"
    replay = manifest.get("zero_transport_replay")
    if not isinstance(replay, dict) or replay.get("transport_requests") != 0:
        return "frozen snapshot manifest lacks a zero-transport replay audit"
    return None


def _replay_and_project(
    raw_output: bytes,
    *,
    m1_cache_dir: Path,
    provenance: AcceptedM1Provenance,
    evidence_report_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Replay all stored M1 queries through the existing cache and deduplicator."""

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app.adapters.arxiv import ArxivAdapter, ArxivAdapterConfig
    from app.core.paper_dedup import deduplicate_papers
    from app.models.embedding import FrozenCandidate
    from app.models.first_round import FirstRoundRun, FirstRoundStatus

    run = FirstRoundRun.model_validate(json.loads(raw_output))
    if run.status is not FirstRoundStatus.SUCCESS or run.config.mode != "real":
        raise ValueError("M1 output is not a successful real run")
    if run.query_plan is None or len(run.query_plan.queries) != 12:
        raise ValueError("M1 output must contain exactly 12 planned queries")
    if run.config.cache_namespace != "first-round:real":
        raise ValueError("M1 output must use the first-round:real cache namespace")
    transport = NetworkForbiddenTransport()
    adapter = ArxivAdapter(
        ArxivAdapterConfig(
            user_agent="research-retrieval-calibrator/0.1 (M2-T01R1 freeze replay)",
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
    records = []
    cache_hits = 0
    for query in run.query_plan.queries:
        records.extend(adapter.search(query, max_results=run.config.max_results_per_query))
        cache_hits += int(adapter.last_observation.cache_hit)
    if transport.request_count != 0:
        raise ValueError("M2_T01_ZERO_TRANSPORT_REPLAY_FAILED")
    if cache_hits != 12:
        raise ValueError("M2_T01_REAL_CACHE_INCOMPLETE")
    deduplicated = deduplicate_papers(records)
    if len(records) != run.raw_candidate_count or len(deduplicated.clusters) != provenance.candidate_count:
        raise ValueError("M2_T01_REAL_CACHE_INCOMPLETE")
    historical = {(candidate.paper_id, candidate.source, candidate.source_id): candidate for candidate in run.candidates}
    frozen: list[FrozenCandidate] = []
    mismatch_count = abs(len(historical) - len(deduplicated.clusters))
    for cluster in deduplicated.clusters:
        record = cluster.canonical_record
        key = (record.paper_id, record.source, record.source_id)
        candidate = historical.get(key)
        if candidate is None or any(
            (
                candidate.title != record.title,
                candidate.authors != record.authors,
                candidate.year != record.year,
                candidate.doi != record.doi,
                candidate.url != record.url,
                candidate.retrieval_paths != cluster.retrieval_paths,
                candidate.cluster_id != cluster.cluster_id,
                candidate.member_source_identities != cluster.source_identities,
                candidate.merge_reasons != cluster.merge_reasons,
            )
        ):
            mismatch_count += 1
            continue
        abstract = record.abstract if record.abstract is not None and record.abstract.strip() else None
        frozen.append(
            FrozenCandidate(
                paper_id=record.paper_id,
                source=record.source,
                source_id=record.source_id,
                title=record.title,
                abstract=abstract,
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
    if mismatch_count:
        raise ValueError("M2_T01_M1_METADATA_MISMATCH")
    candidates = [candidate.model_dump(mode="json") for candidate in sorted(frozen, key=lambda item: item.paper_id)]
    snapshot = {
        "snapshot_version": "m2-candidates.v1",
        "source_phase": "M1",
        "source_merge_commit": provenance.source_merge_commit,
        "source_evidence_report": "evaluation/reports/m1-validation.json",
        "source_evidence_report_sha256": evidence_report_sha256,
        "source_output_sha256": provenance.output_sha256,
        "source_candidate_array_sha256": provenance.candidate_array_sha256,
        "question": run.question,
        "count": len(candidates),
        "candidates": candidates,
    }
    audit = {"transport_requests": transport.request_count, "cache_hits": cache_hits, "query_count": 12}
    manifest = {
        "candidate_count": len(candidates),
        "source_id_coverage": provenance.source_id_coverage,
        "url_coverage": provenance.url_coverage,
        "paper_id_order": [candidate["paper_id"] for candidate in candidates],
        "source_identity_set_sha256": _canonical_json_sha256(sorted((item["source"], item["source_id"]) for item in candidates)),
        "candidate_identity_sha256": _canonical_json_sha256([(item["paper_id"], item["source"], item["source_id"]) for item in candidates]),
        "abstract_present_count": sum(item["abstract"] is not None for item in candidates),
        "abstract_missing_count": sum(item["abstract"] is None for item in candidates),
        "metadata_mismatch_count": mismatch_count,
        "zero_transport_replay": audit,
        "artifact_recovery_classification": "EXACT_HISTORICAL_ARTIFACTS_RECOVERED",
    }
    return snapshot, manifest


def validate_m1_freeze_input(
    raw_output: bytes,
    *,
    expected_output_sha256: str = M1_OUTPUT_SHA256,
    expected_candidate_array_sha256: str = M1_CANDIDATE_ARRAY_SHA256,
) -> str | None:
    """Return a stable reason unless a supplied M1 JSON has freeze provenance.

    The function is pure: it reads neither paths nor caches and performs no
    network or output operation.  The explicit expected hashes let contract
    tests use a synthetic M1 JSON without weakening production defaults.
    """

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
    if not isinstance(candidates, list) or len(candidates) != 33:
        return "M1 output must contain exactly 33 candidates"
    if _canonical_json_sha256(candidates) != expected_candidate_array_sha256:
        return "M1 candidate-array SHA-256 does not match accepted evidence"
    return None


def freeze_candidates(
    *,
    m1_output: Path,
    m1_cache_dir: Path,
    output_dir: Path,
    m1_evidence_report: Path = Path("evaluation/reports/m1-validation.json"),
    expected_output_sha256: str | None = None,
    expected_candidate_array_sha256: str | None = None,
) -> FreezeResult:
    """Check only the historical M1 prerequisites and never write snapshots.

    ``output_dir`` is intentionally unused until the later cache-replay and
    source-projection work is authorized and implemented.
    """

    try:
        provenance = load_accepted_m1_provenance(m1_evidence_report)
    except (TypeError, ValueError) as provenance_error:
        return FreezeResult(
            status="blocked", error_code=FROZEN_INPUT_INVALID, reason=str(provenance_error)
        )
    output_sha256 = expected_output_sha256 or provenance.output_sha256
    candidate_array_sha256 = expected_candidate_array_sha256 or provenance.candidate_array_sha256
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
        expected_output_sha256=output_sha256,
        expected_candidate_array_sha256=candidate_array_sha256,
    )
    if validation_error is not None:
        return FreezeResult(
            status="blocked", error_code=FROZEN_INPUT_INVALID, reason=validation_error
        )
    try:
        snapshot, manifest = _replay_and_project(
            raw_output,
            m1_cache_dir=m1_cache_dir,
            provenance=provenance,
            evidence_report_sha256=hashlib.sha256(m1_evidence_report.read_bytes()).hexdigest(),
        )
    except (AssertionError, OSError, ValueError) as replay_error:
        reason = str(replay_error)
        error_code = (
            reason
            if reason in {
                "M2_T01_ZERO_TRANSPORT_REPLAY_FAILED",
                "M2_T01_REAL_CACHE_INCOMPLETE",
                "M2_T01_M1_METADATA_MISMATCH",
            }
            else FROZEN_INPUT_INVALID
        )
        return FreezeResult(status="blocked", error_code=error_code, reason=reason)
    snapshot_path = output_dir / "m1-candidates.v1.json"
    manifest_path = output_dir / "m1-candidates.v1.manifest.json"
    snapshot_bytes = _render_json_bytes(snapshot)
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    manifest["snapshot_sha256"] = snapshot_sha256
    try:
        _atomic_write(snapshot_path, snapshot_bytes)
        _atomic_write(manifest_path, _render_json_bytes(manifest))
    except OSError as error:
        return FreezeResult(status="blocked", error_code=FROZEN_INPUT_INVALID, reason=str(error))
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
