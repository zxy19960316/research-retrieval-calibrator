"""Generate append-only, zero-network evidence for the M1 intent replay repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.adapters.arxiv import (
    ArxivAdapter,
    ArxivAdapterConfig,
    ArxivResponse,
    ArxivTransportFailure,
)
from app.cli.first_round import render_json
from app.core.first_round import run_first_round
from app.core.intent import canonical_research_intent_bytes
from app.models.first_round import FirstRoundConfig, FirstRoundRun, FirstRoundStatus
from app.models.m1_replay_repair import (
    M1ReplayCandidateAudit,
    M1ReplayRepairManifest,
)
from app.models.project import ResearchIntent

SOURCE_BUNDLE = Path("evaluation/source-artifacts/m1-rebaseline-2026-07-28")
SOURCE_MANIFEST = SOURCE_BUNDLE / "bundle-manifest.json"
SOURCE_FIRST_RUN = SOURCE_BUNDLE / "first-run/first-round.json"
SOURCE_REPLAY = SOURCE_BUNDLE / "replay/first-round.json"
SOURCE_CACHE = SOURCE_BUNDLE / "cache"
REPAIR_BUNDLE = Path("evaluation/source-artifacts/m1-intent-replay-repair-2026-08-03")
REPAIR_FIRST_RUN = REPAIR_BUNDLE / "first-run/first-round.json"
REPAIR_REPLAY = REPAIR_BUNDLE / "replay/first-round.json"
REPAIR_MANIFEST = REPAIR_BUNDLE / "repair-manifest.json"
PROTECTED_CANDIDATE_SNAPSHOT = Path("evaluation/snapshots/m2/m1-candidates.v1.json")

EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "5dad9e6070edececceeead3f8bb3e7f43302f7405fee811dcc7ddb34c84f0849"
)
EXPECTED_SOURCE_FIRST_RUN_SHA256 = (
    "ed4a4d89a247c535e6138644069094d3c59a046bdefd44a79e48a4f861f95afa"
)
EXPECTED_SOURCE_REPLAY_SHA256 = (
    "6caecd1454ff2e0bd945a6a51dac276dfc0ce9fda5716033014e245843d0aab7"
)
EXPECTED_PROTECTED_CANDIDATE_SNAPSHOT_SHA256 = (
    "4a2aec0fd0a1d22adc801fd3bc506e5da89895d1276cd572e2ac64014c162448"
)
REPAIR_VERSION = "m1-frozen-intent-replay-repair.v2"
_INTENT_FIELDS = (
    "object_terms",
    "task_terms",
    "method_terms",
    "scope_terms",
    "exclusions",
    "method_constraint",
    "accepted_paper_roles",
    "revision",
    "frozen_at",
)
_CANDIDATE_FIELDS = (
    "paper_id",
    "source",
    "source_id",
    "title",
    "abstract",
    "authors",
    "year",
    "doi",
    "url",
    "retrieval_paths",
    "cluster_id",
    "member_source_identities",
    "merge_reasons",
)
_CANDIDATE_IDENTITY_FIELDS = tuple(field for field in _CANDIDATE_FIELDS if field != "abstract")


class RepairError(RuntimeError):
    """Stable, non-sensitive failure surface for the repair command."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class OfflineCacheMissTransport:
    """Prohibit every transport request during the repair replay."""

    def __init__(self) -> None:
        self.requests = 0

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> ArxivResponse:
        del url, headers, timeout_seconds
        self.requests += 1
        raise ArxivTransportFailure("OFFLINE_CACHE_MISS")


def main(argv: Sequence[str] | None = None) -> int:
    """Refuse by default and execute only with the explicit offline flag."""

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--execute-offline-repair", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.execute_offline_repair:
        print(json.dumps({"error_code": "REPAIR_EXECUTION_NOT_AUTHORIZED", "status": "refused"}))
        return 2

    try:
        summary = execute_offline_repair()
    except RepairError as error:
        print(json.dumps({"error_code": error.code, "status": "failed"}))
        return 1
    except (OSError, TypeError, ValueError, ValidationError, json.JSONDecodeError):
        print(json.dumps({"error_code": "OFFLINE_REPAIR_FAILED", "status": "failed"}))
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def execute_offline_repair(
    *,
    repository_root: Path = ROOT,
    cache_dir: Path | None = None,
) -> dict[str, object]:
    """Validate fixed historical inputs and publish only the new repair bundle."""

    source_manifest_path = repository_root / SOURCE_MANIFEST
    source_first_path = repository_root / SOURCE_FIRST_RUN
    source_replay_path = repository_root / SOURCE_REPLAY
    source_manifest_bytes = _read_fixed(source_manifest_path, EXPECTED_SOURCE_MANIFEST_SHA256)
    source_first_bytes = _read_fixed(source_first_path, EXPECTED_SOURCE_FIRST_RUN_SHA256)
    source_replay_bytes = _read_fixed(source_replay_path, EXPECTED_SOURCE_REPLAY_SHA256)
    inventory = _source_bundle_inventory(repository_root)
    if not inventory.valid:
        raise RepairError("SOURCE_BUNDLE_INVENTORY_INVALID")
    first_run = _load_run(source_first_bytes)
    historical_replay = _load_run(source_replay_bytes)
    if first_run.intent is None or historical_replay.intent is None:
        raise RepairError("HISTORICAL_INTENT_MISSING")
    if first_run.query_plan is None or historical_replay.query_plan is None:
        raise RepairError("HISTORICAL_QUERY_PLAN_MISSING")
    _validate_intents(first_run.intent, historical_replay.intent)
    if _intent_drift(first_run.intent, historical_replay.intent) != ["intent.frozen_at"]:
        raise RepairError("HISTORICAL_INTENT_DRIFT_NOT_EXACT")

    repaired_replay, transport = _run_cache_only_replay(
        first_run,
        historical_replay,
        repository_root=repository_root,
        cache_dir=cache_dir,
    )
    if repaired_replay.status is not FirstRoundStatus.SUCCESS:
        _validate_repaired_replay(first_run, repaired_replay, transport, None)
    candidate_audit = _candidate_audit(first_run, repaired_replay, repository_root)
    _validate_repaired_replay(first_run, repaired_replay, transport, candidate_audit)

    repaired_replay_bytes = render_json(repaired_replay).encode("utf-8")
    manifest = _repair_manifest(
        source_manifest_bytes=source_manifest_bytes,
        source_first_bytes=source_first_bytes,
        source_replay_bytes=source_replay_bytes,
        first_run=first_run,
        repaired_replay=repaired_replay,
        candidate_audit=candidate_audit,
    )
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    target_bytes = {
        repository_root / REPAIR_FIRST_RUN: source_first_bytes,
        repository_root / REPAIR_REPLAY: repaired_replay_bytes,
        repository_root / REPAIR_MANIFEST: manifest_bytes,
    }
    hashes = _publish_bundle(target_bytes)
    repaired_replay_hash = hashes[repository_root / REPAIR_REPLAY]
    manifest_hash = hashes[repository_root / REPAIR_MANIFEST]
    return {
        "canonical_intent_sha256": manifest["canonical_intent_sha256"],
        "corrected_replay_sha256": repaired_replay_hash,
        "repair_manifest_sha256": manifest_hash,
        "status": "success",
        "transport_requests": transport.requests,
        "cache_hits": repaired_replay.metrics.cache_hits,
    }


def _read_fixed(path: Path, expected_sha256: str) -> bytes:
    data = path.read_bytes()
    if _sha256(data) != expected_sha256:
        raise RepairError("FIXED_HISTORICAL_HASH_MISMATCH")
    return data


def _source_bundle_inventory(repository_root: Path) -> Any:
    """Run the public source inventory audit without introducing an import cycle."""

    from scripts.validate_m1_frozen_intent_replay_repair import (
        validate_source_bundle_inventory,
    )

    return validate_source_bundle_inventory(repository_root)


def _load_run(data: bytes) -> FirstRoundRun:
    try:
        payload = json.loads(data.decode("utf-8"))
        return FirstRoundRun.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValidationError) as error:
        raise RepairError("HISTORICAL_RUN_INVALID") from error


def _validate_intents(first: ResearchIntent, replay: ResearchIntent) -> None:
    try:
        ResearchIntent.model_validate(first.model_dump(mode="json"))
        ResearchIntent.model_validate(replay.model_dump(mode="json"))
    except ValidationError as error:
        raise RepairError("HISTORICAL_INTENT_INVALID") from error


def _intent_drift(first: ResearchIntent, replay: ResearchIntent) -> list[str]:
    first_payload = first.model_dump(mode="json")
    replay_payload = replay.model_dump(mode="json")
    drift: list[str] = []
    for field in _INTENT_FIELDS:
        left = first_payload[field]
        right = replay_payload[field]
        if field == "accepted_paper_roles":
            different = set(left) != set(right)
        else:
            different = left != right
        if different:
            drift.append(f"intent.{field}")
    return drift


def _run_cache_only_replay(
    first_run: FirstRoundRun,
    historical_replay: FirstRoundRun,
    *,
    repository_root: Path,
    cache_dir: Path | None,
) -> tuple[FirstRoundRun, OfflineCacheMissTransport]:
    if first_run.intent is None:
        raise RepairError("HISTORICAL_INTENT_MISSING")
    transport = OfflineCacheMissTransport()
    adapter = _cache_only_adapter(
        first_run.config,
        transport,
        repository_root=repository_root,
        cache_dir=cache_dir,
    )
    now = _sequence_clock(
        [historical_replay.started_at_utc, historical_replay.finished_at_utc]
    )
    monotonic = _sequence_clock([0.0, historical_replay.metrics.elapsed_seconds])
    repaired = run_first_round(
        first_run.question,
        config=first_run.config,
        adapter=adapter,
        now=now,
        monotonic=monotonic,
        frozen_intent=first_run.intent,
    )
    return repaired, transport


def _cache_only_adapter(
    config: FirstRoundConfig,
    transport: OfflineCacheMissTransport,
    *,
    repository_root: Path,
    cache_dir: Path | None,
) -> ArxivAdapter:
    return ArxivAdapter(
        ArxivAdapterConfig(
            endpoint="https://export.arxiv.org/api/query",
            user_agent="research-retrieval-calibrator/m1-replay-repair",
            timeout_seconds=config.timeout_seconds,
            page_size=config.max_results_per_query,
            min_request_interval_seconds=config.min_request_interval_seconds,
            max_attempts=1,
            max_total_results=config.max_results_per_query,
            max_total_attempts=config.max_total_attempts,
            initial_backoff_seconds=0.0,
            cache_dir=cache_dir or repository_root / SOURCE_CACHE,
            cache_schema_version=config.adapter_schema_version,
            cache_namespace=config.cache_namespace,
        ),
        transport=transport,
        monotonic=lambda: 10.0,
        sleeper=lambda _: None,
    )


def _sequence_clock(values: list[Any]) -> Callable[[], Any]:
    remaining = iter(values)

    def read() -> Any:
        try:
            return next(remaining)
        except StopIteration as error:
            raise RepairError("REPLAY_CLOCK_OVERUSED") from error

    return read


def _validate_repaired_replay(
    first_run: FirstRoundRun,
    repaired: FirstRoundRun,
    transport: OfflineCacheMissTransport,
    candidate_audit: M1ReplayCandidateAudit | None,
) -> None:
    if repaired.status is not FirstRoundStatus.SUCCESS:
        raise RepairError("OFFLINE_REPLAY_FAILED")
    if repaired.intent is None or first_run.intent is None:
        raise RepairError("REPAIRED_INTENT_MISSING")
    if repaired.query_plan is None or first_run.query_plan is None:
        raise RepairError("REPAIRED_QUERY_PLAN_MISSING")
    if transport.requests != 0 or repaired.metrics.transport_requests != 0:
        raise RepairError("OFFLINE_REPLAY_TRANSPORT_USED")
    if repaired.metrics.cache_hits != 12 or len(repaired.query_results) != 12:
        raise RepairError("OFFLINE_REPLAY_CACHE_COUNTS_INVALID")
    if not all(result.cache_hit and result.attempt_count == 0 for result in repaired.query_results):
        raise RepairError("OFFLINE_REPLAY_NOT_CACHE_ONLY")
    if candidate_audit is not None:
        if not candidate_audit.candidate_order_equal:
            raise RepairError("OFFLINE_REPLAY_CANDIDATE_ORDER_CHANGED")
        if not candidate_audit.candidate_identity_equal:
            raise RepairError("OFFLINE_REPLAY_CANDIDATE_IDENTITY_CHANGED")
    first_queries = [(query.query_id, query.query_text) for query in first_run.query_plan.queries]
    repaired_queries = [(query.query_id, query.query_text) for query in repaired.query_plan.queries]
    if repaired_queries != first_queries or repaired.query_plan != first_run.query_plan:
        raise RepairError("OFFLINE_REPLAY_QUERY_PLAN_CHANGED")
    if canonical_research_intent_bytes(repaired.intent) != canonical_research_intent_bytes(
        first_run.intent
    ):
        raise RepairError("OFFLINE_REPLAY_INTENT_CHANGED")


def _candidate_audit(
    first_run: FirstRoundRun,
    repaired: FirstRoundRun,
    repository_root: Path,
) -> M1ReplayCandidateAudit:
    """Audit raw candidate evolution and bind replay metadata to the M2 snapshot."""

    first_payload = [candidate.model_dump(mode="json") for candidate in first_run.candidates]
    replay_payload = [candidate.model_dump(mode="json") for candidate in repaired.candidates]
    first_by_id = {candidate["paper_id"]: candidate for candidate in first_payload}
    replay_by_id = {candidate["paper_id"]: candidate for candidate in replay_payload}
    first_order = [candidate["paper_id"] for candidate in first_payload]
    replay_order = [candidate["paper_id"] for candidate in replay_payload]
    candidate_order_equal = first_order == replay_order
    candidate_ids_equal = set(first_by_id) == set(replay_by_id)
    candidate_identity_equal = candidate_ids_equal and all(
        _candidate_projection(first_by_id[paper_id], _CANDIDATE_IDENTITY_FIELDS)
        == _candidate_projection(replay_by_id[paper_id], _CANDIDATE_IDENTITY_FIELDS)
        for paper_id in first_by_id.keys() & replay_by_id.keys()
    )
    delta_fields: set[str] = set()
    for paper_id in first_by_id.keys() | replay_by_id.keys():
        left = first_by_id.get(paper_id)
        right = replay_by_id.get(paper_id)
        if left is None or right is None:
            delta_fields.add("paper_id")
            continue
        delta_fields.update(
            field
            for field in _CANDIDATE_FIELDS
            if left.get(field) != right.get(field)
        )
    _validate_protected_candidate_snapshot(repository_root, replay_payload)
    return cast(
        M1ReplayCandidateAudit,
        M1ReplayCandidateAudit.model_validate(
            {
                "candidate_count": len(replay_payload),
                "candidate_order_equal": candidate_order_equal,
                "candidate_identity_equal": candidate_identity_equal,
                "candidate_payload_equal": first_payload == replay_payload,
                "candidate_delta_fields": sorted(delta_fields),
                "protected_candidate_snapshot_path": PROTECTED_CANDIDATE_SNAPSHOT.as_posix(),
                "protected_candidate_snapshot_sha256": EXPECTED_PROTECTED_CANDIDATE_SNAPSHOT_SHA256,
                "replay_candidate_array_sha256": _sha256(
                    _canonical_json_bytes(replay_payload)
                ),
            }
        ),
    )


def _candidate_projection(payload: dict[str, Any], fields: tuple[str, ...]) -> bytes:
    return _canonical_json_bytes({field: payload.get(field) for field in fields})


def _validate_protected_candidate_snapshot(
    repository_root: Path,
    replay_payload: list[dict[str, Any]],
) -> None:
    snapshot_path = repository_root / PROTECTED_CANDIDATE_SNAPSHOT
    try:
        snapshot_bytes = snapshot_path.read_bytes()
    except OSError as error:
        raise RepairError("PROTECTED_M2_CANDIDATE_SNAPSHOT_UNAVAILABLE") from error
    if _sha256(snapshot_bytes) != EXPECTED_PROTECTED_CANDIDATE_SNAPSHOT_SHA256:
        raise RepairError("PROTECTED_M2_CANDIDATE_SNAPSHOT_HASH_MISMATCH")
    try:
        snapshot = json.loads(snapshot_bytes.decode("utf-8"))
        expected_candidates = snapshot["candidates"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RepairError("PROTECTED_M2_CANDIDATE_SNAPSHOT_INVALID") from error
    if not isinstance(expected_candidates, list) or len(expected_candidates) != 33:
        raise RepairError("PROTECTED_M2_CANDIDATE_SNAPSHOT_INVALID")
    expected_by_id = {
        candidate.get("paper_id"): candidate
        for candidate in expected_candidates
        if isinstance(candidate, dict)
    }
    if len(expected_by_id) != 33 or len(replay_payload) != 33:
        raise RepairError("PROTECTED_M2_CANDIDATE_MISMATCH")
    for candidate in replay_payload:
        paper_id = candidate.get("paper_id")
        protected = expected_by_id.get(paper_id)
        if not isinstance(protected, dict):
            raise RepairError("PROTECTED_M2_CANDIDATE_MISMATCH")
        for field in ("title", "abstract", "source", "source_id"):
            if candidate.get(field) != protected.get(field):
                raise RepairError("PROTECTED_M2_CANDIDATE_MISMATCH")


def _repair_manifest(
    *,
    source_manifest_bytes: bytes,
    source_first_bytes: bytes,
    source_replay_bytes: bytes,
    first_run: FirstRoundRun,
    repaired_replay: FirstRoundRun,
    candidate_audit: M1ReplayCandidateAudit,
) -> dict[str, object]:
    if first_run.intent is None or repaired_replay.query_plan is None:
        raise RepairError("REPAIR_MANIFEST_INPUT_MISSING")
    query_plan_bytes = _canonical_json_bytes(repaired_replay.query_plan.model_dump(mode="json"))
    manifest = {
        "repair_version": REPAIR_VERSION,
        "source_bundle": SOURCE_BUNDLE.as_posix(),
        "source_bundle_manifest_sha256": _sha256(source_manifest_bytes),
        "original_first_run_path": SOURCE_FIRST_RUN.as_posix(),
        "original_first_run_sha256": _sha256(source_first_bytes),
        "original_replay_path": SOURCE_REPLAY.as_posix(),
        "original_replay_sha256": _sha256(source_replay_bytes),
        "drift_fields": ["intent.frozen_at"],
        "corrected_first_run_path": REPAIR_FIRST_RUN.as_posix(),
        "corrected_first_run_sha256": _sha256(source_first_bytes),
        "corrected_replay_path": REPAIR_REPLAY.as_posix(),
        "corrected_replay_sha256": "0" * 64,
        "canonical_intent_sha256": _sha256(canonical_research_intent_bytes(first_run.intent)),
        "candidate_audit": candidate_audit.model_dump(mode="json"),
        "query_plan_sha256": _sha256(query_plan_bytes),
        "zero_transport_replay": {
            "transport_requests": repaired_replay.metrics.transport_requests,
            "cache_hits": repaired_replay.metrics.cache_hits,
            "query_count": len(repaired_replay.query_results),
        },
        "historical_artifacts_modified": False,
    }
    repaired_replay_bytes = render_json(repaired_replay).encode("utf-8")
    manifest["corrected_replay_sha256"] = _sha256(repaired_replay_bytes)
    return cast(
        dict[str, object],
        M1ReplayRepairManifest.model_validate(manifest).model_dump(mode="json"),
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _publish_bundle(target_bytes: dict[Path, bytes]) -> dict[Path, str]:
    """Preflight, stage, and conflict-safely publish the complete three-file bundle."""

    _preflight_bundle_targets(target_bytes)
    bundle = _bundle_root(target_bytes)
    if bundle is None:
        raise RepairError("REPAIR_TARGET_CONFLICT")
    staging = bundle.parent / f".{bundle.name}.staging-{uuid.uuid4().hex}"
    published: list[tuple[Path, bytes]] = []
    try:
        for target, data in target_bytes.items():
            staged = staging / target.relative_to(bundle)
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(data)
            if staged.read_bytes() != data:
                raise RepairError("REPAIR_STAGING_VALIDATION_FAILED")
        for target, data in target_bytes.items():
            if target.exists() and not target.is_symlink():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / target.relative_to(bundle), target)
            published.append((target, data))
    except RepairError:
        _rollback_published(published)
        raise
    except OSError as error:
        _rollback_published(published)
        raise RepairError("REPAIR_PUBLICATION_FAILED") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {path: _sha256(data) for path, data in target_bytes.items()}


def _preflight_bundle_targets(target_bytes: dict[Path, bytes]) -> None:
    """Reject every conflict before staging or creating any target file."""

    bundle = _bundle_root(target_bytes)
    if bundle is None:
        raise RepairError("REPAIR_TARGET_CONFLICT")
    if bundle.is_symlink() or (bundle.exists() and not bundle.is_dir()):
        raise RepairError("REPAIR_TARGET_CONFLICT")
    target_set = set(target_bytes)
    if bundle.is_dir():
        for path in bundle.rglob("*"):
            if (path.is_symlink() or path.is_file()) and path not in target_set:
                raise RepairError("REPAIR_TARGET_CONFLICT")
    for target, data in target_bytes.items():
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise RepairError("REPAIR_TARGET_CONFLICT")
        if target.exists() and target.read_bytes() != data:
            raise RepairError("REPAIR_TARGET_CONFLICT")


def _bundle_root(target_bytes: dict[Path, bytes]) -> Path | None:
    try:
        return Path(os.path.commonpath([str(path) for path in target_bytes]))
    except (OSError, ValueError):
        return None


def _rollback_published(published: list[tuple[Path, bytes]]) -> None:
    for path, data in reversed(published):
        try:
            if path.is_file() and not path.is_symlink() and path.read_bytes() == data:
                path.unlink()
        except OSError:
            continue


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
