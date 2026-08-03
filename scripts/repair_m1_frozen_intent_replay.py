"""Generate append-only, zero-network evidence for the M1 intent replay repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

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

EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "5dad9e6070edececceeead3f8bb3e7f43302f7405fee811dcc7ddb34c84f0849"
)
EXPECTED_SOURCE_FIRST_RUN_SHA256 = (
    "ed4a4d89a247c535e6138644069094d3c59a046bdefd44a79e48a4f861f95afa"
)
EXPECTED_SOURCE_REPLAY_SHA256 = (
    "6caecd1454ff2e0bd945a6a51dac276dfc0ce9fda5716033014e245843d0aab7"
)
REPAIR_VERSION = "m1-frozen-intent-replay-repair.v1"
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


def execute_offline_repair() -> dict[str, object]:
    """Validate fixed historical inputs and write only the new repair bundle."""

    source_manifest_bytes = _read_fixed(SOURCE_MANIFEST, EXPECTED_SOURCE_MANIFEST_SHA256)
    source_first_bytes = _read_fixed(SOURCE_FIRST_RUN, EXPECTED_SOURCE_FIRST_RUN_SHA256)
    source_replay_bytes = _read_fixed(SOURCE_REPLAY, EXPECTED_SOURCE_REPLAY_SHA256)
    first_run = _load_run(source_first_bytes)
    historical_replay = _load_run(source_replay_bytes)
    if first_run.intent is None or historical_replay.intent is None:
        raise RepairError("HISTORICAL_INTENT_MISSING")
    if first_run.query_plan is None or historical_replay.query_plan is None:
        raise RepairError("HISTORICAL_QUERY_PLAN_MISSING")
    _validate_intents(first_run.intent, historical_replay.intent)
    if _intent_drift(first_run.intent, historical_replay.intent) != ["intent.frozen_at"]:
        raise RepairError("HISTORICAL_INTENT_DRIFT_NOT_EXACT")

    repaired_replay, transport = _run_cache_only_replay(first_run, historical_replay)
    _validate_repaired_replay(first_run, repaired_replay, transport)

    repaired_first_hash = _write_if_unchanged(REPAIR_FIRST_RUN, source_first_bytes)
    repaired_replay_bytes = render_json(repaired_replay).encode("utf-8")
    repaired_replay_hash = _write_if_unchanged(REPAIR_REPLAY, repaired_replay_bytes)
    manifest = _repair_manifest(
        source_manifest_bytes=source_manifest_bytes,
        source_first_bytes=source_first_bytes,
        source_replay_bytes=source_replay_bytes,
        first_run=first_run,
        repaired_replay=repaired_replay,
        repaired_first_hash=repaired_first_hash,
        repaired_replay_hash=repaired_replay_hash,
    )
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_hash = _write_if_unchanged(REPAIR_MANIFEST, manifest_bytes)
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
) -> tuple[FirstRoundRun, OfflineCacheMissTransport]:
    if first_run.intent is None:
        raise RepairError("HISTORICAL_INTENT_MISSING")
    transport = OfflineCacheMissTransport()
    adapter = _cache_only_adapter(first_run.config, transport)
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
    repaired = _align_historical_candidate_contract(first_run, repaired)
    return repaired, transport


def _cache_only_adapter(
    config: FirstRoundConfig,
    transport: OfflineCacheMissTransport,
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
            cache_dir=ROOT / SOURCE_CACHE,
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
    if repaired.candidates != first_run.candidates:
        raise RepairError("OFFLINE_REPLAY_CANDIDATES_CHANGED")
    first_queries = [(query.query_id, query.query_text) for query in first_run.query_plan.queries]
    repaired_queries = [(query.query_id, query.query_text) for query in repaired.query_plan.queries]
    if repaired_queries != first_queries or repaired.query_plan != first_run.query_plan:
        raise RepairError("OFFLINE_REPLAY_QUERY_PLAN_CHANGED")
    if canonical_research_intent_bytes(repaired.intent) != canonical_research_intent_bytes(
        first_run.intent
    ):
        raise RepairError("OFFLINE_REPLAY_INTENT_CHANGED")


def _align_historical_candidate_contract(
    first_run: FirstRoundRun,
    repaired: FirstRoundRun,
) -> FirstRoundRun:
    """Keep the accepted M1 candidate shape while preserving source identity."""

    if repaired.status is not FirstRoundStatus.SUCCESS:
        return repaired
    if len(first_run.candidates) != len(repaired.candidates):
        raise RepairError("OFFLINE_REPLAY_CANDIDATE_COUNT_CHANGED")
    for expected, observed in zip(first_run.candidates, repaired.candidates, strict=True):
        expected_payload = expected.model_dump(mode="json")
        observed_payload = observed.model_dump(mode="json")
        expected_payload.pop("abstract", None)
        observed_payload.pop("abstract", None)
        if expected_payload != observed_payload:
            raise RepairError("OFFLINE_REPLAY_CANDIDATE_IDENTITY_CHANGED")
    if repaired.candidates == first_run.candidates:
        return repaired
    return repaired.model_copy(update={"candidates": first_run.candidates})


def _repair_manifest(
    *,
    source_manifest_bytes: bytes,
    source_first_bytes: bytes,
    source_replay_bytes: bytes,
    first_run: FirstRoundRun,
    repaired_replay: FirstRoundRun,
    repaired_first_hash: str,
    repaired_replay_hash: str,
) -> dict[str, object]:
    if first_run.intent is None or repaired_replay.query_plan is None:
        raise RepairError("REPAIR_MANIFEST_INPUT_MISSING")
    candidate_bytes = _canonical_json_bytes(
        [candidate.model_dump(mode="json") for candidate in first_run.candidates]
    )
    query_plan_bytes = _canonical_json_bytes(repaired_replay.query_plan.model_dump(mode="json"))
    return {
        "repair_version": REPAIR_VERSION,
        "source_bundle": SOURCE_BUNDLE.as_posix(),
        "source_bundle_manifest_sha256": _sha256(source_manifest_bytes),
        "original_first_run_path": SOURCE_FIRST_RUN.as_posix(),
        "original_first_run_sha256": _sha256(source_first_bytes),
        "original_replay_path": SOURCE_REPLAY.as_posix(),
        "original_replay_sha256": _sha256(source_replay_bytes),
        "drift_fields": ["intent.frozen_at"],
        "corrected_first_run_path": REPAIR_FIRST_RUN.as_posix(),
        "corrected_first_run_sha256": repaired_first_hash,
        "corrected_replay_path": REPAIR_REPLAY.as_posix(),
        "corrected_replay_sha256": repaired_replay_hash,
        "canonical_intent_sha256": _sha256(canonical_research_intent_bytes(first_run.intent)),
        "candidate_array_sha256": _sha256(candidate_bytes),
        "query_plan_sha256": _sha256(query_plan_bytes),
        "zero_transport_replay": {
            "transport_requests": repaired_replay.metrics.transport_requests,
            "cache_hits": repaired_replay.metrics.cache_hits,
            "query_count": len(repaired_replay.query_results),
        },
        "historical_artifacts_modified": False,
    }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_if_unchanged(path: Path, data: bytes) -> str:
    if path.exists():
        if path.read_bytes() != data:
            raise RepairError("REPAIR_TARGET_CONFLICT")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return _sha256(data)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
