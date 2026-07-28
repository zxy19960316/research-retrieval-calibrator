"""Fail-closed M2-T01 gate for the historical M1 real-source artifacts.

This initial gate deliberately does *not* replay the M1 cache or write an M2
snapshot.  It gives the missing historical source a stable, machine-readable
outcome and exposes a pure provenance validator for the later replay step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

M1_OUTPUT_SHA256 = "069cd8c94d294c68bb051898d4e262f324b7e1f20eff10eabcf22d9bc435d178"
M1_CANDIDATE_ARRAY_SHA256 = "e51eb84d4e772bba324a199478caa5f0983edef0b0dbf4c58fce5f9da5803209"
FROZEN_INPUT_MISSING = "M2_T01_FROZEN_INPUT_MISSING"
FROZEN_INPUT_INVALID = "M2_T01_FROZEN_INPUT_INVALID"


@dataclass(frozen=True)
class FreezeResult:
    """Structured result for the no-write precondition gate."""

    status: str
    error_code: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "error_code": self.error_code,
            "reason": self.reason,
            "status": self.status,
        }


def _canonical_json_sha256(value: object) -> str:
    """Hash the canonical JSON representation used by the M1 evidence."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
    if manifest.get("snapshot_sha256") != _canonical_json_sha256(snapshot):
        return "frozen snapshot manifest SHA-256 does not match snapshot"
    replay = manifest.get("zero_transport_replay")
    if not isinstance(replay, dict) or replay.get("transport_requests") != 0:
        return "frozen snapshot manifest lacks a zero-transport replay audit"
    return None


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
    expected_output_sha256: str = M1_OUTPUT_SHA256,
    expected_candidate_array_sha256: str = M1_CANDIDATE_ARRAY_SHA256,
) -> FreezeResult:
    """Check only the historical M1 prerequisites and never write snapshots.

    ``output_dir`` is intentionally unused until the later cache-replay and
    source-projection work is authorized and implemented.
    """

    del output_dir
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
    error = validate_m1_freeze_input(
        raw_output,
        expected_output_sha256=expected_output_sha256,
        expected_candidate_array_sha256=expected_candidate_array_sha256,
    )
    if error is not None:
        return FreezeResult(status="blocked", error_code=FROZEN_INPUT_INVALID, reason=error)
    return FreezeResult(
        status="blocked",
        error_code=FROZEN_INPUT_INVALID,
        reason="M1 cache replay and snapshot projection are not implemented",
    )


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-output", type=Path, required=True)
    parser.add_argument("--m1-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    result = freeze_candidates(
        m1_output=arguments.m1_output,
        m1_cache_dir=arguments.m1_cache_dir,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0 if result.status == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
