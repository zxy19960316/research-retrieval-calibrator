"""Validate the append-only, zero-network M1 frozen-intent replay repair."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError

from app.core.intent import canonical_research_intent_bytes
from app.models.first_round import FirstRoundRun, FirstRoundStatus
from scripts.repair_m1_frozen_intent_replay import (
    EXPECTED_SOURCE_FIRST_RUN_SHA256,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    EXPECTED_SOURCE_REPLAY_SHA256,
    REPAIR_BUNDLE,
    REPAIR_FIRST_RUN,
    REPAIR_MANIFEST,
    REPAIR_REPLAY,
    SOURCE_BUNDLE,
    SOURCE_FIRST_RUN,
    SOURCE_MANIFEST,
    SOURCE_REPLAY,
    _canonical_json_bytes,
    _intent_drift,
)

REPORT_PATH = Path("evaluation/reports/m1-frozen-intent-replay-repair-2026-08-03.json")
REPORT_VERSION = "m1-frozen-intent-replay-repair.v1"
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
STATUS_M2_RE = re.compile(r"^\|\s*M2\b[^|]*\|\s*([A-Z0-9_]+)\s*\|\s*(\d+)/(\d+)\s*\|", re.MULTILINE)
ABSOLUTE_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/(?:Users|home|tmp)/)")
FORBIDDEN_TEXT = (
    "file://",
    "authorization",
    "bearer ",
    "access_token",
    "api_key",
    "-----begin",
    "traceback",
)

PROTECTED_HASHES = {
    "evaluation/snapshots/m2/m1-candidates.v1.json": "4a2aec0fd0a1d22adc801fd3bc506e5da89895d1276cd572e2ac64014c162448",
    "evaluation/snapshots/m2/m1-candidates.v1.manifest.json": "b6ce7cfab2e5df6b8c84b77fb38c6f573de7429d37a9688d7b4c62e27f8546a8",
    "evaluation/source-artifacts/m2-t02-reranker-candidate-run.json": "0a751dbc35bfa8d07433796113939412bfbdffd4c5c13043c90390bccfc5c35b",
    "evaluation/source-artifacts/m2-t02-reranker-candidate-run-receipt.json": "0c9ca1339a2385a9c6940f8c53bd4711c62b8faf9f3f4cf98710ff465d56ec9a",
    "evaluation/source-artifacts/m2-t03-evidence-classification-run.json": "2bd80d2b6a10bfe531ea2ca3b7570768fd1a78e5d05efca96fc28464d72baee8",
    "evaluation/source-artifacts/m2-t03-evidence-classification-run-receipt.json": "0bc69ad7a4398c8070c2ba96949f7b9a023c4940b83913f38e1ba6cd41beeec4",
}
PROTECTED_REPORT_HASHES = {
    "evaluation/reports/m1-rebaseline-2026-07-28.json": "3336d6a6a1123fd5204f2db1f11cc155dc080f3e440db0f3eab7a2f84d2287fd",
    "evaluation/reports/m1-validation.json": "92d5776c519f8da1e7924db71961a683a471a75dc12be81628e7772d23ace01d",
}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]


def validate_m1_frozen_intent_replay_repair(
    report_path: Path = REPORT_PATH,
    *,
    repository_root: Path = ROOT,
) -> ValidationResult:
    """Return a closed validation result without mutating any repository file."""

    errors: list[str] = []
    _validate_fixed_history(errors, repository_root)
    manifest, _, _ = _validate_bundle(errors, repository_root)
    _validate_status_and_protected_artifacts(errors, repository_root)
    report = _load_json(repository_root / report_path)
    if report is None:
        errors.append("repair report is unavailable or invalid")
    else:
        _validate_report(report, manifest, errors, repository_root)
        _validate_safe_text(report, errors)
    return ValidationResult(not errors, errors)


def _validate_fixed_history(errors: list[str], repository_root: Path) -> None:
    _expect_hash(repository_root / SOURCE_MANIFEST, EXPECTED_SOURCE_MANIFEST_SHA256, errors)
    _expect_hash(repository_root / SOURCE_FIRST_RUN, EXPECTED_SOURCE_FIRST_RUN_SHA256, errors)
    _expect_hash(repository_root / SOURCE_REPLAY, EXPECTED_SOURCE_REPLAY_SHA256, errors)
    for relative, expected in {**PROTECTED_REPORT_HASHES, **PROTECTED_HASHES}.items():
        _expect_hash(repository_root / relative, expected, errors)

    tracked = _git_lines(repository_root, "ls-tree", "-r", "--name-only", "HEAD", "--", SOURCE_BUNDLE.as_posix())
    if not tracked:
        errors.append("historical source bundle has no tracked files")
        return
    for relative in tracked:
        current = repository_root / relative
        committed = _git_bytes(repository_root, "show", f"HEAD:{relative}")
        if current.read_bytes() != committed:
            errors.append(f"historical source bundle changed: {relative}")


def _validate_bundle(
    errors: list[str], repository_root: Path
) -> tuple[dict[str, Any] | None, FirstRoundRun | None, FirstRoundRun | None]:
    bundle = repository_root / REPAIR_BUNDLE
    if not bundle.is_dir():
        errors.append("repair bundle is unavailable")
        return None, None, None
    expected_files = {
        "first-run/first-round.json",
        "replay/first-round.json",
        "repair-manifest.json",
    }
    actual_files = {
        path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        errors.append("repair bundle contains unexpected files or cache copies")
    if any("cache" in path.casefold() or "cert" in path.casefold() for path in actual_files):
        errors.append("repair bundle contains cache or certificate material")

    manifest = _load_json(repository_root / REPAIR_MANIFEST)
    first_payload = _load_json(repository_root / REPAIR_FIRST_RUN)
    replay_payload = _load_json(repository_root / REPAIR_REPLAY)
    first_run = _validate_run(first_payload, "corrected first-run", errors)
    replay = _validate_run(replay_payload, "corrected replay", errors)
    if manifest is None or first_run is None or replay is None:
        return manifest, first_run, replay

    source_first_bytes = (repository_root / SOURCE_FIRST_RUN).read_bytes()
    source_replay_bytes = (repository_root / SOURCE_REPLAY).read_bytes()
    if (repository_root / REPAIR_FIRST_RUN).read_bytes() != source_first_bytes:
        errors.append("corrected first-run is not a byte copy of historical first-run")
    if (repository_root / REPAIR_REPLAY).read_bytes() == source_replay_bytes:
        errors.append("corrected replay is byte-identical to historical replay")
    _validate_manifest(manifest, first_run, replay, errors, repository_root)
    _validate_semantic_replay(first_run, replay, manifest, errors)
    return manifest, first_run, replay


def _validate_run(payload: dict[str, Any] | None, label: str, errors: list[str]) -> FirstRoundRun | None:
    if payload is None:
        errors.append(f"{label} JSON is unavailable")
        return None
    try:
        run = FirstRoundRun.model_validate(payload)
    except ValidationError:
        errors.append(f"{label} fails FirstRoundRun validation")
        return None
    if run.status is not FirstRoundStatus.SUCCESS or run.intent is None or run.query_plan is None:
        errors.append(f"{label} is not a complete successful run")
        return None
    return run


def _validate_manifest(
    manifest: dict[str, Any],
    first_run: FirstRoundRun,
    replay: FirstRoundRun,
    errors: list[str],
    repository_root: Path,
) -> None:
    required = {
        "repair_version",
        "source_bundle",
        "source_bundle_manifest_sha256",
        "original_first_run_path",
        "original_first_run_sha256",
        "original_replay_path",
        "original_replay_sha256",
        "drift_fields",
        "corrected_first_run_path",
        "corrected_first_run_sha256",
        "corrected_replay_path",
        "corrected_replay_sha256",
        "canonical_intent_sha256",
        "candidate_array_sha256",
        "query_plan_sha256",
        "zero_transport_replay",
        "historical_artifacts_modified",
    }
    if not required.issubset(manifest):
        errors.append("repair manifest is missing required fields")
        return
    if manifest["repair_version"] != "m1-frozen-intent-replay-repair.v1":
        errors.append("repair manifest version is invalid")
    if manifest["source_bundle"] != SOURCE_BUNDLE.as_posix():
        errors.append("repair manifest source bundle is invalid")
    if manifest["source_bundle_manifest_sha256"] != EXPECTED_SOURCE_MANIFEST_SHA256:
        errors.append("repair manifest source manifest hash is invalid")
    if manifest["drift_fields"] != ["intent.frozen_at"]:
        errors.append("repair manifest drift fields are not exact")
    if manifest["historical_artifacts_modified"] is not False:
        errors.append("repair manifest claims historical artifacts were modified")

    expected_hashes = {
        "original_first_run_sha256": EXPECTED_SOURCE_FIRST_RUN_SHA256,
        "original_replay_sha256": EXPECTED_SOURCE_REPLAY_SHA256,
        "corrected_first_run_sha256": _sha256((repository_root / REPAIR_FIRST_RUN).read_bytes()),
        "corrected_replay_sha256": _sha256((repository_root / REPAIR_REPLAY).read_bytes()),
    }
    for field, expected in expected_hashes.items():
        if manifest.get(field) != expected:
            errors.append(f"repair manifest hash mismatch: {field}")

    zero_replay = manifest.get("zero_transport_replay")
    if zero_replay != {"transport_requests": 0, "cache_hits": 12, "query_count": 12}:
        errors.append("repair manifest zero-transport replay is invalid")


def _validate_semantic_replay(
    first_run: FirstRoundRun,
    replay: FirstRoundRun,
    manifest: dict[str, Any],
    errors: list[str],
) -> None:
    if first_run.intent is None or replay.intent is None:
        return
    if _intent_drift(first_run.intent, replay.intent):
        errors.append("corrected intent fields are not identical")
    canonical_first = canonical_research_intent_bytes(first_run.intent)
    canonical_replay = canonical_research_intent_bytes(replay.intent)
    if canonical_first != canonical_replay:
        errors.append("corrected canonical intent bytes differ")
    if manifest.get("canonical_intent_sha256") != _sha256(canonical_first):
        errors.append("repair manifest canonical intent hash is invalid")

    if first_run.candidates != replay.candidates:
        errors.append("corrected candidate arrays are not equal")
    candidate_bytes = _canonical_json_bytes(
        [candidate.model_dump(mode="json") for candidate in first_run.candidates]
    )
    if manifest.get("candidate_array_sha256") != _sha256(candidate_bytes):
        errors.append("repair manifest candidate array hash is invalid")
    if first_run.query_plan is None or replay.query_plan is None:
        return
    first_queries = [(query.query_id, query.query_text) for query in first_run.query_plan.queries]
    replay_queries = [(query.query_id, query.query_text) for query in replay.query_plan.queries]
    if first_queries != replay_queries or first_run.query_plan != replay.query_plan:
        errors.append("corrected query IDs, text, or plan semantics differ")
    query_plan_bytes = _canonical_json_bytes(first_run.query_plan.model_dump(mode="json"))
    if manifest.get("query_plan_sha256") != _sha256(query_plan_bytes):
        errors.append("repair manifest query-plan hash is invalid")

    if replay.metrics.transport_requests != 0 or replay.metrics.cache_hits != 12:
        errors.append("corrected replay transport/cache counts are invalid")
    if len(replay.query_results) != 12 or not all(
        item.cache_hit and item.attempt_count == 0 and item.http_status is None
        for item in replay.query_results
    ):
        errors.append("corrected replay is not a 12-hit cache-only replay")


def _validate_status_and_protected_artifacts(errors: list[str], repository_root: Path) -> None:
    try:
        status = (repository_root / "STATUS.md").read_text(encoding="utf-8")
    except OSError:
        errors.append("STATUS.md is unavailable")
        return
    match = STATUS_M2_RE.search(status)
    if match is None or match.group(1) != "IN_PROGRESS" or match.group(2) != "2" or match.group(3) != "5":
        errors.append("STATUS.md must retain M2 IN_PROGRESS 2/5")


def _validate_report(
    report: dict[str, Any],
    manifest: dict[str, Any] | None,
    errors: list[str],
    repository_root: Path,
) -> None:
    required = {
        "report_version",
        "task_id",
        "implementation_commit",
        "validated_commit",
        "root_cause",
        "repair",
        "historical_bundle",
        "m2_historical_evidence",
        "m2_progress",
        "human_review",
        "m2_t04",
        "scoring_eligible",
        "source_artifacts",
        "repair_artifact",
        "commands",
        "test_totals",
        "evidence_types",
        "not_run",
    }
    if not required.issubset(report):
        errors.append("repair report is missing required fields")
        return
    if report["report_version"] != REPORT_VERSION or report["task_id"] != "M1-FROZEN-INTENT-REPLAY-REPAIR":
        errors.append("repair report identity is invalid")
    for field in ("implementation_commit", "validated_commit"):
        value = report.get(field)
        if not isinstance(value, str) or SHA1_RE.fullmatch(value) is None:
            errors.append(f"repair report {field} is not a Git SHA-1")
        elif _git(repository_root, "merge-base", "--is-ancestor", value, "HEAD") != 0:
            errors.append(f"repair report {field} is not an ancestor of HEAD")
    if report["root_cause"] != "replay reconstructed ResearchIntent using replay started_at":
        errors.append("repair report root cause is invalid")
    if report["repair"] != "replay reuses the exact first-run frozen ResearchIntent":
        errors.append("repair report repair statement is invalid")
    historical_bundle = report["historical_bundle"]
    if not isinstance(historical_bundle, dict) or historical_bundle.get("status") != "unchanged":
        errors.append("repair report historical bundle status is invalid")
    if report["m2_progress"] != "IN_PROGRESS 2/5":
        errors.append("repair report M2 progress is invalid")
    if report["human_review"] != "not started" or report["m2_t04"] != "not started":
        errors.append("repair report out-of-scope gates are invalid")
    if report["scoring_eligible"] is not False:
        errors.append("repair report must disable scoring")

    source = report["source_artifacts"]
    expected_source = {
        "source_bundle_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "original_first_run_sha256": EXPECTED_SOURCE_FIRST_RUN_SHA256,
        "original_replay_sha256": EXPECTED_SOURCE_REPLAY_SHA256,
        "drift_fields": ["intent.frozen_at"],
    }
    if source != expected_source:
        errors.append("repair report source artifact binding is invalid")
    repair_artifact = report["repair_artifact"]
    expected_repair_artifact = {
        "manifest_sha256": _sha256((repository_root / REPAIR_MANIFEST).read_bytes()),
        "corrected_first_run_sha256": None if manifest is None else manifest.get("corrected_first_run_sha256"),
        "corrected_replay_sha256": None if manifest is None else manifest.get("corrected_replay_sha256"),
        "canonical_intent_sha256": None if manifest is None else manifest.get("canonical_intent_sha256"),
        "candidate_array_sha256": None if manifest is None else manifest.get("candidate_array_sha256"),
        "query_plan_sha256": None if manifest is None else manifest.get("query_plan_sha256"),
    }
    if manifest is None or not isinstance(repair_artifact, dict):
        errors.append("repair report repair artifact binding is unavailable")
    elif any(repair_artifact.get(key) != value for key, value in expected_repair_artifact.items()):
        errors.append("repair report repair artifact binding is invalid")
    _validate_commands_and_totals(report, errors)
    if report["evidence_types"] != {
        "offline_cache_replay": "recorded_cache_replay",
        "real_arxiv_requests": "not_run",
        "model_runs": "not_run",
        "human_review": "not_started",
    }:
        errors.append("repair report evidence types are invalid")
    if report["not_run"] != {
        "real_arxiv_requests": True,
        "model_runs": True,
        "human_review": True,
        "m2_t04": True,
    }:
        errors.append("repair report not-run gates are invalid")


def _validate_commands_and_totals(report: dict[str, Any], errors: list[str]) -> None:
    commands = report.get("commands")
    if not isinstance(commands, list) or not commands:
        errors.append("repair report commands must be non-empty")
    else:
        for command in commands:
            if not isinstance(command, dict) or not isinstance(command.get("name"), str):
                errors.append("repair report command entry is invalid")
            elif command.get("exit_code") != 0:
                errors.append(f"repair report command is not green: {command['name']}")
    totals = report.get("test_totals")
    if not isinstance(totals, dict) or totals.get("new_repair_contract") != {"passed": 13, "failed": 0}:
        errors.append("repair report test totals do not bind the new contract suite")


def _validate_safe_text(value: object, errors: list[str]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _validate_safe_text(item, errors)
    elif isinstance(value, list):
        for item in value:
            _validate_safe_text(item, errors)
    elif isinstance(value, str):
        if ABSOLUTE_PATH_RE.match(value):
            errors.append("repair evidence contains an absolute path")
        if any(token in value.casefold() for token in FORBIDDEN_TEXT):
            errors.append("repair evidence contains forbidden credential or raw-error text")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _expect_hash(path: Path, expected: str, errors: list[str]) -> None:
    try:
        actual = _sha256(path.read_bytes())
    except OSError:
        errors.append(f"fixed artifact unavailable: {path.as_posix()}")
        return
    if actual != expected:
        errors.append(f"fixed artifact hash mismatch: {path.as_posix()}")


def _git(repository_root: Path, *arguments: str) -> int:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        capture_output=True,
        check=False,
    ).returncode


def _git_lines(repository_root: Path, *arguments: str) -> list[str]:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return result.stdout.decode("utf-8").splitlines()


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    return result.stdout


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    result = validate_m1_frozen_intent_replay_repair()
    if not result.valid:
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1
    print("PASS: M1 frozen intent replay repair, historical immutability, and M2 gate are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
