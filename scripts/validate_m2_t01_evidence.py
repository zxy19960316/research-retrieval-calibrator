"""Validate the completed M2-T01 evidence without loading an embedding model.

The task can be implemented while its historical M1 inputs or real BGE-M3 run
remain unavailable.  Therefore the absence of the final report is valid only
while ``STATUS.md`` still makes no M2-T01 completion claim.  Once the report
exists, this validator is deliberately fail-closed: it reads committed JSON
and Git blobs only, and never imports a provider, downloads a model, or makes
a network request.

Completed report contract (``m2-t01-embedding.v1``):

* provenance: ``baseline_commit``, ``implementation_commit_a``,
  ``snapshot_commit_b``, ``validated_commit``, and ``validated_inputs``;
* ``candidate_snapshot`` and ``vector_snapshot`` objects containing their
  paths, Git-blob SHA-256 values, manifest paths/blob hashes, and unambiguous
  artifact hashes: exact file bytes for candidates and canonical JSON for
  vectors;
* green ``automated_checks``, explicit ``fake_evidence`` and
  ``real_model_evidence``, M1 metadata audit, and ``status_after_evidence``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = Path("evaluation/reports/m2-t01-embedding.json")
REPORT_VERSION = "m2-t01-embedding.v1"
BASELINE_COMMIT = "0eb45fc22d10adb72cb66aa45494333057080bc1"
M1_REBASELINE_REPORT = "evaluation/reports/m1-rebaseline-2026-07-28.json"
M1_REBASELINE_SOURCE_BUNDLE = "evaluation/source-artifacts/m1-rebaseline-2026-07-28"
M1_REBASELINE_CANDIDATE_ARRAY_SHA256 = "e51eb84d4e772bba324a199478caa5f0983edef0b0dbf4c58fce5f9da5803209"
SHA1 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
STATUS_ROW = re.compile(
    r"^\|\s*(M\d+)\b[^|]*\|\s*([A-Z0-9_]+)\s*\|\s*(\d+)/(\d+)\s*\|",
    re.MULTILINE,
)
REQUIRED_CHECKS = {
    "focused_embedding_tests",
    "full_regression",
    "ruff",
    "mypy",
    "project_docs_validation",
    "m0_evidence_validation",
    "m1_evidence_validation",
    "m2_t01_evidence_validation",
    "pip_check",
}


@dataclass(frozen=True)
class EvidenceValidationResult:
    valid: bool
    errors: list[str]


def _git(*arguments: str, repository_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )


def _git_bytes(
    *arguments: str, repository_root: Path
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )


def _safe_path(value: object, repository_root: Path) -> Path | None:
    if not isinstance(value, str):
        return None
    candidate = Path(value)
    windows_candidate = PureWindowsPath(value)
    if (
        candidate.is_absolute()
        or windows_candidate.is_absolute()
        or ".." in candidate.parts
        or ".." in windows_candidate.parts
    ):
        return None
    resolved = (repository_root / candidate).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError:
        return None
    return resolved


def _blob_bytes(commit: str, path: str, repository_root: Path) -> bytes | None:
    shown = _git_bytes(
        "show", f"{commit}:{path.replace('\\', '/')}", repository_root=repository_root
    )
    return shown.stdout if shown.returncode == 0 else None


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parse_status(repository_root: Path, errors: list[str]) -> dict[str, tuple[str, int, int]]:
    try:
        text = (repository_root / "STATUS.md").read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"cannot read STATUS.md: {error}")
        return {}
    phases: dict[str, tuple[str, int, int]] = {}
    for phase, state, completed, total in STATUS_ROW.findall(text):
        if phase in phases:
            errors.append(f"STATUS.md declares duplicate current state for {phase}")
            continue
        phases[phase] = (state, int(completed), int(total))
    return phases


def _validate_precompletion_status(repository_root: Path, errors: list[str]) -> None:
    phases = _parse_status(repository_root, errors)
    if phases.get("M1") != ("COMPLETE", 4, 4):
        errors.append("pre-completion STATUS.md must retain M1 COMPLETE 4/4")
    if phases.get("M2") not in {("READY", 0, 5), ("IN_PROGRESS", 0, 5)}:
        errors.append("no M2-T01 report requires M2 READY/IN_PROGRESS 0/5")
    if phases.get("M3") != ("BLOCKED_BY_M2", 0, 5):
        errors.append("pre-completion STATUS.md must retain M3 BLOCKED_BY_M2 0/5")
    try:
        status = (repository_root / "STATUS.md").read_text(encoding="utf-8")
    except OSError:
        return
    if "M2-T01 COMPLETE" in status.replace("`", ""):
        errors.append("STATUS.md claims M2-T01 COMPLETE without its evidence report")


def _is_ancestor(commit: object, repository_root: Path) -> bool:
    return isinstance(commit, str) and bool(SHA1.fullmatch(commit)) and _git(
        "merge-base", "--is-ancestor", commit, "HEAD", repository_root=repository_root
    ).returncode == 0


def _validate_provenance(
    payload: dict[str, Any], errors: list[str], repository_root: Path
) -> str | None:
    if payload.get("baseline_commit") != BASELINE_COMMIT:
        errors.append("baseline_commit must equal the M1 merge baseline")
    ordered = [
        payload.get("baseline_commit"),
        payload.get("implementation_commit_a"),
        payload.get("snapshot_commit_b"),
        payload.get("validated_commit"),
    ]
    if not all(_is_ancestor(commit, repository_root) for commit in ordered):
        errors.append("baseline, implementation, snapshot, and validated commits must be HEAD ancestors")
        return None
    for earlier, later in pairwise(ordered):
        if not isinstance(earlier, str) or not isinstance(later, str):
            continue
        if _git("merge-base", "--is-ancestor", earlier, later, repository_root=repository_root).returncode != 0:
            errors.append("implementation/data/validated commits are not in required ancestry order")
            break
    validated_commit = payload.get("validated_commit")
    return validated_commit if isinstance(validated_commit, str) else None


def _validate_inputs(
    payload: dict[str, Any], validated_commit: str | None, errors: list[str], repository_root: Path
) -> None:
    inputs = payload.get("validated_inputs")
    if not isinstance(inputs, list) or not inputs:
        errors.append("validated_inputs must be a non-empty list")
        return
    seen: set[str] = set()
    for item in inputs:
        if not isinstance(item, dict):
            errors.append("validated input entry must be an object")
            continue
        path, expected = item.get("path"), item.get("sha256")
        if not isinstance(path, str) or path in seen or _safe_path(path, repository_root) is None:
            errors.append("validated input path is missing, duplicate, or unsafe")
            continue
        seen.add(path)
        if not isinstance(expected, str) or SHA256.fullmatch(expected) is None:
            errors.append(f"validated input has invalid sha256: {path}")
            continue
        contents = _blob_bytes(validated_commit, path, repository_root) if validated_commit else None
        if contents is None or hashlib.sha256(contents).hexdigest() != expected:
            errors.append(f"validated input hash mismatch: {path}")


def _snapshot_fields(
    payload: dict[str, Any], name: str, hash_field: str, errors: list[str], repository_root: Path
) -> tuple[str, str, str, str, str] | None:
    snapshot = payload.get(name)
    if not isinstance(snapshot, dict):
        errors.append(f"{name} must be an object")
        return None
    path = snapshot.get("path")
    blob_sha256 = snapshot.get("blob_sha256")
    manifest_path = snapshot.get("manifest_path")
    manifest_blob_sha256 = snapshot.get("manifest_blob_sha256")
    artifact_sha256 = snapshot.get(hash_field)
    if (
        not isinstance(path, str)
        or not isinstance(blob_sha256, str)
        or not isinstance(manifest_path, str)
        or not isinstance(manifest_blob_sha256, str)
        or not isinstance(artifact_sha256, str)
        or _safe_path(path, repository_root) is None
        or _safe_path(manifest_path, repository_root) is None
        or SHA256.fullmatch(blob_sha256) is None
        or SHA256.fullmatch(manifest_blob_sha256) is None
        or SHA256.fullmatch(artifact_sha256) is None
    ):
        errors.append(f"{name} has unsafe paths or invalid hashes")
        return None
    return path, blob_sha256, manifest_path, manifest_blob_sha256, artifact_sha256


def _load_snapshot_json(
    commit: str | None,
    fields: tuple[str, str, str, str, str] | None,
    name: str,
    errors: list[str],
    repository_root: Path,
) -> tuple[bytes, bytes, dict[str, Any], dict[str, Any]] | None:
    if commit is None or fields is None:
        return None
    path, expected_blob, manifest_path, expected_manifest_blob, _ = fields
    raw, raw_manifest = _blob_bytes(commit, path, repository_root), _blob_bytes(commit, manifest_path, repository_root)
    if raw is None or hashlib.sha256(raw).hexdigest() != expected_blob:
        errors.append(f"{name} Git-blob SHA-256 does not match")
        return None
    if raw_manifest is None or hashlib.sha256(raw_manifest).hexdigest() != expected_manifest_blob:
        errors.append(f"{name} manifest Git-blob SHA-256 does not match")
        return None
    try:
        snapshot, manifest = json.loads(raw), json.loads(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        errors.append(f"{name} snapshot or manifest is invalid JSON: {error}")
        return None
    if not isinstance(snapshot, dict) or not isinstance(manifest, dict):
        errors.append(f"{name} snapshot and manifest must be JSON objects")
        return None
    return raw, raw_manifest, snapshot, manifest


def _validate_candidate_snapshot(
    payload: dict[str, Any], commit: str | None, errors: list[str], repository_root: Path
) -> str | None:
    fields = _snapshot_fields(payload, "candidate_snapshot", "snapshot_sha256", errors, repository_root)
    loaded = _load_snapshot_json(commit, fields, "candidate snapshot", errors, repository_root)
    if fields is None or loaded is None:
        return None
    raw_snapshot, _, snapshot, manifest = loaded
    if snapshot.get("snapshot_version") != "m2-candidates.v1" or snapshot.get("count") != 33:
        errors.append("candidate snapshot must be m2-candidates.v1 with count 33")
    if snapshot.get("source_phase") != "M1" or snapshot.get("source_merge_commit") != BASELINE_COMMIT:
        errors.append("candidate snapshot must retain its M1 baseline provenance")
    candidates = snapshot.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 33:
        errors.append("candidate snapshot must contain exactly 33 candidates")
    else:
        paper_ids: list[str] = []
        source_identities: list[tuple[str, str]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                errors.append("candidate snapshot entries must be objects")
                continue
            paper_id, source, source_id, url = (
                candidate.get("paper_id"),
                candidate.get("source"),
                candidate.get("source_id"),
                candidate.get("url"),
            )
            if (
                not isinstance(paper_id, str)
                or not paper_id.strip()
                or not isinstance(source, str)
                or not source.strip()
                or not isinstance(source_id, str)
                or not source_id.strip()
            ):
                errors.append("candidate snapshot has a blank paper/source identity")
                continue
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                errors.append("candidate snapshot has a non-HTTP(S) URL")
            abstract = candidate.get("abstract")
            if isinstance(abstract, str) and abstract.strip().casefold() in {
                "no abstract",
                "not provided",
                "no abstract available",
            }:
                errors.append("candidate snapshot contains a fabricated abstract placeholder")
            paper_ids.append(paper_id)
            source_identities.append((source, source_id))
        if len(paper_ids) != len(set(paper_ids)):
            errors.append("candidate snapshot has duplicate paper IDs")
        if len(source_identities) != len(set(source_identities)):
            errors.append("candidate snapshot has duplicate source identities")
    exact_snapshot_sha256 = hashlib.sha256(raw_snapshot).hexdigest()
    if manifest.get("snapshot_sha256") != exact_snapshot_sha256 or fields[4] != exact_snapshot_sha256:
        errors.append("candidate snapshot exact SHA-256 does not match manifest/report")
    if (
        snapshot.get("source_evidence_report") != M1_REBASELINE_REPORT
        or snapshot.get("source_candidate_array_sha256") != M1_REBASELINE_CANDIDATE_ARRAY_SHA256
        or manifest.get("artifact_source_classification") != "M1_REBASELINE_SOURCE_BUNDLE"
        or manifest.get("source_bundle") != M1_REBASELINE_SOURCE_BUNDLE
    ):
        errors.append("candidate snapshot has invalid M1 rebaseline provenance")
    if manifest.get("source_id_coverage") != 1.0 or manifest.get("url_coverage") != 1.0:
        errors.append("candidate snapshot manifest must record 1.0 source-ID and URL coverage")
    replay = manifest.get("zero_transport_replay")
    if (
        manifest.get("candidate_count") != 33
        or not isinstance(replay, dict)
        or replay.get("transport_requests") != 0
        or replay.get("cache_hits") != 12
        or replay.get("query_count") != 12
        or replay.get("empty_cache_entry_count") != 1
        or manifest.get("metadata_mismatch_count") != 0
    ):
        errors.append("candidate snapshot manifest has invalid zero-transport replay audit")
    return exact_snapshot_sha256


def _validate_vector_snapshot(
    payload: dict[str, Any], commit: str | None, candidate_sha256: str | None, errors: list[str], repository_root: Path
) -> None:
    fields = _snapshot_fields(payload, "vector_snapshot", "canonical_sha256", errors, repository_root)
    loaded = _load_snapshot_json(commit, fields, "vector snapshot", errors, repository_root)
    if fields is None or loaded is None:
        return
    _, _, snapshot, manifest = loaded
    if snapshot.get("snapshot_version") != "m2-embedding-v1":
        errors.append("vector snapshot must use m2-embedding-v1")
    if candidate_sha256 is None or snapshot.get("candidate_snapshot_sha256") != candidate_sha256:
        errors.append("vector snapshot must reference the validated candidate snapshot")
    canonical = _canonical_sha256(snapshot)
    if manifest.get("vector_snapshot_sha256") != canonical or fields[4] != canonical:
        errors.append("vector snapshot canonical SHA-256 does not match manifest/report")
    records = snapshot.get("records")
    if not isinstance(records, list) or len(records) != 34:
        errors.append("vector snapshot must contain exactly 33 candidate and one query record")
        return
    input_ids: set[str] = set()
    dimensions: set[int] = set()
    query_count = 0
    for record in records:
        if not isinstance(record, dict):
            errors.append("vector snapshot records must be objects")
            continue
        input_id, dimension, vector = record.get("input_id"), record.get("dimension"), record.get("vector")
        if not isinstance(input_id, str) or not input_id or input_id in input_ids:
            errors.append("vector snapshot has blank or duplicate input IDs")
        elif input_id.startswith("query:"):
            query_count += 1
        input_ids.add(input_id) if isinstance(input_id, str) else None
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            errors.append("vector snapshot has a non-positive dimension")
            continue
        dimensions.add(dimension)
        if not isinstance(vector, list) or len(vector) != dimension or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
            for value in vector
        ):
            errors.append("vector snapshot contains a dimension mismatch or non-finite value")
    if query_count != 1 or len(dimensions) != 1:
        errors.append("vector snapshot must have one query and one positive uniform dimension")
    if manifest.get("candidate_count") != 33 or manifest.get("query_count") != 1:
        errors.append("vector manifest must record 33 candidate vectors and one query vector")


def _validate_checks(payload: dict[str, Any], errors: list[str]) -> None:
    checks = payload.get("automated_checks")
    if not isinstance(checks, list):
        errors.append("automated_checks must be a list")
        return
    by_name: dict[str, dict[str, Any]] = {}
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get("name"), str):
            errors.append("automated check must be an object with a name")
            continue
        name = check["name"]
        if name in by_name:
            errors.append(f"duplicate automated check: {name}")
        by_name[name] = check
    for name in sorted(REQUIRED_CHECKS - set(by_name)):
        errors.append(f"missing automated check: {name}")
    for name, check in by_name.items():
        if name in REQUIRED_CHECKS and (
            check.get("evidence_type") != "automated"
            or check.get("status") != "passed"
            or check.get("exit_code") != 0
        ):
            errors.append(f"automated check is not green: {name}")


def _validate_model_evidence(payload: dict[str, Any], errors: list[str]) -> None:
    fake = payload.get("fake_evidence")
    if not isinstance(fake, dict) or fake.get("classification") != "deterministic_fake":
        errors.append("fake_evidence must be explicitly classified deterministic_fake")
    else:
        descriptor = fake.get("provider")
        if not isinstance(descriptor, dict) or descriptor.get("provider_name") != "deterministic_fake" or descriptor.get("cache_namespace") != "embedding:fake":
            errors.append("fake evidence must use the fixed deterministic fake descriptor/namespace")

    real = payload.get("real_model_evidence")
    if not isinstance(real, dict) or real.get("classification") != "real":
        errors.append("real_model_evidence must be explicitly classified real")
        return
    descriptor = real.get("provider")
    if not isinstance(descriptor, dict):
        errors.append("real model evidence must include its provider descriptor")
        return
    revision, namespace, dimension = (
        descriptor.get("model_revision"),
        descriptor.get("cache_namespace"),
        descriptor.get("dimension"),
    )
    if descriptor.get("provider_name") == "deterministic_fake" or descriptor.get("model_id") != "BAAI/bge-m3":
        errors.append("real model evidence must identify BAAI/bge-m3, never fake")
    if not isinstance(revision, str) or not revision.strip() or revision.casefold() in {"main", "latest"}:
        errors.append("real model evidence requires an immutable model revision")
    if not isinstance(namespace, str) or not namespace.strip() or namespace == "embedding:fake":
        errors.append("fake and real cache namespaces must differ")
    if descriptor.get("embedding_mode") != "dense" or not isinstance(dimension, int) or dimension <= 0:
        errors.append("real model evidence must record a positive dense dimension")
    if real.get("candidate_vector_count") != 33 or real.get("query_vector_count") != 1:
        errors.append("real model evidence must record 33 candidate vectors and one query vector")
    if real.get("non_finite_count") != 0:
        errors.append("real model evidence must record zero non-finite vectors")
    live, replay = real.get("live_cache"), real.get("replay_cache")
    if not isinstance(live, dict) or live.get("cache_hits") != 0 or not isinstance(live.get("provider_call_count"), int) or live["provider_call_count"] <= 0:
        errors.append("real live cache evidence must have zero hits and positive provider calls")
    if not isinstance(replay, dict) or replay.get("provider_call_count") != 0 or replay.get("cache_misses") != 0 or replay.get("vector_arrays_equal") is not True:
        errors.append("real replay must have zero provider calls/misses and equal vectors")


def _validate_status(payload: dict[str, Any], errors: list[str], repository_root: Path) -> None:
    expected = {"m2": "IN_PROGRESS 1/5", "m3": "BLOCKED_BY_M2"}
    if payload.get("status_after_evidence") != expected:
        errors.append("status_after_evidence must declare M2 IN_PROGRESS 1/5 and M3 BLOCKED_BY_M2")
    phases = _parse_status(repository_root, errors)
    if phases.get("M1") != ("COMPLETE", 4, 4):
        errors.append("completed M2-T01 evidence must retain M1 COMPLETE 4/4")
    if phases.get("M2") != ("IN_PROGRESS", 1, 5):
        errors.append("completed M2-T01 evidence requires M2 IN_PROGRESS 1/5")
    if phases.get("M3") != ("BLOCKED_BY_M2", 0, 5):
        errors.append("completed M2-T01 evidence must retain M3 BLOCKED_BY_M2 0/5")


def validate_m2_t01_evidence(
    report_path: Path = REPORT_PATH, *, repository_root: Path = ROOT
) -> EvidenceValidationResult:
    """Return a validation result for either truthful pre-completion or completion state."""

    errors: list[str] = []
    absolute_report = report_path if report_path.is_absolute() else repository_root / report_path
    if not absolute_report.exists():
        _validate_precompletion_status(repository_root, errors)
        return EvidenceValidationResult(not errors, errors)
    try:
        payload = json.loads(absolute_report.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return EvidenceValidationResult(False, [f"cannot load M2-T01 report: {error}"])
    if not isinstance(payload, dict):
        return EvidenceValidationResult(False, ["M2-T01 report must be a JSON object"])
    if payload.get("report_version") != REPORT_VERSION:
        errors.append(f"report_version must be {REPORT_VERSION}")
    if payload.get("phase") != "M2" or payload.get("task_id") != "M2-T01":
        errors.append("report must identify phase M2 and task M2-T01")
    validated_commit = _validate_provenance(payload, errors, repository_root)
    _validate_inputs(payload, validated_commit, errors, repository_root)
    candidate_sha256 = _validate_candidate_snapshot(payload, validated_commit, errors, repository_root)
    _validate_vector_snapshot(payload, validated_commit, candidate_sha256, errors, repository_root)
    _validate_checks(payload, errors)
    _validate_model_evidence(payload, errors)
    m1_provenance = payload.get("m1_provenance")
    if not isinstance(m1_provenance, dict) or m1_provenance.get("metadata_projection_mismatch_count") != 0:
        errors.append("M1 provenance audit must record metadata_projection_mismatch_count = 0")
    _validate_status(payload, errors, repository_root)
    return EvidenceValidationResult(not errors, errors)


def main() -> int:
    result = validate_m2_t01_evidence()
    for error in result.errors:
        print(f"ERROR: {error}")
    if not result.valid:
        return 1
    if (ROOT / REPORT_PATH).exists():
        print("PASS: completed M2-T01 evidence, snapshots, provenance, and status gate are valid.")
    else:
        print("PASS: M2-T01 remains truthfully pre-completion; no completed evidence report exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
