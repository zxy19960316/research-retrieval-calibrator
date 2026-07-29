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
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.embedding import build_embedding_input_from_fields, build_query_embedding_input

REPORT_PATH = Path("evaluation/reports/m2-t01-embedding.json")
REPORT_VERSION = "m2-t01-embedding.v1"
BGE_RUN_AUDIT_PATH = "evaluation/reports/m2-t01-bge-run-2026-07-29.json"
BGE_RUN_AUDIT_VERSION = "m2-t01-bge-run.v1"
M2_IMPLEMENTATION_COMMIT = "84cd61261c4f496e6b8255ad44c3849ed98841c2"
M2_SNAPSHOT_COMMIT = "438e7ad79a8d9e8085ea7eccefce68f5be85abec"
BGE_M3_WEIGHT_SHA256 = "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"
BGE_M3_WEIGHT_SIZE_BYTES = 2271145830
BASELINE_COMMIT = "0eb45fc22d10adb72cb66aa45494333057080bc1"
M1_REBASELINE_REPORT = "evaluation/reports/m1-rebaseline-2026-07-28.json"
M1_REBASELINE_SOURCE_BUNDLE = "evaluation/source-artifacts/m1-rebaseline-2026-07-28"
M1_REBASELINE_CANDIDATE_ARRAY_SHA256 = "e51eb84d4e772bba324a199478caa5f0983edef0b0dbf4c58fce5f9da5803209"
SHA1 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
HF_COMMIT = re.compile(r"[0-9a-f]{40}")
BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3_FLAGEMBEDDING_VERSION = "1.3.5"
B1_CANDIDATE_SNAPSHOT_SHA256 = "4a2aec0fd0a1d22adc801fd3bc506e5da89895d1276cd572e2ac64014c162448"
B2_PATHS = (
    "evaluation/snapshots/m2/m1-candidates.v1.json",
    "evaluation/snapshots/m2/m1-candidates.v1.manifest.json",
    "evaluation/snapshots/m2/bge-m3-dense-v1.json",
    "evaluation/snapshots/m2/bge-m3-dense-v1.manifest.json",
    BGE_RUN_AUDIT_PATH,
)
FORBIDDEN_PROVIDER_VERSIONS = frozenset({"optional", "unknown", "latest", "unavailable"})
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


@dataclass(frozen=True)
class CandidateEvidenceContext:
    snapshot_sha256: str
    expected_input_ids: frozenset[str]
    expected_text_sha256_by_input_id: dict[str, str]


@dataclass(frozen=True)
class VectorEvidenceContext:
    canonical_sha256: str
    provider: dict[str, object]
    runtime: dict[str, object]
    stats: dict[str, object]


@dataclass(frozen=True)
class BgeRunAuditContext:
    live: dict[str, object]
    replay: dict[str, object]
    canonical_sha256: str


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
) -> CandidateEvidenceContext | None:
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
    expected_text_sha256_by_input_id: dict[str, str] = {}
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
            title = candidate.get("title")
            abstract = candidate.get("abstract")
            if not isinstance(title, str) or not isinstance(abstract, (str, type(None))):
                errors.append("candidate snapshot lacks production embedding text fields")
                continue
            try:
                embedding_input = build_embedding_input_from_fields(
                    paper_id=paper_id,
                    title=title,
                    abstract=abstract,
                    source_snapshot_sha256=B1_CANDIDATE_SNAPSHOT_SHA256,
                )
            except ValueError:
                errors.append("candidate snapshot has invalid production embedding text")
                continue
            expected_text_sha256_by_input_id[embedding_input.input_id] = embedding_input.text_sha256
        if len(paper_ids) != len(set(paper_ids)):
            errors.append("candidate snapshot has duplicate paper IDs")
        if len(source_identities) != len(set(source_identities)):
            errors.append("candidate snapshot has duplicate source identities")
    exact_snapshot_sha256 = hashlib.sha256(raw_snapshot).hexdigest()
    if (
        exact_snapshot_sha256 != B1_CANDIDATE_SNAPSHOT_SHA256
        or manifest.get("snapshot_sha256") != exact_snapshot_sha256
        or fields[4] != exact_snapshot_sha256
    ):
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
    question = snapshot.get("question")
    if not isinstance(question, str):
        errors.append("candidate snapshot question must be available for query binding")
    else:
        try:
            query_input = build_query_embedding_input(question, B1_CANDIDATE_SNAPSHOT_SHA256)
        except ValueError:
            errors.append("candidate snapshot question has invalid production embedding text")
        else:
            expected_text_sha256_by_input_id[query_input.input_id] = query_input.text_sha256
    if len(expected_text_sha256_by_input_id) != 34:
        errors.append("candidate snapshot must derive exactly 33 paper and one query embedding inputs")
    return CandidateEvidenceContext(
        snapshot_sha256=exact_snapshot_sha256,
        expected_input_ids=frozenset(expected_text_sha256_by_input_id),
        expected_text_sha256_by_input_id=expected_text_sha256_by_input_id,
    )


def _validate_vector_snapshot(
    payload: dict[str, Any],
    commit: str | None,
    candidate_context: CandidateEvidenceContext | None,
    errors: list[str],
    repository_root: Path,
) -> VectorEvidenceContext | None:
    fields = _snapshot_fields(payload, "vector_snapshot", "canonical_sha256", errors, repository_root)
    loaded = _load_snapshot_json(commit, fields, "vector snapshot", errors, repository_root)
    if fields is None or loaded is None:
        return None
    _, _, snapshot, manifest = loaded
    if snapshot.get("snapshot_version") != "m2-embedding-v1":
        errors.append("vector snapshot must use m2-embedding-v1")
    if (
        candidate_context is None
        or snapshot.get("candidate_snapshot_sha256") != candidate_context.snapshot_sha256
        or manifest.get("candidate_snapshot_sha256") != candidate_context.snapshot_sha256
        or candidate_context.snapshot_sha256 != B1_CANDIDATE_SNAPSHOT_SHA256
    ):
        errors.append("vector snapshot must reference the validated candidate snapshot")
    canonical = _canonical_sha256(snapshot)
    if manifest.get("vector_snapshot_sha256") != canonical or fields[4] != canonical:
        errors.append("vector snapshot canonical SHA-256 does not match manifest/report")
    records = snapshot.get("records")
    if not isinstance(records, list) or len(records) != 34:
        errors.append("vector snapshot must contain exactly 33 candidate and one query record")
        return None
    provider = snapshot.get("provider")
    runtime = manifest.get("runtime")
    if not isinstance(provider, dict) or not isinstance(manifest.get("provider"), dict) or provider != manifest.get("provider"):
        errors.append("vector snapshot and manifest providers must match exactly")
        provider = {}
    _validate_real_provider_contract(provider, runtime, errors)
    input_ids: set[str] = set()
    non_unit_norm_count = 0
    for record in records:
        if not isinstance(record, dict):
            errors.append("vector snapshot records must be objects")
            continue
        input_id, dimension, vector = record.get("input_id"), record.get("dimension"), record.get("vector")
        if not isinstance(input_id, str) or not input_id or input_id in input_ids:
            errors.append("vector snapshot has blank or duplicate input IDs")
        input_ids.add(input_id) if isinstance(input_id, str) else None
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension != 1024:
            errors.append("vector snapshot records must be exactly 1024-dimensional")
            continue
        if not isinstance(vector, list) or len(vector) != dimension or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
            for value in vector
        ):
            errors.append("vector snapshot contains a dimension mismatch or non-finite value")
            continue
        if not isinstance(input_id, str) or record.get("text_sha256") != (
            candidate_context.expected_text_sha256_by_input_id.get(input_id) if candidate_context else None
        ):
            errors.append("vector snapshot record text SHA-256 does not bind to the B1 input")
        if record.get("descriptor") != provider:
            errors.append("vector snapshot record descriptor must match snapshot provider")
        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            non_unit_norm_count += 1
    if candidate_context is None or input_ids != candidate_context.expected_input_ids:
        errors.append("vector snapshot input IDs must exactly match the B1-derived input set")
    if manifest.get("candidate_count") != 33 or manifest.get("query_count") != 1:
        errors.append("vector manifest must record 33 candidate vectors and one query vector")
    if manifest.get("evidence_type") != "real":
        errors.append("vector manifest must be explicitly classified real")
    audit = manifest.get("vector_norm_audit")
    if audit != {
        "checked_count": 34,
        "non_unit_norm_count": non_unit_norm_count,
        "relative_tolerance": 0.001,
        "absolute_tolerance": 0.001,
    }:
        errors.append("vector manifest norm audit does not match actual vectors")
    if non_unit_norm_count:
        errors.append("vector snapshot contains non-unit vectors")
    stats = manifest.get("stats")
    if not isinstance(stats, dict):
        stats = {}
    return VectorEvidenceContext(
        canonical_sha256=canonical,
        provider=provider,
        runtime=runtime if isinstance(runtime, dict) else {},
        stats=stats,
    )


def _validate_real_provider_contract(
    provider: object, runtime: object, errors: list[str]
) -> None:
    """Reject unverifiable BGE-M3 descriptors and fabricated runtime metadata."""

    if not isinstance(provider, dict):
        errors.append("real model evidence must include its provider descriptor")
        return
    if provider.get("provider_name") != "bge_m3" or provider.get("model_id") != BGE_M3_MODEL_ID:
        errors.append("real model evidence must identify BAAI/bge-m3, never fake")
    revision = provider.get("model_revision")
    if not isinstance(revision, str) or HF_COMMIT.fullmatch(revision) is None or revision != BGE_M3_MODEL_REVISION:
        errors.append("real model evidence requires the fixed full 40-character model revision")
    library_version = provider.get("provider_library_version")
    if provider.get("provider_library") != "FlagEmbedding" or library_version != BGE_M3_FLAGEMBEDDING_VERSION:
        errors.append("real model evidence requires FlagEmbedding 1.3.5")
    if provider.get("embedding_mode") != "dense" or provider.get("dimension") != 1024 or provider.get("normalized") is not True:
        errors.append("real model evidence must record normalized 1024-dimensional dense vectors")
    namespace = provider.get("cache_namespace")
    if not isinstance(namespace, str) or not namespace.strip() or namespace == "embedding:fake":
        errors.append("fake and real cache namespaces must differ")
    if not isinstance(runtime, dict):
        errors.append("real model evidence must include complete runtime metadata")
        return
    for field in (
        "python_version",
        "flagembedding_version",
        "torch_version",
        "transformers_version",
        "huggingface_hub_version",
        "numpy_version",
    ):
        value = runtime.get(field)
        if not isinstance(value, str) or not value.strip() or value.casefold() in FORBIDDEN_PROVIDER_VERSIONS:
            errors.append(f"real runtime has an invalid {field}")
    if runtime.get("model_revision") != revision:
        errors.append("real runtime model revision does not match provider")
    if runtime.get("flagembedding_version") != BGE_M3_FLAGEMBEDDING_VERSION or runtime.get("flagembedding_version") != library_version:
        errors.append("real runtime FlagEmbedding version does not match provider")
    if runtime.get("use_fp16") is not False:
        errors.append("real runtime use_fp16 must be false")
    if runtime.get("device_request") not in {None, "cpu", "cuda", "cuda:0", "mps"}:
        errors.append("real runtime has an invalid device_request")


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


def _validate_model_evidence(
    payload: dict[str, Any], vector_context: VectorEvidenceContext | None, errors: list[str]
) -> None:
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
    _validate_real_provider_contract(real.get("provider"), real.get("runtime"), errors)
    if vector_context is None or real.get("provider") != vector_context.provider:
        errors.append("real model evidence provider must match vector manifest")
    if vector_context is None or real.get("runtime") != vector_context.runtime:
        errors.append("real model evidence runtime must match vector manifest")
    if real.get("candidate_vector_count") != 33 or real.get("query_vector_count") != 1:
        errors.append("real model evidence must record 33 candidate vectors and one query vector")
    if real.get("non_finite_count") != 0:
        errors.append("real model evidence must record zero non-finite vectors")
    if real.get("non_unit_norm_count") != 0:
        errors.append("real model evidence must record zero non-unit vectors")
    live, replay = real.get("live_cache"), real.get("replay_cache")
    if not isinstance(live, dict) or live.get("cache_hits") != 0 or not isinstance(live.get("provider_call_count"), int) or live["provider_call_count"] <= 0:
        errors.append("real live cache evidence must have zero hits and positive provider calls")
    if not isinstance(replay, dict) or replay.get("provider_call_count") != 0 or replay.get("cache_misses") != 0 or replay.get("vector_arrays_equal") is not True:
        errors.append("real replay must have zero provider calls/misses and equal vectors")


def _validate_bge_run_audit(
    audit: object, vector: VectorEvidenceContext | None, errors: list[str]
) -> BgeRunAuditContext | None:
    if not isinstance(audit, dict):
        errors.append("B2 run audit must be a JSON object")
        return None
    if (audit.get("report_version"), audit.get("phase"), audit.get("task_id")) != (
        BGE_RUN_AUDIT_VERSION, "M2", "M2-T01"
    ) or audit.get("a6_2_commit") != M2_IMPLEMENTATION_COMMIT or audit.get("evidence_type") != "real":
        errors.append("B2 run audit has invalid identity or implementation commit")
    policy = audit.get("download_policy")
    if not isinstance(policy, dict) or policy.get("allowlist") != "BGE_M3_REQUIRED_FILES" or policy.get("max_workers") != 1 or audit.get("xet_disabled") is not True or audit.get("onnx_file_count") != 0:
        errors.append("B2 run audit has invalid controlled download evidence")
    if audit.get("pytorch_model_sha256") != BGE_M3_WEIGHT_SHA256 or audit.get("pytorch_model_size_bytes") != BGE_M3_WEIGHT_SIZE_BYTES:
        errors.append("B2 run audit has invalid weight identity")
    expected_preflight = {"status": "success", "model_id": BGE_M3_MODEL_ID, "model_revision": BGE_M3_MODEL_REVISION, "dimension": 1024, "provider_library_version": BGE_M3_FLAGEMBEDDING_VERSION, "device": "cpu"}
    online, offline = audit.get("online_preflight"), audit.get("offline_preflight")
    if not isinstance(online, dict) or any(online.get(k) != v for k, v in expected_preflight.items()):
        errors.append("B2 run audit has invalid online preflight")
    if not isinstance(offline, dict) or any(offline.get(k) != v for k, v in expected_preflight.items()) or offline.get("offline") is not True:
        errors.append("B2 run audit has invalid offline preflight")
    live, replay = audit.get("live"), audit.get("replay")
    expected_live = {"cache_corrupt_count": 0, "cache_hits": 0, "cache_misses": 34, "provider_call_count": 17, "provider_input_count": 34}
    expected_replay = {"cache_corrupt_count": 0, "cache_hits": 34, "cache_misses": 0, "provider_call_count": 0, "provider_input_count": 0, "empty_model_cache_file_count": 0}
    if live != expected_live or replay != expected_replay or audit.get("vector_arrays_equal") is not True or audit.get("exact_bytes_equal") is not True:
        errors.append("B2 run audit has invalid live/replay evidence")
    canonical = audit.get("vector_snapshot_canonical_sha256")
    if not isinstance(canonical, str) or SHA256.fullmatch(canonical) is None or vector is None or canonical != vector.canonical_sha256 or live != vector.stats:
        errors.append("B2 run audit does not bind to vector manifest")
    return BgeRunAuditContext(live=live if isinstance(live, dict) else {}, replay=replay if isinstance(replay, dict) else {}, canonical_sha256=canonical if isinstance(canonical, str) else "")


def _validate_committed_b2_precompletion(repository_root: Path, errors: list[str]) -> None:
    present = [(repository_root / path).exists() for path in B2_PATHS[2:]]
    if not any(present):
        return
    if not all(present):
        errors.append("B2 evidence bundle is partial")
        return
    blobs = {path: _blob_bytes(M2_SNAPSHOT_COMMIT, path, repository_root) for path in B2_PATHS}
    if any(value is None for value in blobs.values()):
        errors.append("committed B2 evidence bundle is missing")
        return
    def fields(path: str, manifest: str, key: str) -> dict[str, str]:
        raw, raw_manifest = blobs[path], blobs[manifest]
        assert raw is not None and raw_manifest is not None
        artifact_hash = hashlib.sha256(raw).hexdigest() if key == "snapshot_sha256" else _canonical_sha256(json.loads(raw))
        return {"path": path, "blob_sha256": hashlib.sha256(raw).hexdigest(), "manifest_path": manifest, "manifest_blob_sha256": hashlib.sha256(raw_manifest).hexdigest(), key: artifact_hash}
    payload = {"candidate_snapshot": fields(B2_PATHS[0], B2_PATHS[1], "snapshot_sha256"), "vector_snapshot": fields(B2_PATHS[2], B2_PATHS[3], "canonical_sha256")}
    candidate = _validate_candidate_snapshot(payload, M2_SNAPSHOT_COMMIT, errors, repository_root)
    vector = _validate_vector_snapshot(payload, M2_SNAPSHOT_COMMIT, candidate, errors, repository_root)
    raw_audit = blobs[BGE_RUN_AUDIT_PATH]
    assert raw_audit is not None
    try:
        _validate_bge_run_audit(json.loads(raw_audit), vector, errors)
    except (UnicodeDecodeError, json.JSONDecodeError):
        errors.append("B2 run audit is invalid JSON")


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
        _validate_committed_b2_precompletion(repository_root, errors)
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
    candidate_context = _validate_candidate_snapshot(payload, validated_commit, errors, repository_root)
    vector_context = _validate_vector_snapshot(payload, validated_commit, candidate_context, errors, repository_root)
    _validate_checks(payload, errors)
    _validate_model_evidence(payload, vector_context, errors)
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
        print("PASS: M2-T01 remains truthfully pre-completion; committed B2 real-model evidence is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
