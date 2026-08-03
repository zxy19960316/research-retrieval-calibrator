"""Validate the first-phase M2-T03 human adjudication bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.evidence_classification import source_text_sha256
from app.core.intent import canonical_research_intent_bytes
from app.models.embedding import FrozenCandidate, FrozenCandidateSnapshot
from app.models.evidence_classification import (
    EvidenceClassificationRecord,
    EvidenceClassifierDescriptor,
)
from app.models.first_round import FirstRoundRun
from app.models.m1_replay_repair import M1ReplayRepairManifest
from app.models.m2_t03_human_adjudication import (
    ArtifactBinding,
    HumanAdjudicationBundle,
    HumanAdjudicationReceipt,
    HumanAdjudicationReport,
)
from scripts.build_m2_t03_human_adjudication_bundle import (
    BUNDLE_PATH,
    CANDIDATE_SNAPSHOT,
    CLASSIFICATION_RESULT,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_CANDIDATE_SNAPSHOT_SHA256,
    EXPECTED_CLASSIFICATION_RESULT_SHA256,
    EXPECTED_M1_REPAIR_FIRST_RUN_SHA256,
    EXPECTED_M1_REPAIR_MANIFEST_SHA256,
    EXPECTED_MAIN_MERGE_COMMIT,
    EXPECTED_COMMANDS,
    EXPECTED_EXIT_CODES,
    EXPECTED_PROTOCOL_SHA256,
    REVIEW_DISALLOWED_SOURCES,
    REVIEW_SOURCE_FIELDS,
    M1_REPAIR_FIRST_RUN,
    M1_REPAIR_MANIFEST,
    PROTOCOL_PATH,
    RECEIPT_PATH,
    REPORT_PATH,
    REVIEW_TEMPLATE_PATH,
    RUNNER_PATH,
    STATUS_PATH,
)

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STATUS_ROW = re.compile(r"^\|\s*(M\d+)\b[^|]*\|\s*([A-Z0-9_]+)\s*\|\s*(\d+/\d+)\s*\|", re.MULTILINE)
_FORBIDDEN_KEY_RE = re.compile(r"(?:score|rank|selection|weight|total_score)", re.IGNORECASE)
_FORBIDDEN_TEXT_RE = re.compile(
    r"(?:file://|authorization(?:\s+headers?)?|cookie|bearer\s|access[_-]?token|api[_-]?key|"
    r"credentials?|passwords?|tokens?|raw exceptions?|tracebacks?)",
    re.IGNORECASE,
)
_MACHINE_TEMPLATE_KEYS = {
    "machine_advisory",
    "evidence_slot",
    "support_level",
    "reason",
    "supporting_excerpt",
}
_COMPLETED_RESULT_PATH = Path("evaluation/source-artifacts/m2-t03-human-adjudication-result.json")
_COMPLETED_RECEIPT_PATH = Path(
    "evaluation/source-artifacts/m2-t03-human-adjudication-result-receipt.json"
)
_COMPLETED_REPORT_PATH = Path("evaluation/reports/m2-t03-human-adjudication-result.json")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular(path: Path) -> bytes:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise OSError
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if stat.S_ISLNK(opened.st_mode) or not stat.S_ISREG(opened.st_mode):
            raise OSError
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise OSError
        data = handle.read()
        final = os.fstat(handle.fileno())
        if (final.st_dev, final.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise OSError
        return data


def _load_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = _read_regular(path)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError
    return raw, cast(dict[str, Any], value)


def _git(repository_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=repository_root, capture_output=True, check=False, text=True
    )


def _git_bytes(repository_root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments], cwd=repository_root, capture_output=True, check=False
    )


def _is_ancestor(repository_root: Path, commit: str) -> bool:
    if _SHA1_RE.fullmatch(commit) is None:
        return False
    return _git(repository_root, "merge-base", "--is-ancestor", commit, "HEAD").returncode == 0


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _historical_status_bytes(
    repository_root: Path, generated_from_commit: str, errors: list[str]
) -> bytes | None:
    if not _is_ancestor(repository_root, generated_from_commit):
        errors.append("generated_from_commit is not an ancestor of HEAD")
        return None
    completed = _git_bytes(
        repository_root, "show", f"{generated_from_commit}:{STATUS_PATH.as_posix()}"
    )
    if completed.returncode != 0:
        errors.append("historical STATUS.md is unavailable")
        return None
    status = completed.stdout
    try:
        decoded = status.decode("utf-8")
    except UnicodeDecodeError:
        errors.append("historical STATUS.md is not UTF-8")
        return None
    rows = {phase: (state, count) for phase, state, count in _STATUS_ROW.findall(decoded)}
    if rows.get("M2") != ("IN_PROGRESS", "2/5"):
        errors.append("historical STATUS.md does not retain M2 IN_PROGRESS 2/5")
    if rows.get("M3") != ("BLOCKED_BY_M2", "0/5"):
        errors.append("historical STATUS.md does not retain M3 BLOCKED_BY_M2 0/5")
    return status


def _binding_matches(
    binding: ArtifactBinding,
    *,
    repository_root: Path,
    expected_path: Path,
    expected_sha256: str,
) -> bool:
    if binding.path != expected_path.as_posix() or binding.sha256 != expected_sha256:
        return False
    try:
        return _sha256(_read_regular(repository_root / expected_path)) == expected_sha256
    except OSError:
        return False


def _validate_privacy(value: object, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and _FORBIDDEN_KEY_RE.search(key):
                errors.append("bundle contains a forbidden score or selection field")
            _validate_privacy(item, errors)
    elif isinstance(value, list | tuple):
        for item in value:
            _validate_privacy(item, errors)
    elif isinstance(value, str) and (
        value.startswith(("/", "\\\\"))
        or bool(re.match(r"^[A-Za-z]:[\\/]", value))
        or _FORBIDDEN_TEXT_RE.search(value)
    ):
        errors.append("bundle contains forbidden path, credential, or raw-error text")


def _validate_intent(
    bundle: HumanAdjudicationBundle, repository_root: Path, errors: list[str]
) -> None:
    context = bundle.intent_context
    if not _binding_matches(
        context.source_artifact,
        repository_root=repository_root,
        expected_path=M1_REPAIR_FIRST_RUN,
        expected_sha256=EXPECTED_M1_REPAIR_FIRST_RUN_SHA256,
    ):
        errors.append("bundle does not bind the repaired M1 first-run artifact")
    if not _binding_matches(
        context.repair_manifest,
        repository_root=repository_root,
        expected_path=M1_REPAIR_MANIFEST,
        expected_sha256=EXPECTED_M1_REPAIR_MANIFEST_SHA256,
    ):
        errors.append("bundle does not bind the M1 repair manifest")
    try:
        manifest = M1ReplayRepairManifest.model_validate(
            _load_json(repository_root / M1_REPAIR_MANIFEST)[1]
        )
        first_run = FirstRoundRun.model_validate(
            _load_json(repository_root / M1_REPAIR_FIRST_RUN)[1]
        )
        if first_run.intent is None:
            raise ValueError
        if context.research_intent != first_run.intent:
            raise ValueError
        if _sha256(canonical_research_intent_bytes(context.research_intent)) != context.canonical_intent_sha256:
            raise ValueError
        if context.canonical_intent_sha256 != manifest.canonical_intent_sha256:
            raise ValueError
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, ValidationError):
        errors.append("bundle ResearchIntent is not identical to the repaired first-run intent")


def _validate_candidates_and_advisories(
    bundle: HumanAdjudicationBundle, repository_root: Path, errors: list[str]
) -> None:
    if not _binding_matches(
        bundle.candidate_snapshot,
        repository_root=repository_root,
        expected_path=CANDIDATE_SNAPSHOT,
        expected_sha256=EXPECTED_CANDIDATE_SNAPSHOT_SHA256,
    ):
        errors.append("bundle does not bind the protected M2 candidate snapshot")
    if not _binding_matches(
        bundle.classification_result,
        repository_root=repository_root,
        expected_path=CLASSIFICATION_RESULT,
        expected_sha256=EXPECTED_CLASSIFICATION_RESULT_SHA256,
    ):
        errors.append("bundle does not bind the fixed M2-T03 classification result")
    try:
        snapshot_payload = _load_json(repository_root / CANDIDATE_SNAPSHOT)[1]
        raw_candidates = snapshot_payload["candidates"]
        if not isinstance(raw_candidates, list):
            raise TypeError
        snapshot = FrozenCandidateSnapshot(
            snapshot_version=snapshot_payload["snapshot_version"],
            question=snapshot_payload["question"],
            candidates=[FrozenCandidate.model_validate(item) for item in raw_candidates],
        )
        result = _load_json(repository_root / CLASSIFICATION_RESULT)[1]
        descriptor = EvidenceClassifierDescriptor.model_validate(result["classifier"])
        records = [EvidenceClassificationRecord.model_validate(item) for item in result["records"]]
        records_by_id = {record.paper_id: record for record in records}
        if len(snapshot.candidates) != EXPECTED_CANDIDATE_COUNT or len(records_by_id) != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        expected_order = tuple(candidate.paper_id for candidate in snapshot.candidates)
        actual_order = tuple(item.context.paper_id for item in bundle.items)
        if bundle.paper_id_order != expected_order or actual_order != expected_order:
            raise ValueError
        for item, candidate in zip(bundle.items, snapshot.candidates, strict=True):
            record = records_by_id.get(candidate.paper_id)
            if record is None:
                raise ValueError
            expected_hash = source_text_sha256(candidate.title, candidate.abstract)
            if item.context.model_dump(mode="json") != {
                "abstract": candidate.abstract,
                "paper_id": candidate.paper_id,
                "source_identity": {
                    "source": candidate.source,
                    "source_id": candidate.source_id,
                    "url": candidate.url,
                },
                "source_text_sha256": expected_hash,
                "title": candidate.title,
            }:
                raise ValueError
            if item.machine_advisory.paper_id != record.paper_id:
                raise ValueError
            if item.machine_advisory.classifier_descriptor != descriptor:
                raise ValueError
            if item.machine_advisory.source_text_sha256 != record.source_text_sha256:
                raise ValueError
            if item.machine_advisory.model_dump(mode="json") != {
                "advisory_only": True,
                "classifier_descriptor": record.classifier_descriptor.model_dump(mode="json"),
                "classification_version": record.classification_version,
                "evidence_type": "deterministic_fake",
                "evidence_slot": record.evidence_slot,
                "paper_id": record.paper_id,
                "reason": record.reason,
                "source_text_sha256": record.source_text_sha256,
                "state": record.state,
                "support_level": record.support_level,
                "supporting_excerpt": record.supporting_excerpt,
            }:
                raise ValueError
            if any(value is not None for value in item.human_adjudication.model_dump().values()):
                raise ValueError
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationError):
        errors.append("bundle context or advisory classification does not match protected inputs")


def _validate_policy(
    bundle: HumanAdjudicationBundle, repository_root: Path, errors: list[str]
) -> None:
    if bundle.review_policy.source_fields != REVIEW_SOURCE_FIELDS:
        errors.append("review policy source_fields are not the frozen protocol order")
    if bundle.review_policy.disallowed_sources != REVIEW_DISALLOWED_SOURCES:
        errors.append("review policy disallowed_sources are not the frozen protocol order")
    protocol = bundle.review_policy.protocol
    if protocol.path != PROTOCOL_PATH.as_posix() or protocol.sha256 != EXPECTED_PROTOCOL_SHA256:
        errors.append("review protocol binding is invalid")
    else:
        try:
            if _sha256(_read_regular(repository_root / PROTOCOL_PATH)) != EXPECTED_PROTOCOL_SHA256:
                errors.append("review protocol bytes do not match the frozen hash")
        except OSError:
            errors.append("review protocol is unavailable")


def _validate_review_template(
    template_payload: dict[str, Any],
    bundle: HumanAdjudicationBundle,
    bundle_raw: bytes,
    repository_root: Path,
    errors: list[str],
) -> None:
    expected_keys = {
        "template_version",
        "pending_bundle",
        "review_protocol",
        "candidate_count",
        "paper_id_order",
        "items",
    }
    if set(template_payload) != expected_keys:
        errors.append("blind review template has unexpected top-level fields")
        return
    if template_payload["template_version"] != "m2-t03-human-adjudication-template.v1":
        errors.append("blind review template version is invalid")
    if template_payload["pending_bundle"] != {
        "path": BUNDLE_PATH.as_posix(),
        "sha256": _sha256(bundle_raw),
    }:
        errors.append("blind review template bundle binding is invalid")
    if template_payload["review_protocol"] != bundle.review_policy.protocol.model_dump(mode="json"):
        errors.append("blind review template protocol binding is invalid")
    if template_payload["candidate_count"] != EXPECTED_CANDIDATE_COUNT:
        errors.append("blind review template candidate count is invalid")
    if template_payload["paper_id_order"] != list(bundle.paper_id_order):
        errors.append("blind review template paper ID order is invalid")
    items = template_payload["items"]
    if not isinstance(items, list) or len(items) != EXPECTED_CANDIDATE_COUNT:
        errors.append("blind review template does not contain exactly 33 items")
        return
    expected_human_fields = {
        "verdict",
        "evidence_slot",
        "support_level",
        "grounded_reason",
        "supporting_excerpt",
        "reviewer_id",
        "reviewed_at_utc",
        "notes",
    }
    for expected_item, actual_item in zip(bundle.items, items, strict=True):
        if not isinstance(actual_item, dict):
            errors.append("blind review template item is not an object")
            continue
        if set(actual_item) != {
            "paper_id",
            "title",
            "abstract",
            "source_identity",
            "source_text_sha256",
            "human_adjudication",
        }:
            errors.append("blind review template item exposes a forbidden field")
            continue
        if actual_item["human_adjudication"] != dict.fromkeys(expected_human_fields):
            errors.append("blind review template human fields are not empty")
        if actual_item["human_adjudication"] and set(actual_item["human_adjudication"]) != expected_human_fields:
            errors.append("blind review template human field set is invalid")
        expected_context = expected_item.context.model_dump(mode="json")
        actual_context = {key: actual_item[key] for key in expected_context}
        if actual_context != expected_context:
            errors.append("blind review template context does not match pending bundle")
        if any(key in actual_item for key in _MACHINE_TEMPLATE_KEYS):
            errors.append("blind review template exposes machine advisory content")


def _validate_receipt(
    receipt: HumanAdjudicationReceipt,
    bundle_raw: bytes,
    template_raw: bytes,
    repository_root: Path,
    errors: list[str],
) -> None:
    if receipt.bundle_path != BUNDLE_PATH.as_posix() or receipt.bundle_sha256 != _sha256(bundle_raw):
        errors.append("receipt bundle hash binding is invalid")
    if receipt.review_template != ArtifactBinding(
        path=REVIEW_TEMPLATE_PATH.as_posix(), sha256=_sha256(template_raw)
    ):
        errors.append("receipt review template binding is invalid")
    try:
        runner_matches = receipt.runner_path == RUNNER_PATH.as_posix() and receipt.runner_sha256 == _sha256(
            _read_regular(repository_root / RUNNER_PATH)
        )
    except OSError:
        runner_matches = False
    if not runner_matches:
        errors.append("receipt runner binding is invalid")
    if not _is_ancestor(repository_root, receipt.generated_from_commit):
        errors.append("receipt generated_from_commit is not an ancestor of HEAD")


def _validate_report(
    report: HumanAdjudicationReport,
    receipt_raw: bytes,
    bundle_raw: bytes,
    template_raw: bytes,
    bundle: HumanAdjudicationBundle,
    historical_status: bytes | None,
    repository_root: Path,
    errors: list[str],
) -> None:
    expected_artifacts = {
        BUNDLE_PATH.as_posix(): _sha256(bundle_raw),
        RECEIPT_PATH.as_posix(): _sha256(receipt_raw),
        REVIEW_TEMPLATE_PATH.as_posix(): _sha256(template_raw),
    }
    actual_artifacts = {item.path: item.sha256 for item in report.artifact}
    if actual_artifacts != expected_artifacts:
        errors.append("report artifact hashes are invalid")
    if not _is_ancestor(repository_root, report.generated_from_commit):
        errors.append("report generated_from_commit is not an ancestor of HEAD")
    if report.generated_at_utc != bundle.generated_at_utc:
        errors.append("bundle and report generated_at_utc values differ")
    if report.commands != EXPECTED_COMMANDS:
        errors.append("report command set is not fixed")
    if report.exit_codes != EXPECTED_EXIT_CODES:
        errors.append("report exit code map is not fixed")
    expected_inputs = None
    if historical_status is not None:
        expected_inputs = (
            ArtifactBinding(
                path=M1_REPAIR_FIRST_RUN.as_posix(), sha256=EXPECTED_M1_REPAIR_FIRST_RUN_SHA256
            ),
            ArtifactBinding(
                path=M1_REPAIR_MANIFEST.as_posix(), sha256=EXPECTED_M1_REPAIR_MANIFEST_SHA256
            ),
            ArtifactBinding(
                path=CANDIDATE_SNAPSHOT.as_posix(), sha256=EXPECTED_CANDIDATE_SNAPSHOT_SHA256
            ),
            ArtifactBinding(
                path=CLASSIFICATION_RESULT.as_posix(),
                sha256=EXPECTED_CLASSIFICATION_RESULT_SHA256,
            ),
            bundle.review_policy.protocol,
            ArtifactBinding(path=STATUS_PATH.as_posix(), sha256=_sha256(historical_status)),
        )
    if expected_inputs is None or report.input_artifacts != expected_inputs:
        errors.append("report input artifact bindings are invalid")


def validate_bundle(
    bundle_path: Path = BUNDLE_PATH,
    *,
    repository_root: Path = ROOT,
) -> ValidationResult:
    errors: list[str] = []
    historical_status: bytes | None = None
    try:
        bundle_raw, bundle_payload = _load_json(
            bundle_path if bundle_path.is_absolute() else repository_root / bundle_path
        )
        receipt_raw, receipt_payload = _load_json(repository_root / RECEIPT_PATH)
        _, report_payload = _load_json(repository_root / REPORT_PATH)
        template_raw, template_payload = _load_json(repository_root / REVIEW_TEMPLATE_PATH)
        bundle = HumanAdjudicationBundle.model_validate(bundle_payload)
        receipt = HumanAdjudicationReceipt.model_validate(receipt_payload)
        report = HumanAdjudicationReport.model_validate(report_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, ValidationError):
        return ValidationResult(False, ["cannot load closed M2-T03 human adjudication artifacts"])

    if not _is_ancestor(repository_root, EXPECTED_MAIN_MERGE_COMMIT):
        if repository_root.resolve() == ROOT.resolve():
            errors.append("PR #14 main merge commit is not an ancestor of HEAD")
    if bundle.generated_from_commit != receipt.generated_from_commit:
        errors.append("bundle and receipt generation commits differ")
    if bundle.generated_from_commit != report.generated_from_commit:
        errors.append("bundle and report generation commits differ")
    if bundle.candidate_count != EXPECTED_CANDIDATE_COUNT or len(bundle.items) != EXPECTED_CANDIDATE_COUNT:
        errors.append("bundle does not contain exactly 33 review items")
    if bundle.review_policy.machine_labels_are_advisory is not True:
        errors.append("machine classifications are not marked advisory_only")
    if bundle.review_policy.human_fields_initially_empty is not True:
        errors.append("human fields are not marked initially empty")
    if bundle.status_boundary.m2_progress != "IN_PROGRESS 2/5":
        errors.append("bundle advances M2 beyond IN_PROGRESS 2/5")
    if bundle.status_boundary.scoring_eligible is not False:
        errors.append("bundle is incorrectly scoring eligible")
    historical_status = _historical_status_bytes(
        repository_root, bundle.generated_from_commit, errors
    )
    _validate_intent(bundle, repository_root, errors)
    _validate_candidates_and_advisories(bundle, repository_root, errors)
    _validate_policy(bundle, repository_root, errors)
    _validate_receipt(receipt, bundle_raw, template_raw, repository_root, errors)
    _validate_report(
        report,
        receipt_raw,
        bundle_raw,
        template_raw,
        bundle,
        historical_status,
        repository_root,
        errors,
    )
    _validate_review_template(template_payload, bundle, bundle_raw, repository_root, errors)
    _validate_privacy(bundle_payload, errors)
    _validate_privacy(receipt_payload, errors)
    _validate_privacy(report_payload, errors)
    return ValidationResult(not errors, errors)


def validate_current_m2_gate(
    repository_root: Path = ROOT,
    *,
    completed_result_validator: Callable[[Path], ValidationResult | bool] | None = None,
) -> ValidationResult:
    """Validate only the current worktree transition gate, not pending history."""

    errors: list[str] = []
    try:
        status = _read_regular(repository_root / STATUS_PATH).decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return ValidationResult(False, ["current STATUS.md is unavailable"])
    rows = {phase: (state, count) for phase, state, count in _STATUS_ROW.findall(status)}
    if rows.get("M3") != ("BLOCKED_BY_M2", "0/5"):
        errors.append("current STATUS.md does not retain M3 BLOCKED_BY_M2 0/5")
    current_m2 = rows.get("M2")
    if current_m2 == ("IN_PROGRESS", "2/5"):
        return ValidationResult(not errors, errors)
    if current_m2 != ("IN_PROGRESS", "3/5"):
        errors.append("current STATUS.md has an unsupported M2 transition")
        return ValidationResult(False, errors)
    result_paths = (
        repository_root / _COMPLETED_RESULT_PATH,
        repository_root / _COMPLETED_RECEIPT_PATH,
        repository_root / _COMPLETED_REPORT_PATH,
    )
    if not all(path.is_file() for path in result_paths):
        errors.append("M2 3/5 requires all completed human-result artifacts")
        return ValidationResult(False, errors)
    if completed_result_validator is None:
        errors.append("M2 3/5 requires a passing completed-result validator")
        return ValidationResult(False, errors)
    try:
        outcome = completed_result_validator(repository_root)
    except Exception:
        return ValidationResult(False, ["completed-result validator failed"])
    if isinstance(outcome, ValidationResult):
        if not outcome.valid:
            errors.append("completed-result validator did not pass")
    elif outcome is not True:
        errors.append("completed-result validator did not pass")
    return ValidationResult(not errors, errors)


def main() -> int:
    result = validate_bundle()
    if not result.valid:
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1
    print("PASS: M2-T03 human adjudication context/review bundle is closed, advisory-only, and pending human review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
