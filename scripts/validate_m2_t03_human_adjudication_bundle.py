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
    M1_REPAIR_FIRST_RUN,
    M1_REPAIR_MANIFEST,
    RECEIPT_PATH,
    REPORT_PATH,
    RUNNER_PATH,
    STATUS_PATH,
)

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STATUS_ROW = re.compile(r"^\|\s*(M\d+)\b[^|]*\|\s*([A-Z0-9_]+)\s*\|\s*(\d+/\d+)\s*\|", re.MULTILINE)
_FORBIDDEN_KEY_RE = re.compile(r"(?:score|rank|selection|weight|total_score)", re.IGNORECASE)
_FORBIDDEN_TEXT_RE = re.compile(
    r"(?:file://|authorization|cookie|bearer\s|access[_-]?token|api[_-]?key|password|raw exception|traceback)",
    re.IGNORECASE,
)
_ABSOLUTE_PATH_RE = re.compile(r"(?:^[A-Za-z]:[\\/]|^\\\\|/(?:Users|home|tmp)/)")


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


def _is_ancestor(repository_root: Path, commit: str) -> bool:
    if _SHA1_RE.fullmatch(commit) is None:
        return False
    return _git(repository_root, "merge-base", "--is-ancestor", commit, "HEAD").returncode == 0


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


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
        _ABSOLUTE_PATH_RE.search(value) or _FORBIDDEN_TEXT_RE.search(value)
    ):
        errors.append("bundle contains forbidden path, credential, or raw-error text")


def _validate_status(repository_root: Path, errors: list[str]) -> None:
    try:
        text = _read_regular(repository_root / STATUS_PATH).decode("utf-8")
        rows = {phase: (state, count) for phase, state, count in _STATUS_ROW.findall(text)}
        if rows.get("M2") != ("IN_PROGRESS", "2/5"):
            errors.append("STATUS.md does not retain M2 IN_PROGRESS 2/5")
        if rows.get("M3") != ("BLOCKED_BY_M2", "0/5"):
            errors.append("STATUS.md does not retain M3 BLOCKED_BY_M2 0/5")
    except (OSError, UnicodeDecodeError):
        errors.append("STATUS.md is unavailable")


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


def _validate_receipt(
    receipt: HumanAdjudicationReceipt,
    bundle_raw: bytes,
    repository_root: Path,
    errors: list[str],
) -> None:
    if receipt.bundle_path != BUNDLE_PATH.as_posix() or receipt.bundle_sha256 != _sha256(bundle_raw):
        errors.append("receipt bundle hash binding is invalid")
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
    repository_root: Path,
    errors: list[str],
) -> None:
    expected_artifacts = {
        BUNDLE_PATH.as_posix(): _sha256(bundle_raw),
        RECEIPT_PATH.as_posix(): _sha256(receipt_raw),
    }
    actual_artifacts = {item.path: item.sha256 for item in report.artifact}
    if actual_artifacts != expected_artifacts:
        errors.append("report artifact hashes are invalid")
    if not _is_ancestor(repository_root, report.generated_from_commit):
        errors.append("report generated_from_commit is not an ancestor of HEAD")
    if report.input_artifacts != (
        ArtifactBinding(path=M1_REPAIR_FIRST_RUN.as_posix(), sha256=EXPECTED_M1_REPAIR_FIRST_RUN_SHA256),
        ArtifactBinding(path=M1_REPAIR_MANIFEST.as_posix(), sha256=EXPECTED_M1_REPAIR_MANIFEST_SHA256),
        ArtifactBinding(path=CANDIDATE_SNAPSHOT.as_posix(), sha256=EXPECTED_CANDIDATE_SNAPSHOT_SHA256),
        ArtifactBinding(path=CLASSIFICATION_RESULT.as_posix(), sha256=EXPECTED_CLASSIFICATION_RESULT_SHA256),
        ArtifactBinding(path=STATUS_PATH.as_posix(), sha256=_sha256(_read_regular(repository_root / STATUS_PATH))),
    ):
        errors.append("report input artifact bindings are invalid")


def validate_bundle(
    bundle_path: Path = BUNDLE_PATH,
    *,
    repository_root: Path = ROOT,
) -> ValidationResult:
    errors: list[str] = []
    try:
        bundle_raw, bundle_payload = _load_json(
            bundle_path if bundle_path.is_absolute() else repository_root / bundle_path
        )
        receipt_raw, receipt_payload = _load_json(repository_root / RECEIPT_PATH)
        _, report_payload = _load_json(repository_root / REPORT_PATH)
        bundle = HumanAdjudicationBundle.model_validate(bundle_payload)
        receipt = HumanAdjudicationReceipt.model_validate(receipt_payload)
        report = HumanAdjudicationReport.model_validate(report_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, ValidationError):
        return ValidationResult(False, ["cannot load closed M2-T03 human adjudication artifacts"])

    if not _is_ancestor(repository_root, EXPECTED_MAIN_MERGE_COMMIT):
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
    _validate_intent(bundle, repository_root, errors)
    _validate_candidates_and_advisories(bundle, repository_root, errors)
    _validate_receipt(receipt, bundle_raw, repository_root, errors)
    _validate_report(report, receipt_raw, bundle_raw, repository_root, errors)
    _validate_status(repository_root, errors)
    _validate_privacy(bundle_payload, errors)
    _validate_privacy(receipt_payload, errors)
    _validate_privacy(report_payload, errors)
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
