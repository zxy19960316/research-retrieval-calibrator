"""Build the offline, human-pending M2-T03 context and review bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
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
    M2_T03_HUMAN_ADJUDICATION_TEMPLATE_VERSION,
    ArtifactBinding,
    CandidateContext,
    HumanAdjudicationBundle,
    HumanAdjudicationFields,
    HumanAdjudicationReceipt,
    HumanAdjudicationReport,
    HumanReviewItem,
    MachineAdvisoryClassification,
    ResearchIntentContext,
    ReviewPolicy,
    SourceBinding,
    StatusBoundary,
)

M1_REPAIR_FIRST_RUN = Path(
    "evaluation/source-artifacts/m1-intent-replay-repair-2026-08-03/first-run/first-round.json"
)
M1_REPAIR_MANIFEST = Path(
    "evaluation/source-artifacts/m1-intent-replay-repair-2026-08-03/repair-manifest.json"
)
CANDIDATE_SNAPSHOT = Path("evaluation/snapshots/m2/m1-candidates.v1.json")
CLASSIFICATION_RESULT = Path(
    "evaluation/source-artifacts/m2-t03-evidence-classification-run.json"
)
STATUS_PATH = Path("STATUS.md")
PROTOCOL_PATH = Path("docs/reviews/m2-t03-human-adjudication-protocol.md")
BUNDLE_PATH = Path("evaluation/source-artifacts/m2-t03-human-adjudication-bundle.json")
RECEIPT_PATH = Path(
    "evaluation/source-artifacts/m2-t03-human-adjudication-bundle-receipt.json"
)
REVIEW_TEMPLATE_PATH = Path(
    "evaluation/source-artifacts/m2-t03-human-adjudication-review-template.json"
)
REPORT_PATH = Path("evaluation/reports/m2-t03-human-adjudication.json")
RUNNER_PATH = Path("scripts/build_m2_t03_human_adjudication_bundle.py")

EXPECTED_MAIN_MERGE_COMMIT = "b51fb5e3de86b0c1c49ebf5cbf02be227b1e9651"
EXPECTED_M1_REPAIR_FIRST_RUN_SHA256 = (
    "ed4a4d89a247c535e6138644069094d3c59a046bdefd44a79e48a4f861f95afa"
)
EXPECTED_M1_REPAIR_MANIFEST_SHA256 = (
    "fb9b0f2db164f6b40b24802e2466f5a744589ee64e9bc65a77295c8444236df3"
)
EXPECTED_CANDIDATE_SNAPSHOT_SHA256 = (
    "4a2aec0fd0a1d22adc801fd3bc506e5da89895d1276cd572e2ac64014c162448"
)
EXPECTED_CLASSIFICATION_RESULT_SHA256 = (
    "2bd80d2b6a10bfe531ea2ca3b7570768fd1a78e5d05efca96fc28464d72baee8"
)
EXPECTED_PROTOCOL_SHA256 = (
    "41b02244d0b183a87f4089ce5814ddf8eb6505c4f5f2d0ea9c4ba2d09c063cb2"
)
EXPECTED_CANDIDATE_COUNT = 33
GENERATED_AT_UTC = datetime(2026, 8, 3, 12, 30, tzinfo=UTC)
EXPECTED_COMMANDS = (
    "python scripts/build_m2_t03_human_adjudication_bundle.py --execute-offline",
)
EXPECTED_EXIT_CODES = {"build_bundle": 0}
REVIEW_SOURCE_FIELDS = (
    "paper_id",
    "title",
    "abstract",
    "source",
    "source_id",
    "url",
    "ResearchIntent",
)
REVIEW_DISALLOWED_SOURCES = (
    "paper_full_text",
    "citation_count",
    "author_reputation",
    "journal_rank",
    "reranker_output",
    "dense_output",
    "selection_output",
    "user_feedback",
    "network_supplementation",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_STATUS_ROW = re.compile(
    r"^\|\s*(M\d+)\b[^|]*\|\s*([A-Z0-9_]+)\s*\|\s*(\d+/\d+)\s*\|",
    re.MULTILINE,
)


class BundleError(RuntimeError):
    """Stable, non-sensitive failure surface for bundle generation."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _regular_bytes(path: Path) -> bytes:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if stat.S_ISLNK(opened.st_mode) or not stat.S_ISREG(opened.st_mode):
                raise ValueError
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise ValueError
            data = handle.read()
            final = os.fstat(handle.fileno())
            if (final.st_dev, final.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise ValueError
            return data
    except (OSError, ValueError):
        raise BundleError("FIXED_INPUT_INVALID") from None


def _fixed_bytes(repository_root: Path, relative_path: Path, expected: str) -> bytes:
    data = _regular_bytes(repository_root / relative_path)
    if _sha256(data) != expected:
        raise BundleError("FIXED_INPUT_HASH_MISMATCH")
    return data


def _json(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BundleError("FIXED_INPUT_INVALID") from None
    if not isinstance(value, dict):
        raise BundleError("FIXED_INPUT_INVALID")
    return cast(dict[str, Any], value)


def _git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=repository_root, capture_output=True, check=False, text=True
    )
    if completed.returncode != 0:
        raise BundleError("GIT_METADATA_UNAVAILABLE")
    return completed.stdout.strip()


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments], cwd=repository_root, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise BundleError("GIT_METADATA_UNAVAILABLE")
    return completed.stdout


def _require_main_ancestor(repository_root: Path) -> str:
    head = _git(repository_root, "rev-parse", "HEAD")
    if _SHA1_RE.fullmatch(head) is None:
        raise BundleError("GIT_METADATA_UNAVAILABLE")
    if repository_root.resolve() == ROOT.resolve():
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", EXPECTED_MAIN_MERGE_COMMIT, head],
            cwd=repository_root,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise BundleError("MAIN_MERGE_NOT_ANCESTOR")
    return head


def _relative_binding(repository_root: Path, path: Path, expected: str) -> ArtifactBinding:
    _fixed_bytes(repository_root, path, expected)
    return ArtifactBinding(path=path.as_posix(), sha256=expected)


def _historical_status_bytes(repository_root: Path, generated_from_commit: str) -> bytes:
    if _SHA1_RE.fullmatch(generated_from_commit) is None:
        raise BundleError("GIT_METADATA_UNAVAILABLE")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", generated_from_commit, "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise BundleError("GENERATED_COMMIT_NOT_ANCESTOR")
    status = _git_bytes(repository_root, "show", f"{generated_from_commit}:{STATUS_PATH.as_posix()}")
    try:
        decoded = status.decode("utf-8")
    except UnicodeDecodeError:
        raise BundleError("STATUS_BOUNDARY_INVALID") from None
    rows = {phase: (state, count) for phase, state, count in _STATUS_ROW.findall(decoded)}
    if rows.get("M2") != ("IN_PROGRESS", "2/5") or rows.get("M3") != (
        "BLOCKED_BY_M2",
        "0/5",
    ):
        raise BundleError("STATUS_BOUNDARY_INVALID")
    return status


def _protocol_binding(repository_root: Path) -> ArtifactBinding:
    _fixed_bytes(repository_root, PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256)
    return ArtifactBinding(path=PROTOCOL_PATH.as_posix(), sha256=EXPECTED_PROTOCOL_SHA256)


def _load_intent_context(repository_root: Path) -> ResearchIntentContext:
    first_run_bytes = _fixed_bytes(
        repository_root, M1_REPAIR_FIRST_RUN, EXPECTED_M1_REPAIR_FIRST_RUN_SHA256
    )
    manifest_bytes = _fixed_bytes(
        repository_root, M1_REPAIR_MANIFEST, EXPECTED_M1_REPAIR_MANIFEST_SHA256
    )
    try:
        manifest = M1ReplayRepairManifest.model_validate(_json(manifest_bytes))
        first_run = FirstRoundRun.model_validate(_json(first_run_bytes))
        if first_run.intent is None:
            raise ValueError
        canonical = canonical_research_intent_bytes(first_run.intent)
        if _sha256(canonical) != manifest.canonical_intent_sha256:
            raise ValueError
        if manifest.corrected_first_run_path != M1_REPAIR_FIRST_RUN.as_posix():
            raise ValueError
        if manifest.corrected_first_run_sha256 != EXPECTED_M1_REPAIR_FIRST_RUN_SHA256:
            raise ValueError
    except (TypeError, ValueError, ValidationError):
        raise BundleError("REPAIRED_INTENT_INVALID") from None
    intent_payload = json.loads(canonical.decode("utf-8"))
    return ResearchIntentContext(
        source_artifact=ArtifactBinding(
            path=M1_REPAIR_FIRST_RUN.as_posix(), sha256=EXPECTED_M1_REPAIR_FIRST_RUN_SHA256
        ),
        repair_manifest=ArtifactBinding(
            path=M1_REPAIR_MANIFEST.as_posix(), sha256=EXPECTED_M1_REPAIR_MANIFEST_SHA256
        ),
        canonical_intent_sha256=manifest.canonical_intent_sha256,
        research_intent=intent_payload,
    )


def _load_candidates(
    repository_root: Path,
) -> tuple[FrozenCandidateSnapshot, ArtifactBinding]:
    snapshot_bytes = _fixed_bytes(
        repository_root, CANDIDATE_SNAPSHOT, EXPECTED_CANDIDATE_SNAPSHOT_SHA256
    )
    try:
        payload = _json(snapshot_bytes)
        raw_candidates = payload["candidates"]
        if not isinstance(raw_candidates, list):
            raise TypeError
        candidates = [FrozenCandidate.model_validate(item) for item in raw_candidates]
        snapshot = FrozenCandidateSnapshot(
            snapshot_version=payload["snapshot_version"],
            question=payload["question"],
            candidates=candidates,
        )
    except (TypeError, ValueError, ValidationError):
        raise BundleError("CANDIDATE_SNAPSHOT_INVALID") from None
    if len(snapshot.candidates) != EXPECTED_CANDIDATE_COUNT:
        raise BundleError("CANDIDATE_SNAPSHOT_INVALID")
    return snapshot, ArtifactBinding(
        path=CANDIDATE_SNAPSHOT.as_posix(), sha256=EXPECTED_CANDIDATE_SNAPSHOT_SHA256
    )


def _load_classifications(
    repository_root: Path,
) -> tuple[dict[str, EvidenceClassificationRecord], EvidenceClassifierDescriptor, ArtifactBinding]:
    result_bytes = _fixed_bytes(
        repository_root, CLASSIFICATION_RESULT, EXPECTED_CLASSIFICATION_RESULT_SHA256
    )
    payload = _json(result_bytes)
    try:
        records_value = payload["records"]
        descriptor = EvidenceClassifierDescriptor.model_validate(payload["classifier"])
        if not isinstance(records_value, list) or len(records_value) != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        records = [EvidenceClassificationRecord.model_validate(item) for item in records_value]
    except (KeyError, TypeError, ValueError, ValidationError):
        raise BundleError("CLASSIFICATION_RESULT_INVALID") from None
    by_id = {record.paper_id: record for record in records}
    if len(by_id) != EXPECTED_CANDIDATE_COUNT or any(
        record.classifier_descriptor != descriptor for record in records
    ):
        raise BundleError("CLASSIFICATION_RESULT_INVALID")
    return (
        by_id,
        descriptor,
        ArtifactBinding(path=CLASSIFICATION_RESULT.as_posix(), sha256=EXPECTED_CLASSIFICATION_RESULT_SHA256),
    )


def _build_bundle(
    repository_root: Path,
    generated_from_commit: str,
    protocol_binding: ArtifactBinding,
) -> HumanAdjudicationBundle:
    intent_context = _load_intent_context(repository_root)
    snapshot, snapshot_binding = _load_candidates(repository_root)
    records, descriptor, classification_binding = _load_classifications(repository_root)
    paper_id_order = tuple(candidate.paper_id for candidate in snapshot.candidates)
    if set(paper_id_order) != set(records):
        raise BundleError("CLASSIFICATION_CANDIDATE_ID_MISMATCH")

    items: list[HumanReviewItem] = []
    for candidate in snapshot.candidates:
        record = records[candidate.paper_id]
        if candidate.abstract is None:
            raise BundleError("CANDIDATE_ABSTRACT_MISSING")
        text_hash = source_text_sha256(candidate.title, candidate.abstract)
        if record.source_text_sha256 != text_hash:
            raise BundleError("CLASSIFICATION_SOURCE_HASH_MISMATCH")
        source_identity = SourceBinding(
            source=candidate.source,
            source_id=candidate.source_id,
            url=candidate.url,
        )
        context = CandidateContext(
            paper_id=candidate.paper_id,
            title=candidate.title,
            abstract=candidate.abstract,
            source_identity=source_identity,
            source_text_sha256=text_hash,
        )
        advisory = MachineAdvisoryClassification(
            advisory_only=True,
            evidence_type="deterministic_fake",
            paper_id=record.paper_id,
            evidence_slot=record.evidence_slot,
            support_level=record.support_level,
            reason=record.reason,
            supporting_excerpt=record.supporting_excerpt,
            source_text_sha256=record.source_text_sha256,
            classifier_descriptor=descriptor,
            classification_version=record.classification_version,
            state=record.state,
        )
        items.append(
            HumanReviewItem(
                context=context,
                machine_advisory=advisory,
                human_adjudication=HumanAdjudicationFields(),
            )
        )

    policy_value = _json(_regular_bytes(repository_root / CLASSIFICATION_RESULT))["policy"]
    try:
        policy = cast(dict[str, Any], policy_value)
        disallowed_sources = tuple(str(value) for value in policy["disallowed_sources"])
    except (KeyError, TypeError, ValueError):
        raise BundleError("CLASSIFICATION_POLICY_INVALID") from None
    if disallowed_sources != REVIEW_DISALLOWED_SOURCES:
        raise BundleError("CLASSIFICATION_POLICY_INVALID")
    return HumanAdjudicationBundle(
        bundle_version="m2-t03-human-adjudication.v1",
        phase="M2",
        task_id="M2-T03",
        generated_from_commit=generated_from_commit,
        generated_at_utc=GENERATED_AT_UTC,
        candidate_count=EXPECTED_CANDIDATE_COUNT,
        paper_id_order=paper_id_order,
        candidate_snapshot=snapshot_binding,
        classification_result=classification_binding,
        intent_context=intent_context,
        review_policy=ReviewPolicy(
            machine_labels_are_advisory=True,
            human_fields_initially_empty=True,
            human_review_status="pending",
            source_fields=REVIEW_SOURCE_FIELDS,
            disallowed_sources=disallowed_sources,
            protocol=protocol_binding,
            network_forbidden=True,
            real_model_run=False,
            real_arxiv_requests=False,
            m2_t04_started=False,
        ),
        status_boundary=StatusBoundary(
            m2_progress="IN_PROGRESS 2/5",
            m3_progress="BLOCKED_BY_M2 0/5",
            scoring_eligible=False,
            human_review="pending",
            m2_t04="not_started",
        ),
        items=tuple(items),
    )


def _json_bytes(model: Any) -> bytes:
    value = model.model_dump(mode="json") if hasattr(model, "model_dump") else model
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _review_template_payload(bundle: HumanAdjudicationBundle, bundle_sha256: str) -> dict[str, Any]:
    return {
        "template_version": M2_T03_HUMAN_ADJUDICATION_TEMPLATE_VERSION,
        "pending_bundle": {
            "path": BUNDLE_PATH.as_posix(),
            "sha256": bundle_sha256,
        },
        "review_protocol": bundle.review_policy.protocol.model_dump(mode="json"),
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "paper_id_order": list(bundle.paper_id_order),
        "items": [
            {
                **item.context.model_dump(mode="json"),
                "human_adjudication": HumanAdjudicationFields().model_dump(mode="json"),
            }
            for item in bundle.items
        ],
    }


def _validate_publish_parent(repository_root: Path, destination: Path) -> list[Path]:
    missing: list[Path] = []
    current = destination.parent
    root = repository_root.resolve()
    while current.resolve() != root:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            missing.append(current)
            current = current.parent
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise BundleError("RESULT_CONFLICT")
        break
    return missing


def _publish_conflict_safe(repository_root: Path, targets: Mapping[Path, bytes]) -> None:
    """Publish all pending artifacts atomically, or remove only this run's files."""

    destinations: list[tuple[Path, bytes, bool]] = []
    missing_parent_paths: set[Path] = set()
    try:
        for relative_path, data in targets.items():
            if not isinstance(relative_path, Path) or not isinstance(data, bytes):
                raise BundleError("RESULT_CONFLICT")
            try:
                ArtifactBinding(path=relative_path.as_posix(), sha256=_sha256(data))
            except ValidationError:
                raise BundleError("RESULT_CONFLICT") from None
            destination = repository_root / relative_path
            missing_parent_paths.update(_validate_publish_parent(repository_root, destination))
            try:
                metadata = destination.lstat()
            except FileNotFoundError:
                destinations.append((destination, data, False))
                continue
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise BundleError("RESULT_CONFLICT")
            if _regular_bytes(destination) != data:
                raise BundleError("RESULT_CONFLICT")
            destinations.append((destination, data, True))

        created_directories: list[Path] = []
        for directory in sorted(missing_parent_paths, key=lambda value: len(value.parts)):
            if not directory.exists():
                directory.mkdir()
                created_directories.append(directory)

        staged: list[tuple[Path, Path, bytes]] = []
        owned_staging: list[Path] = []
        created_destinations: list[tuple[Path, bytes]] = []
        publish_succeeded = False
        try:
            for destination, data, already_exists in destinations:
                if already_exists:
                    continue
                temporary = destination.with_name(
                    f".{destination.name}.m2-t03-tmp-{os.getpid()}-{uuid.uuid4().hex}"
                )
                temporary.write_bytes(data)
                owned_staging.append(temporary)
                if _regular_bytes(temporary) != data:
                    raise BundleError("RESULT_PUBLISH_FAILED")
                staged.append((destination, temporary, data))

            for destination, temporary, data in staged:
                destination_metadata: os.stat_result | None
                try:
                    destination_metadata = destination.lstat()
                except FileNotFoundError:
                    destination_metadata = None
                if destination_metadata is not None:
                    if stat.S_ISLNK(destination_metadata.st_mode) or not stat.S_ISREG(
                        destination_metadata.st_mode
                    ):
                        raise BundleError("RESULT_CONFLICT")
                    if _regular_bytes(destination) != data:
                        raise BundleError("RESULT_CONFLICT")
                    temporary.unlink(missing_ok=True)
                    continue
                os.replace(temporary, destination)
                created_destinations.append((destination, data))
            publish_succeeded = True
        except BundleError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError):
            raise BundleError("RESULT_PUBLISH_FAILED") from None
        finally:
            for temporary in owned_staging:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            if not publish_succeeded:
                for destination, data in reversed(created_destinations):
                    try:
                        metadata = destination.lstat()
                        if stat.S_ISREG(metadata.st_mode) and _regular_bytes(destination) == data:
                            destination.unlink()
                    except OSError:
                        pass
            if not publish_succeeded and created_directories:
                for directory in reversed(created_directories):
                    try:
                        directory.rmdir()
                    except OSError:
                        pass
    except BundleError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise BundleError("RESULT_PUBLISH_FAILED") from None


def build_bundle(*, repository_root: Path = ROOT) -> dict[str, str | int]:
    generated_from_commit = _require_main_ancestor(repository_root)
    historical_status = _historical_status_bytes(repository_root, generated_from_commit)
    protocol_binding = _protocol_binding(repository_root)
    bundle = _build_bundle(repository_root, generated_from_commit, protocol_binding)
    bundle_bytes = _json_bytes(bundle)
    bundle_sha256 = _sha256(bundle_bytes)
    template_bytes = _json_bytes(_review_template_payload(bundle, bundle_sha256))
    template_sha256 = _sha256(template_bytes)
    runner_bytes = _regular_bytes(repository_root / RUNNER_PATH)
    runner_sha256 = _sha256(runner_bytes)
    receipt = HumanAdjudicationReceipt(
        receipt_version="m2-t03-human-adjudication-receipt.v1",
        phase="M2",
        task_id="M2-T03",
        generated_from_commit=generated_from_commit,
        bundle_path=BUNDLE_PATH.as_posix(),
        bundle_sha256=bundle_sha256,
        review_template=ArtifactBinding(
            path=REVIEW_TEMPLATE_PATH.as_posix(), sha256=template_sha256
        ),
        runner_path=RUNNER_PATH.as_posix(),
        runner_sha256=runner_sha256,
        candidate_count=EXPECTED_CANDIDATE_COUNT,
        human_fields_empty=True,
        network_requests=0,
        real_model_run=False,
        real_arxiv_requests=False,
        human_review="not_started",
        m2_t04="not_started",
        scoring_eligible=False,
        exit_code=0,
    )
    receipt_bytes = _json_bytes(receipt)
    report = HumanAdjudicationReport(
        report_version="m2-t03-human-adjudication-report.v1",
        phase="M2",
        task_id="M2-T03",
        generated_from_commit=generated_from_commit,
        generated_at_utc=GENERATED_AT_UTC,
        input_artifacts=(
            bundle.intent_context.source_artifact,
            bundle.intent_context.repair_manifest,
            bundle.candidate_snapshot,
            bundle.classification_result,
            protocol_binding,
            ArtifactBinding(path=STATUS_PATH.as_posix(), sha256=_sha256(historical_status)),
        ),
        artifact=(
            ArtifactBinding(path=BUNDLE_PATH.as_posix(), sha256=bundle_sha256),
            ArtifactBinding(path=RECEIPT_PATH.as_posix(), sha256=_sha256(receipt_bytes)),
            ArtifactBinding(path=REVIEW_TEMPLATE_PATH.as_posix(), sha256=template_sha256),
        ),
        candidate_count=EXPECTED_CANDIDATE_COUNT,
        machine_advisory_count=EXPECTED_CANDIDATE_COUNT,
        human_fields_nonempty_count=0,
        human_review="pending",
        real_model_run="not_run",
        real_arxiv_requests="not_run",
        m2_t04="not_started",
        scoring_eligible=False,
        status_after_bundle="M2 IN_PROGRESS 2/5; M3 BLOCKED_BY_M2 0/5",
        commands=EXPECTED_COMMANDS,
        exit_codes=EXPECTED_EXIT_CODES,
    )
    report_bytes = _json_bytes(report)
    _publish_conflict_safe(
        repository_root,
        {
            BUNDLE_PATH: bundle_bytes,
            RECEIPT_PATH: receipt_bytes,
            REVIEW_TEMPLATE_PATH: template_bytes,
            REPORT_PATH: report_bytes,
        },
    )
    return {
        "bundle_sha256": bundle_sha256,
        "receipt_sha256": _sha256(receipt_bytes),
        "template_sha256": template_sha256,
        "report_sha256": _sha256(report_bytes),
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-offline", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.execute_offline:
        print(json.dumps({"error_code": "BUNDLE_EXECUTION_NOT_AUTHORIZED", "status": "refused"}))
        return 2
    try:
        summary = build_bundle()
    except BundleError as error:
        print(json.dumps({"error_code": error.code, "status": "failed"}))
        return 1
    except (OSError, TypeError, ValueError, ValidationError, json.JSONDecodeError):
        print(json.dumps({"error_code": "BUNDLE_BUILD_FAILED", "status": "failed"}))
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
