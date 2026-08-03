"""Run the fixed, title/abstract-only M2-T03 classification contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if __package__ in {None, ""}:
    repository_root_text = str(_REPOSITORY_ROOT)
    if repository_root_text not in sys.path:
        sys.path.insert(0, repository_root_text)

from app.adapters.evidence_classification import (
    DeterministicFakeEvidenceClassifier,
    EvidenceClassifierProvider,
)
from app.core.evidence_classification import (
    build_classification_input,
    classify_evidence_batch,
    validate_classification_record,
)
from app.models.embedding import FrozenCandidate
from app.models.enums import EvidenceSlot, SupportLevel
from app.models.evidence_classification import (
    EvidenceClassificationBatch,
    EvidenceClassificationError,
    EvidenceClassificationInput,
    EvidenceClassificationQualityDiagnostics,
    EvidenceClassificationRecord,
    EvidenceClassificationState,
    EvidenceClassificationWarningCode,
    EvidenceClassifierDescriptor,
)

CANDIDATE_SNAPSHOT_PATH = Path("evaluation/snapshots/m2/m1-candidates.v1.json")
CANDIDATE_MANIFEST_PATH = Path("evaluation/snapshots/m2/m1-candidates.v1.manifest.json")
RERANKER_RESULT_PATH = Path("evaluation/source-artifacts/m2-t02-reranker-candidate-run.json")
RERANKER_RECEIPT_PATH = Path(
    "evaluation/source-artifacts/m2-t02-reranker-candidate-run-receipt.json"
)
FORMAL_RESULT_PATH = Path(
    "evaluation/source-artifacts/m2-t03-evidence-classification-run.json"
)
FORMAL_RECEIPT_PATH = Path(
    "evaluation/source-artifacts/m2-t03-evidence-classification-run-receipt.json"
)

EXPECTED_CANDIDATE_SNAPSHOT_SHA256 = (
    "4a2aec0fd0a1d22adc801fd3bc506e5da89895d1276cd572e2ac64014c162448"
)
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "b6ce7cfab2e5df6b8c84b77fb38c6f573de7429d37a9688d7b4c62e27f8546a8"
)
EXPECTED_RERANKER_RESULT_SHA256 = (
    "0a751dbc35bfa8d07433796113939412bfbdffd4c5c13043c90390bccfc5c35b"
)
EXPECTED_RERANKER_RECEIPT_SHA256 = (
    "0c9ca1339a2385a9c6940f8c53bd4711c62b8faf9f3f4cf98710ff465d56ec9a"
)
EXPECTED_CANDIDATE_COUNT = 33
EXPECTED_PAPER_ID_ORDER = (
    "arxiv:1401.6891",
    "arxiv:1611.00097",
    "arxiv:1804.03257",
    "arxiv:2005.04961",
    "arxiv:2007.12731",
    "arxiv:2210.15912",
    "arxiv:2306.03535",
    "arxiv:2306.10044",
    "arxiv:2310.16146",
    "arxiv:2402.12352",
    "arxiv:2408.15002",
    "arxiv:2408.15545",
    "arxiv:2412.05447",
    "arxiv:2412.15232",
    "arxiv:2501.02157",
    "arxiv:2503.02922",
    "arxiv:2504.08768",
    "arxiv:2507.08945",
    "arxiv:2508.09995",
    "arxiv:2508.20514",
    "arxiv:2509.01042",
    "arxiv:2509.08032",
    "arxiv:2510.26824",
    "arxiv:2511.05498",
    "arxiv:2511.10014",
    "arxiv:2511.14362",
    "arxiv:2512.12760",
    "arxiv:2601.14662",
    "arxiv:2602.17856",
    "arxiv:2604.16317",
    "arxiv:2604.16416",
    "arxiv:2604.25256",
    "arxiv:2606.22375",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_LOCAL_ABSOLUTE_PATH_RE = re.compile(r"(?:^[A-Za-z]:[\\/]|^\\\\|/(?:Users|home|tmp)/)")
_FORBIDDEN_TEXT_RE = re.compile(
    r"(?:file://|authorization|cookie|bearer\s|access[_-]?token|api[_-]?key|"
    r"password|raw exception|traceback)",
    re.IGNORECASE,
)
_PUBLISH_LOCK = threading.RLock()

_RUN_REPORT_FIELDS = {
    "report_version",
    "phase",
    "task_id",
    "decision_status",
    "input",
    "classifier",
    "policy",
    "execution",
    "records",
}
_RUN_INPUT_FIELDS = {
    "candidate_snapshot_path",
    "candidate_snapshot_sha256",
    "candidate_manifest_path",
    "candidate_manifest_sha256",
    "reranker_result_path",
    "reranker_result_sha256",
    "reranker_receipt_path",
    "reranker_receipt_sha256",
    "candidate_count",
    "paper_id_order",
}
_RUN_POLICY_FIELDS = {
    "input_fields",
    "disallowed_sources",
    "ordering",
    "network_forbidden",
    "order_independent",
    "evidence_type_separation",
}
_RUN_EXECUTION_FIELDS = {
    "evidence_type",
    "provider_call_count",
    "record_count",
    "title_only_count",
    "rejected_count",
    "slot_distribution",
    "support_level_distribution",
    "quality_diagnostics",
}
_RECORD_FIELDS = {
    "paper_id",
    "evidence_slot",
    "support_level",
    "reason",
    "supporting_excerpt",
    "source_text_sha256",
    "classifier_descriptor",
    "classification_version",
    "state",
}
_RECEIPT_FIELDS = {
    "report_version",
    "phase",
    "task_id",
    "execution_commit",
    "result_path",
    "result_sha256",
    "receipt_path",
    "runner_path",
    "runner_sha256",
    "candidate_snapshot_path",
    "candidate_snapshot_sha256",
    "candidate_manifest_path",
    "candidate_manifest_sha256",
    "reranker_result_path",
    "reranker_result_sha256",
    "reranker_receipt_path",
    "reranker_receipt_sha256",
    "runtime",
    "exit_code",
    "run_count",
    "decision_status",
    "formal_result_created",
    "evidence_type",
    "real_model_run",
    "human_judged",
}


@dataclass(frozen=True)
class EvidenceClassificationRunPaths:
    """Fixed repository paths, injectable only for isolated tests."""

    candidate_snapshot_path: Path | str = CANDIDATE_SNAPSHOT_PATH
    candidate_manifest_path: Path | str = CANDIDATE_MANIFEST_PATH
    reranker_result_path: Path | str = RERANKER_RESULT_PATH
    reranker_receipt_path: Path | str = RERANKER_RECEIPT_PATH
    result_path: Path | str = FORMAL_RESULT_PATH
    receipt_path: Path | str = FORMAL_RECEIPT_PATH


@dataclass(frozen=True)
class _FixedInputContext:
    candidates: tuple[FrozenCandidate, ...]
    paper_id_order: tuple[str, ...]
    candidate_snapshot_sha256: str
    candidate_manifest_sha256: str
    reranker_result_sha256: str
    reranker_receipt_sha256: str


class CountingEvidenceClassifierProvider:
    """Count provider calls while preserving the Protocol boundary."""

    def __init__(self, provider: EvidenceClassifierProvider) -> None:
        self._provider = provider
        self.provider_call_count = 0

    @property
    def descriptor(self) -> EvidenceClassifierDescriptor:
        return self._provider.descriptor

    def classify(self, inputs: Sequence[EvidenceClassificationInput]) -> Sequence[object]:
        self.provider_call_count += 1
        return self._provider.classify(inputs)

    @property
    def quality_diagnostics(self) -> EvidenceClassificationQualityDiagnostics | None:
        diagnostics = getattr(self._provider, "quality_diagnostics", None)
        return diagnostics if isinstance(diagnostics, EvidenceClassificationQualityDiagnostics) else None


def _operational_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _REPOSITORY_ROOT / candidate


def _is_link_or_junction(path: Path, metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    checker = getattr(path, "is_junction", None)
    if callable(checker):
        try:
            return bool(checker())
        except OSError:
            return True
    return False


def _read_regular_file(path: Path) -> bytes:
    try:
        metadata = path.lstat()
        if _is_link_or_junction(path, metadata) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if _is_link_or_junction(path, opened) or not stat.S_ISREG(opened.st_mode):
                raise ValueError
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise ValueError
            payload = handle.read()
            final = os.fstat(handle.fileno())
            if (final.st_dev, final.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise ValueError
            return payload
    except Exception:  # noqa: BLE001
        raise EvidenceClassificationError("FIXED_INPUT_INVALID") from None


def _read_json_file(path: Path) -> tuple[bytes, object]:
    raw = _read_regular_file(path)
    try:
        return raw, json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001
        raise EvidenceClassificationError("FIXED_INPUT_INVALID") from None


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError
    return cast(dict[str, object], value)


def _require_sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError
    return value


def _validate_candidate_inputs(paths: EvidenceClassificationRunPaths) -> _FixedInputContext:
    snapshot_raw, snapshot_value = _read_json_file(_operational_path(paths.candidate_snapshot_path))
    manifest_raw, manifest_value = _read_json_file(_operational_path(paths.candidate_manifest_path))
    snapshot_sha256 = hashlib.sha256(snapshot_raw).hexdigest()
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if (
        snapshot_sha256 != EXPECTED_CANDIDATE_SNAPSHOT_SHA256
        or manifest_sha256 != EXPECTED_CANDIDATE_MANIFEST_SHA256
    ):
        raise EvidenceClassificationError("FIXED_INPUT_INVALID")
    try:
        snapshot = _as_mapping(snapshot_value)
        manifest = _as_mapping(manifest_value)
        if snapshot["snapshot_version"] != "m2-candidates.v1":
            raise ValueError
        candidate_values = snapshot["candidates"]
        if not isinstance(candidate_values, list) or len(candidate_values) != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        if snapshot["count"] != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        if manifest["candidate_count"] != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        if manifest["snapshot_sha256"] != EXPECTED_CANDIDATE_SNAPSHOT_SHA256:
            raise ValueError
        if manifest["abstract_present_count"] != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        if manifest["abstract_missing_count"] != 0:
            raise ValueError
        raw_order = manifest["paper_id_order"]
        if raw_order != list(EXPECTED_PAPER_ID_ORDER):
            raise ValueError
        candidates: list[FrozenCandidate] = []
        for raw_candidate in candidate_values:
            candidate = FrozenCandidate.model_validate(raw_candidate)
            if candidate.abstract is None or candidate.paper_id not in EXPECTED_PAPER_ID_ORDER:
                raise ValueError
            candidates.append(candidate)
        paper_ids = tuple(candidate.paper_id for candidate in candidates)
        if paper_ids != EXPECTED_PAPER_ID_ORDER:
            raise ValueError
        if len(set(paper_ids)) != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
    except (TypeError, ValueError, KeyError):
        raise EvidenceClassificationError("FIXED_INPUT_INVALID") from None

    reranker_result_raw, reranker_result_value = _read_json_file(
        _operational_path(paths.reranker_result_path)
    )
    reranker_receipt_raw, reranker_receipt_value = _read_json_file(
        _operational_path(paths.reranker_receipt_path)
    )
    reranker_result_sha256 = hashlib.sha256(reranker_result_raw).hexdigest()
    reranker_receipt_sha256 = hashlib.sha256(reranker_receipt_raw).hexdigest()
    if (
        reranker_result_sha256 != EXPECTED_RERANKER_RESULT_SHA256
        or reranker_receipt_sha256 != EXPECTED_RERANKER_RECEIPT_SHA256
    ):
        raise EvidenceClassificationError("FIXED_INPUT_INVALID")
    try:
        reranker_result = _as_mapping(reranker_result_value)
        reranker_receipt = _as_mapping(reranker_receipt_value)
        if (
            reranker_result["phase"] != "M2"
            or reranker_result["task_id"] != "M2-T02"
            or reranker_result["decision_status"] != "candidate_reranking_completed"
        ):
            raise ValueError
        result_input = _as_mapping(reranker_result["input"])
        if (
            result_input["candidate_snapshot_path"] != CANDIDATE_SNAPSHOT_PATH.as_posix()
            or result_input["candidate_snapshot_sha256"] != EXPECTED_CANDIDATE_SNAPSHOT_SHA256
            or result_input["candidate_count"] != EXPECTED_CANDIDATE_COUNT
            or result_input["paper_id_order"] != list(EXPECTED_PAPER_ID_ORDER)
        ):
            raise ValueError
        result_records = reranker_result["records"]
        if not isinstance(result_records, list) or len(result_records) != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        reranker_ids = []
        for raw_record in result_records:
            record = _as_mapping(raw_record)
            paper_id = record["paper_id"]
            if not isinstance(paper_id, str):
                raise TypeError
            reranker_ids.append(paper_id)
        if set(reranker_ids) != set(EXPECTED_PAPER_ID_ORDER):
            raise ValueError
        if (
            reranker_receipt["phase"] != "M2"
            or reranker_receipt["task_id"] != "M2-T02"
            or reranker_receipt["result_path"] != RERANKER_RESULT_PATH.as_posix()
            or reranker_receipt["result_sha256"] != EXPECTED_RERANKER_RESULT_SHA256
            or reranker_receipt["candidate_snapshot_path"] != CANDIDATE_SNAPSHOT_PATH.as_posix()
            or reranker_receipt["candidate_snapshot_sha256"] != EXPECTED_CANDIDATE_SNAPSHOT_SHA256
            or reranker_receipt["candidate_manifest_path"] != CANDIDATE_MANIFEST_PATH.as_posix()
            or reranker_receipt["candidate_manifest_sha256"] != EXPECTED_CANDIDATE_MANIFEST_SHA256
            or reranker_receipt["exit_code"] != 0
            or reranker_receipt["formal_result_created"] is not True
            or reranker_receipt["full_candidate_reranking_run"] is not True
        ):
            raise ValueError
    except (TypeError, ValueError, KeyError):
        raise EvidenceClassificationError("FIXED_INPUT_INVALID") from None
    return _FixedInputContext(
        candidates=tuple(candidates),
        paper_id_order=paper_ids,
        candidate_snapshot_sha256=snapshot_sha256,
        candidate_manifest_sha256=manifest_sha256,
        reranker_result_sha256=reranker_result_sha256,
        reranker_receipt_sha256=reranker_receipt_sha256,
    )


def _distribution(
    records: Sequence[EvidenceClassificationRecord],
) -> tuple[dict[str, int], dict[str, int]]:
    slots = {slot.value: 0 for slot in EvidenceSlot}
    support_levels = {level.value: 0 for level in SupportLevel}
    for record in records:
        if record.evidence_slot is not None:
            slots[record.evidence_slot.value] += 1
        if record.support_level is not None:
            support_levels[record.support_level.value] += 1
    return slots, support_levels


def _build_run_report(
    context: _FixedInputContext,
    batch: EvidenceClassificationBatch,
    provider_call_count: int,
    quality_diagnostics: EvidenceClassificationQualityDiagnostics | None,
) -> dict[str, object]:
    records = [record.model_dump(mode="json") for record in batch.records]
    slot_distribution, support_distribution = _distribution(batch.records)
    title_only_count = sum(
        record.state is EvidenceClassificationState.TITLE_ONLY for record in batch.records
    )
    rejected_count = sum(
        record.state
        in {EvidenceClassificationState.REJECTED, EvidenceClassificationState.INSUFFICIENT_SOURCE_TEXT}
        for record in batch.records
    )
    diagnostics = quality_diagnostics or _derive_quality_diagnostics(batch.records)
    return {
        "report_version": "m2-t03-evidence-classification-run.v1",
        "phase": "M2",
        "task_id": "M2-T03",
        "decision_status": (
            "classification_completed" if rejected_count == 0 else "classification_partial_failure"
        ),
        "input": {
            "candidate_snapshot_path": CANDIDATE_SNAPSHOT_PATH.as_posix(),
            "candidate_snapshot_sha256": context.candidate_snapshot_sha256,
            "candidate_manifest_path": CANDIDATE_MANIFEST_PATH.as_posix(),
            "candidate_manifest_sha256": context.candidate_manifest_sha256,
            "reranker_result_path": RERANKER_RESULT_PATH.as_posix(),
            "reranker_result_sha256": context.reranker_result_sha256,
            "reranker_receipt_path": RERANKER_RECEIPT_PATH.as_posix(),
            "reranker_receipt_sha256": context.reranker_receipt_sha256,
            "candidate_count": len(context.candidates),
            "paper_id_order": list(context.paper_id_order),
        },
        "classifier": batch.classifier_descriptor.model_dump(mode="json"),
        "policy": {
            "input_fields": [
                "paper_id",
                "title",
                "abstract",
                "source_text_sha256",
                "classifier_descriptor",
                "classification_version",
            ],
            "disallowed_sources": [
                "paper_full_text",
                "citation_count",
                "author_reputation",
                "journal_rank",
                "reranker_output",
                "dense_output",
                "selection_output",
                "user_feedback",
                "network_supplementation",
            ],
            "ordering": "paper_id_ascending",
            "network_forbidden": True,
            "order_independent": True,
            "evidence_type_separation": "deterministic_fake_only",
        },
        "execution": {
            "evidence_type": "deterministic_fake",
            "provider_call_count": provider_call_count,
            "record_count": len(batch.records),
            "title_only_count": title_only_count,
            "rejected_count": rejected_count,
            "slot_distribution": slot_distribution,
            "support_level_distribution": support_distribution,
            "quality_diagnostics": diagnostics.model_dump(mode="json"),
        },
        "records": records,
    }


def _derive_quality_diagnostics(
    records: Sequence[EvidenceClassificationRecord],
) -> EvidenceClassificationQualityDiagnostics:
    slots = {record.evidence_slot for record in records if record.evidence_slot is not None}
    support_levels = {
        record.support_level for record in records if record.support_level is not None
    }
    rejected_count = sum(
        record.state
        in {EvidenceClassificationState.REJECTED, EvidenceClassificationState.INSUFFICIENT_SOURCE_TEXT}
        for record in records
    )
    warnings: list[EvidenceClassificationWarningCode] = []
    if len(slots) < 3:
        warnings.append("FEWER_THAN_THREE_SLOTS_OBSERVED")
    if support_levels and len(support_levels) == 1:
        warnings.append("ALL_RECORDS_SAME_SUPPORT_LEVEL")
    if not {SupportLevel.INDIRECT, SupportLevel.HYPOTHETICAL} & support_levels:
        warnings.append("NO_INDIRECT_OR_HYPOTHETICAL_RECORDS")
    if rejected_count == 0:
        warnings.append("NO_REJECTED_OR_UNCERTAIN_RECORDS")
    return EvidenceClassificationQualityDiagnostics(
        observed_slot_count=len(slots),
        observed_support_level_count=len(support_levels),
        all_records_same_support_level=bool(support_levels) and len(support_levels) == 1,
        generic_marker_only_count=0,
        ambiguous_rejection_count=0,
        warnings=warnings,
    )


def _serialized_json(value: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise EvidenceClassificationError("INVALID_OUTPUT") from None


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        value = result.stdout.strip()
        if _SHA1_RE.fullmatch(value) is None:
            raise ValueError
        return value
    except Exception:  # noqa: BLE001
        return "0" * 40


def _runner_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    except OSError:
        raise EvidenceClassificationError("INVALID_OUTPUT") from None


def _build_receipt(report: Mapping[str, object], *, formal_result_created: bool) -> dict[str, object]:
    result_bytes = _serialized_json(report)
    return {
        "report_version": "m2-t03-evidence-classification-run-receipt.v1",
        "phase": "M2",
        "task_id": "M2-T03",
        "execution_commit": _git_head(),
        "result_path": FORMAL_RESULT_PATH.as_posix(),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "receipt_path": FORMAL_RECEIPT_PATH.as_posix(),
        "runner_path": "scripts/run_m2_t03_evidence_classification.py",
        "runner_sha256": _runner_sha256(),
        "candidate_snapshot_path": CANDIDATE_SNAPSHOT_PATH.as_posix(),
        "candidate_snapshot_sha256": EXPECTED_CANDIDATE_SNAPSHOT_SHA256,
        "candidate_manifest_path": CANDIDATE_MANIFEST_PATH.as_posix(),
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "reranker_result_path": RERANKER_RESULT_PATH.as_posix(),
        "reranker_result_sha256": EXPECTED_RERANKER_RESULT_SHA256,
        "reranker_receipt_path": RERANKER_RECEIPT_PATH.as_posix(),
        "reranker_receipt_sha256": EXPECTED_RERANKER_RECEIPT_SHA256,
        "runtime": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "classifier_runtime": "stdlib",
        },
        "exit_code": 0,
        "run_count": 1,
        "decision_status": report["decision_status"],
        "formal_result_created": formal_result_created,
        "evidence_type": "deterministic_fake",
        "real_model_run": False,
        "human_judged": False,
    }


def _validate_safe_strings(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _validate_safe_strings(item)
    elif isinstance(value, list):
        for item in value:
            _validate_safe_strings(item)
    elif isinstance(value, str) and (
        _LOCAL_ABSOLUTE_PATH_RE.search(value) or _FORBIDDEN_TEXT_RE.search(value)
    ):
        raise ValueError


def _reject_forbidden_output_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = key.casefold()
            if (
                lowered in {"score", "weight", "rank", "selection", "selection_rank"}
                or lowered.endswith(("_score", "_weight", "_rank"))
            ):
                raise ValueError
            _reject_forbidden_output_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_output_keys(item)


def validate_classification_run_report(value: object) -> None:
    """Validate the formal closed M2-T03 run report without model runtimes."""

    try:
        report = _as_mapping(value)
        if set(report) != _RUN_REPORT_FIELDS:
            raise ValueError
        if (
            report["report_version"] != "m2-t03-evidence-classification-run.v1"
            or report["phase"] != "M2"
            or report["task_id"] != "M2-T03"
            or report["decision_status"]
            not in {"classification_completed", "classification_partial_failure"}
        ):
            raise ValueError
        input_data = _as_mapping(report["input"])
        policy = _as_mapping(report["policy"])
        execution = _as_mapping(report["execution"])
        if set(input_data) != _RUN_INPUT_FIELDS or set(policy) != _RUN_POLICY_FIELDS:
            raise ValueError
        if set(execution) != _RUN_EXECUTION_FIELDS:
            raise ValueError
        if input_data != {
            "candidate_snapshot_path": CANDIDATE_SNAPSHOT_PATH.as_posix(),
            "candidate_snapshot_sha256": EXPECTED_CANDIDATE_SNAPSHOT_SHA256,
            "candidate_manifest_path": CANDIDATE_MANIFEST_PATH.as_posix(),
            "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
            "reranker_result_path": RERANKER_RESULT_PATH.as_posix(),
            "reranker_result_sha256": EXPECTED_RERANKER_RESULT_SHA256,
            "reranker_receipt_path": RERANKER_RECEIPT_PATH.as_posix(),
            "reranker_receipt_sha256": EXPECTED_RERANKER_RECEIPT_SHA256,
            "candidate_count": EXPECTED_CANDIDATE_COUNT,
            "paper_id_order": list(EXPECTED_PAPER_ID_ORDER),
        }:
            raise ValueError
        expected_policy = {
            "input_fields": [
                "paper_id",
                "title",
                "abstract",
                "source_text_sha256",
                "classifier_descriptor",
                "classification_version",
            ],
            "disallowed_sources": [
                "paper_full_text",
                "citation_count",
                "author_reputation",
                "journal_rank",
                "reranker_output",
                "dense_output",
                "selection_output",
                "user_feedback",
                "network_supplementation",
            ],
            "ordering": "paper_id_ascending",
            "network_forbidden": True,
            "order_independent": True,
            "evidence_type_separation": "deterministic_fake_only",
        }
        if policy != expected_policy:
            raise ValueError
        descriptor = EvidenceClassifierDescriptor.model_validate(report["classifier"])
        if descriptor.evidence_type != "deterministic_fake":
            raise ValueError
        context = _validate_candidate_inputs(EvidenceClassificationRunPaths())
        inputs = {
            candidate.paper_id: build_classification_input(candidate, descriptor)
            for candidate in context.candidates
        }
        raw_records = report["records"]
        if not isinstance(raw_records, list) or len(raw_records) != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        records: list[EvidenceClassificationRecord] = []
        for raw_record in raw_records:
            record_mapping = _as_mapping(raw_record)
            if set(record_mapping) != _RECORD_FIELDS:
                raise ValueError
            record = EvidenceClassificationRecord.model_validate(record_mapping)
            source = inputs.get(record.paper_id)
            if source is None:
                raise ValueError
            validate_classification_record(record, source)
            records.append(record)
        if [record.paper_id for record in records] != sorted(EXPECTED_PAPER_ID_ORDER):
            raise ValueError
        if len({record.paper_id for record in records}) != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        slot_distribution, support_distribution = _distribution(records)
        title_only_count = sum(
            record.state is EvidenceClassificationState.TITLE_ONLY for record in records
        )
        rejected_count = sum(
            record.state
            in {EvidenceClassificationState.REJECTED, EvidenceClassificationState.INSUFFICIENT_SOURCE_TEXT}
            for record in records
        )
        expected_execution = {
            "evidence_type": "deterministic_fake",
            "provider_call_count": 1,
            "record_count": EXPECTED_CANDIDATE_COUNT,
            "title_only_count": title_only_count,
            "rejected_count": rejected_count,
            "slot_distribution": slot_distribution,
            "support_level_distribution": support_distribution,
        }
        quality_diagnostics = EvidenceClassificationQualityDiagnostics.model_validate(
            execution["quality_diagnostics"]
        )
        if (
            quality_diagnostics.observed_slot_count
            != sum(value > 0 for value in slot_distribution.values())
            or quality_diagnostics.observed_support_level_count
            != sum(value > 0 for value in support_distribution.values())
            or quality_diagnostics.generic_marker_only_count > rejected_count
            or quality_diagnostics.ambiguous_rejection_count > rejected_count
        ):
            raise ValueError
        if {
            key: value for key, value in execution.items() if key != "quality_diagnostics"
        } != expected_execution:
            raise ValueError
        expected_status = (
            "classification_completed" if rejected_count == 0 else "classification_partial_failure"
        )
        if report["decision_status"] != expected_status:
            raise ValueError
        _reject_forbidden_output_keys(report)
        _validate_safe_strings(report)
    except EvidenceClassificationError:
        raise
    except Exception:  # noqa: BLE001
        raise EvidenceClassificationError("INVALID_OUTPUT") from None


def validate_classification_receipt(value: object, *, report: Mapping[str, object] | None = None) -> None:
    """Validate the closed receipt and optionally bind it to a report's bytes."""

    try:
        receipt = _as_mapping(value)
        if set(receipt) != _RECEIPT_FIELDS:
            raise ValueError
        if (
            receipt["report_version"] != "m2-t03-evidence-classification-run-receipt.v1"
            or receipt["phase"] != "M2"
            or receipt["task_id"] != "M2-T03"
            or not isinstance(receipt["execution_commit"], str)
            or _SHA1_RE.fullmatch(receipt["execution_commit"]) is None
            or receipt["result_path"] != FORMAL_RESULT_PATH.as_posix()
            or receipt["receipt_path"] != FORMAL_RECEIPT_PATH.as_posix()
            or receipt["runner_path"] != "scripts/run_m2_t03_evidence_classification.py"
            or receipt["candidate_snapshot_path"] != CANDIDATE_SNAPSHOT_PATH.as_posix()
            or receipt["candidate_manifest_path"] != CANDIDATE_MANIFEST_PATH.as_posix()
            or receipt["reranker_result_path"] != RERANKER_RESULT_PATH.as_posix()
            or receipt["reranker_receipt_path"] != RERANKER_RECEIPT_PATH.as_posix()
            or receipt["candidate_snapshot_sha256"] != EXPECTED_CANDIDATE_SNAPSHOT_SHA256
            or receipt["candidate_manifest_sha256"] != EXPECTED_CANDIDATE_MANIFEST_SHA256
            or receipt["reranker_result_sha256"] != EXPECTED_RERANKER_RESULT_SHA256
            or receipt["reranker_receipt_sha256"] != EXPECTED_RERANKER_RECEIPT_SHA256
            or receipt["runner_sha256"] != _runner_sha256()
            or receipt["exit_code"] != 0
            or receipt["run_count"] != 1
            or receipt["formal_result_created"] is not True
            or receipt["evidence_type"] != "deterministic_fake"
            or receipt["real_model_run"] is not False
            or receipt["human_judged"] is not False
        ):
            raise ValueError
        _require_sha256(receipt["result_sha256"])
        if report is not None:
            expected_hash = hashlib.sha256(_serialized_json(report)).hexdigest()
            if receipt["result_sha256"] != expected_hash:
                raise ValueError
        _validate_safe_strings(receipt)
    except EvidenceClassificationError:
        raise
    except Exception:  # noqa: BLE001
        raise EvidenceClassificationError("INVALID_OUTPUT") from None


def _existing_bytes(path: Path) -> bytes | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise EvidenceClassificationError("RESULT_CONFLICT") from None
    if _is_link_or_junction(path, metadata) or not stat.S_ISREG(metadata.st_mode):
        raise EvidenceClassificationError("RESULT_CONFLICT")
    try:
        return path.read_bytes()
    except OSError:
        raise EvidenceClassificationError("RESULT_CONFLICT") from None


def _require_parent_directory(path: Path) -> None:
    try:
        metadata = path.parent.lstat()
        if _is_link_or_junction(path.parent, metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError
    except Exception:  # noqa: BLE001
        raise EvidenceClassificationError("RESULT_CONFLICT") from None


def _write_temp(path: Path, payload: bytes) -> Path:
    temporary_path: Path | None = None
    try:
        raw_fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
        os.close(raw_fd)
        temporary_path = Path(temporary_name)
        with temporary_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary_path
    except (OSError, TypeError, ValueError):
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise EvidenceClassificationError("RESULT_PUBLICATION_FAILED") from None


def _cleanup_temp(path: Path | None) -> None:
    if path is None:
        return
    try:
        metadata = path.lstat()
        if not _is_link_or_junction(path, metadata) and stat.S_ISREG(metadata.st_mode):
            path.unlink()
    except (FileNotFoundError, OSError):
        pass


def _restore_targets(
    published: Sequence[Path], snapshots: Mapping[Path, bytes | None]
) -> None:
    for target in reversed(published):
        try:
            snapshot = snapshots[target]
            if snapshot is None:
                target.unlink(missing_ok=True)
            else:
                with target.open("wb") as handle:
                    handle.write(snapshot)
                    handle.flush()
                    os.fsync(handle.fileno())
        except (OSError, KeyError):
            pass


def publish_classification_run(
    report: Mapping[str, object],
    *,
    destination: Path | str = FORMAL_RESULT_PATH,
    receipt: Mapping[str, object] | None = None,
    receipt_destination: Path | str = FORMAL_RECEIPT_PATH,
) -> Literal["PUBLISHED", "REUSED"]:
    """Publish one report, or the report/receipt pair, atomically and idempotently."""

    validate_classification_run_report(report)
    report_bytes = _serialized_json(report)
    targets: list[tuple[Path, bytes]] = [(_operational_path(destination), report_bytes)]
    if receipt is not None:
        validate_classification_receipt(receipt, report=report)
        targets.append((_operational_path(receipt_destination), _serialized_json(receipt)))
    with _PUBLISH_LOCK:
        for path, _ in targets:
            _require_parent_directory(path)
        snapshots = {path: _existing_bytes(path) for path, _ in targets}
        if all(snapshots[path] == payload for path, payload in targets):
            return "REUSED"
        if any(snapshots[path] is not None for path, _ in targets):
            raise EvidenceClassificationError("RESULT_CONFLICT")
        temporary_paths: list[Path] = []
        published: list[Path] = []
        try:
            for path, payload in targets:
                temporary_paths.append(_write_temp(path, payload))
            for path, temporary in zip((path for path, _ in targets), temporary_paths, strict=True):
                if _existing_bytes(path) is not None:
                    raise EvidenceClassificationError("RESULT_CONFLICT")
                os.replace(temporary, path)
                published.append(path)
            return "PUBLISHED"
        except EvidenceClassificationError:
            _restore_targets(published, snapshots)
            raise
        except (OSError, TypeError, ValueError):
            _restore_targets(published, snapshots)
            raise EvidenceClassificationError("RESULT_PUBLICATION_FAILED") from None
        finally:
            for temporary in temporary_paths:
                _cleanup_temp(temporary)


def run_m2_t03_evidence_classification(
    *,
    execute_deterministic_fake: bool = False,
    paths: EvidenceClassificationRunPaths | None = None,
    provider_factory: Callable[[EvidenceClassifierDescriptor], EvidenceClassifierProvider]
    | None = None,
    publish_result: bool = False,
) -> dict[str, object]:
    """Run the fixed 33-candidate contract and optionally publish its two artifacts."""

    if execute_deterministic_fake is not True:
        raise EvidenceClassificationError("INVALID_INPUT")
    selected_paths = paths or EvidenceClassificationRunPaths()
    context = _validate_candidate_inputs(selected_paths)
    selected_factory = provider_factory or (
        lambda _descriptor: DeterministicFakeEvidenceClassifier()
    )
    try:
        probe_descriptor = DeterministicFakeEvidenceClassifier().descriptor
        provider = selected_factory(probe_descriptor)
        counting_provider = CountingEvidenceClassifierProvider(provider)
        if counting_provider.descriptor.evidence_type != "deterministic_fake":
            raise EvidenceClassificationError("INVALID_INPUT")
        inputs = [build_classification_input(candidate, counting_provider.descriptor) for candidate in context.candidates]
        batch = classify_evidence_batch(inputs, counting_provider)
        report = _build_run_report(
            context,
            batch,
            counting_provider.provider_call_count,
            counting_provider.quality_diagnostics,
        )
        validate_classification_run_report(report)
        if publish_result:
            receipt = _build_receipt(report, formal_result_created=True)
            publish_classification_run(
                report,
                destination=selected_paths.result_path,
                receipt=receipt,
                receipt_destination=selected_paths.receipt_path,
            )
        return report
    except EvidenceClassificationError:
        raise
    except Exception:  # noqa: BLE001
        raise EvidenceClassificationError("PROVIDER_UNAVAILABLE") from None


def main(argv: Sequence[str] | None = None) -> int:
    """Accept only the explicit deterministic-fake execution flag."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments != ["--execute-deterministic-fake"]:
        return 2
    try:
        run_m2_t03_evidence_classification(
            execute_deterministic_fake=True,
            publish_result=True,
        )
    except EvidenceClassificationError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
