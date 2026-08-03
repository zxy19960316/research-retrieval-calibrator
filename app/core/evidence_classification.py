"""Deterministic, fail-closed orchestration for M2-T03 classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from app.adapters.evidence_classification import EvidenceClassifierProvider
from app.models.evidence_classification import (
    EVIDENCE_CLASSIFICATION_VERSION,
    EvidenceClassificationBatch,
    EvidenceClassificationBatchState,
    EvidenceClassificationError,
    EvidenceClassificationInput,
    EvidenceClassificationRecord,
    EvidenceClassificationState,
    EvidenceClassifierDescriptor,
    build_grounded_reason,
    source_text_sha256,
)
from app.models.embedding import FrozenCandidate

__all__ = [
    "build_classification_input",
    "build_grounded_reason",
    "classify_evidence_batch",
    "source_text_sha256",
    "validate_classification_record",
]


def build_classification_input(
    candidate: FrozenCandidate,
    descriptor: EvidenceClassifierDescriptor,
    *,
    classification_version: str = EVIDENCE_CLASSIFICATION_VERSION,
) -> EvidenceClassificationInput:
    """Project a frozen candidate into the title/abstract-only classifier input."""

    try:
        return EvidenceClassificationInput(
            paper_id=candidate.paper_id,
            title=candidate.title,
            abstract=candidate.abstract,
            source_text_sha256=source_text_sha256(candidate.title, candidate.abstract),
            classifier_descriptor=descriptor,
            classification_version=classification_version,
        )
    except (TypeError, ValueError, ValidationError):
        raise EvidenceClassificationError("INVALID_INPUT") from None


def validate_classification_record(
    record: EvidenceClassificationRecord,
    source: EvidenceClassificationInput,
) -> None:
    """Validate one provider record against its exact source input."""

    if not isinstance(record, EvidenceClassificationRecord):
        raise EvidenceClassificationError("INVALID_OUTPUT")
    try:
        record.validate_against_input(source)
        if record.state in {
            EvidenceClassificationState.CLASSIFIED,
            EvidenceClassificationState.TITLE_ONLY,
        }:
            assert record.support_level is not None
            assert record.supporting_excerpt is not None
            expected_reason = build_grounded_reason(
                record.support_level,
                record.supporting_excerpt,
                title_only=record.state is EvidenceClassificationState.TITLE_ONLY,
            )
            if record.reason != expected_reason:
                raise ValueError("reason is not the declared grounded reason grammar")
            if (
                record.support_level.value == "INDIRECT"
                and ("间接" not in record.reason or "仍需" not in record.reason or "验证" not in record.reason)
            ):
                raise ValueError("INDIRECT reason must include the validation caveat")
    except (AssertionError, TypeError, ValueError):
        raise EvidenceClassificationError("INVALID_OUTPUT") from None


def classify_evidence_batch(
    inputs: Sequence[EvidenceClassificationInput],
    provider: EvidenceClassifierProvider,
) -> EvidenceClassificationBatch:
    """Classify a canonicalized input batch with all-or-nothing provider validation."""

    if isinstance(inputs, (str, bytes)):
        raise EvidenceClassificationError("INVALID_INPUT")
    try:
        raw_inputs = list(inputs)
    except (TypeError, ValueError):
        raise EvidenceClassificationError("INVALID_INPUT") from None
    if not raw_inputs:
        raise EvidenceClassificationError("INVALID_INPUT")

    validated_inputs: list[EvidenceClassificationInput] = []
    for item in raw_inputs:
        try:
            if isinstance(item, EvidenceClassificationInput):
                validated = EvidenceClassificationInput.model_validate(item.model_dump(mode="python"))
            else:
                validated = EvidenceClassificationInput.model_validate(item)
        except (TypeError, ValueError, ValidationError):
            raise EvidenceClassificationError("INVALID_INPUT") from None
        validated_inputs.append(validated)

    paper_ids = [item.paper_id for item in validated_inputs]
    if len(paper_ids) != len(set(paper_ids)):
        raise EvidenceClassificationError("DUPLICATE_PAPER_ID")

    try:
        descriptor_value = provider.descriptor
        descriptor = EvidenceClassifierDescriptor.model_validate(
            descriptor_value.model_dump(mode="python")
            if isinstance(descriptor_value, EvidenceClassifierDescriptor)
            else descriptor_value
        )
    except (AttributeError, TypeError, ValueError, ValidationError):
        raise EvidenceClassificationError("PROVIDER_UNAVAILABLE") from None

    if any(item.classifier_descriptor != descriptor for item in validated_inputs):
        raise EvidenceClassificationError("INVALID_INPUT")

    canonical_inputs = sorted(validated_inputs, key=lambda item: item.paper_id)
    try:
        raw_outputs = provider.classify(canonical_inputs)
    except EvidenceClassificationError:
        raise
    except Exception:  # noqa: BLE001
        raise EvidenceClassificationError("PROVIDER_UNAVAILABLE") from None
    if isinstance(raw_outputs, (str, bytes)) or not isinstance(raw_outputs, Sequence):
        raise EvidenceClassificationError("INVALID_OUTPUT")
    if len(raw_outputs) != len(canonical_inputs):
        raise EvidenceClassificationError("INVALID_OUTPUT")

    input_by_paper_id = {item.paper_id: item for item in canonical_inputs}
    records_by_paper_id: dict[str, EvidenceClassificationRecord] = {}
    for raw_output in raw_outputs:
        try:
            payload: object
            if isinstance(raw_output, EvidenceClassificationRecord):
                payload = raw_output.model_dump(mode="python")
            elif isinstance(raw_output, Mapping):
                payload = raw_output
            else:
                raise TypeError
            record = EvidenceClassificationRecord.model_validate(payload)
        except (TypeError, ValueError, ValidationError):
            raise EvidenceClassificationError("INVALID_OUTPUT") from None
        if record.paper_id in records_by_paper_id:
            raise EvidenceClassificationError("INVALID_OUTPUT")
        source = input_by_paper_id.get(record.paper_id)
        if source is None:
            raise EvidenceClassificationError("PAPER_ID_MISMATCH")
        validate_classification_record(record, source)
        records_by_paper_id[record.paper_id] = record

    if set(records_by_paper_id) != set(input_by_paper_id):
        raise EvidenceClassificationError("PAPER_ID_MISMATCH")
    records = [records_by_paper_id[paper_id] for paper_id in sorted(input_by_paper_id)]
    state = (
        EvidenceClassificationBatchState.PARTIAL_FAILURE
        if any(
            record.state
            in {
                EvidenceClassificationState.REJECTED,
                EvidenceClassificationState.INSUFFICIENT_SOURCE_TEXT,
            }
            for record in records
        )
        else EvidenceClassificationBatchState.COMPLETE
    )
    try:
        return EvidenceClassificationBatch(
            state=state,
            records=records,
            classifier_descriptor=descriptor,
            classification_version=EVIDENCE_CLASSIFICATION_VERSION,
        )
    except (TypeError, ValueError, ValidationError):
        raise EvidenceClassificationError("INVALID_OUTPUT") from None
