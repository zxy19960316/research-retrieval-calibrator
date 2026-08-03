"""Closed red-green contracts for M2-T03 evidence-slot classification."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from app.adapters.evidence_classification import (
    DeterministicFakeEvidenceClassifier,
    EvidenceClassifierProvider,
    deterministic_fake_descriptor,
)
from app.core.evidence_classification import (
    classify_evidence_batch,
    source_text_sha256,
    validate_classification_record,
)
from app.models.enums import EvidenceSlot, SupportLevel
from app.models.evidence_classification import (
    EVIDENCE_CLASSIFICATION_VERSION,
    EvidenceClassificationBatch,
    EvidenceClassificationError,
    EvidenceClassificationInput,
    EvidenceClassificationRecord,
    EvidenceClassificationState,
    EvidenceClassifierDescriptor,
    serialize_source_text,
)


def _input(
    paper_id: str = "arxiv:test-1",
    *,
    title: str = "Graph retrieval for scientific literature",
    abstract: str | None = "We evaluate the method on a benchmark dataset.",
    **overrides: object,
) -> EvidenceClassificationInput:
    payload: dict[str, object] = {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "source_text_sha256": source_text_sha256(title, abstract),
        "classifier_descriptor": deterministic_fake_descriptor(),
        "classification_version": EVIDENCE_CLASSIFICATION_VERSION,
    }
    payload.update(overrides)
    return EvidenceClassificationInput.model_validate(payload)


def test_public_contracts_and_fixed_vocabularies_are_importable() -> None:
    provider = _accepts_provider_protocol(DeterministicFakeEvidenceClassifier())
    assert provider.descriptor == deterministic_fake_descriptor()
    assert EvidenceSlot.PROBLEM_EXISTENCE.value == "PROBLEM_EXISTENCE"
    assert SupportLevel.INDIRECT.value == "INDIRECT"
    assert EvidenceClassificationState.TITLE_ONLY.value == "TITLE_ONLY"


def _accepts_provider_protocol(provider: EvidenceClassifierProvider) -> EvidenceClassifierProvider:
    return provider


def test_source_serialization_and_sha256_are_recomputable_after_normalization() -> None:
    title = "  A title\r\n"
    abstract = "A first line\r\nA second line  "
    serialized = serialize_source_text(title, abstract)
    assert serialized == '{"abstract":"A first line\\nA second line","title":"A title"}'
    assert source_text_sha256(title, abstract) == hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()
    record = _input(title=title, abstract=abstract)
    assert record.title == "A title"
    assert record.abstract == "A first line\nA second line"


def test_contracts_forbid_unknown_fields_and_blank_text() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvidenceClassifierDescriptor.model_validate(
            deterministic_fake_descriptor().model_dump(mode="python") | {"unexpected": True}
        )
    with pytest.raises(ValidationError):
        EvidenceClassificationInput.model_validate(
            {
                "paper_id": "arxiv:test-1",
                "title": " ",
                "abstract": "valid",
                "source_text_sha256": "0" * 64,
                "classifier_descriptor": deterministic_fake_descriptor(),
                "classification_version": EVIDENCE_CLASSIFICATION_VERSION,
            }
        )
    with pytest.raises(ValidationError):
        EvidenceClassificationInput.model_validate(
            {
                "paper_id": "arxiv:test-1",
                "title": "valid",
                "abstract": "\t",
                "source_text_sha256": "0" * 64,
                "classifier_descriptor": deterministic_fake_descriptor(),
                "classification_version": EVIDENCE_CLASSIFICATION_VERSION,
            }
        )
    with pytest.raises(ValidationError):
        _input(source_text_sha256="0" * 64)
    with pytest.raises(ValidationError):
        _input(unexpected=True)


def test_direct_classification_is_grounded_in_source_text() -> None:
    item = _input(
        title="A graph retrieval method for literature discovery",
        abstract="We propose a graph retrieval method and demonstrate it on a benchmark dataset.",
    )
    result = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier())
    assert result.records[0].state is EvidenceClassificationState.CLASSIFIED
    assert result.records[0].support_level is SupportLevel.DIRECT
    assert result.records[0].supporting_excerpt in {item.title, item.abstract}
    validate_classification_record(result.records[0], item)


def test_indirect_classification_requires_explicit_validation_caveat() -> None:
    item = _input(
        title="A neighboring-domain graph method",
        abstract=(
            "This related-domain method may inform literature retrieval, but its effectiveness "
            "for the target problem remains to be validated."
        ),
    )
    result = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier())
    record = result.records[0]
    assert record.support_level is SupportLevel.INDIRECT
    assert "间接" in record.reason
    assert "仍需" in record.reason
    assert "验证" in record.reason


def test_hypothetical_classification_does_not_claim_verified_results() -> None:
    item = _input(
        title="A possible graph approach",
        abstract="This could be explored as a hypothesis for future literature retrieval.",
    )
    record = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier()).records[0]
    assert record.support_level is SupportLevel.HYPOTHETICAL
    assert all(word not in record.reason for word in ("已验证", "证明了", "demonstrates"))


def test_missing_abstract_can_be_explicit_title_only() -> None:
    item = _input(
        title="Graph retrieval benchmark",
        abstract=None,
    )
    record = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier()).records[0]
    assert record.state is EvidenceClassificationState.TITLE_ONLY
    assert record.supporting_excerpt == item.title
    assert "title-only" in record.reason


def test_title_without_reliable_evidence_is_rejected_not_defaulted() -> None:
    item = _input(title="A study", abstract=None)
    record = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier()).records[0]
    assert record.state is EvidenceClassificationState.INSUFFICIENT_SOURCE_TEXT
    assert record.evidence_slot is None
    assert record.support_level is None
    assert record.supporting_excerpt is None


def test_title_and_abstract_without_valid_text_fail_closed_at_input_boundary() -> None:
    with pytest.raises(ValidationError):
        EvidenceClassificationInput.model_validate(
            {
                "paper_id": "arxiv:test-1",
                "title": " ",
                "abstract": " ",
                "source_text_sha256": "0" * 64,
                "classifier_descriptor": deterministic_fake_descriptor(),
                "classification_version": EVIDENCE_CLASSIFICATION_VERSION,
            }
        )


@pytest.mark.parametrize("field", ["evidence_slot", "support_level"])
def test_unknown_closed_enum_values_are_rejected(field: str) -> None:
    item = _input()
    record_payload: dict[str, object] = {
        "state": "CLASSIFIED",
        "paper_id": item.paper_id,
        "evidence_slot": EvidenceSlot.CURRENT_METHODS,
        "support_level": SupportLevel.DIRECT,
        "reason": "来源摘录直接支持该证据槽位：We evaluate the method on a benchmark dataset。",
        "supporting_excerpt": item.abstract,
        "source_text_sha256": item.source_text_sha256,
        "classifier_descriptor": item.classifier_descriptor,
        "classification_version": EVIDENCE_CLASSIFICATION_VERSION,
    }
    record_payload[field] = "UNKNOWN"
    with pytest.raises(ValidationError):
        EvidenceClassificationRecord.model_validate(record_payload)


def test_empty_reason_is_rejected() -> None:
    item = _input()
    with pytest.raises(ValidationError):
        EvidenceClassificationRecord(
            state="CLASSIFIED",
            paper_id=item.paper_id,
            evidence_slot=EvidenceSlot.EVALUATION_BASIS,
            support_level=SupportLevel.DIRECT,
            reason=" ",
            supporting_excerpt=item.abstract,
            source_text_sha256=item.source_text_sha256,
            classifier_descriptor=item.classifier_descriptor,
            classification_version=EVIDENCE_CLASSIFICATION_VERSION,
        )


def test_record_cannot_claim_excerpt_outside_title_or_abstract() -> None:
    item = _input()
    record = EvidenceClassificationRecord(
        state="CLASSIFIED",
        paper_id=item.paper_id,
        evidence_slot=EvidenceSlot.EVALUATION_BASIS,
        support_level=SupportLevel.DIRECT,
        reason="来源摘录直接支持该证据槽位：not in the source。",
        supporting_excerpt="not in the source",
        source_text_sha256=item.source_text_sha256,
        classifier_descriptor=item.classifier_descriptor,
        classification_version=EVIDENCE_CLASSIFICATION_VERSION,
    )
    with pytest.raises(EvidenceClassificationError):
        validate_classification_record(record, item)


def test_reason_with_external_assertion_is_rejected() -> None:
    item = _input()
    record = EvidenceClassificationRecord(
        state="CLASSIFIED",
        paper_id=item.paper_id,
        evidence_slot=EvidenceSlot.EVALUATION_BASIS,
        support_level=SupportLevel.DIRECT,
        reason="来源摘录直接支持该证据槽位：We evaluate the method on a benchmark dataset。它证明了临床疗效。",
        supporting_excerpt=item.abstract,
        source_text_sha256=item.source_text_sha256,
        classifier_descriptor=item.classifier_descriptor,
        classification_version=EVIDENCE_CLASSIFICATION_VERSION,
    )
    with pytest.raises(EvidenceClassificationError):
        validate_classification_record(record, item)


def test_batch_rejects_conflicting_duplicate_paper_ids() -> None:
    first = _input()
    conflicting = _input(title="Different title")
    with pytest.raises(EvidenceClassificationError) as captured:
        classify_evidence_batch([first, conflicting], DeterministicFakeEvidenceClassifier())
    assert captured.value.code == "DUPLICATE_PAPER_ID"


def test_batch_contract_is_closed_and_has_one_descriptor() -> None:
    item = _input()
    record = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier()).records[0]
    batch = EvidenceClassificationBatch(
        state="COMPLETE",
        records=[record],
        classifier_descriptor=item.classifier_descriptor,
        classification_version=EVIDENCE_CLASSIFICATION_VERSION,
    )
    assert batch.records == [record]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvidenceClassificationBatch.model_validate(
            batch.model_dump(mode="python") | {"selection_rank": 1}
        )


def test_public_error_codes_are_stable_and_closed() -> None:
    for code in (
        "INVALID_INPUT",
        "PROVIDER_UNAVAILABLE",
        "INVALID_OUTPUT",
        "DUPLICATE_PAPER_ID",
        "SOURCE_HASH_MISMATCH",
        "RESULT_CONFLICT",
    ):
        assert EvidenceClassificationError(code).code == code
    with pytest.raises(ValueError, match="Unknown evidence classification error code"):
        EvidenceClassificationError("NOT_A_CODE")
