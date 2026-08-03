"""Deterministic fake and provider-boundary tests for M2-T03."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.adapters.evidence_classification import (
    DeterministicFakeEvidenceClassifier,
    EvidenceClassifierProvider,
)
from app.core.evidence_classification import classify_evidence_batch
from app.models.evidence_classification import EvidenceClassificationInput
from tests.contract.test_m2_t03_evidence_classification import _input


class _FailingProvider:
    descriptor = DeterministicFakeEvidenceClassifier().descriptor

    def classify(self, inputs: Sequence[EvidenceClassificationInput]) -> list[object]:
        del inputs
        raise RuntimeError("provider failure must not leak")


class _MalformedProvider:
    descriptor = DeterministicFakeEvidenceClassifier().descriptor

    def classify(self, inputs: Sequence[EvidenceClassificationInput]) -> list[object]:
        return [
            {
                "state": "CLASSIFIED",
                "paper_id": inputs[0].paper_id,
                "evidence_slot": "UNKNOWN",
                "support_level": "DIRECT",
                "reason": "invalid",
                "supporting_excerpt": inputs[0].title,
                "source_text_sha256": inputs[0].source_text_sha256,
                "classifier_descriptor": inputs[0].classifier_descriptor,
                "classification_version": inputs[0].classification_version,
            }
        ]


def _accepts_provider_protocol(provider: EvidenceClassifierProvider) -> EvidenceClassifierProvider:
    return provider


def test_provider_protocol_accepts_deterministic_fake_without_runtime_imports() -> None:
    assert _accepts_provider_protocol(DeterministicFakeEvidenceClassifier()) is not None


def test_repeated_fake_runs_are_identical() -> None:
    items = [_input("arxiv:test-2"), _input("arxiv:test-1")]
    provider = DeterministicFakeEvidenceClassifier()
    first = classify_evidence_batch(items, provider)
    second = classify_evidence_batch(items, provider)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_reordered_inputs_preserve_paper_id_keyed_outputs() -> None:
    items = [_input("arxiv:test-2"), _input("arxiv:test-1")]
    provider = DeterministicFakeEvidenceClassifier()
    first = classify_evidence_batch(items, provider)
    second = classify_evidence_batch(list(reversed(items)), provider)
    first_by_id = {record.paper_id: record.model_dump(mode="json") for record in first.records}
    second_by_id = {record.paper_id: record.model_dump(mode="json") for record in second.records}
    assert first_by_id == second_by_id


def test_provider_exception_is_an_explicit_failure_without_partial_output() -> None:
    with pytest.raises(Exception) as captured:
        classify_evidence_batch([_input()], _FailingProvider())
    assert getattr(captured.value, "code", None) == "PROVIDER_UNAVAILABLE"


def test_malformed_provider_output_is_an_explicit_failure() -> None:
    with pytest.raises(Exception) as captured:
        classify_evidence_batch([_input()], _MalformedProvider())
    assert getattr(captured.value, "code", None) == "INVALID_OUTPUT"


def test_classification_output_has_no_scoring_or_selection_fields() -> None:
    result = classify_evidence_batch([_input()], DeterministicFakeEvidenceClassifier())
    payload = result.model_dump(mode="json")
    serialized_keys = str(payload)
    for forbidden in ("score", "weight", "total_score", "selection_rank", "rank"):
        assert forbidden not in serialized_keys
