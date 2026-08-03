"""Deterministic fake and provider-boundary tests for M2-T03."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.adapters.evidence_classification import (
    _ClassificationOutcome,
    _build_quality_diagnostics,
    DeterministicFakeEvidenceClassifier,
    EvidenceClassifierProvider,
    _collect_matches,
    _source_fragments,
)
from app.core.evidence_classification import classify_evidence_batch
from app.models.enums import EvidenceSlot, SupportLevel
from app.models.evidence_classification import (
    EvidenceClassificationInput,
    EvidenceClassificationState,
)
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


def _fragment_texts(source: str) -> list[str]:
    return [fragment for fragment, _, _ in _source_fragments(source, "abstract")]


def test_dotted_identifier_is_one_fragment_and_selected_excerpt_preserves_prefix() -> None:
    source = "Clinfo.ai: An Open-Source Retrieval-Augmented System"
    fragments = _source_fragments(source, "abstract")

    assert [fragment for fragment, _, _ in fragments] == [source]
    assert source[fragments[0][2] : fragments[0][2] + len(fragments[0][0])] == fragments[0][0]

    item = _input(title=source, abstract=None)
    record = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier()).records[0]
    assert record.supporting_excerpt is not None
    assert "Clinfo.ai" in record.supporting_excerpt


def test_decimal_is_not_truncated_in_fragment_or_excerpt() -> None:
    source = "The method achieves a mean Average Precision of 59.70%."
    assert _fragment_texts(source) == [source]

    item = _input(title="A scientific paper", abstract=source)
    record = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier()).records[0]
    assert record.supporting_excerpt is not None
    assert "59.70%" in record.supporting_excerpt
    assert "59." not in record.supporting_excerpt.replace("59.70%", "")


@pytest.mark.parametrize(
    "source",
    [
        "We use e.g. a graph method. It is evaluated.",
        "We use i.e. a graph method. It is evaluated.",
        "The result follows et al. in the cited study. It is evaluated.",
        "Fig. 2 shows the pipeline. It is evaluated.",
        "Dr. Smith presents the method. It is evaluated.",
        "The method is compared vs. a baseline. It is evaluated.",
    ],
)
def test_common_abbreviations_do_not_create_internal_fragments(source: str) -> None:
    fragments = _fragment_texts(source)

    assert len(fragments) == 2
    assert "It is evaluated." in fragments[1]


@pytest.mark.parametrize(
    "source",
    [
        "The contract targets v1.2. It is evaluated.",
        "The runner uses Python 3.12. It is evaluated.",
        "The identifier is 10.1000/example.doi. It is evaluated.",
        "The endpoint is example.org/path. It is evaluated.",
    ],
)
def test_versions_urls_and_dois_remain_inside_their_fragment(source: str) -> None:
    fragments = _fragment_texts(source)

    assert len(fragments) == 2
    assert "It is evaluated." in fragments[1]


def test_real_sentence_ending_still_splits_in_order() -> None:
    source = "We implement a retrieval pipeline. It is evaluated on a benchmark."

    assert _fragment_texts(source) == [
        "We implement a retrieval pipeline.",
        "It is evaluated on a benchmark.",
    ]


def test_scanner_preserves_order_offsets_and_exact_substrings() -> None:
    source = "Python 3.12. We implement a pipeline."
    fragments = _source_fragments(source, "abstract")

    assert [fragment for fragment, _, _ in fragments] == [
        "Python 3.12.",
        "We implement a pipeline.",
    ]
    assert all(
        source[offset : offset + len(fragment)] == fragment
        for fragment, _, offset in fragments
    )
    assert [offset for _, _, offset in fragments] == sorted(
        offset for _, _, offset in fragments
    )


def test_evidence_match_marker_and_excerpt_share_source_offset() -> None:
    item = _input(
        title="Cross-domain retrieval method",
        abstract="The method transfers to a related-domain task. It demonstrates strong results.",
    )

    matches = _collect_matches(item)

    assert matches
    for candidate in matches:
        evidence = candidate.evidence
        source = item.title if evidence.matched_source == "title" else item.abstract
        assert source is not None
        assert source[
            evidence.excerpt_start : evidence.excerpt_start + len(evidence.supporting_excerpt)
        ] == evidence.supporting_excerpt
        assert evidence.matched_marker.casefold() in evidence.supporting_excerpt.casefold()


def test_excerpt_does_not_absorb_direct_marker_from_next_sentence() -> None:
    item = _input(
        title="A cross-domain retrieval method",
        abstract="The method transfers to a related-domain task. It demonstrates strong results.",
    )

    record = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier()).records[0]

    assert record.support_level is SupportLevel.INDIRECT
    assert record.supporting_excerpt == "The method transfers to a related-domain task."
    assert "demonstrates" not in record.supporting_excerpt


def _diagnostic_outcomes(
    support_levels: Sequence[SupportLevel | str],
) -> list[_ClassificationOutcome]:
    slots = list(EvidenceSlot)
    return [
        _ClassificationOutcome(
            record={"evidence_slot": slots[index], "support_level": support_level},
            rejection_kind="none",
        )
        for index, support_level in enumerate(support_levels)
    ]


@pytest.mark.parametrize(
    ("support_levels", "same_support_warning", "missing_support_warning"),
    [
        ([SupportLevel.DIRECT, SupportLevel.DIRECT, SupportLevel.DIRECT], True, True),
        ([SupportLevel.DIRECT, SupportLevel.INDIRECT, SupportLevel.DIRECT], False, False),
        ([SupportLevel.DIRECT, SupportLevel.HYPOTHETICAL, SupportLevel.DIRECT], False, False),
        ([SupportLevel.INDIRECT, SupportLevel.INDIRECT, SupportLevel.INDIRECT], True, False),
    ],
)
def test_support_level_diagnostic_warnings_are_type_explicit(
    support_levels: Sequence[SupportLevel],
    same_support_warning: bool,
    missing_support_warning: bool,
) -> None:
    diagnostics = _build_quality_diagnostics(_diagnostic_outcomes(support_levels))
    warnings = set(diagnostics.warnings)

    assert ("ALL_RECORDS_SAME_SUPPORT_LEVEL" in warnings) is same_support_warning
    assert ("NO_INDIRECT_OR_HYPOTHETICAL_RECORDS" in warnings) is missing_support_warning


def test_support_level_diagnostics_accept_serialized_values() -> None:
    diagnostics = _build_quality_diagnostics(
        _diagnostic_outcomes(
            [SupportLevel.DIRECT.value, SupportLevel.INDIRECT.value, SupportLevel.DIRECT.value]
        )
    )

    assert "ALL_RECORDS_SAME_SUPPORT_LEVEL" not in diagnostics.warnings
    assert "NO_INDIRECT_OR_HYPOTHETICAL_RECORDS" not in diagnostics.warnings


def test_unknown_support_level_fails_closed_in_diagnostics() -> None:
    with pytest.raises(ValueError):
        _build_quality_diagnostics(_diagnostic_outcomes(["UNKNOWN_SUPPORT_LEVEL"]))


def test_title_only_marker_uses_title_excerpt() -> None:
    item = _input(
        title="Benchmark for graph retrieval",
        abstract="This paper examines a scientific corpus.",
    )

    record = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier()).records[0]

    assert record.evidence_slot is EvidenceSlot.EVALUATION_BASIS
    assert record.supporting_excerpt == item.title


def test_title_marker_does_not_fallback_to_unrelated_abstract() -> None:
    item = _input(
        title="Pipeline for graph retrieval",
        abstract="This paper concerns a broad research topic.",
    )

    record = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier()).records[0]

    assert record.evidence_slot is EvidenceSlot.IMPLEMENTATION_PATH
    assert record.supporting_excerpt == item.title
    assert record.supporting_excerpt not in (item.abstract or "")


@pytest.mark.parametrize(
    "title",
    [
        "A system for research",
        "A dataset for science",
        "Results from a study",
        "An application for research",
    ],
)
def test_generic_system_dataset_results_application_words_do_not_classify_alone(
    title: str,
) -> None:
    item = _input(title=title, abstract="A broad resource is described.")

    record = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier()).records[0]

    assert record.state is EvidenceClassificationState.REJECTED
    assert record.evidence_slot is None
    assert record.support_level is None
    assert record.supporting_excerpt is None


def test_support_level_is_bound_to_the_selected_sentence() -> None:
    item = _input(
        title="Cross-domain retrieval method",
        abstract="It demonstrates a generic property in a separate sentence.",
    )

    record = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier()).records[0]

    assert record.evidence_slot is EvidenceSlot.METHOD_TRANSFERABILITY
    assert record.support_level is SupportLevel.INDIRECT
    assert record.supporting_excerpt == item.title
    assert record.supporting_excerpt not in (item.abstract or "")


def test_equal_low_specificity_matches_are_rejected() -> None:
    item = _input(
        title="A system and dataset",
        abstract="A broad resource is described.",
    )

    record = classify_evidence_batch([item], DeterministicFakeEvidenceClassifier()).records[0]

    assert record.state is EvidenceClassificationState.REJECTED
    assert record.evidence_slot is None
    assert record.support_level is None
    assert record.supporting_excerpt is None


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
