"""Direct red-green contracts for the M2-T02 model and provider boundary."""

from __future__ import annotations

import hashlib
import math

import pytest
from pydantic import ValidationError

from app.adapters.reranking import RerankerProvider
from app.models.reranking import (
    ProviderRawScore,
    RerankerInput,
    RerankerModelDescriptor,
    RerankerRunState,
    RerankerTaskError,
    RerankRecord,
    RerankRun,
)


def _descriptor_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "provider_name": "synthetic",
        "model_id": "synthetic-cross-encoder",
        "model_revision": "a" * 40,
        "provider_library": "tests",
        "provider_library_version": "1",
        "input_format_version": "m2-reranker-title-abstract-v1",
        "cache_namespace": "reranker:synthetic",
    }
    payload.update(overrides)
    return payload


def _descriptor(**overrides: object) -> RerankerModelDescriptor:
    return RerankerModelDescriptor.model_validate(_descriptor_payload(**overrides))


def _input_payload(**overrides: object) -> dict[str, object]:
    text = "title:\nA source-backed paper"
    payload: dict[str, object] = {
        "paper_id": "arxiv:2401.00001",
        "text": text,
        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    payload.update(overrides)
    return payload


def _scored_record_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "state": "SCORED",
        "paper_id": "arxiv:2401.00001",
        "raw_score": 2.0,
        "normalized_score": 0.5,
        "descriptor": _descriptor().model_dump(mode="json"),
        "input_sha256": _input_payload()["input_sha256"],
    }
    payload.update(overrides)
    return payload


def _accepts_provider_protocol(provider: RerankerProvider) -> RerankerProvider:
    return provider


def test_reranker_provider_protocol_is_importable_without_core_dependency() -> None:
    assert _accepts_provider_protocol is not None


def test_reranker_task_error_exposes_only_declared_stable_codes() -> None:
    for code in ("INVALID_INPUT", "PROVIDER_UNAVAILABLE", "INVALID_OUTPUT"):
        assert RerankerTaskError(code).code == code
    with pytest.raises(ValueError, match="Unknown reranker task error code"):
        RerankerTaskError("NOT_A_DECLARED_CODE")


@pytest.mark.parametrize(
    "field",
    [
        "provider_name",
        "model_id",
        "provider_library",
        "provider_library_version",
        "cache_namespace",
    ],
)
def test_descriptor_rejects_blank_identity_fields(field: str) -> None:
    with pytest.raises(ValidationError, match="non-blank"):
        _descriptor(**{field: "  "})


@pytest.mark.parametrize("revision", ["latest", "main", "unknown", "a" * 39, "A" * 40, "g" * 40])
def test_descriptor_requires_immutable_lowercase_40_hex_revision(revision: str) -> None:
    with pytest.raises(ValidationError, match="40-character|immutable"):
        _descriptor(model_revision=revision)


def test_descriptor_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RerankerModelDescriptor.model_validate(_descriptor_payload(unexpected=True))


@pytest.mark.parametrize(
    "input_format_version",
    ["m2-reranker-title-abstract-v1", "m2-reranker-title-abstract-v2"],
)
def test_descriptor_accepts_explicit_v1_and_v2_input_formats(input_format_version: str) -> None:
    assert _descriptor(input_format_version=input_format_version).input_format_version == input_format_version


@pytest.mark.parametrize("input_format_version", ["", " ", "\t", "m2-reranker-title-abstract-v3"])
def test_descriptor_rejects_blank_or_unrecognized_input_format(input_format_version: str) -> None:
    with pytest.raises(ValidationError, match="input format"):
        _descriptor(input_format_version=input_format_version)


def test_reranker_input_requires_exact_utf8_text_sha_and_closed_fields() -> None:
    input_record = RerankerInput.model_validate(_input_payload())
    assert input_record.input_sha256 == hashlib.sha256(input_record.text.encode("utf-8")).hexdigest()

    with pytest.raises(ValidationError, match="paper_id"):
        RerankerInput.model_validate(_input_payload(paper_id=" "))
    with pytest.raises(ValidationError, match="text"):
        RerankerInput.model_validate(_input_payload(text=" "))
    with pytest.raises(ValidationError, match="input_sha256"):
        RerankerInput.model_validate(_input_payload(input_sha256="a" * 64))
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RerankerInput.model_validate(_input_payload(unexpected=True))


@pytest.mark.parametrize("raw_score", [-1, 0, 1, 0.25])
def test_provider_raw_score_accepts_finite_int_and_float(raw_score: float) -> None:
    assert ProviderRawScore(paper_id="arxiv:2401.00001", raw_score=raw_score).raw_score == raw_score


@pytest.mark.parametrize("raw_score", [True, "1.0", float("nan"), float("inf"), float("-inf")])
def test_provider_raw_score_rejects_bool_string_and_non_finite_values(raw_score: object) -> None:
    with pytest.raises(ValidationError, match="finite|number|boolean"):
        ProviderRawScore(paper_id="arxiv:2401.00001", raw_score=raw_score)


def test_provider_raw_score_rejects_blank_id_and_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="paper_id"):
        ProviderRawScore(paper_id=" ", raw_score=1.0)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProviderRawScore.model_validate(
            {"paper_id": "arxiv:2401.00001", "raw_score": 1.0, "unexpected": True}
        )


def test_scored_record_requires_finite_raw_and_normalized_scores_with_provenance() -> None:
    record = RerankRecord.model_validate(_scored_record_payload())
    assert record.state is RerankerRunState.SCORED
    assert record.descriptor == _descriptor()

    for field in ("raw_score", "normalized_score", "descriptor", "input_sha256"):
        payload = _scored_record_payload()
        payload.pop(field)
        with pytest.raises(ValidationError):
            RerankRecord.model_validate(payload)
    for normalized_score in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValidationError, match="normalized|finite"):
            RerankRecord.model_validate(_scored_record_payload(normalized_score=normalized_score))
    with pytest.raises(ValidationError, match="raw|finite"):
        RerankRecord.model_validate(_scored_record_payload(raw_score=float("nan")))


@pytest.mark.parametrize("state", ["NOT_RUN", "PROVIDER_UNAVAILABLE", "INVALID_OUTPUT"])
def test_non_scored_record_cannot_carry_numeric_scores(state: str) -> None:
    payload = _scored_record_payload(state=state, raw_score=None, normalized_score=None)
    assert RerankRecord.model_validate(payload).state.value == state
    with pytest.raises(ValidationError, match="non-SCORED|score"):
        RerankRecord.model_validate(_scored_record_payload(state=state))


def test_rerank_run_enforces_state_record_cardinality_identity_and_descriptor_consistency() -> None:
    first = RerankRecord.model_validate(_scored_record_payload())
    second = RerankRecord.model_validate(
        _scored_record_payload(paper_id="arxiv:2401.00002", input_sha256="b" * 64)
    )
    assert RerankRun(state="SCORED", records=[first, second]).state is RerankerRunState.SCORED
    assert RerankRun(state="NOT_RUN", records=[]).records == []

    with pytest.raises(ValidationError, match="SCORED|records"):
        RerankRun(state="SCORED", records=[])
    with pytest.raises(ValidationError, match="NOT_RUN|records"):
        RerankRun(state="NOT_RUN", records=[first])
    for state in ("PROVIDER_UNAVAILABLE", "INVALID_OUTPUT"):
        with pytest.raises(ValidationError, match="partial|records"):
            RerankRun(state=state, records=[first])
    with pytest.raises(ValidationError, match="paper_id"):
        RerankRun(state="SCORED", records=[first, first])

    other_descriptor = _descriptor(model_revision="b" * 40)
    inconsistent = RerankRecord.model_validate(
        _scored_record_payload(
            paper_id="arxiv:2401.00003",
            input_sha256="c" * 64,
            descriptor=other_descriptor.model_dump(mode="json"),
        )
    )
    with pytest.raises(ValidationError, match="descriptor"):
        RerankRun(state="SCORED", records=[first, inconsistent])


def test_model_contract_uses_only_finite_numeric_values() -> None:
    assert math.isfinite(ProviderRawScore(paper_id="arxiv:2401.00001", raw_score=0.0).raw_score)
