"""Closed-schema contracts for the M1-T04 first-round boundary."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models.dedup import SourceIdentity
from app.models.first_round import (
    CandidateOutput,
    FailureReport,
    FirstRoundConfig,
    FirstRoundStatus,
    QueryExecutionResult,
    RunMetrics,
)


def test_config_rejects_unbounded_or_inconsistent_limits() -> None:
    with pytest.raises(ValidationError):
        FirstRoundConfig(cache_dir=Path("cache"), mode="recorded", max_results_per_query=0)
    with pytest.raises(ValidationError):
        FirstRoundConfig(cache_dir=Path("cache"), mode="recorded", max_total_candidates=61)
    with pytest.raises(ValidationError):
        FirstRoundConfig(cache_dir=Path("cache"), mode="recorded", timeout_seconds=121)
    with pytest.raises(ValidationError):
        FirstRoundConfig.model_validate(
            {"cache_dir": "cache", "mode": "recorded", "unexpected": True}
        )


def test_real_mode_requires_a_positive_request_interval_but_recorded_can_disable_it() -> None:
    with pytest.raises(ValidationError):
        FirstRoundConfig(
            cache_dir=Path("cache"),
            mode="real",
            min_request_interval_seconds=0.0,
        )

    config = FirstRoundConfig(
        cache_dir=Path("cache"),
        mode="recorded",
        min_request_interval_seconds=0.0,
    )

    assert config.min_request_interval_seconds == 0.0


def test_visible_candidate_requires_source_identity_and_http_url() -> None:
    with pytest.raises(ValidationError):
        CandidateOutput.model_validate({"paper_id": "arxiv:1", "source": "arxiv"})

    payload = _candidate_payload()
    payload["source_id"] = " "
    with pytest.raises(ValidationError):
        CandidateOutput.model_validate(payload)

    payload = _candidate_payload()
    payload["url"] = "ftp://arxiv.org/abs/1"
    with pytest.raises(ValidationError, match="HTTP\\(S\\)"):
        CandidateOutput.model_validate(payload)


def test_failure_and_run_contracts_forbid_unknown_fields() -> None:
    failure = FailureReport(error_code="ARXIV_UNAVAILABLE", scope="run")
    with pytest.raises(ValidationError):
        FailureReport.model_validate({**failure.model_dump(), "unexpected": True})

    with pytest.raises(ValidationError):
        RunMetrics.model_validate({"unexpected": True})


def test_query_execution_requires_stable_failure_shape() -> None:
    with pytest.raises(ValidationError):
        QueryExecutionResult(
            query_id="Q1",
            status="success",
            candidate_count=0,
            attempt_count=1,
            http_status=None,
            cache_hit=False,
            error_code="ARXIV_UNAVAILABLE",
        )
    with pytest.raises(ValidationError):
        QueryExecutionResult(
            query_id="Q1",
            status="failed",
            candidate_count=1,
            attempt_count=1,
            http_status=503,
            cache_hit=False,
        )


def test_status_values_are_json_stable() -> None:
    assert FirstRoundStatus.SUCCESS.value == "success"
    assert FirstRoundStatus.PARTIAL_SUCCESS.value == "partial_success"
    assert FirstRoundStatus.FAILED.value == "failed"


def _candidate_payload() -> dict[str, object]:
    return {
        "paper_id": "arxiv:2401.00001",
        "source": "arxiv",
        "source_id": "2401.00001",
        "title": "A Source-Backed Paper",
        "authors": ["Ada Author"],
        "year": 2024,
        "doi": None,
        "url": "https://arxiv.org/abs/2401.00001",
        "retrieval_paths": ["Q1"],
        "cluster_id": "arxiv:2401.00001",
        "member_source_identities": [
            SourceIdentity(
                source="arxiv",
                source_id="2401.00001",
                url="https://arxiv.org/abs/2401.00001",
            )
        ],
        "merge_reasons": [],
    }
