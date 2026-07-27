"""Closed, JSON-ready contracts for the bounded M1-T04 first-round run."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.dedup import DedupDecision, SourceIdentity
from app.models.planning import QueryPlan
from app.models.project import ResearchIntent


class FirstRoundStatus(StrEnum):
    """Terminal status of one first-round retrieval run."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class FirstRoundConfig(BaseModel):
    """Hard limits and provenance for one bounded retrieval run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_results_per_query: int = Field(default=5, ge=1, le=5)
    max_total_candidates: int = Field(default=60, ge=1, le=60)
    max_total_attempts: int = Field(default=20, ge=1, le=100)
    timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    min_request_interval_seconds: float = Field(default=3.0, ge=0, le=60)
    cache_dir: Path
    mode: Literal["recorded", "real"]
    cache_namespace: str = Field(default="first-round:recorded", min_length=1, max_length=100)
    adapter_schema_version: Literal["m1-t04.v2"] = "m1-t04.v2"

    @model_validator(mode="after")
    def require_real_mode_rate_limit(self) -> Self:
        if self.mode == "real" and self.min_request_interval_seconds < 1.0:
            raise ValueError("real mode requires min_request_interval_seconds >= 1.0")
        return self


class QueryExecutionResult(BaseModel):
    """One query's source observation, including an honest isolated failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1)
    status: Literal["success", "failed"]
    candidate_count: int = Field(ge=0)
    attempt_count: int = Field(ge=0)
    http_status: int | None = Field(default=None, ge=100, le=599)
    cache_hit: bool
    error_code: str | None = Field(default=None, min_length=1)

    @field_validator("query_id", "error_code")
    @classmethod
    def reject_blank_identifiers(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Query identifiers and error codes must be non-blank")
        return value

    @model_validator(mode="after")
    def require_consistent_status(self) -> Self:
        if self.status == "success" and self.error_code is not None:
            raise ValueError("Successful query executions cannot carry an error code")
        if self.status == "failed" and self.candidate_count != 0:
            raise ValueError("Failed query executions cannot carry candidates")
        if self.status == "failed" and self.error_code is None:
            raise ValueError("Failed query executions require an error code")
        return self


class CandidateOutput(BaseModel):
    """A source-backed, user-visible candidate copied only from deduplication output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[str]
    year: int | None = Field(default=None, ge=1900, le=2100)
    doi: str | None = None
    url: str = Field(min_length=1)
    retrieval_paths: list[str] = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    member_source_identities: list[SourceIdentity] = Field(min_length=1)
    merge_reasons: list[DedupDecision]

    @field_validator(
        "paper_id",
        "source",
        "source_id",
        "title",
        "url",
        "cluster_id",
        "doi",
    )
    @classmethod
    def reject_blank_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Candidate text fields must be non-blank when present")
        return value

    @field_validator("authors", "retrieval_paths")
    @classmethod
    def reject_blank_list_values(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("Candidate list values must be non-blank")
        return values

    @model_validator(mode="after")
    def require_visible_source_identity(self) -> Self:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("Visible candidates require an HTTP(S) URL")
        return self


class FailureReport(BaseModel):
    """A bounded, JSON-safe report for a run or individual-query failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    error_code: str = Field(min_length=1)
    scope: Literal["run", "query"]
    query_id: str | None = Field(default=None, min_length=1)
    reason: str | None = Field(default=None, min_length=1, max_length=1000)

    @field_validator("error_code", "query_id", "reason")
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Failure-report text must be non-blank when present")
        return value

    @model_validator(mode="after")
    def require_scope_provenance(self) -> Self:
        if (self.scope == "query") != (self.query_id is not None):
            raise ValueError("Query failures require query_id; run failures must not carry query_id")
        return self


class RunMetrics(BaseModel):
    """Raw and displayed counts plus bounded provenance and quality measurements."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_candidate_count: int = Field(ge=0)
    deduplicated_candidate_count: int = Field(ge=0)
    manual_review_count: int = Field(ge=0)
    source_id_coverage: float = Field(ge=0, le=1)
    url_coverage: float = Field(ge=0, le=1)
    metadata_hallucination_rate: float = Field(ge=0, le=1)
    metadata_projection_mismatch_count: int = Field(ge=0)
    candidate_budget_reached: bool
    transport_requests: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    configured_min_request_interval_seconds: float = Field(ge=0)
    request_start_offsets_seconds: list[float] = Field(default_factory=list)
    minimum_observed_request_start_delta_seconds: float | None = Field(default=None, ge=0)
    rate_limit_wait_count: int = Field(ge=0)
    rate_limit_wait_seconds: float = Field(ge=0)


class FirstRoundRun(BaseModel):
    """Complete first-round provenance and terminal output for JSON rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    status: FirstRoundStatus
    started_at_utc: datetime
    finished_at_utc: datetime
    question: str = Field(min_length=1)
    config: FirstRoundConfig
    intent: ResearchIntent | None = None
    query_plan: QueryPlan | None = None
    query_results: list[QueryExecutionResult]
    candidates: list[CandidateOutput]
    raw_candidate_count: int = Field(ge=0)
    deduplicated_candidate_count: int = Field(ge=0)
    metrics: RunMetrics
    failure: FailureReport | None = None
    error_code: str | None = Field(default=None, min_length=1)

    @field_validator("run_id", "question", "error_code")
    @classmethod
    def reject_blank_run_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Run text fields must be non-blank when present")
        return value

    @field_validator("started_at_utc", "finished_at_utc")
    @classmethod
    def require_utc_timestamps(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("Run timestamps must be UTC-aware")
        return value

    @model_validator(mode="after")
    def require_consistent_terminal_state(self) -> Self:
        if self.finished_at_utc < self.started_at_utc:
            raise ValueError("Run finish time cannot precede start time")
        if self.raw_candidate_count != self.metrics.raw_candidate_count:
            raise ValueError("Run raw candidate count must match metrics")
        if self.deduplicated_candidate_count != self.metrics.deduplicated_candidate_count:
            raise ValueError("Run deduplicated candidate count must match metrics")
        if len(self.candidates) != self.deduplicated_candidate_count:
            raise ValueError("Displayed candidates must match the deduplicated candidate count")
        if self.status is FirstRoundStatus.FAILED:
            if self.candidates or self.deduplicated_candidate_count != 0:
                raise ValueError("Failed runs cannot carry displayed candidates")
            if self.failure is None or self.failure.scope != "run" or self.error_code is None:
                raise ValueError("Failed runs require a run-scoped failure and error code")
            if self.failure.error_code != self.error_code:
                raise ValueError("Failed run error_code must match its failure report")
        elif self.failure is not None or self.error_code is not None:
            raise ValueError("Successful runs cannot carry a run-level failure")
        elif not self.candidates:
            raise ValueError("Successful runs require at least one displayed candidate")
        return self
