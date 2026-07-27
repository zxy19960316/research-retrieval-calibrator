"""Immutable-derived contracts for deterministic paper normalization and deduplication."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.paper import PaperRecord


class SourceIdentity(BaseModel):
    """A preserved source identity from one original paper record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    url: str = Field(min_length=1)

    @field_validator("source", "source_id", "url")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Source identity values must be non-blank")
        return value


class NormalizedPaper(BaseModel):
    """A paper record plus derived values; the embedded record is never altered."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record: PaperRecord
    canonical_doi: str | None
    canonical_arxiv_id: str | None
    normalized_title: str = Field(min_length=1)
    title_tokens: tuple[str, ...] = Field(min_length=1)
    normalized_authors: tuple[str, ...]


class DedupReason(str, Enum):
    EXACT_DOI = "EXACT_DOI"
    EXACT_ARXIV_ID = "EXACT_ARXIV_ID"
    EXACT_NORMALIZED_TITLE_WITH_AUTHOR = "EXACT_NORMALIZED_TITLE_WITH_AUTHOR"
    HIGH_TITLE_SIMILARITY_WITH_AUTHOR = "HIGH_TITLE_SIMILARITY_WITH_AUTHOR"
    CROSS_LANGUAGE_POSSIBLE_DUPLICATE = "CROSS_LANGUAGE_POSSIBLE_DUPLICATE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    IDENTITY_CONFLICT = "IDENTITY_CONFLICT"
    TRANSITIVE_BRIDGE_RISK = "TRANSITIVE_BRIDGE_RISK"


class PaperDeduplicationError(ValueError):
    """A stable rejection for incompatible duplicate source observations."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class DedupDecision(BaseModel):
    """A reproducible pairwise decision with the evidence used to make it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left_paper_id: str = Field(min_length=1)
    right_paper_id: str = Field(min_length=1)
    action: Literal["auto_merge", "manual_review", "keep_separate"]
    reason: DedupReason
    title_similarity: float | None = Field(default=None, ge=0, le=1)
    author_overlap: float | None = Field(default=None, ge=0, le=1)
    year_difference: int | None = Field(default=None, ge=0)


class DedupCluster(BaseModel):
    """A stable component of original records joined only by automatic merges."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cluster_id: str = Field(min_length=1)
    canonical_record: PaperRecord
    member_records: list[PaperRecord] = Field(min_length=1)
    retrieval_paths: list[str] = Field(min_length=1)
    source_identities: list[SourceIdentity] = Field(min_length=1)
    merge_reasons: list[DedupDecision]

    @field_validator("retrieval_paths")
    @classmethod
    def require_sorted_nonblank_unique_paths(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("Retrieval paths must be non-blank")
        if values != sorted(set(values)):
            raise ValueError("Retrieval paths must be sorted and unique")
        return values

    @field_validator("source_identities")
    @classmethod
    def require_sorted_unique_source_identities(
        cls, values: list[SourceIdentity]
    ) -> list[SourceIdentity]:
        keys = [(value.source, value.source_id, value.url) for value in values]
        if keys != sorted(set(keys)):
            raise ValueError("Source identities must be sorted and unique")
        return values

    @model_validator(mode="after")
    def retain_canonical_record(self) -> DedupCluster:
        if self.canonical_record.paper_id not in {
            record.paper_id for record in self.member_records
        }:
            raise ValueError("Canonical record must be a cluster member")
        return self


class DeduplicationResult(BaseModel):
    """The complete deterministic output, including non-merge review decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clusters: list[DedupCluster]
    decisions: list[DedupDecision]

    @model_validator(mode="after")
    def require_unique_cluster_ids(self) -> DeduplicationResult:
        cluster_ids = [cluster.cluster_id for cluster in self.clusters]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("Cluster IDs must be unique")
        return self


class DedupConfig(BaseModel):
    """Centralized conservative thresholds for deterministic similarity matching."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title_token_jaccard_threshold: float = Field(default=0.92, ge=0, le=1)
    title_sequence_ratio_threshold: float = Field(default=0.96, ge=0, le=1)
    author_jaccard_threshold: float = Field(default=0.50, ge=0, le=1)
    allowed_year_difference: int = Field(default=1, ge=0)
