"""Closed, JSON-ready contracts for M2-T01 embedding provenance and vectors."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from app.models.dedup import SourceIdentity

EmbeddingErrorCode = Literal[
    "FROZEN_SNAPSHOT_MISSING",
    "FROZEN_SNAPSHOT_HASH_MISMATCH",
    "INVALID_EMBEDDING_INPUT",
    "DUPLICATE_EMBEDDING_ID",
    "MODEL_REVISION_UNPINNED",
    "EMBEDDING_PROVIDER_UNAVAILABLE",
    "EMBEDDING_PROVIDER_FAILED",
    "EMBEDDING_COUNT_MISMATCH",
    "EMBEDDING_DIMENSION_MISMATCH",
    "EMBEDDING_NON_FINITE",
    "EMBEDDING_CACHE_CORRUPT",
    "EMBEDDING_OUTPUT_WRITE_FAILED",
]

EMBEDDING_ERROR_CODES: tuple[EmbeddingErrorCode, ...] = (
    "FROZEN_SNAPSHOT_MISSING",
    "FROZEN_SNAPSHOT_HASH_MISMATCH",
    "INVALID_EMBEDDING_INPUT",
    "DUPLICATE_EMBEDDING_ID",
    "MODEL_REVISION_UNPINNED",
    "EMBEDDING_PROVIDER_UNAVAILABLE",
    "EMBEDDING_PROVIDER_FAILED",
    "EMBEDDING_COUNT_MISMATCH",
    "EMBEDDING_DIMENSION_MISMATCH",
    "EMBEDDING_NON_FINITE",
    "EMBEDDING_CACHE_CORRUPT",
    "EMBEDDING_OUTPUT_WRITE_FAILED",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HF_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PLACEHOLDER_ABSTRACTS = frozenset({"no abstract", "not provided", "no abstract available"})
BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_FORBIDDEN_PROVIDER_VERSIONS = frozenset({"optional", "unknown", "latest", "unavailable"})


class EmbeddingTaskError(ValueError):
    """A stable, machine-readable error at an M2-T01 boundary."""

    def __init__(self, code: EmbeddingErrorCode | str) -> None:
        if code not in EMBEDDING_ERROR_CODES:
            raise ValueError(f"Unknown embedding task error code: {code}")
        self.code: EmbeddingErrorCode = code
        super().__init__(code)


class EmbeddingModelDescriptor(BaseModel):
    """The complete identity of one dense embedding provider configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str
    provider_library: str = Field(min_length=1)
    provider_library_version: str = Field(min_length=1)
    embedding_mode: Literal["dense"]
    input_format_version: Literal["m2-title-abstract-v1"]
    normalized: StrictBool
    dimension: int = Field(gt=0)
    cache_namespace: str = Field(min_length=1)

    @field_validator(
        "provider_name",
        "model_id",
        "provider_library",
        "provider_library_version",
        "cache_namespace",
    )
    @classmethod
    def reject_blank_descriptor_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Embedding descriptor values must be non-blank")
        return value

    @model_validator(mode="after")
    def require_fake_real_separation_and_pinned_real_revision(self) -> Self:
        if self.provider_name == "deterministic_fake":
            fake_identity = (
                self.model_id == "sha256-vector"
                and self.model_revision == "fake-v1"
                and self.provider_library == "stdlib"
                and self.provider_library_version == "3.12"
                and self.embedding_mode == "dense"
                and self.input_format_version == "m2-title-abstract-v1"
                and self.normalized is True
                and self.dimension == 16
                and self.cache_namespace == "embedding:fake"
            )
            if not fake_identity:
                raise ValueError("deterministic fake descriptors must use the fixed fake identity")
            return self

        if self.provider_name == "bge_m3":
            if self.model_id != BGE_M3_MODEL_ID:
                raise ValueError("BGE-M3 descriptors must use the fixed BGE-M3 model ID")
            if not _HF_COMMIT_RE.fullmatch(self.model_revision):
                raise ValueError("BGE-M3 descriptors require a lowercase 40-character HF commit revision")
            if self.model_revision != BGE_M3_MODEL_REVISION:
                raise ValueError("BGE-M3 descriptors must use the fixed BGE-M3 model revision")
            if self.provider_library != "FlagEmbedding":
                raise ValueError("BGE-M3 descriptors must identify FlagEmbedding")
            if self.provider_library_version.casefold() in _FORBIDDEN_PROVIDER_VERSIONS:
                raise ValueError("BGE-M3 descriptors require a resolved provider library version")
            if self.embedding_mode != "dense" or self.dimension != 1024 or self.normalized is not True:
                raise ValueError("BGE-M3 descriptors must describe normalized 1024-dimensional dense vectors")
        elif not self.model_revision.strip() or not _HF_COMMIT_RE.fullmatch(self.model_revision):
            raise ValueError("real embedding descriptors require an immutable model revision")
        if self.cache_namespace == "embedding:fake":
            raise ValueError("real embedding descriptors must use a namespace distinct from fake")
        return self


class FrozenCandidate(BaseModel):
    """One M1-source-backed candidate before any M2 ranking behavior occurs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract: str | None = None
    authors: list[str]
    year: int | None = Field(default=None, ge=1900, le=2100)
    doi: str | None = None
    url: str = Field(min_length=1)
    language: Literal["zh", "en"]
    categories: list[str]
    retrieval_paths: list[str] = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    member_source_identities: list[SourceIdentity] = Field(min_length=1)

    @field_validator("paper_id", "source", "source_id", "title", "doi", "url", "cluster_id")
    @classmethod
    def reject_blank_candidate_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Frozen candidate text fields must be non-blank when present")
        return value

    @field_validator("abstract")
    @classmethod
    def reject_placeholder_abstracts(cls, value: str | None) -> str | None:
        if value is not None and value.strip().casefold() in _PLACEHOLDER_ABSTRACTS:
            raise ValueError("Frozen candidates must not use placeholder abstracts")
        return value

    @field_validator("authors", "categories", "retrieval_paths")
    @classmethod
    def reject_blank_list_values(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("Frozen candidate list values must be non-blank")
        if len(values) != len(set(values)):
            raise ValueError("Frozen candidate list values must be unique")
        return values

    @model_validator(mode="after")
    def require_source_provenance(self) -> Self:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("Frozen candidates require an HTTP(S) URL")
        identities = [
            (identity.source, identity.source_id, identity.url)
            for identity in self.member_source_identities
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Frozen candidate member source identities must be unique")
        if any(not identity.url.startswith(("http://", "https://")) for identity in self.member_source_identities):
            raise ValueError("Frozen candidate source identities require HTTP(S) URLs")
        return self


class FrozenCandidateSnapshot(BaseModel):
    """A closed collection of unique, source-backed frozen candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_version: Literal["m2-candidates.v1"] = "m2-candidates.v1"
    question: str = Field(min_length=1)
    candidates: list[FrozenCandidate] = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Frozen candidate snapshot question must be non-blank")
        return value

    @model_validator(mode="after")
    def require_unique_candidate_identities(self) -> Self:
        paper_ids = [candidate.paper_id for candidate in self.candidates]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("Frozen candidate snapshot paper IDs must be unique")
        source_identities = [(candidate.source, candidate.source_id) for candidate in self.candidates]
        if len(source_identities) != len(set(source_identities)):
            raise ValueError("Frozen candidate snapshot source identities must be unique")
        return self


class EmbeddingInput(BaseModel):
    """One deterministic text input derived from a frozen paper or original query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_id: str = Field(min_length=1)
    input_kind: Literal["paper", "query"]
    paper_id: str | None = None
    query_id: str | None = None
    input_format_version: Literal["m2-title-abstract-v1"]
    text: str = Field(min_length=1)
    text_sha256: str = Field(min_length=64, max_length=64)
    source_snapshot_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("input_id", "paper_id", "query_id")
    @classmethod
    def reject_blank_input_identifiers(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Embedding input identifiers must be non-blank when present")
        return value

    @field_validator("text_sha256", "source_snapshot_sha256")
    @classmethod
    def require_lowercase_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("Embedding SHA-256 values must be lowercase 64-character hex")
        return value

    @model_validator(mode="after")
    def require_exact_identity_and_text_hash(self) -> Self:
        has_paper = self.paper_id is not None
        has_query = self.query_id is not None
        if has_paper == has_query:
            raise ValueError("Embedding inputs require exactly one paper_id or query_id")
        if (self.input_kind == "paper") != has_paper:
            raise ValueError("Embedding input kind must match its source identity")
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_sha256:
            raise ValueError("Embedding input text_sha256 must hash the UTF-8 text exactly")
        return self


class EmbeddingVectorRecord(BaseModel):
    """One validated finite dense vector paired with its full descriptor identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_id: str = Field(min_length=1)
    text_sha256: str = Field(min_length=64, max_length=64)
    descriptor: EmbeddingModelDescriptor
    dimension: int = Field(gt=0)
    vector: list[float] = Field(min_length=1)

    @field_validator("input_id")
    @classmethod
    def reject_blank_vector_input_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Embedding vector input_id must be non-blank")
        return value

    @field_validator("text_sha256")
    @classmethod
    def validate_vector_text_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("Embedding vector text_sha256 must be lowercase 64-character hex")
        return value

    @model_validator(mode="after")
    def require_matching_finite_vector(self) -> Self:
        if self.dimension != self.descriptor.dimension or len(self.vector) != self.dimension:
            raise ValueError("Embedding vector dimension must match descriptor and vector length")
        if not all(math.isfinite(value) for value in self.vector):
            raise ValueError("Embedding vectors must contain only finite values")
        return self


class EmbeddingCacheEntry(BaseModel):
    """The serializable content of one descriptor- and text-scoped cache entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cache_key: str = Field(min_length=64, max_length=64)
    text_sha256: str = Field(min_length=64, max_length=64)
    descriptor: EmbeddingModelDescriptor
    dimension: int = Field(gt=0)
    vector: list[float] = Field(min_length=1)

    @field_validator("cache_key", "text_sha256")
    @classmethod
    def require_cache_hashes(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("Embedding cache hashes must be lowercase 64-character hex")
        return value

    @model_validator(mode="after")
    def require_matching_finite_cached_vector(self) -> Self:
        if self.dimension != self.descriptor.dimension or len(self.vector) != self.dimension:
            raise ValueError("Embedding cache dimension must match descriptor and vector length")
        if not all(math.isfinite(value) for value in self.vector):
            raise ValueError("Embedding cache vectors must contain only finite values")
        return self


class EmbeddingRunStats(BaseModel):
    """Auditable counts for one local embedding cache-orchestration run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    cache_corrupt_count: int = Field(default=0, ge=0)
    provider_call_count: int = Field(default=0, ge=0)
    provider_input_count: int = Field(default=0, ge=0)
