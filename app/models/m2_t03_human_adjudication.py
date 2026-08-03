"""Closed contracts for the first, human-pending M2-T03 review bundle."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import EvidenceSlot, SupportLevel
from app.models.evidence_classification import (
    EvidenceClassificationState,
    EvidenceClassifierDescriptor,
)
from app.models.project import ResearchIntent

M2_T03_HUMAN_ADJUDICATION_VERSION = "m2-t03-human-adjudication.v1"
M2_T03_HUMAN_ADJUDICATION_RECEIPT_VERSION = "m2-t03-human-adjudication-receipt.v1"
M2_T03_HUMAN_ADJUDICATION_REPORT_VERSION = "m2-t03-human-adjudication-report.v1"


class SourceBinding(BaseModel):
    """One source identity copied from the protected candidate snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    url: str = Field(min_length=1)


class ArtifactBinding(BaseModel):
    """A relative repository path bound to its raw UTF-8 bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HumanAdjudicationFields(BaseModel):
    """The empty human fields carried by every newly generated review item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["CONFIRM", "REVISE", "REJECT"] | None = None
    evidence_slot: EvidenceSlot | None = None
    support_level: SupportLevel | None = None
    grounded_reason: str | None = Field(default=None, max_length=1200)
    supporting_excerpt: str | None = Field(default=None, max_length=1000)
    reviewer_id: str | None = Field(default=None, max_length=200)
    reviewed_at_utc: datetime | None = None
    notes: str | None = Field(default=None, max_length=2000)


class ResearchIntentContext(BaseModel):
    """The authoritative intent and the repaired M1 bytes that supplied it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_artifact: ArtifactBinding
    repair_manifest: ArtifactBinding
    canonical_intent_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    research_intent: ResearchIntent


class CandidateContext(BaseModel):
    """Title/abstract/source identity context shown to a human reviewer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract: str = Field(min_length=1)
    source_identity: SourceBinding
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MachineAdvisoryClassification(BaseModel):
    """A deterministic label copied for orientation, never a human decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    advisory_only: Literal[True]
    evidence_type: Literal["deterministic_fake"]
    paper_id: str = Field(min_length=1)
    evidence_slot: EvidenceSlot | None = None
    support_level: SupportLevel | None = None
    reason: str = Field(min_length=1, max_length=1200)
    supporting_excerpt: str | None = Field(default=None, max_length=1000)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    classifier_descriptor: EvidenceClassifierDescriptor
    classification_version: Literal["m2-t03-evidence-classification.v1"]
    state: EvidenceClassificationState


class HumanReviewItem(BaseModel):
    """One review item with immutable context and empty adjudication fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context: CandidateContext
    machine_advisory: MachineAdvisoryClassification
    human_adjudication: HumanAdjudicationFields


class ReviewPolicy(BaseModel):
    """Explicit first-phase review boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    machine_labels_are_advisory: Literal[True]
    human_fields_initially_empty: Literal[True]
    human_review_status: Literal["pending"]
    source_fields: tuple[Literal["paper_id", "title", "abstract", "source_identity"], ...]
    disallowed_sources: tuple[str, ...]
    network_forbidden: Literal[True]
    real_model_run: Literal[False]
    real_arxiv_requests: Literal[False]
    m2_t04_started: Literal[False]


class StatusBoundary(BaseModel):
    """The project gate that this bundle is not allowed to advance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    m2_progress: Literal["IN_PROGRESS 2/5"]
    m3_progress: Literal["BLOCKED_BY_M2 0/5"]
    scoring_eligible: Literal[False]
    human_review: Literal["pending"]
    m2_t04: Literal["not_started"]


class HumanAdjudicationBundle(BaseModel):
    """Complete M2-T03 first-phase context/review bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_version: Literal["m2-t03-human-adjudication.v1"]
    phase: Literal["M2"]
    task_id: Literal["M2-T03"]
    generated_from_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    generated_at_utc: datetime
    candidate_count: Literal[33]
    paper_id_order: tuple[str, ...] = Field(min_length=33, max_length=33)
    candidate_snapshot: ArtifactBinding
    classification_result: ArtifactBinding
    intent_context: ResearchIntentContext
    review_policy: ReviewPolicy
    status_boundary: StatusBoundary
    items: tuple[HumanReviewItem, ...] = Field(min_length=33, max_length=33)


class HumanAdjudicationReceipt(BaseModel):
    """Receipt for the offline bundle generation operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_version: Literal["m2-t03-human-adjudication-receipt.v1"]
    phase: Literal["M2"]
    task_id: Literal["M2-T03"]
    generated_from_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    bundle_path: str = Field(min_length=1)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_path: str = Field(min_length=1)
    runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_count: Literal[33]
    human_fields_empty: Literal[True]
    network_requests: Literal[0]
    real_model_run: Literal[False]
    real_arxiv_requests: Literal[False]
    human_review: Literal["not_started"]
    m2_t04: Literal["not_started"]
    scoring_eligible: Literal[False]
    exit_code: Literal[0]


class HumanAdjudicationReport(BaseModel):
    """Summary evidence for the human-pending first phase."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_version: Literal["m2-t03-human-adjudication-report.v1"]
    phase: Literal["M2"]
    task_id: Literal["M2-T03"]
    generated_from_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    generated_at_utc: datetime
    input_artifacts: tuple[ArtifactBinding, ...] = Field(min_length=1)
    artifact: tuple[ArtifactBinding, ArtifactBinding]
    candidate_count: Literal[33]
    machine_advisory_count: Literal[33]
    human_fields_nonempty_count: Literal[0]
    human_review: Literal["pending"]
    real_model_run: Literal["not_run"]
    real_arxiv_requests: Literal["not_run"]
    m2_t04: Literal["not_started"]
    scoring_eligible: Literal[False]
    status_after_bundle: Literal["M2 IN_PROGRESS 2/5; M3 BLOCKED_BY_M2 0/5"]
    commands: tuple[str, ...] = Field(min_length=1)
    exit_codes: dict[str, Literal[0]]
