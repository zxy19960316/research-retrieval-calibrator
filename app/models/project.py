"""Contracts for retrieval projects and their frozen research intent."""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import MethodConstraint, ProjectStage

_PROJECT_ID_PATTERN = r"^RRC-\d{4}-\d{4}$"
_ROUND_STAGES = {
    0: {ProjectStage.INIT, ProjectStage.CLARIFYING},
    1: {ProjectStage.ROUND1_SEARCHING, ProjectStage.WAITING_FOR_FEEDBACK},
    2: {
        ProjectStage.ROUND2_SEARCHING,
        ProjectStage.WAITING_FOR_DIRECTION_CONFIRMATION,
        ProjectStage.FINALIZED,
    },
}


def _validate_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Datetime must be UTC-aware")
    return value


class ResearchIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_terms: list[str] = Field(min_length=1)
    task_terms: list[str] = Field(min_length=1)
    method_terms: list[str] = Field(min_length=1)
    scope_terms: list[str] = Field(min_length=1)
    exclusions: list[str] = Field(default_factory=list)
    method_constraint: MethodConstraint
    accepted_paper_roles: set[str] = Field(min_length=1)
    revision: int = Field(ge=1)
    frozen_at: datetime

    @field_validator("frozen_at")
    @classmethod
    def validate_frozen_at(cls, value: datetime) -> datetime:
        return _validate_utc(value)

    @model_validator(mode="after")
    def validate_non_blank_terms(self) -> "ResearchIntent":
        collections = (
            self.object_terms,
            self.task_terms,
            self.method_terms,
            self.scope_terms,
            self.exclusions,
        )
        if any(any(not term.strip() for term in terms) for terms in collections):
            raise ValueError("Research intent terms must be non-blank")
        if any(not role.strip() for role in self.accepted_paper_roles):
            raise ValueError("Accepted paper roles must be non-blank")
        return self


class RetrievalProject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(pattern=_PROJECT_ID_PATTERN)
    stage: ProjectStage
    round_number: int = Field(ge=0, le=2)
    original_input: str = Field(min_length=1)
    current_intent: ResearchIntent | None = None
    round1_selection_ids: list[str] | None = None
    final_selection_ids: list[str] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timestamps(cls, value: datetime) -> datetime:
        return _validate_utc(value)

    @model_validator(mode="after")
    def validate_project_invariants(self) -> "RetrievalProject":
        if self.stage not in _ROUND_STAGES[self.round_number]:
            raise ValueError("Project stage must match round number")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.round1_selection_ids is not None:
            if not 15 <= len(self.round1_selection_ids) <= 20:
                raise ValueError("Round-one selection must contain 15 to 20 papers")
            if any(not paper_id.strip() for paper_id in self.round1_selection_ids):
                raise ValueError("Round-one selection IDs must be non-blank")
        if self.final_selection_ids is not None:
            if len(self.final_selection_ids) > 10:
                raise ValueError("Final selection cannot exceed 10 papers")
            if any(not paper_id.strip() for paper_id in self.final_selection_ids):
                raise ValueError("Final selection IDs must be non-blank")
        if self.stage is ProjectStage.WAITING_FOR_FEEDBACK and self.round1_selection_ids is None:
            raise ValueError("Waiting for feedback requires a round-one selection")
        if self.stage is ProjectStage.FINALIZED and self.final_selection_ids is None:
            raise ValueError("Finalized projects require a final selection")
        return self
