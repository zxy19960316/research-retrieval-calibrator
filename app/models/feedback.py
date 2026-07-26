"""User feedback contract with relevance-specific aspect rules."""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import FeedbackAspect, Relevance


class FeedbackRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(pattern=r"^RRC-\d{4}-\d{4}$")
    paper_id: str = Field(min_length=1)
    relevance: Relevance
    aspects: list[FeedbackAspect] = Field(default_factory=list)
    raw_text: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("Datetime must be UTC-aware")
        return value

    @model_validator(mode="after")
    def validate_aspects(self) -> "FeedbackRecord":
        if self.relevance is Relevance.PARTIAL and not self.aspects:
            raise ValueError("PARTIAL feedback requires at least one aspect")
        if self.relevance is not Relevance.PARTIAL and self.aspects:
            raise ValueError("Only PARTIAL feedback accepts aspects")
        return self
