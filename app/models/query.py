"""Query and query-revision provenance contracts."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.enums import QueryBranch, QueryBreadth


class QueryRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_paper_id: str | None = None
    rule: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1)
    round_number: int = Field(ge=1, le=2)
    branch: QueryBranch
    breadth: QueryBreadth
    language: str = Field(pattern=r"^(zh|en)$")
    query_text: str = Field(min_length=1)
    weight: float = Field(ge=0, le=1)
    revision_sources: list[QueryRevision] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_round_provenance(self) -> "Query":
        if self.round_number == 1 and self.revision_sources:
            raise ValueError("Round-one queries cannot consume feedback revisions")
        if self.round_number == 2 and not self.revision_sources:
            raise ValueError("Round-two queries require revision provenance")
        return self
