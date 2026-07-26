"""Source-backed paper record contract."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PaperRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_id: str
    title: str = Field(min_length=1)
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1900, le=2100)
    doi: str | None = None
    url: str
    language: str = Field(pattern=r"^(zh|en)$")
    retrieval_paths: list[str] = Field(min_length=1)
    user_visible: bool = False

    @model_validator(mode="after")
    def validate_source_identity(self) -> "PaperRecord":
        if any(not path.strip() for path in self.retrieval_paths):
            raise ValueError("Retrieval paths must be non-blank")
        if self.user_visible and (not self.source_id.strip() or not self.url.startswith(("http://", "https://"))):
            raise ValueError("User-visible papers require source_id and HTTP(S) URL")
        return self
