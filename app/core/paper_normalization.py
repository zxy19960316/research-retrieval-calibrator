"""Pure, offline normalization of source-backed paper metadata."""

from __future__ import annotations

import re
import unicodedata

from app.models.dedup import NormalizedPaper
from app.models.paper import PaperRecord

_DOI_PREFIXES = (
    "https://dx.doi.org/",
    "https://doi.org/",
    "http://doi.org/",
    "doi:",
)
_DOI_PATTERN = re.compile(r"10\.\d{4,9}/\S+")
_MODERN_ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}(?:v[1-9]\d*)?$")
_LEGACY_ARXIV_ID = re.compile(r"[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v[1-9]\d*)?$")
_ARXIV_VERSION = re.compile(r"v\d+$")
_TRAILING_CITATION_PUNCTUATION = ".,;:)]}>\"'"


def canonicalize_doi(value: str | None) -> str | None:
    """Canonicalize an explicitly supplied DOI without inferring one from other fields."""

    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    for prefix in _DOI_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
            break
    normalized = normalized.rstrip(_TRAILING_CITATION_PUNCTUATION)
    if not normalized or not _DOI_PATTERN.fullmatch(normalized):
        return None
    return normalized


def canonicalize_arxiv_id(record: PaperRecord) -> str | None:
    """Return only a source-confirmed modern or canonical legacy arXiv identity."""

    if record.source != "arxiv":
        return None
    source_id = unicodedata.normalize("NFKC", record.source_id).strip()
    if not (_MODERN_ARXIV_ID.fullmatch(source_id) or _LEGACY_ARXIV_ID.fullmatch(source_id)):
        return None
    return _ARXIV_VERSION.sub("", source_id)


def normalize_title(value: str) -> tuple[str, tuple[str, ...]]:
    """Normalize typography into deterministic token boundaries without translating or stemming."""

    normalized = _normalize_meaningful_characters(value)
    tokens = tuple(normalized.split())
    return " ".join(tokens), tokens


def normalize_author(value: str) -> str:
    """Normalize one author string without changing token order or guessing name parts."""

    return _normalize_meaningful_characters(value)


def normalize_paper(record: PaperRecord) -> NormalizedPaper:
    """Build derived comparison fields while retaining the untouched original record."""

    normalized_title, title_tokens = normalize_title(record.title)
    return NormalizedPaper(
        record=record,
        canonical_doi=canonicalize_doi(record.doi),
        canonical_arxiv_id=canonicalize_arxiv_id(record),
        normalized_title=normalized_title,
        title_tokens=title_tokens,
        normalized_authors=tuple(
            normalized
            for author in record.authors
            if (normalized := normalize_author(author))
        ),
    )


def _normalize_meaningful_characters(value: str) -> str:
    canonical = unicodedata.normalize("NFKC", value).strip().casefold()
    token_boundaries = "".join(
        character if character.isalnum() else " " for character in canonical
    )
    return " ".join(token_boundaries.split())
