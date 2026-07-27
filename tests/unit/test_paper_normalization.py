from __future__ import annotations

import copy

import pytest

from app.core.paper_normalization import (
    canonicalize_arxiv_id,
    canonicalize_doi,
    normalize_author,
    normalize_paper,
    normalize_title,
)
from app.models.paper import PaperRecord


def _paper(**overrides: object) -> PaperRecord:
    values: dict[str, object] = {
        "paper_id": "paper-1",
        "source": "crossref",
        "source_id": "source-1",
        "title": "Graph-Based Retrieval: A Study",
        "abstract": "A source-backed abstract.",
        "authors": ["Ada Author"],
        "year": 2024,
        "doi": None,
        "url": "https://example.test/paper-1",
        "language": "en",
        "retrieval_paths": ["Q1"],
    }
    values.update(overrides)
    return PaperRecord.model_validate(values)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.1000/XYZ", "10.1000/xyz"),
        ("doi:10.1000/xyz", "10.1000/xyz"),
        ("https://doi.org/10.1000/XYZ", "10.1000/xyz"),
        ("http://doi.org/10.1000/XYZ.", "10.1000/xyz"),
        ("https://dx.doi.org/10.1000/XYZ)", "10.1000/xyz"),
        ("not a doi", None),
        ("", None),
        (None, None),
    ],
)
def test_canonicalize_doi_never_guesses(raw: str | None, expected: str | None) -> None:
    assert canonicalize_doi(raw) == expected


def test_title_and_author_normalization_are_deterministic_and_conservative() -> None:
    left_title, left_tokens = normalize_title("Graph-Based Retrieval: A Study")
    right_title, right_tokens = normalize_title(" graph based retrieval — a study ")

    assert left_title == right_title == "graph based retrieval a study"
    assert left_tokens == right_tokens == ("graph", "based", "retrieval", "a", "study")
    assert normalize_author("Ａda,  Author") == "ada author"
    assert normalize_author("A. Author") != normalize_author("Ada Author")
    assert normalize_title("中文 检索：研究")[0] == "中文 检索 研究"


def test_arxiv_identity_requires_confirmed_source_and_removes_version_only() -> None:
    assert canonicalize_arxiv_id(
        _paper(source="arxiv", source_id="2401.00001v2")
    ) == "2401.00001"
    assert canonicalize_arxiv_id(
        _paper(source="arxiv", source_id="math.GT/0309136v2")
    ) == "math.GT/0309136"
    assert canonicalize_arxiv_id(
        _paper(source="arxiv", source_id="math.GT/0309136v0")
    ) is None
    assert canonicalize_arxiv_id(
        _paper(source="crossref", source_id="2401.00001v2")
    ) is None


def test_normalized_paper_retains_original_record_without_mutation() -> None:
    record = _paper(doi="doi:10.1000/XYZ")
    before = copy.deepcopy(record.model_dump())

    normalized = normalize_paper(record)

    assert normalized.record is record
    assert normalized.canonical_doi == "10.1000/xyz"
    assert normalized.canonical_arxiv_id is None
    assert normalized.normalized_title == "graph based retrieval a study"
    assert normalized.title_tokens == ("graph", "based", "retrieval", "a", "study")
    assert normalized.normalized_authors == ("ada author",)
    assert record.model_dump() == before
