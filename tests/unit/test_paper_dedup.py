from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest

from app.adapters.arxiv import ArxivAdapter, ArxivAdapterConfig, ArxivResponse
from app.core.paper_dedup import deduplicate_papers
from app.models.dedup import DedupReason
from app.models.enums import QueryBranch, QueryBreadth
from app.models.paper import PaperRecord
from app.models.query import Query

FIXTURES = Path(__file__).parents[1] / "fixtures" / "dedup"


def _paper(paper_id: str, **overrides: object) -> PaperRecord:
    values: dict[str, object] = {
        "paper_id": paper_id,
        "source": "crossref",
        "source_id": f"source-{paper_id}",
        "title": "Graph-Based Retrieval: A Study",
        "abstract": "A source-backed abstract.",
        "authors": ["Ada Author"],
        "year": 2024,
        "doi": None,
        "url": f"https://example.test/{paper_id}",
        "language": "en",
        "retrieval_paths": ["Q1"],
    }
    values.update(overrides)
    return PaperRecord.model_validate(values)


def _fixture(name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")))


def _summary(records: list[PaperRecord]) -> dict[str, int]:
    result = deduplicate_papers(records)
    actions = [decision.action for decision in result.decisions]
    return {
        "cluster_count": len(result.clusters),
        "auto_merge_count": actions.count("auto_merge"),
        "manual_review_count": actions.count("manual_review"),
        "keep_separate_count": actions.count("keep_separate"),
    }


class _RecordedTransport:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def get(self, *_: object, **__: object) -> ArxivResponse:
        return ArxivResponse(200, self._body, {})


def _recorded_query(query_id: str) -> Query:
    return Query(
        query_id=query_id,
        round_number=1,
        branch=QueryBranch.DIRECT_INTERSECTION,
        breadth=QueryBreadth.MEDIUM,
        language="en",
        query_text='all:"radiation shielding"',
        weight=1.0,
    )


def _recorded_adapter() -> ArxivAdapter:
    return ArxivAdapter(
        ArxivAdapterConfig(
            user_agent="rrc-m1-t03r1-tests/1.0",
            timeout_seconds=1.0,
            page_size=1,
            min_request_interval_seconds=0.0,
            max_attempts=1,
            max_total_results=1,
            max_total_attempts=1,
            initial_backoff_seconds=0.0,
        ),
        transport=_RecordedTransport((Path(__file__).parents[1] / "fixtures" / "arxiv" / "normal.xml").read_bytes()),
    )


def test_exact_doi_merges_and_preserves_paths_sources_and_original_records() -> None:
    records = [
        _paper(
            "b",
            source="arxiv",
            source_id="2401.00001",
            doi="doi:10.1000/X",
            retrieval_paths=["Q2"],
        ),
        _paper(
            "a",
            source="crossref",
            source_id="doi:10.1000/x",
            doi="10.1000/x",
            retrieval_paths=["Q1", "Q2"],
        ),
    ]
    before = copy.deepcopy([record.model_dump() for record in records])

    result = deduplicate_papers(records)

    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert cluster.cluster_id == "doi:10.1000/x"
    assert cluster.canonical_record.paper_id == "a"
    assert [record.paper_id for record in cluster.member_records] == ["a", "b"]
    assert cluster.retrieval_paths == ["Q1", "Q2"]
    assert {(identity.source, identity.source_id) for identity in cluster.source_identities} == {
        ("arxiv", "2401.00001"),
        ("crossref", "doi:10.1000/x"),
    }
    assert [decision.reason for decision in cluster.merge_reasons] == [DedupReason.EXACT_DOI]
    assert [record.model_dump() for record in records] == before


def test_exact_arxiv_and_title_author_and_high_similarity_merges() -> None:
    arxiv_result = deduplicate_papers(
        [
            _paper("arxiv-a", source="arxiv", source_id="2401.00001v2", retrieval_paths=["Q1"]),
            _paper("arxiv-b", source="arxiv", source_id="2401.00001", retrieval_paths=["Q2"]),
        ]
    )
    title_result = deduplicate_papers(
        [
            _paper("title-a", title="Graph-Based Retrieval: A Study"),
            _paper("title-b", title="graph based retrieval a study", year=2025),
        ]
    )
    similar_result = deduplicate_papers(
        [
            _paper("similar-a", title="Deterministic Graph Based Retrieval for Scientific Literature Discovery and Evaluation Benchmark Studies", authors=["Ada Author", "Ben Author"]),
            _paper("similar-b", title="Deterministic Graph Based Retrieval for Scientific Literature Discovery and Evaluation Benchmark Studies Robust", authors=["Ada Author", "Ben Author", "Cara Author"]),
        ]
    )

    assert arxiv_result.decisions[0].reason is DedupReason.EXACT_ARXIV_ID
    assert title_result.decisions[0].reason is DedupReason.EXACT_NORMALIZED_TITLE_WITH_AUTHOR
    assert similar_result.decisions[0].reason is DedupReason.HIGH_TITLE_SIMILARITY_WITH_AUTHOR


def test_insufficient_author_evidence_and_negative_cases_never_auto_merge() -> None:
    same_title = deduplicate_papers(
        [_paper("a", authors=["Ada Author"]), _paper("b", authors=["Ben Author"])]
    )
    initials = deduplicate_papers(
        [_paper("c", authors=["A. Author"]), _paper("d", authors=["Ada Author"])]
    )

    assert same_title.decisions[0].action != "auto_merge"
    assert initials.decisions[0].action != "auto_merge"


def test_possible_cross_language_and_identity_conflict_require_manual_review() -> None:
    cross_language = deduplicate_papers(
        [
            _paper("zh", title="图检索研究", language="zh", authors=["Ada Author"], year=2024),
            _paper("en", title="Graph retrieval study", language="en", authors=["Ada Author"], year=2024),
        ]
    )
    identity_conflict = deduplicate_papers(
        [
            _paper("id-a", source="arxiv", source_id="2401.00001", doi="10.1000/x"),
            _paper("id-b", source="arxiv", source_id="2401.00002", doi="10.1000/x"),
        ]
    )

    assert cross_language.decisions[0].action == "manual_review"
    assert cross_language.decisions[0].reason is DedupReason.CROSS_LANGUAGE_POSSIBLE_DUPLICATE
    assert identity_conflict.decisions[0].action == "manual_review"
    assert identity_conflict.decisions[0].reason is DedupReason.IDENTITY_CONFLICT


def test_duplicate_arxiv_paper_id_coalesces_query_observations_without_mutation() -> None:
    records = [
        _paper(
            "arxiv:2401.00001",
            source="arxiv",
            source_id="2401.00001",
            title="Recorded Paper",
            authors=["Ada Author"],
            doi=None,
            url="https://arxiv.org/abs/2401.00001",
            retrieval_paths=["Q-direct"],
        ),
        _paper(
            "arxiv:2401.00001",
            source="arxiv",
            source_id="2401.00001",
            title="Recorded Paper",
            authors=["Ada Author"],
            doi=None,
            url="https://arxiv.org/abs/2401.00001",
            retrieval_paths=["Q-method"],
        ),
    ]
    before = copy.deepcopy([record.model_dump() for record in records])

    forward = deduplicate_papers(records)
    reverse = deduplicate_papers(list(reversed(records)))
    repeated = deduplicate_papers(records)

    assert forward == reverse == repeated
    assert len(forward.clusters) == 1
    assert forward.clusters[0].retrieval_paths == ["Q-direct", "Q-method"]
    assert [record.model_dump() for record in records] == before


def test_recorded_arxiv_adapter_results_coalesce_across_two_queries() -> None:
    direct = _recorded_adapter().search(_recorded_query("Q-direct"), max_results=1)
    method = _recorded_adapter().search(_recorded_query("Q-method"), max_results=1)
    before = copy.deepcopy([record.model_dump() for record in [*direct, *method]])

    result = deduplicate_papers([*direct, *method])

    assert [record.paper_id for record in direct] == [record.paper_id for record in method]
    assert len(result.clusters) == 1
    assert result.clusters[0].retrieval_paths == ["Q-direct", "Q-method"]
    assert [record.model_dump() for record in [*direct, *method]] == before


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_id", "2401.00002"),
        ("doi", "10.1000/conflicting-doi"),
        ("title", "A conflicting title"),
        ("url", "https://arxiv.org/abs/2401.00001?conflict=1"),
    ],
)
def test_duplicate_paper_id_with_conflicting_metadata_fails_stably(
    field: str, replacement: object
) -> None:
    first = _paper(
        "arxiv:2401.00001",
        source="arxiv",
        source_id="2401.00001",
        doi="10.1000/original-doi",
        url="https://arxiv.org/abs/2401.00001",
    )
    second = first.model_copy(update={field: replacement}, deep=True)

    with pytest.raises(ValueError, match="DUPLICATE_PAPER_ID_CONFLICT") as raised:
        deduplicate_papers([first, second])

    assert getattr(raised.value, "code", None) == "DUPLICATE_PAPER_ID_CONFLICT"


def test_negative_fixture_duplicate_paper_id_conflicts_fail_stably() -> None:
    fixture = _fixture("negative")
    base = _paper(
        "arxiv:2401.00001",
        source="arxiv",
        source_id="2401.00001",
        doi="10.1000/original-doi",
        url="https://arxiv.org/abs/2401.00001",
    )

    for case in fixture["duplicate_paper_id_conflict_cases"]:
        conflicting = base.model_copy(update={case["field"]: case["replacement"]}, deep=True)
        with pytest.raises(ValueError, match="DUPLICATE_PAPER_ID_CONFLICT"):
            deduplicate_papers([base, conflicting])


def test_identity_conflict_uses_unique_stable_fallback_cluster_ids() -> None:
    records = [
        _paper("id-a", source="arxiv", source_id="2401.00001", doi="10.1000/x"),
        _paper("id-b", source="arxiv", source_id="2401.00002", doi="10.1000/x"),
    ]

    forward = deduplicate_papers(records)
    reverse = deduplicate_papers(list(reversed(records)))

    assert forward == reverse
    assert [(decision.action, decision.reason) for decision in forward.decisions] == [
        ("manual_review", DedupReason.IDENTITY_CONFLICT)
    ]
    assert [cluster.cluster_id for cluster in forward.clusters] == ["paper:id-a", "paper:id-b"]


def test_similarity_bridge_does_not_transitively_merge_three_records() -> None:
    titles = [
        (
            "robust deterministic graph retrieval for comprehensive scientific literature discovery "
            "evaluation benchmark methodology applications workflows protocols datasets validation "
            "reproducibility transparency governance systems architecture analysis implementation "
            "traceability auditability"
        ),
        (
            "robust deterministic graph retrieval for comprehensive scientific literature discovery "
            "evaluation benchmark methodology applications workflows protocols datasets validation "
            "reproducibility transparency governance systems architecture analysis implementation "
            "traceability assurance"
        ),
        (
            "robust deterministic graph retrieval for comprehensive scientific literature discovery "
            "evaluation benchmark methodology applications workflows protocols datasets validation "
            "reproducibility transparency governance systems architecture analysis interpretability "
            "traceability assurance"
        ),
    ]
    records = [_paper(paper_id, title=title) for paper_id, title in zip(["a", "b", "c"], titles)]

    forward = deduplicate_papers(records)
    reverse = deduplicate_papers(list(reversed(records)))
    decisions = {(decision.left_paper_id, decision.right_paper_id): decision for decision in forward.decisions}

    assert forward == reverse
    assert [[record.paper_id for record in cluster.member_records] for cluster in forward.clusters] == [
        ["a", "b"],
        ["c"],
    ]
    assert decisions[("b", "c")].action == "manual_review"
    assert decisions[("b", "c")].reason is DedupReason.TRANSITIVE_BRIDGE_RISK


def test_author_bridge_does_not_transitively_merge_three_records() -> None:
    records = [
        _paper("a", authors=["Alice"], title="Shared title"),
        _paper("b", authors=["Alice", "Bob"], title="Shared title"),
        _paper("c", authors=["Bob"], title="Shared title"),
    ]

    forward = deduplicate_papers(records)
    reverse = deduplicate_papers(list(reversed(records)))
    decisions = {(decision.left_paper_id, decision.right_paper_id): decision for decision in forward.decisions}

    assert forward == reverse
    assert [[record.paper_id for record in cluster.member_records] for cluster in forward.clusters] == [
        ["a", "b"],
        ["c"],
    ]
    assert decisions[("b", "c")].action == "manual_review"
    assert decisions[("b", "c")].reason is DedupReason.TRANSITIVE_BRIDGE_RISK


def test_order_invariance_cluster_identity_and_idempotence() -> None:
    records = [
        _paper("b", doi="10.1000/x", retrieval_paths=["Q2"]),
        _paper("a", doi="doi:10.1000/X", retrieval_paths=["Q1"]),
        _paper("single", title="Independent paper", authors=["Ben Author"]),
    ]

    forward = deduplicate_papers(records)
    reversed_result = deduplicate_papers(list(reversed(records)))
    repeated = deduplicate_papers(records)

    assert forward == reversed_result == repeated


@pytest.mark.parametrize("fixture_name", ["positive", "negative", "manual_review"])
def test_frozen_fixtures_are_order_invariant_and_preserve_inputs(fixture_name: str) -> None:
    fixture = _fixture(fixture_name)
    payloads = fixture["records"]
    assert isinstance(payloads, list)
    before = copy.deepcopy(payloads)
    records = [PaperRecord.model_validate(payload) for payload in payloads]

    forward = deduplicate_papers(records)
    reverse = deduplicate_papers(list(reversed(records)))

    assert forward == reverse
    assert payloads == before


def test_positive_fixture_matches_frozen_summary() -> None:
    fixture = _fixture("positive")
    records = [PaperRecord.model_validate(payload) for payload in fixture["records"]]

    assert _summary(records) == fixture["expected"]


def test_negative_fixture_has_zero_false_auto_merges_for_protected_pairs() -> None:
    fixture = _fixture("negative")
    records = [PaperRecord.model_validate(payload) for payload in fixture["records"]]
    protected_pairs = {tuple(sorted(pair)) for pair in fixture["protected_pairs"]}

    result = deduplicate_papers(records)
    false_auto_merges = [
        decision
        for decision in result.decisions
        if tuple(sorted((decision.left_paper_id, decision.right_paper_id))) in protected_pairs
        and decision.action == "auto_merge"
    ]

    assert false_auto_merges == []
    assert fixture["expected"]["false_auto_merge_count"] == 0


def test_manual_review_fixture_matches_each_protected_decision() -> None:
    fixture = _fixture("manual_review")
    records = [PaperRecord.model_validate(payload) for payload in fixture["records"]]
    decisions = {
        tuple(sorted((decision.left_paper_id, decision.right_paper_id))): decision
        for decision in deduplicate_papers(records).decisions
    }

    for expected in fixture["expected_manual_reviews"]:
        decision = decisions[tuple(sorted(expected["pair"]))]
        assert decision.action == "manual_review"
        assert decision.reason.value == expected["reason"]
