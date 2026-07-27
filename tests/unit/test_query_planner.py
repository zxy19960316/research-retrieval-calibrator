from datetime import UTC, datetime

import pytest

from app.core.intent import freeze_research_intent
from app.core.query_planner import build_query_plan
from app.models.enums import MethodConstraint, QueryBranch, QueryBreadth
from app.models.planning import (
    IntentDraft,
    PlanningError,
    QueryExpansion,
    RetrievalIntentField,
    TermEvidence,
    TermSource,
)


def _frozen_intent(*, exclusions: list[str] | None = None, chinese: bool = False):
    if chinese:
        values = {
            "object_terms": ["辐射屏蔽"],
            "task_terms": ["设计"],
            "method_terms": ["迁移学习"],
            "scope_terms": ["核工程"],
            "source_language": "zh",
        }
    else:
        values = {
            "object_terms": ["radiation shielding"],
            "task_terms": ["design"],
            "method_terms": ["transfer learning"],
            "scope_terms": ["nuclear engineering"],
            "source_language": "en",
        }
    evidence = {
        field: [TermEvidence(term=term, source=TermSource.ORIGINAL_INPUT)]
        for field, terms in values.items()
        if field.endswith("_terms")
        for term in terms
    }
    field_evidence = {
        "object": evidence["object_terms"],
        "task": evidence["task_terms"],
        "method": evidence["method_terms"],
        "scope": evidence["scope_terms"],
        "accepted_paper_roles": [TermEvidence(term="method", source=TermSource.DETERMINISTIC_RULE)],
    }
    if exclusions:
        field_evidence["exclusions"] = [
            TermEvidence(term=term, source=TermSource.ORIGINAL_INPUT) for term in exclusions
        ]
    draft = IntentDraft(
        original_input="test intent",
        **values,
        exclusions=exclusions or [],
        method_constraint=MethodConstraint.PREFERRED,
        accepted_paper_roles={"method"},
        revision=1,
        field_evidence=field_evidence,
    )
    return freeze_research_intent(draft, datetime(2026, 1, 1, tzinfo=UTC))


def test_same_intent_has_deterministic_twelve_query_plan() -> None:
    intent = _frozen_intent()
    first = build_query_plan("RRC-2026-0001", intent)
    second = build_query_plan("RRC-2026-0001", intent)

    assert first.plan_id == second.plan_id
    assert first.queries == second.queries
    assert len(first.queries) == 12
    assert {query.branch for query in first.queries} == set(QueryBranch)
    for branch in QueryBranch:
        assert {query.breadth for query in first.queries if query.branch is branch} == set(QueryBreadth)
    assert len({query.query_id for query in first.queries}) == 12
    assert all(query.language == "en" and not query.revision_sources for query in first.queries)
    assert all(query.weight > 0 for query in first.queries)
    assert sum(query.weight for query in first.queries) == pytest.approx(1.0)


def test_branch_templates_have_required_constraints() -> None:
    plan = build_query_plan("RRC-2026-0001", _frozen_intent())
    direct = next(query for query in plan.queries if query.branch is QueryBranch.DIRECT_INTERSECTION and query.breadth is QueryBreadth.NARROW)
    problem = next(query for query in plan.queries if query.branch is QueryBranch.PROBLEM_DOMAIN and query.breadth is QueryBreadth.NARROW)
    method = next(query for query in plan.queries if query.branch is QueryBranch.METHOD_DOMAIN and query.breadth is QueryBreadth.NARROW)
    bridge = next(query for query in plan.queries if query.branch is QueryBranch.BRIDGE_DOMAIN and query.breadth is QueryBreadth.NARROW)

    assert "transfer learning" in direct.query_text
    assert "transfer learning" not in problem.query_text
    assert "radiation shielding" not in method.query_text
    assert any(word in bridge.query_text for word in ("application", "adaptation", "transfer", "framework", "methodology", "benchmark"))


def test_chinese_intent_produces_english_arxiv_queries() -> None:
    plan = build_query_plan("RRC-2026-0001", _frozen_intent(chinese=True))

    assert all(query.query_text.isascii() for query in plan.queries)
    assert any("radiation shielding" in query.query_text for query in plan.queries)


def test_exclusion_uses_exact_normalized_match_not_substring() -> None:
    intent = _frozen_intent(exclusions=[" linear "])
    plan = build_query_plan(
        "RRC-2026-0001",
        intent,
        positive_expansions=[
            QueryExpansion(
                target_field="method",
                term_en="linear",
                source=TermSource.LLM_FAKE,
            ),
            QueryExpansion(
                target_field="method",
                term_en="nonlinear",
                source=TermSource.LLM_FAKE,
            ),
        ],
    )

    assert [item.term_en for item in plan.positive_expansions] == ["nonlinear"]
    assert plan.excluded_term_conflicts[0].term == "linear"
    assert all('all:"linear"' not in query.query_text for query in plan.queries)
    assert any('all:"nonlinear"' in query.query_text for query in plan.queries)


def test_unmapped_required_chinese_term_refuses_query_plan() -> None:
    intent = _frozen_intent().model_copy(update={"object_terms": ["\u672a\u77e5\u672f\u8bed"]})

    with pytest.raises(PlanningError, match="UNMAPPED_RETRIEVAL_TERM"):
        build_query_plan("RRC-2026-0001", intent)


def test_canonical_arxiv_queries_have_allowed_prefixes_operators_and_quoting() -> None:
    plan = build_query_plan("RRC-2026-0001", _frozen_intent())

    assert len(plan.queries) == 12
    for query in plan.queries:
        assert 'all:"' in query.query_text
        assert "%" not in query.query_text and "+" not in query.query_text
        assert '"research"' not in query.query_text
        assert all(operator in {"AND", "OR", "ANDNOT"} for operator in query.query_text.replace("(", " ").replace(")", " ").split() if operator in {"AND", "OR", "ANDNOT"})


def test_allowed_expansion_enters_target_medium_and_wide_queries_only() -> None:
    plan = build_query_plan(
        "RRC-2026-0001",
        _frozen_intent(),
        positive_expansions=[
            QueryExpansion(
                target_field="object",
                term_en="radiation barrier",
                source=TermSource.LLM_FAKE,
            )
        ],
    )

    matching = [query for query in plan.queries if 'all:"radiation barrier"' in query.query_text]
    assert matching
    assert {query.breadth for query in matching} == {QueryBreadth.MEDIUM, QueryBreadth.WIDE}


def test_clock_injection_is_reproducible_without_fixed_default_time() -> None:
    intent = _frozen_intent()
    fixed = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
    first = build_query_plan("RRC-2026-0001", intent, generated_at_utc=fixed)
    second = build_query_plan("RRC-2026-0001", intent, generated_at_utc=fixed)
    default = build_query_plan("RRC-2026-0001", intent)

    assert first.generated_at_utc == second.generated_at_utc == fixed
    assert first.plan_id == second.plan_id == default.plan_id
    assert default.generated_at_utc.year >= 2026


def test_exclusions_close_fixed_synonym_and_bridge_paths_with_provenance() -> None:
    synonym_plan = build_query_plan(
        "RRC-2026-0001", _frozen_intent(exclusions=["domain adaptation"])
    )
    bridge_plan = build_query_plan("RRC-2026-0001", _frozen_intent(exclusions=["transfer"]))

    assert all("domain adaptation" not in query.query_text for query in synonym_plan.queries)
    assert all('all:"transfer"' not in query.query_text for query in bridge_plan.queries)
    assert synonym_plan.excluded_term_conflicts[0].positive_source is TermSource.DETERMINISTIC_RULE
    assert bridge_plan.excluded_term_conflicts[0].positive_source is TermSource.DETERMINISTIC_RULE


def test_exclusions_close_design_synonym_and_preserve_exact_matching() -> None:
    plan = build_query_plan(
        "RRC-2026-0001",
        _frozen_intent(exclusions=["optimization", "linear"]),
        positive_expansions=[
            QueryExpansion(
                target_field=RetrievalIntentField.METHOD,
                term_en="nonlinear",
                source=TermSource.LLM_FAKE,
            )
        ],
    )

    assert all("optimization" not in query.query_text for query in plan.queries)
    assert all('all:"linear"' not in query.query_text for query in plan.queries)
    assert any('all:"nonlinear"' in query.query_text for query in plan.queries)


def test_exclusion_mapping_rejects_unmapped_chinese_and_core_conflicts() -> None:
    with pytest.raises(PlanningError, match="UNMAPPED_RETRIEVAL_TERM"):
        build_query_plan("RRC-2026-0001", _frozen_intent(exclusions=["未知术语"]))

    with pytest.raises(PlanningError, match="CONTRADICTORY_INTENT"):
        build_query_plan("RRC-2026-0001", _frozen_intent(exclusions=["迁移学习"]))


@pytest.mark.parametrize("target_field", ("accepted_paper_roles", "exclusions"))
def test_expansion_field_is_restricted_and_exclusions_never_reach_any_query_text(
    target_field: str,
) -> None:
    with pytest.raises(ValueError, match="INVALID_QUERY_PLAN"):
        QueryExpansion(
            target_field=target_field,
            term_en="survey",
            source=TermSource.LLM_FAKE,
        )

    plan = build_query_plan(
        "RRC-2026-0001",
        _frozen_intent(exclusions=["radiation barrier"]),
        positive_expansions=[
            QueryExpansion(
                target_field=RetrievalIntentField.OBJECT,
                term_en="radiation barrier",
                source=TermSource.LLM_FAKE,
            )
        ],
    )

    assert plan.positive_expansions == []
    assert all("radiation barrier" not in query.query_text for query in plan.queries)
    assert plan.excluded_term_conflicts[0].positive_source is TermSource.LLM_FAKE
