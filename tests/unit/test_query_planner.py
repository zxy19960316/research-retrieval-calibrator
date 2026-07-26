from datetime import UTC, datetime

import pytest

from app.core.intent import freeze_research_intent
from app.core.query_planner import build_query_plan
from app.models.enums import MethodConstraint, QueryBranch, QueryBreadth
from app.models.planning import IntentDraft, TermEvidence, TermSource


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

    assert first == second
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
    plan = build_query_plan("RRC-2026-0001", intent, positive_expansions=["linear", "nonlinear"])

    assert "linear" not in plan.positive_expansions
    assert "nonlinear" in plan.positive_expansions
    assert plan.excluded_term_conflicts[0].term == "linear"
