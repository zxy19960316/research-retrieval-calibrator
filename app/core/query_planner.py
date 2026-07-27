"""Deterministic M1-T01 four-branch, three-breadth canonical arXiv planning."""

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime

from app.core.intent import build_retrieval_term
from app.core.text_normalization import normalise_text
from app.models.enums import QueryBranch, QueryBreadth
from app.models.planning import (
    ExclusionTerm,
    IntentField,
    PlanningError,
    QueryExpansion,
    QueryExpression,
    QueryPlan,
    TermConflict,
    TermSource,
)
from app.models.project import ResearchIntent
from app.models.query import Query

PLANNING_CONFIG_VERSION = "m1-t01.v2"
_BRANCH_WEIGHTS = {
    QueryBranch.DIRECT_INTERSECTION: 0.35,
    QueryBranch.PROBLEM_DOMAIN: 0.25,
    QueryBranch.METHOD_DOMAIN: 0.20,
    QueryBranch.BRIDGE_DOMAIN: 0.20,
}
_BRIDGE_TERMS = ("application", "adaptation", "transfer", "framework", "methodology", "benchmark")
_SYNONYMS = {
    "transfer learning": ("domain adaptation",),
    "radiation shielding": ("radiation protection",),
    "design": ("optimization",),
    "nuclear engineering": ("nuclear technology",),
}
_FIELD_BY_GROUP = {
    "object": IntentField.OBJECT,
    "task": IntentField.TASK,
    "method": IntentField.METHOD,
    "scope": IntentField.SCOPE,
}


def build_exclusion_set(intent: ResearchIntent) -> dict[str, ExclusionTerm]:
    """Map exclusions for comparison while retaining their original audit input."""

    exclusions: dict[str, ExclusionTerm] = {}
    for original_text in intent.exclusions:
        mapped = build_retrieval_term(
            original_text,
            source=TermSource.ORIGINAL_INPUT,
            target_field=IntentField.EXCLUSIONS,
        )
        canonical = normalise_text(mapped.retrieval_text_en)
        if canonical in exclusions:
            raise PlanningError("INVALID_QUERY_PLAN", "duplicate canonical exclusion")
        exclusions[canonical] = ExclusionTerm(
            original_text=original_text,
            canonical_text_en=canonical,
            source=TermSource.ORIGINAL_INPUT,
        )
    return exclusions


def _filter_expansions(
    exclusions: dict[str, ExclusionTerm], positive_expansions: Sequence[QueryExpansion]
) -> tuple[list[QueryExpansion], list[TermConflict]]:
    kept: list[QueryExpansion] = []
    conflicts: list[TermConflict] = []
    for expansion in positive_expansions:
        canonical = normalise_text(expansion.term_en)
        exclusion = exclusions.get(canonical)
        if exclusion is None:
            kept.append(expansion)
        else:
            conflicts.append(
                TermConflict(
                    term=canonical,
                    positive_source=expansion.source,
                    exclusion_source=exclusion.source,
                    exclusion_original_text=exclusion.original_text,
                )
            )
    return kept, conflicts


def _filter_deterministic_terms(
    terms: Sequence[str], exclusions: dict[str, ExclusionTerm], conflicts: list[TermConflict]
) -> list[str]:
    kept: list[str] = []
    for term in terms:
        canonical = normalise_text(term)
        exclusion = exclusions.get(canonical)
        if exclusion is None:
            kept.append(term)
        else:
            conflicts.append(
                TermConflict(
                    term=canonical,
                    positive_source=TermSource.DETERMINISTIC_RULE,
                    exclusion_source=exclusion.source,
                    exclusion_original_text=exclusion.original_text,
                )
            )
    return kept


def _mapped_groups(intent: ResearchIntent) -> dict[str, list[str]]:
    values = {
        "object": intent.object_terms,
        "task": intent.task_terms,
        "method": intent.method_terms,
        "scope": intent.scope_terms,
    }
    return {
        group: [
            build_retrieval_term(
                value, source=TermSource.ORIGINAL_INPUT, target_field=_FIELD_BY_GROUP[group]
            ).retrieval_text_en
            for value in terms
        ]
        for group, terms in values.items()
    }


def _groups_with_expansions(
    groups: dict[str, list[str]], expansions: Sequence[QueryExpansion]
) -> dict[str, list[str]]:
    expanded = {name: list(terms) for name, terms in groups.items()}
    for expansion in expansions:
        for group, field in _FIELD_BY_GROUP.items():
            if expansion.target_field.value == field.value:
                expanded[group].append(expansion.term_en)
    return expanded


def _expression(
    groups: Sequence[Sequence[str]], anchors: Sequence[str]
) -> QueryExpression:
    return QueryExpression(
        required_groups=tuple(tuple(dict.fromkeys(group)) for group in groups),
        anchor_terms=tuple(anchors),
    )


def _narrow_expression(
    branch: QueryBranch, groups: dict[str, list[str]], bridge_terms: Sequence[str]
) -> QueryExpression:
    if branch is QueryBranch.DIRECT_INTERSECTION:
        return _expression([[groups["object"][0]], [groups["task"][0]], [groups["method"][0]]], [groups["object"][0]])
    if branch is QueryBranch.PROBLEM_DOMAIN:
        return _expression([[groups["object"][0]], [groups["task"][0]], [groups["scope"][0]]], [groups["object"][0]])
    if branch is QueryBranch.METHOD_DOMAIN:
        return _expression([[groups["method"][0]], [groups["task"][0]]], [groups["method"][0]])
    return _expression(
        [[groups["object"][0]], [groups["task"][0]], [groups["method"][0]], [bridge_terms[0]]],
        [groups["object"][0]],
    )


def _medium_expression(
    branch: QueryBranch, groups: dict[str, list[str]], bridge_terms: Sequence[str]
) -> QueryExpression:
    if branch is QueryBranch.DIRECT_INTERSECTION:
        return _expression([groups["object"], groups["task"], groups["method"]], [groups["object"][0]])
    if branch is QueryBranch.PROBLEM_DOMAIN:
        return _expression([groups["object"], groups["task"], groups["scope"]], [groups["object"][0]])
    if branch is QueryBranch.METHOD_DOMAIN:
        return _expression([groups["method"], groups["task"]], [groups["method"][0]])
    return _expression([groups["object"], groups["method"], bridge_terms], [groups["object"][0]])


def _wide_expression(
    branch: QueryBranch, groups: dict[str, list[str]], bridge_terms: Sequence[str]
) -> QueryExpression:
    if branch is QueryBranch.DIRECT_INTERSECTION:
        return _expression([groups["object"], [*groups["task"], *groups["method"]]], [groups["object"][0]])
    if branch is QueryBranch.PROBLEM_DOMAIN:
        return _expression([groups["object"], [*groups["task"], *groups["scope"]]], [groups["object"][0]])
    if branch is QueryBranch.METHOD_DOMAIN:
        return _expression([groups["method"]], [groups["method"][0]])
    return _expression(
        [[*groups["object"], *groups["task"]], [*groups["method"], *bridge_terms]],
        [groups["object"][0]],
    )


def _stable_id(
    project_id: str, revision: int, branch: QueryBranch, breadth: QueryBreadth, text: str
) -> str:
    payload = json.dumps(
        [project_id, revision, branch.value, breadth.value, text, PLANNING_CONFIG_VERSION],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return f"Q1-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def build_query_plan(
    project_id: str,
    intent: ResearchIntent,
    *,
    positive_expansions: Sequence[QueryExpansion] | None = None,
    generated_at_utc: datetime | None = None,
) -> QueryPlan:
    """Build 12 canonical, unencoded arXiv queries from a frozen intent."""

    if not isinstance(intent, ResearchIntent):
        raise PlanningError("INVALID_INTENT_STRUCTURE")
    exclusions = build_exclusion_set(intent)
    groups = _mapped_groups(intent)
    if any(normalise_text(term) in exclusions for terms in groups.values() for term in terms):
        raise PlanningError("CONTRADICTORY_INTENT")
    expansions, conflicts = _filter_expansions(exclusions, positive_expansions or ())
    synonym_groups = {
        name: _filter_deterministic_terms(
            [synonym for term in terms for synonym in _SYNONYMS.get(normalise_text(term), ())],
            exclusions,
            conflicts,
        )
        for name, terms in groups.items()
    }
    expanded_groups = _groups_with_expansions(groups, expansions)
    for name, synonyms in synonym_groups.items():
        expanded_groups[name].extend(synonyms)
    bridge_terms = _filter_deterministic_terms(_BRIDGE_TERMS, exclusions, conflicts)
    if not bridge_terms:
        raise PlanningError("INVALID_QUERY_PLAN", "bridge vocabulary exhausted")

    queries: list[Query] = []
    expressions: dict[str, QueryExpression] = {}
    for branch in QueryBranch:
        for breadth in QueryBreadth:
            expression = {
                QueryBreadth.NARROW: _narrow_expression(branch, groups, bridge_terms),
                QueryBreadth.MEDIUM: _medium_expression(branch, expanded_groups, bridge_terms),
                QueryBreadth.WIDE: _wide_expression(branch, expanded_groups, bridge_terms),
            }[breadth]
            text = expression.serialize()
            query_id = _stable_id(project_id, intent.revision, branch, breadth, text)
            queries.append(
                Query(
                    query_id=query_id,
                    round_number=1,
                    branch=branch,
                    breadth=breadth,
                    language="en",
                    query_text=text,
                    weight=_BRANCH_WEIGHTS[branch] / 3,
                )
            )
            expressions[query_id] = expression
    plan_seed = json.dumps(
        [project_id, intent.revision, [query.query_id for query in queries]],
        separators=(",", ":"),
    )
    return QueryPlan(
        plan_id=f"QP-{hashlib.sha256(plan_seed.encode('utf-8')).hexdigest()[:16]}",
        project_id=project_id,
        round_number=1,
        intent_revision=intent.revision,
        source="arxiv",
        planning_config_version=PLANNING_CONFIG_VERSION,
        branch_weights=_BRANCH_WEIGHTS,
        queries=queries,
        expressions=expressions,
        generated_at_utc=generated_at_utc or datetime.now(UTC),
        exclusions=list(exclusions.values()),
        positive_expansions=expansions,
        excluded_term_conflicts=conflicts,
    )
