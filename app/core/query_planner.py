"""Deterministic M1-T01 four-branch, three-breadth canonical arXiv planning."""

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime

from app.core.intent import build_retrieval_term
from app.core.text_normalization import normalise_text
from app.models.enums import QueryBranch, QueryBreadth
from app.models.planning import (
    IntentField,
    PlanningError,
    QueryExpansion,
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
_WIDE_TERMS = {
    QueryBranch.DIRECT_INTERSECTION: ("framework",),
    QueryBranch.PROBLEM_DOMAIN: ("application",),
    QueryBranch.METHOD_DOMAIN: ("methodology",),
    QueryBranch.BRIDGE_DOMAIN: ("benchmark",),
}
_SYNONYMS = {
    "transfer learning": ("domain adaptation",),
    "radiation shielding": ("radiation protection",),
    "design": ("optimization",),
}
_FIELD_BY_GROUP = {
    "object": IntentField.OBJECT,
    "task": IntentField.TASK,
    "method": IntentField.METHOD,
    "scope": IntentField.SCOPE,
}


def _term_clause(term: str, field: str = "all") -> str:
    return f'{field}:"{term}"'


def _or_group(terms: Sequence[str]) -> str:
    unique = list(dict.fromkeys(terms))
    if not unique:
        raise PlanningError("INVALID_INTENT_STRUCTURE")
    clauses = [_term_clause(term) for term in unique]
    return clauses[0] if len(clauses) == 1 else "(" + " OR ".join(clauses) + ")"


def _narrow_terms(terms: Sequence[str]) -> str:
    if not terms:
        raise PlanningError("INVALID_INTENT_STRUCTURE")
    return _term_clause(terms[0])


def build_exclusion_set(intent: ResearchIntent) -> dict[str, str]:
    """Map exclusions through the retrieval pipeline while retaining audit text."""

    exclusions: dict[str, str] = {}
    for original_text in intent.exclusions:
        mapped = build_retrieval_term(
            original_text,
            source=TermSource.ORIGINAL_INPUT,
            target_field=IntentField.EXCLUSIONS,
        )
        exclusions[normalise_text(mapped.retrieval_text_en)] = original_text
    return exclusions


def _components(
    branch: QueryBranch, groups: dict[str, list[str]], bridge_terms: Sequence[str]
) -> list[list[str]]:
    if branch is QueryBranch.DIRECT_INTERSECTION:
        return [groups["object"], groups["task"], groups["method"]]
    if branch is QueryBranch.PROBLEM_DOMAIN:
        return [groups["object"], groups["task"], groups["scope"]]
    if branch is QueryBranch.METHOD_DOMAIN:
        return [groups["method"], groups["task"]]
    return [groups["object"], groups["task"], groups["method"], list(bridge_terms)]


def _template(
    branch: QueryBranch,
    breadth: QueryBreadth,
    groups: dict[str, list[str]],
    expanded_groups: dict[str, list[str]],
    bridge_terms: Sequence[str],
    wide_terms: dict[QueryBranch, list[str]],
) -> str:
    components = _components(branch, groups, bridge_terms)
    if breadth is QueryBreadth.NARROW:
        return " AND ".join(_narrow_terms(component) for component in components)
    if breadth is QueryBreadth.MEDIUM:
        return " AND ".join(
            _or_group(component)
            for component in _components(branch, expanded_groups, bridge_terms)
        )
    wide_components = _components(branch, expanded_groups, bridge_terms)
    if branch is QueryBranch.BRIDGE_DOMAIN:
        wide_components[-1] = list(bridge_terms[:3])
    return " AND ".join(
        [_or_group(component) for component in wide_components]
        + [_or_group(wide_terms[branch])]
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


def _filter_expansions(
    exclusions: dict[str, str], positive_expansions: Sequence[QueryExpansion]
) -> tuple[list[QueryExpansion], list[TermConflict]]:
    kept: list[QueryExpansion] = []
    conflicts: list[TermConflict] = []
    for expansion in positive_expansions:
        term = normalise_text(expansion.term_en)
        if term in exclusions:
            conflicts.append(
                TermConflict(
                    term=term,
                    positive_source=expansion.source,
                    exclusion_source=TermSource.ORIGINAL_INPUT,
                )
            )
        else:
            kept.append(expansion)
    return kept, conflicts


def _filter_deterministic_terms(
    terms: Sequence[str], exclusions: dict[str, str], conflicts: list[TermConflict]
) -> list[str]:
    kept: list[str] = []
    for term in terms:
        canonical = normalise_text(term)
        if canonical in exclusions:
            conflicts.append(
                TermConflict(
                    term=canonical,
                    positive_source=TermSource.DETERMINISTIC_RULE,
                    exclusion_source=TermSource.ORIGINAL_INPUT,
                )
            )
        else:
            kept.append(term)
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
    for terms in groups.values():
        if any(normalise_text(term) in exclusions for term in terms):
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
    wide_terms = {
        branch: _filter_deterministic_terms(terms, exclusions, conflicts)
        for branch, terms in _WIDE_TERMS.items()
    }
    queries: list[Query] = []
    for branch in QueryBranch:
        for breadth in QueryBreadth:
            text = _template(branch, breadth, groups, expanded_groups, bridge_terms, wide_terms)
            queries.append(
                Query(
                    query_id=_stable_id(project_id, intent.revision, branch, breadth, text),
                    round_number=1,
                    branch=branch,
                    breadth=breadth,
                    language="en",
                    query_text=text,
                    weight=_BRANCH_WEIGHTS[branch] / 3,
                )
            )
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
        generated_at_utc=generated_at_utc or datetime.now(UTC),
        exclusions=list(exclusions),
        positive_expansions=expansions,
        excluded_term_conflicts=conflicts,
    )
