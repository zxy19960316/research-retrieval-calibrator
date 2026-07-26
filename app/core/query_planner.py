"""Deterministic M1-T01 four-branch, three-breadth arXiv query planning."""

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime

from app.core.intent import normalise_term
from app.models.enums import QueryBranch, QueryBreadth
from app.models.planning import PlanningError, QueryPlan, TermConflict, TermSource
from app.models.project import ResearchIntent
from app.models.query import Query

PLANNING_CONFIG_VERSION = "m1-t01.v1"
_DETERMINISTIC_GENERATED_AT = datetime(2026, 1, 1, tzinfo=UTC)
_BRANCH_WEIGHTS = {
    QueryBranch.DIRECT_INTERSECTION: 0.35,
    QueryBranch.PROBLEM_DOMAIN: 0.25,
    QueryBranch.METHOD_DOMAIN: 0.20,
    QueryBranch.BRIDGE_DOMAIN: 0.20,
}
_BRIDGE_TERMS = ("application", "adaptation", "transfer", "framework", "methodology", "benchmark")
_TRANSLATIONS = {
    "辐射屏蔽": "radiation shielding",
    "核反应堆": "nuclear reactor",
    "核工程": "nuclear engineering",
    "迁移学习": "transfer learning",
    "深度学习": "deep learning",
    "机器学习": "machine learning",
    "设计": "design",
    "评估": "evaluation",
    "方法": "method",
}
_SYNONYMS = {
    "transfer learning": ("domain adaptation",),
    "radiation shielding": ("radiation protection",),
    "design": ("optimization",),
}


def _english_term(value: str) -> str:
    translated = value
    for chinese, english in sorted(_TRANSLATIONS.items(), key=lambda item: -len(item[0])):
        translated = translated.replace(chinese, english)
    translated = re.sub(r"[^\x00-\x7f]+", " ", translated)
    return " ".join(translated.split()).casefold()


def _english_terms(values: Sequence[str]) -> list[str]:
    return [term for value in values if (term := _english_term(value))]


def _quoted(term: str) -> str:
    return f'"{term}"'


def _or_group(terms: Sequence[str]) -> str:
    unique = list(dict.fromkeys(terms))
    if not unique:
        return '"research"'
    if len(unique) == 1:
        return _quoted(unique[0])
    return "(" + " OR ".join(_quoted(term) for term in unique) + ")"


def _narrow_terms(terms: Sequence[str]) -> str:
    return _quoted(terms[0]) if terms else '"research"'


def _synonym_terms(terms: Sequence[str]) -> list[str]:
    return [synonym for term in terms for synonym in _SYNONYMS.get(normalise_term(term), ())]


def _template(branch: QueryBranch, breadth: QueryBreadth, groups: dict[str, list[str]]) -> str:
    object_terms = groups["object"]
    task_terms = groups["task"]
    method_terms = groups["method"]
    scope_terms = groups["scope"]
    if branch is QueryBranch.DIRECT_INTERSECTION:
        components = [object_terms, task_terms, method_terms]
    elif branch is QueryBranch.PROBLEM_DOMAIN:
        components = [object_terms, task_terms, scope_terms]
    elif branch is QueryBranch.METHOD_DOMAIN:
        components = [method_terms, task_terms]
    else:
        components = [object_terms, task_terms, method_terms, list(_BRIDGE_TERMS)]

    if breadth is QueryBreadth.NARROW:
        return " AND ".join(_narrow_terms(component) for component in components)
    if breadth is QueryBreadth.MEDIUM:
        return " AND ".join(_or_group(component + _synonym_terms(component)) for component in components)
    wide_components = list(components)
    if branch is QueryBranch.DIRECT_INTERSECTION:
        wide_components[-1] = method_terms + _synonym_terms(method_terms)
    if branch is QueryBranch.BRIDGE_DOMAIN:
        wide_components[-1] = list(_BRIDGE_TERMS[:3])
    return " AND ".join(_or_group(component) for component in wide_components)


def _stable_id(project_id: str, revision: int, branch: QueryBranch, breadth: QueryBreadth, text: str) -> str:
    payload = json.dumps(
        [project_id, revision, branch.value, breadth.value, text, PLANNING_CONFIG_VERSION],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return f"Q1-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _filter_expansions(
    exclusions: Sequence[str], positive_expansions: Sequence[str]
) -> tuple[list[str], list[TermConflict]]:
    excluded = {normalise_term(term) for term in exclusions}
    kept: list[str] = []
    conflicts: list[TermConflict] = []
    for term in positive_expansions:
        if normalise_term(term) in excluded:
            conflicts.append(
                TermConflict(
                    term=normalise_term(term),
                    positive_source=TermSource.DETERMINISTIC_RULE,
                    exclusion_source=TermSource.ORIGINAL_INPUT,
                )
            )
        else:
            kept.append(term)
    return kept, conflicts


def build_query_plan(
    project_id: str,
    intent: ResearchIntent,
    *,
    positive_expansions: Sequence[str] | None = None,
) -> QueryPlan:
    """Build a byte-stable first-round plan from a frozen M0 ResearchIntent only."""

    if not isinstance(intent, ResearchIntent):
        raise PlanningError("INVALID_INTENT_STRUCTURE")
    expansions, conflicts = _filter_expansions(intent.exclusions, positive_expansions or ())
    groups = {
        "object": _english_terms(intent.object_terms),
        "task": _english_terms(intent.task_terms),
        "method": _english_terms(intent.method_terms),
        "scope": _english_terms(intent.scope_terms),
    }
    queries: list[Query] = []
    for branch in QueryBranch:
        for breadth in QueryBreadth:
            text = _template(branch, breadth, groups)
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
        generated_at_utc=_DETERMINISTIC_GENERATED_AT,
        exclusions=list(intent.exclusions),
        positive_expansions=expansions,
        excluded_term_conflicts=conflicts,
    )
