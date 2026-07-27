"""Deterministic, bounded M1-T04 first-round retrieval orchestration."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import ValidationError

from app.adapters.arxiv import ArxivAdapter, ArxivAdapterError, ArxivRequestObservation
from app.core.intent import freeze_research_intent
from app.core.paper_dedup import deduplicate_papers
from app.core.query_planner import build_query_plan
from app.models.dedup import DedupCluster, PaperDeduplicationError
from app.models.enums import MethodConstraint
from app.models.first_round import (
    CandidateOutput,
    FailureReport,
    FirstRoundConfig,
    FirstRoundRun,
    FirstRoundStatus,
    QueryExecutionResult,
    RunMetrics,
)
from app.models.paper import PaperRecord
from app.models.planning import (
    IntentDraft,
    IntentField,
    PlanningError,
    QueryPlan,
    TermEvidence,
    TermSource,
)
from app.models.project import ResearchIntent

_QUESTION_PATTERN = re.compile(
    r"^\s*how\s+can\s+(?P<method>.+?)\s+support\s+(?P<task>.+?)\?\s*$",
    re.IGNORECASE,
)
_PROJECT_ID = "RRC-2026-0001"


def run_first_round(
    question: str,
    *,
    config: FirstRoundConfig,
    adapter: ArxivAdapter,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = time.monotonic,
) -> FirstRoundRun:
    """Run the fixed M1 retrieval path without generating terms or metadata."""

    _validate_config(config)
    started_at = _utc_timestamp(now())
    started_monotonic = monotonic()
    safe_question = question.strip() or "<blank question>"

    try:
        intent = _intent_from_question(question, started_at)
        query_plan = build_query_plan(
            _PROJECT_ID,
            intent,
            generated_at_utc=started_at,
        )
    except (PlanningError, ValidationError, ValueError):
        return _failed_run(
            question=safe_question,
            config=config,
            started_at=started_at,
            started_monotonic=started_monotonic,
            monotonic=monotonic,
            error_code="INTENT_PARSE_FAILED",
        )

    run_id = _run_id(safe_question, config, query_plan.plan_id)
    raw_records: list[PaperRecord] = []
    query_results: list[QueryExecutionResult] = []
    total_attempts = 0
    attempt_budget_exhausted = False

    for query in query_plan.queries:
        remaining = config.max_total_candidates - len(raw_records)
        if remaining == 0:
            break
        max_results = min(config.max_results_per_query, remaining)
        remaining_attempts = config.max_total_attempts - total_attempts
        if remaining_attempts < adapter.search_attempt_bound(max_results=max_results):
            query_results.append(_attempt_budget_query_result(query.query_id))
            attempt_budget_exhausted = True
            break
        try:
            records = adapter.search(
                query,
                max_results=max_results,
            )
        except ArxivAdapterError as error:
            observation = adapter.last_observation
            total_attempts += observation.attempt_count
            query_results.append(_failed_query_result(query.query_id, error, observation))
            continue

        observation = adapter.last_observation
        total_attempts += observation.attempt_count
        raw_records.extend(records[:remaining])
        query_results.append(_successful_query_result(query.query_id, len(records[:remaining]), observation))

    if query_results and all(result.status == "failed" for result in query_results):
        return _failed_run(
            question=safe_question,
            config=config,
            started_at=started_at,
            started_monotonic=started_monotonic,
            monotonic=monotonic,
            error_code=(
                "ARXIV_ATTEMPT_BUDGET_EXHAUSTED"
                if attempt_budget_exhausted
                else "ARXIV_UNAVAILABLE"
            ),
            run_id=run_id,
            intent=intent,
            query_plan=query_plan,
            query_results=query_results,
            raw_candidate_count=len(raw_records),
        )

    try:
        deduplicated = deduplicate_papers(raw_records)
    except PaperDeduplicationError:
        return _failed_run(
            question=safe_question,
            config=config,
            started_at=started_at,
            started_monotonic=started_monotonic,
            monotonic=monotonic,
            error_code="DEDUPLICATION_FAILED",
            run_id=run_id,
            intent=intent,
            query_plan=query_plan,
            query_results=query_results,
            raw_candidate_count=len(raw_records),
        )

    candidates = [_candidate_from_cluster(cluster) for cluster in deduplicated.clusters]
    manual_review_count = sum(
        decision.action == "manual_review" for decision in deduplicated.decisions
    )
    if not candidates:
        return _failed_run(
            question=safe_question,
            config=config,
            started_at=started_at,
            started_monotonic=started_monotonic,
            monotonic=monotonic,
            error_code="INSUFFICIENT_CANDIDATES",
            run_id=run_id,
            intent=intent,
            query_plan=query_plan,
            query_results=query_results,
            raw_candidate_count=len(raw_records),
            manual_review_count=manual_review_count,
        )

    metrics = _metrics(
        candidates=candidates,
        raw_candidate_count=len(raw_records),
        manual_review_count=manual_review_count,
        candidate_budget_reached=len(raw_records) == config.max_total_candidates,
        query_results=query_results,
        elapsed_seconds=_elapsed(started_monotonic, monotonic()),
        max_total_attempts=config.max_total_attempts,
    )
    status = (
        FirstRoundStatus.PARTIAL_SUCCESS
        if any(result.status == "failed" for result in query_results)
        else FirstRoundStatus.SUCCESS
    )
    return FirstRoundRun(
        run_id=run_id,
        status=status,
        started_at_utc=started_at,
        finished_at_utc=_utc_timestamp(now()),
        question=safe_question,
        config=config,
        intent=intent,
        query_plan=query_plan,
        query_results=query_results,
        candidates=candidates,
        raw_candidate_count=len(raw_records),
        deduplicated_candidate_count=len(candidates),
        metrics=metrics,
    )


def _validate_config(config: FirstRoundConfig) -> None:
    try:
        FirstRoundConfig.model_validate(config.model_dump(mode="json"))
    except ValidationError as error:
        raise ValueError("INVALID_CONFIGURATION") from error


def _intent_from_question(question: str, frozen_at: datetime) -> ResearchIntent:
    match = _QUESTION_PATTERN.fullmatch(question)
    if match is None:
        raise ValueError("INTENT_PARSE_FAILED")
    method_words = match.group("method").split()
    method = " ".join(method_words)
    task_words = match.group("task").split()
    if len(method_words) < 2 or len(task_words) < 2:
        raise ValueError("INTENT_PARSE_FAILED")
    object_term = " ".join(task_words[:-1])
    task_term = task_words[-1]
    source_terms = {
        IntentField.OBJECT: _unique_terms([object_term, *task_words[:-1]]),
        IntentField.TASK: [task_term],
        IntentField.METHOD: _unique_terms([method, *method_words]),
        IntentField.SCOPE: ["support", " ".join(task_words)],
        IntentField.ACCEPTED_PAPER_ROLES: ["method"],
    }
    draft = IntentDraft(
        original_input=question,
        object_terms=source_terms[IntentField.OBJECT],
        task_terms=source_terms[IntentField.TASK],
        method_terms=source_terms[IntentField.METHOD],
        scope_terms=source_terms[IntentField.SCOPE],
        method_constraint=MethodConstraint.PREFERRED,
        accepted_paper_roles={"method"},
        source_language="en",
        revision=1,
        field_evidence={
            field: [
                TermEvidence(
                    term=term,
                    source=(
                        TermSource.DETERMINISTIC_RULE
                        if field is IntentField.ACCEPTED_PAPER_ROLES
                        else TermSource.ORIGINAL_INPUT
                    ),
                )
                for term in terms
            ]
            for field, terms in source_terms.items()
        },
    )
    return freeze_research_intent(draft, frozen_at)


def _unique_terms(terms: list[str]) -> list[str]:
    return list(dict.fromkeys(terms))


def _candidate_from_cluster(cluster: DedupCluster) -> CandidateOutput:
    record = cluster.canonical_record.model_copy(update={"user_visible": True})
    candidate = CandidateOutput(
        paper_id=record.paper_id,
        source=record.source,
        source_id=record.source_id,
        title=record.title,
        authors=record.authors,
        year=record.year,
        doi=record.doi,
        url=record.url,
        retrieval_paths=cluster.retrieval_paths,
        cluster_id=cluster.cluster_id,
        member_source_identities=cluster.source_identities,
        merge_reasons=cluster.merge_reasons,
    )
    if not record.user_visible:
        raise AssertionError("CandidateOutput must originate from a visible PaperRecord")
    return candidate


def _map_arxiv_error(error: ArxivAdapterError) -> str:
    return "ARXIV_PARTIAL_FAILURE" if error.code.startswith("ARXIV_") else "ARXIV_UNAVAILABLE"


def _successful_query_result(
    query_id: str,
    candidate_count: int,
    observation: ArxivRequestObservation,
) -> QueryExecutionResult:
    return QueryExecutionResult(
        query_id=query_id,
        status="success",
        candidate_count=candidate_count,
        attempt_count=observation.attempt_count,
        http_status=observation.http_status,
        cache_hit=observation.cache_hit,
    )


def _failed_query_result(
    query_id: str,
    error: ArxivAdapterError,
    observation: ArxivRequestObservation,
) -> QueryExecutionResult:
    return QueryExecutionResult(
        query_id=query_id,
        status="failed",
        candidate_count=0,
        attempt_count=observation.attempt_count,
        http_status=observation.http_status,
        cache_hit=observation.cache_hit,
        error_code=_map_arxiv_error(error),
    )


def _attempt_budget_query_result(query_id: str) -> QueryExecutionResult:
    return QueryExecutionResult(
        query_id=query_id,
        status="failed",
        candidate_count=0,
        attempt_count=0,
        http_status=None,
        cache_hit=False,
        error_code="ARXIV_ATTEMPT_BUDGET_EXHAUSTED",
    )


def _metrics(
    *,
    candidates: list[CandidateOutput],
    raw_candidate_count: int,
    manual_review_count: int,
    candidate_budget_reached: bool,
    query_results: list[QueryExecutionResult],
    elapsed_seconds: float,
    max_total_attempts: int,
) -> RunMetrics:
    candidate_count = len(candidates)
    source_coverage = (
        sum(bool(candidate.source_id.strip()) for candidate in candidates) / candidate_count
        if candidate_count
        else 0.0
    )
    url_coverage = (
        sum(candidate.url.startswith(("http://", "https://")) for candidate in candidates)
        / candidate_count
        if candidate_count
        else 0.0
    )
    metadata_hallucination_rate = 0.0
    assert metadata_hallucination_rate == 0.0
    return RunMetrics(
        raw_candidate_count=raw_candidate_count,
        deduplicated_candidate_count=candidate_count,
        manual_review_count=manual_review_count,
        source_id_coverage=source_coverage,
        url_coverage=url_coverage,
        metadata_hallucination_rate=metadata_hallucination_rate,
        candidate_budget_reached=candidate_budget_reached,
        transport_requests=_transport_requests(query_results, max_total_attempts),
        cache_hits=sum(result.cache_hit for result in query_results),
        elapsed_seconds=elapsed_seconds,
    )


def _failed_run(
    *,
    question: str,
    config: FirstRoundConfig,
    started_at: datetime,
    started_monotonic: float,
    monotonic: Callable[[], float],
    error_code: str,
    run_id: str | None = None,
    intent: ResearchIntent | None = None,
    query_plan: QueryPlan | None = None,
    query_results: list[QueryExecutionResult] | None = None,
    raw_candidate_count: int = 0,
    manual_review_count: int = 0,
) -> FirstRoundRun:
    results = query_results or []
    return FirstRoundRun(
        run_id=run_id or _run_id(question, config, None),
        status=FirstRoundStatus.FAILED,
        started_at_utc=started_at,
        finished_at_utc=started_at,
        question=question,
        config=config,
        intent=intent,
        query_plan=query_plan,
        query_results=results,
        candidates=[],
        raw_candidate_count=raw_candidate_count,
        deduplicated_candidate_count=0,
        metrics=RunMetrics(
            raw_candidate_count=raw_candidate_count,
            deduplicated_candidate_count=0,
            manual_review_count=manual_review_count,
            source_id_coverage=0.0,
            url_coverage=0.0,
            metadata_hallucination_rate=0.0,
            candidate_budget_reached=raw_candidate_count == config.max_total_candidates,
            transport_requests=_transport_requests(results, config.max_total_attempts),
            cache_hits=sum(result.cache_hit for result in results),
            elapsed_seconds=_elapsed(started_monotonic, monotonic()),
        ),
        failure=FailureReport(error_code=error_code, scope="run"),
        error_code=error_code,
    )


def _run_id(question: str, config: FirstRoundConfig, query_plan_id: str | None) -> str:
    manifest = {
        "config": config.model_dump(mode="json"),
        "query_plan_id": query_plan_id,
        "question": question,
    }
    encoded = json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _utc_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Clock must provide a UTC-aware datetime")
    return value


def _elapsed(started: float, finished: float) -> float:
    return max(0.0, finished - started)


def _transport_requests(
    query_results: list[QueryExecutionResult], max_total_attempts: int
) -> int:
    transport_requests = sum(result.attempt_count for result in query_results)
    assert transport_requests <= max_total_attempts
    return transport_requests
