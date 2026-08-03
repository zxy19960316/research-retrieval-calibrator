"""Replaceable provider boundary and deterministic fake for M2-T03."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from app.models.enums import EvidenceSlot, SupportLevel
from app.models.evidence_classification import (
    EVIDENCE_CLASSIFICATION_VERSION,
    MAX_SUPPORTING_EXCERPT_LENGTH,
    EvidenceClassificationInput,
    EvidenceClassificationQualityDiagnostics,
    EvidenceClassificationState,
    EvidenceClassifierDescriptor,
    build_grounded_reason,
)

_FAKE_DESCRIPTOR = EvidenceClassifierDescriptor(
    provider_name="deterministic_fake",
    classifier_id="m2-t03-lexical-evidence-slot",
    classifier_revision="fake-v2-grounded-fragment",
    prompt_revision="none",
    schema_version=EVIDENCE_CLASSIFICATION_VERSION,
    evidence_type="deterministic_fake",
)


@dataclass(frozen=True)
class _MarkerRule:
    marker: str
    pattern: re.Pattern[str]
    specificity: int
    high_specificity: bool


def _make_rule(
    pattern: str,
    *,
    marker: str | None = None,
    specificity: int = 1,
    high_specificity: bool = True,
) -> _MarkerRule:
    expression = (
        rf"\b(?:{pattern})\b"
        if pattern.isascii()
        else re.escape(pattern)
    )
    return _MarkerRule(
        marker=marker or pattern,
        pattern=re.compile(expression, re.IGNORECASE),
        specificity=specificity,
        high_specificity=high_specificity,
    )


_SLOT_MARKERS: tuple[tuple[EvidenceSlot, tuple[_MarkerRule, ...]], ...] = (
    (
        EvidenceSlot.PROBLEM_EXISTENCE,
        (
            _make_rule("challenge", specificity=3),
            _make_rule("limitation", specificity=3),
            _make_rule("limited", specificity=3),
            _make_rule("lack(?: of)?", marker="lack of", specificity=3),
            _make_rule("remains difficult", specificity=4),
            _make_rule("difficult", specificity=3),
            _make_rule("struggle(?:s)?", marker="struggle", specificity=3),
            _make_rule("overload", specificity=2),
            _make_rule("complex", specificity=2),
            _make_rule("问题", specificity=3),
            _make_rule("挑战", specificity=3),
            _make_rule("困难", specificity=3),
            _make_rule("缺乏", specificity=3),
            _make_rule("problem", high_specificity=False),
            _make_rule("need", high_specificity=False),
            _make_rule("gap", high_specificity=False),
        ),
    ),
    (
        EvidenceSlot.CURRENT_METHODS,
        (
            _make_rule(
                "we propose(?: a)?(?: (?:method|approach))?",
                marker="we propose",
                specificity=3,
            ),
            _make_rule("existing approaches", specificity=3),
            _make_rule("current framework", specificity=3),
            _make_rule("graph(?: retrieval)? approach", specificity=4),
            _make_rule("we introduce", specificity=2),
            _make_rule("we present", specificity=2),
            _make_rule("方法", specificity=3),
            _make_rule("框架", specificity=3),
            _make_rule("method", high_specificity=False),
            _make_rule("approach", high_specificity=False),
            _make_rule("framework", high_specificity=False),
            _make_rule("retrieval", high_specificity=False),
            _make_rule("graph", high_specificity=False),
        ),
    ),
    (
        EvidenceSlot.METHOD_TRANSFERABILITY,
        (
            _make_rule("cross[- ]domain", marker="cross-domain", specificity=4),
            _make_rule("related[- ]domain", marker="related-domain", specificity=4),
            _make_rule("transfer(?: to|able)?", marker="transfer", specificity=3),
            _make_rule("similar task", specificity=4),
            _make_rule("adjacent domain", specificity=4),
            _make_rule("neighboring", specificity=3),
            _make_rule("analog", specificity=3),
            _make_rule("multimodal", specificity=3),
            _make_rule("迁移", specificity=4),
            _make_rule("邻近", specificity=3),
            _make_rule("类比", specificity=3),
            _make_rule("跨域", specificity=4),
            _make_rule("domain", high_specificity=False),
            _make_rule("task", high_specificity=False),
            _make_rule("similar", high_specificity=False),
        ),
    ),
    (
        EvidenceSlot.IMPLEMENTATION_PATH,
        (
            _make_rule("we implement", specificity=4),
            _make_rule("we develop a system", specificity=5),
            _make_rule("pipeline", specificity=4),
            _make_rule("architecture", specificity=4),
            _make_rule("open[- ]source implementation", marker="open-source implementation", specificity=5),
            _make_rule("engineering solution", specificity=4),
            _make_rule("implementation", specificity=4),
            _make_rule("deploy(?:ment)?", marker="deployment", specificity=3),
            _make_rule("部署", specificity=4),
            _make_rule("实现", specificity=4),
            _make_rule("system", high_specificity=False),
            _make_rule("application", high_specificity=False),
            _make_rule("platform", high_specificity=False),
            _make_rule("code", high_specificity=False),
            _make_rule("available", high_specificity=False),
        ),
    ),
    (
        EvidenceSlot.EVALUATION_BASIS,
        (
            _make_rule("evaluation(?:s)?", marker="evaluation", specificity=3),
            _make_rule("evaluated on", specificity=4),
            _make_rule("experiment(?:s|al)?", marker="experiment", specificity=4),
            _make_rule("benchmark results", specificity=5),
            _make_rule("benchmark(?:s|ed|ing)?", marker="benchmark", specificity=4),
            _make_rule("accuracy", specificity=4),
            _make_rule("precision", specificity=4),
            _make_rule("recall", specificity=4),
            _make_rule("ndcg", specificity=4),
            _make_rule("mAP", specificity=4),
            _make_rule("f1", specificity=4),
            _make_rule("评估", specificity=4),
            _make_rule("实验", specificity=4),
            _make_rule("基准", specificity=4),
            _make_rule("结果", specificity=3),
            _make_rule("dataset(?:s)?", marker="dataset", high_specificity=False),
            _make_rule("results", high_specificity=False),
            _make_rule("test(?:s)?", marker="test", high_specificity=False),
            _make_rule("data", high_specificity=False),
        ),
    ),
)


def _support_patterns(values: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(_make_rule(value, high_specificity=False).pattern for value in values)


_HYPOTHETICAL_MARKERS = _support_patterns(
    (
        "could",
        "may be",
        "might",
        "possible",
        "potential",
        "hypothesis",
        "hypothetical",
        "future",
        "可能",
        "假设",
        "探索",
    )
)
_INDIRECT_MARKERS = _support_patterns(
    (
        "related[- ]domain",
        "neighboring",
        "transfer",
        "transferable",
        "analog",
        "similar task",
        "adjacent",
        "cross[- ]domain",
        "may inform",
        "remains to be validated",
        "仍需",
        "间接",
        "迁移",
        "邻近",
        "类比",
    )
)
_DIRECT_MARKERS = _support_patterns(
    (
        "we propose",
        "we introduce",
        "we present",
        "we develop",
        "we study",
        "we demonstrate",
        "we show",
        "demonstrate",
        "evaluated",
        "experiments",
        "benchmark",
        "results",
        "pipeline",
        "architecture",
        "implementation",
        "direct",
        "证明",
        "展示",
    )
)


@dataclass(frozen=True)
class _EvidenceMatch:
    slot: EvidenceSlot
    support_level: SupportLevel
    matched_marker: str
    matched_source: Literal["title", "abstract"]
    supporting_excerpt: str
    specificity: int


@dataclass(frozen=True)
class _ScoredEvidenceMatch:
    evidence: _EvidenceMatch
    match_score: int
    high_specificity_count: int
    generic_marker_count: int
    source_priority: int
    excerpt_start: int


_RejectionKind = Literal["none", "no_match", "generic_only", "ambiguous"]


@dataclass(frozen=True)
class _ClassificationOutcome:
    record: dict[str, object]
    rejection_kind: _RejectionKind


class EvidenceClassifierProvider(Protocol):
    """A provider that classifies only pre-validated title/abstract inputs."""

    @property
    def descriptor(self) -> EvidenceClassifierDescriptor: ...

    def classify(self, inputs: Sequence[EvidenceClassificationInput]) -> Sequence[object]: ...


def deterministic_fake_descriptor() -> EvidenceClassifierDescriptor:
    """Return the fixed descriptor used for offline contract evidence."""

    return _FAKE_DESCRIPTOR


class DeterministicFakeEvidenceClassifier:
    """A lexical, deterministic fake whose output is never real-model evidence."""

    def __init__(self) -> None:
        self._quality_diagnostics: EvidenceClassificationQualityDiagnostics | None = None

    @property
    def descriptor(self) -> EvidenceClassifierDescriptor:
        return _FAKE_DESCRIPTOR

    @property
    def quality_diagnostics(self) -> EvidenceClassificationQualityDiagnostics | None:
        return self._quality_diagnostics

    def classify(self, inputs: Sequence[EvidenceClassificationInput]) -> list[dict[str, object]]:
        outcomes = [self._classify_one(item) for item in inputs]
        records = [outcome.record for outcome in outcomes]
        self._quality_diagnostics = _build_quality_diagnostics(outcomes)
        return records

    def _classify_one(self, item: EvidenceClassificationInput) -> _ClassificationOutcome:
        match, rejection_kind = _select_match(item)
        if match is None:
            state = (
                EvidenceClassificationState.INSUFFICIENT_SOURCE_TEXT
                if item.abstract is None
                else EvidenceClassificationState.REJECTED
            )
            return _ClassificationOutcome(
                record=_rejection(item, state, rejection_kind),
                rejection_kind=rejection_kind,
            )

        state = (
            EvidenceClassificationState.TITLE_ONLY
            if item.abstract is None
            else EvidenceClassificationState.CLASSIFIED
        )
        return _ClassificationOutcome(
            record=_classified(
                item,
                slot=match.evidence.slot,
                support_level=match.evidence.support_level,
                state=state,
                excerpt=match.evidence.supporting_excerpt,
                title_only=state is EvidenceClassificationState.TITLE_ONLY,
            ),
            rejection_kind="none",
        )


def _source_fragments(
    source: str, source_name: Literal["title", "abstract"]
) -> list[tuple[str, Literal["title", "abstract"], int]]:
    fragments: list[tuple[str, Literal["title", "abstract"], int]] = []
    for match in re.finditer(r"[^.!?。！？\n]+(?:[.!?。！？]+|$)", source):
        raw = match.group(0)
        fragment = raw.strip()
        if fragment:
            leading = len(raw) - len(raw.lstrip())
            fragments.append((fragment, source_name, match.start() + leading))
    return fragments


def _candidate_fragments(
    item: EvidenceClassificationInput,
) -> list[tuple[str, Literal["title", "abstract"], int]]:
    fragments = _source_fragments(item.title, "title")
    if item.abstract is not None:
        fragments.extend(_source_fragments(item.abstract, "abstract"))
    return fragments


def _bounded_excerpt(fragment: str, marker_start: int) -> str:
    if len(fragment) <= MAX_SUPPORTING_EXCERPT_LENGTH:
        return fragment
    start = max(0, marker_start - MAX_SUPPORTING_EXCERPT_LENGTH // 3)
    end = min(len(fragment), start + MAX_SUPPORTING_EXCERPT_LENGTH)
    start = max(0, end - MAX_SUPPORTING_EXCERPT_LENGTH)
    return fragment[start:end].strip()


def _select_support_level(text: str) -> SupportLevel:
    has_hypothetical = any(marker.search(text) for marker in _HYPOTHETICAL_MARKERS)
    has_indirect = any(marker.search(text) for marker in _INDIRECT_MARKERS)
    has_direct = any(marker.search(text) for marker in _DIRECT_MARKERS)
    if has_hypothetical and not has_direct:
        return SupportLevel.HYPOTHETICAL
    if has_indirect and not has_direct:
        return SupportLevel.INDIRECT
    return SupportLevel.DIRECT


def _collect_matches(item: EvidenceClassificationInput) -> list[_ScoredEvidenceMatch]:
    matches: list[_ScoredEvidenceMatch] = []
    for fragment, source_name, source_offset in _candidate_fragments(item):
        for slot, rules in _SLOT_MARKERS:
            hits = [(rule, rule.pattern.search(fragment)) for rule in rules]
            matched = [(rule, hit) for rule, hit in hits if hit is not None]
            if not matched:
                continue
            high = [rule for rule, _ in matched if rule.high_specificity]
            generic = [rule for rule, _ in matched if not rule.high_specificity]
            primary_rule, primary_hit = max(
                matched,
                key=lambda pair: (pair[0].specificity, len(pair[0].marker), pair[0].marker),
            )
            assert primary_hit is not None
            excerpt = _bounded_excerpt(fragment, primary_hit.start())
            matches.append(
                _ScoredEvidenceMatch(
                    evidence=_EvidenceMatch(
                        slot=slot,
                        support_level=_select_support_level(excerpt),
                        matched_marker=primary_rule.marker,
                        matched_source=source_name,
                        supporting_excerpt=excerpt,
                        specificity=sum(rule.specificity for rule, _ in matched),
                    ),
                    match_score=3 * len(high) + len(generic),
                    high_specificity_count=len(high),
                    generic_marker_count=len(generic),
                    source_priority=0 if source_name == "title" else 1,
                    excerpt_start=source_offset + primary_hit.start(),
                )
            )
    return matches


def _select_match(
    item: EvidenceClassificationInput,
) -> tuple[_ScoredEvidenceMatch | None, _RejectionKind]:
    candidates = _collect_matches(item)
    if not candidates:
        return None, "no_match"
    if not any(candidate.high_specificity_count for candidate in candidates):
        return None, "generic_only"

    best_score = max(candidate.match_score for candidate in candidates)
    best = [candidate for candidate in candidates if candidate.match_score == best_score]
    best_specificity = max(candidate.evidence.specificity for candidate in best)
    best = [candidate for candidate in best if candidate.evidence.specificity == best_specificity]
    if len(best) > 1 and len(
        {
            (candidate.source_priority, candidate.excerpt_start)
            for candidate in best
        }
    ) == 1:
        return None, "ambiguous"
    winner = min(
        best,
        key=lambda candidate: (
            candidate.source_priority,
            candidate.excerpt_start,
            list(EvidenceSlot).index(candidate.evidence.slot),
        ),
    )
    return winner, "none"


def _select_excerpt(item: EvidenceClassificationInput, slot: EvidenceSlot) -> str:
    """Select an excerpt only when it is bound to a matching slot marker."""

    match, _ = _select_match(item)
    if match is None or match.evidence.slot is not slot:
        raise ValueError("no grounded excerpt for the requested evidence slot")
    return match.evidence.supporting_excerpt


def _build_quality_diagnostics(
    outcomes: Sequence[_ClassificationOutcome],
) -> EvidenceClassificationQualityDiagnostics:
    slots = {
        record["evidence_slot"]
        for outcome in outcomes
        if (record := outcome.record).get("evidence_slot") is not None
    }
    support_levels = {
        record["support_level"]
        for outcome in outcomes
        if (record := outcome.record).get("support_level") is not None
    }
    generic_only_count = sum(outcome.rejection_kind == "generic_only" for outcome in outcomes)
    ambiguous_count = sum(outcome.rejection_kind == "ambiguous" for outcome in outcomes)
    rejected_count = sum(outcome.rejection_kind != "none" for outcome in outcomes)
    warnings: list[str] = []
    if len(slots) < 3:
        warnings.append("FEWER_THAN_THREE_SLOTS_OBSERVED")
    if support_levels and len(support_levels) == 1:
        warnings.append("ALL_RECORDS_SAME_SUPPORT_LEVEL")
    if not {SupportLevel.INDIRECT.value, SupportLevel.HYPOTHETICAL.value} & support_levels:
        warnings.append("NO_INDIRECT_OR_HYPOTHETICAL_RECORDS")
    if generic_only_count >= max(1, len(outcomes) // 2):
        warnings.append("GENERIC_MARKER_DOMINANCE")
    if rejected_count == 0:
        warnings.append("NO_REJECTED_OR_UNCERTAIN_RECORDS")
    return EvidenceClassificationQualityDiagnostics(
        observed_slot_count=len(slots),
        observed_support_level_count=len(support_levels),
        all_records_same_support_level=bool(support_levels) and len(support_levels) == 1,
        generic_marker_only_count=generic_only_count,
        ambiguous_rejection_count=ambiguous_count,
        warnings=warnings,  # type: ignore[arg-type]
    )


def _classified(
    item: EvidenceClassificationInput,
    *,
    slot: EvidenceSlot,
    support_level: SupportLevel,
    state: EvidenceClassificationState,
    excerpt: str,
    title_only: bool = False,
) -> dict[str, object]:
    return {
        "paper_id": item.paper_id,
        "evidence_slot": slot,
        "support_level": support_level,
        "reason": build_grounded_reason(support_level, excerpt, title_only=title_only),
        "supporting_excerpt": excerpt,
        "source_text_sha256": item.source_text_sha256,
        "classifier_descriptor": item.classifier_descriptor,
        "classification_version": item.classification_version,
        "state": state,
    }


def _rejection(
    item: EvidenceClassificationInput,
    state: EvidenceClassificationState,
    rejection_kind: _RejectionKind,
) -> dict[str, object]:
    if state is EvidenceClassificationState.INSUFFICIENT_SOURCE_TEXT:
        reason = "title-only source text is insufficient for a reliable evidence-slot classification."
    elif rejection_kind == "generic_only":
        reason = "The source contains only generic markers and does not provide a reliable evidence-slot basis."
    elif rejection_kind == "ambiguous":
        reason = "The source evidence-slot markers are ambiguous and do not provide a reliable evidence-slot basis."
    else:
        reason = "The title and abstract do not provide a reliable evidence-slot basis."
    return {
        "paper_id": item.paper_id,
        "evidence_slot": None,
        "support_level": None,
        "reason": reason,
        "supporting_excerpt": None,
        "source_text_sha256": item.source_text_sha256,
        "classifier_descriptor": item.classifier_descriptor,
        "classification_version": item.classification_version,
        "state": state,
    }
