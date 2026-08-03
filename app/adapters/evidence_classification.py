"""Replaceable provider boundary and deterministic fake for M2-T03."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from app.models.enums import EvidenceSlot, SupportLevel
from app.models.evidence_classification import (
    EVIDENCE_CLASSIFICATION_VERSION,
    MAX_SUPPORTING_EXCERPT_LENGTH,
    EvidenceClassificationInput,
    EvidenceClassificationState,
    EvidenceClassifierDescriptor,
    build_grounded_reason,
)

_FAKE_DESCRIPTOR = EvidenceClassifierDescriptor(
    provider_name="deterministic_fake",
    classifier_id="m2-t03-lexical-evidence-slot",
    classifier_revision="fake-v1",
    prompt_revision="none",
    schema_version=EVIDENCE_CLASSIFICATION_VERSION,
    evidence_type="deterministic_fake",
)

_SLOT_MARKERS: tuple[tuple[EvidenceSlot, tuple[str, ...]], ...] = (
    (
        EvidenceSlot.EVALUATION_BASIS,
        (
            "evaluation",
            "evaluate",
            "evaluated",
            "experiment",
            "benchmark",
            "dataset",
            "results",
            "accuracy",
            "precision",
            "recall",
            "评估",
            "实验",
            "基准",
            "数据集",
            "结果",
        ),
    ),
    (
        EvidenceSlot.IMPLEMENTATION_PATH,
        (
            "pipeline",
            "architecture",
            "implementation",
            "implement",
            "application",
            "platform",
            "system",
            "code",
            "available",
            "部署",
            "实现",
            "系统",
            "平台",
        ),
    ),
    (
        EvidenceSlot.METHOD_TRANSFERABILITY,
        (
            "transfer",
            "transferable",
            "related-domain",
            "neighboring",
            "analog",
            "similar task",
            "adjacent",
            "cross-domain",
            "multimodal",
            "迁移",
            "邻近",
            "类比",
            "跨域",
        ),
    ),
    (
        EvidenceSlot.PROBLEM_EXISTENCE,
        (
            "problem",
            "challenge",
            "difficult",
            "limited",
            "lack",
            "need",
            "overload",
            "complex",
            "问题",
            "挑战",
            "困难",
            "缺乏",
            "需要",
        ),
    ),
    (
        EvidenceSlot.CURRENT_METHODS,
        (
            "method",
            "approach",
            "framework",
            "retrieval",
            "graph",
            "propose",
            "introduce",
            "present",
            "方法",
            "框架",
            "检索",
            "图",
        ),
    ),
)

_HYPOTHETICAL_MARKERS = (
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
_INDIRECT_MARKERS = (
    "related-domain",
    "neighboring",
    "transfer",
    "transferable",
    "analog",
    "similar task",
    "adjacent",
    "cross-domain",
    "may inform",
    "remains to be validated",
    "仍需",
    "间接",
    "迁移",
    "邻近",
    "类比",
)
_DIRECT_MARKERS = (
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
    "addresses",
    "provides",
    "直接",
    "证明",
    "展示",
)


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

    @property
    def descriptor(self) -> EvidenceClassifierDescriptor:
        return _FAKE_DESCRIPTOR

    def classify(self, inputs: Sequence[EvidenceClassificationInput]) -> list[dict[str, object]]:
        return [self._classify_one(item) for item in inputs]

    def _classify_one(self, item: EvidenceClassificationInput) -> dict[str, object]:
        text = f"{item.title}\n{item.abstract or ''}".casefold()
        slot = _select_slot(text)
        if item.abstract is None:
            if slot is None:
                return _rejection(item, EvidenceClassificationState.INSUFFICIENT_SOURCE_TEXT)
            excerpt = item.title[:MAX_SUPPORTING_EXCERPT_LENGTH]
            return _classified(
                item,
                slot=slot,
                support_level=SupportLevel.DIRECT,
                state=EvidenceClassificationState.TITLE_ONLY,
                excerpt=excerpt,
                title_only=True,
            )
        if slot is None:
            return _rejection(item, EvidenceClassificationState.REJECTED)

        support_level = _select_support_level(text)
        excerpt = _select_excerpt(item, slot)
        return _classified(
            item,
            slot=slot,
            support_level=support_level,
            state=EvidenceClassificationState.CLASSIFIED,
            excerpt=excerpt,
        )


def _select_slot(text: str) -> EvidenceSlot | None:
    for slot, markers in _SLOT_MARKERS:
        if any(marker in text for marker in markers):
            return slot
    return None


def _select_support_level(text: str) -> SupportLevel:
    has_hypothetical = any(marker in text for marker in _HYPOTHETICAL_MARKERS)
    has_indirect = any(marker in text for marker in _INDIRECT_MARKERS)
    has_direct = any(marker in text for marker in _DIRECT_MARKERS)
    if has_hypothetical and not has_direct:
        return SupportLevel.HYPOTHETICAL
    if has_indirect and not has_direct:
        return SupportLevel.INDIRECT
    return SupportLevel.DIRECT


def _select_excerpt(item: EvidenceClassificationInput, slot: EvidenceSlot) -> str:
    source = item.abstract or item.title
    lowered = source.casefold()
    markers = dict(_SLOT_MARKERS)[slot]
    for sentence in re.split(r"(?<=[.!?。！？])\s+", source):
        candidate = sentence.strip()
        if candidate and any(marker in candidate.casefold() for marker in markers):
            return candidate[:MAX_SUPPORTING_EXCERPT_LENGTH]
    for marker in markers:
        index = lowered.find(marker)
        if index >= 0:
            start = max(0, index - 80)
            return source[start : start + MAX_SUPPORTING_EXCERPT_LENGTH].strip()
    return source[:MAX_SUPPORTING_EXCERPT_LENGTH].strip()


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
) -> dict[str, object]:
    reason = (
        "title-only source text is insufficient for a reliable evidence-slot classification."
        if state is EvidenceClassificationState.INSUFFICIENT_SOURCE_TEXT
        else "The title and abstract do not provide a reliable evidence-slot basis."
    )
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
