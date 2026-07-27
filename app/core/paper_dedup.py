"""Conservative deterministic clustering for normalized source-backed papers."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Literal, NamedTuple
from unicodedata import normalize as unicode_normalize

from app.core.paper_normalization import normalize_paper
from app.models.dedup import (
    DedupCluster,
    DedupConfig,
    DedupDecision,
    DeduplicationResult,
    DedupReason,
    NormalizedPaper,
    PaperDeduplicationError,
    SourceIdentity,
)
from app.models.paper import PaperRecord

DEFAULT_DEDUP_CONFIG = DedupConfig()


class _PairMetrics(NamedTuple):
    title_similarity: float
    author_overlap: float
    year_difference: int | None


def deduplicate_papers(
    records: Sequence[PaperRecord], *, config: DedupConfig = DEFAULT_DEDUP_CONFIG
) -> DeduplicationResult:
    """Classify every stable pair and cluster only conservatively justified duplicates."""

    ordered = _coalesce_observations(records)
    normalized = {record.paper_id: normalize_paper(record) for record in ordered}
    parent = {record.paper_id: record.paper_id for record in ordered}
    base_decisions = _base_pair_decisions(ordered, normalized, config)
    final_decisions = dict(base_decisions)

    for reason in (
        DedupReason.EXACT_DOI,
        DedupReason.EXACT_ARXIV_ID,
        DedupReason.EXACT_NORMALIZED_TITLE_WITH_AUTHOR,
        DedupReason.HIGH_TITLE_SIMILARITY_WITH_AUTHOR,
    ):
        for pair_key, base_decision in base_decisions.items():
            if base_decision.action != "auto_merge" or base_decision.reason is not reason:
                continue
            left = normalized[pair_key[0]]
            right = normalized[pair_key[1]]
            if _would_conflict(parent, normalized, left, right):
                final_decisions[pair_key] = _manual_decision(
                    base_decision, DedupReason.IDENTITY_CONFLICT
                )
                continue
            if _requires_complete_link(base_decision) and not _has_complete_auto_merge_link(
                parent, base_decisions, left, right
            ):
                final_decisions[pair_key] = _manual_decision(
                    base_decision, DedupReason.TRANSITIVE_BRIDGE_RISK
                )
                continue
            _union(parent, left.record.paper_id, right.record.paper_id)

    decisions = [final_decisions[pair_key] for pair_key in base_decisions]

    clusters = _materialize_clusters(ordered, normalized, parent, decisions)
    return DeduplicationResult(clusters=clusters, decisions=decisions)


def _base_pair_decisions(
    ordered: Sequence[PaperRecord],
    normalized: Mapping[str, NormalizedPaper],
    config: DedupConfig,
) -> dict[tuple[str, str], DedupDecision]:
    """Classify every stable pair before clustering can affect any decision."""

    return {
        (left_record.paper_id, right_record.paper_id): _classify_pair(
            normalized[left_record.paper_id], normalized[right_record.paper_id], config
        )
        for left_index, left_record in enumerate(ordered)
        for right_record in ordered[left_index + 1 :]
    }


def _manual_decision(decision: DedupDecision, reason: DedupReason) -> DedupDecision:
    return decision.model_copy(update={"action": "manual_review", "reason": reason})


def _classify_pair(
    left: NormalizedPaper, right: NormalizedPaper, config: DedupConfig
) -> DedupDecision:
    token_jaccard = _jaccard(set(left.title_tokens), set(right.title_tokens))
    sequence_ratio = SequenceMatcher(
        None, left.normalized_title, right.normalized_title, autojunk=False
    ).ratio()
    author_jaccard = _jaccard(set(left.normalized_authors), set(right.normalized_authors))
    year_difference = _year_difference(left.record.year, right.record.year)
    metrics = _PairMetrics(sequence_ratio, author_jaccard, year_difference)

    if _identities_conflict(left, right):
        return _decision(left, right, "manual_review", DedupReason.IDENTITY_CONFLICT, metrics)
    if left.canonical_doi and left.canonical_doi == right.canonical_doi:
        return _decision(left, right, "auto_merge", DedupReason.EXACT_DOI, metrics)
    if left.canonical_arxiv_id and left.canonical_arxiv_id == right.canonical_arxiv_id:
        return _decision(left, right, "auto_merge", DedupReason.EXACT_ARXIV_ID, metrics)

    same_language = left.record.language == right.record.language
    compatible_year = year_difference is None or year_difference <= config.allowed_year_difference
    shared_author = bool(set(left.normalized_authors) & set(right.normalized_authors))
    high_title_similarity = (
        token_jaccard >= config.title_token_jaccard_threshold
        and sequence_ratio >= config.title_sequence_ratio_threshold
    )
    high_author_overlap = author_jaccard >= config.author_jaccard_threshold

    if (
        same_language
        and compatible_year
        and left.normalized_title == right.normalized_title
        and shared_author
    ):
        return _decision(
            left, right, "auto_merge", DedupReason.EXACT_NORMALIZED_TITLE_WITH_AUTHOR, metrics
        )
    if same_language and compatible_year and high_title_similarity and high_author_overlap:
        return _decision(
            left, right, "auto_merge", DedupReason.HIGH_TITLE_SIMILARITY_WITH_AUTHOR, metrics
        )
    if not same_language and (shared_author or high_title_similarity):
        return _decision(
            left, right, "manual_review", DedupReason.CROSS_LANGUAGE_POSSIBLE_DUPLICATE, metrics
        )
    if (
        (high_title_similarity and not high_author_overlap)
        or (not compatible_year and (shared_author or high_title_similarity))
        or (high_author_overlap and not high_title_similarity)
    ):
        return _decision(left, right, "manual_review", DedupReason.INSUFFICIENT_EVIDENCE, metrics)
    return _decision(left, right, "keep_separate", DedupReason.INSUFFICIENT_EVIDENCE, metrics)


def _decision(
    left: NormalizedPaper,
    right: NormalizedPaper,
    action: Literal["auto_merge", "manual_review", "keep_separate"],
    reason: DedupReason,
    metrics: _PairMetrics,
) -> DedupDecision:
    return DedupDecision(
        left_paper_id=left.record.paper_id,
        right_paper_id=right.record.paper_id,
        action=action,
        reason=reason,
        title_similarity=metrics.title_similarity,
        author_overlap=metrics.author_overlap,
        year_difference=metrics.year_difference,
    )


def _identities_conflict(left: NormalizedPaper, right: NormalizedPaper) -> bool:
    return (
        left.canonical_doi == right.canonical_doi
        and left.canonical_doi is not None
        and left.canonical_arxiv_id is not None
        and right.canonical_arxiv_id is not None
        and left.canonical_arxiv_id != right.canonical_arxiv_id
    ) or (
        left.canonical_arxiv_id == right.canonical_arxiv_id
        and left.canonical_arxiv_id is not None
        and left.canonical_doi is not None
        and right.canonical_doi is not None
        and left.canonical_doi != right.canonical_doi
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _year_difference(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return abs(left - right)


def _coalesce_observations(records: Sequence[PaperRecord]) -> list[PaperRecord]:
    """Merge repeated observations only when every non-path field agrees safely."""

    groups: dict[str, list[PaperRecord]] = defaultdict(list)
    for record in records:
        groups[record.paper_id].append(record)

    coalesced: list[PaperRecord] = []
    for paper_id in sorted(groups):
        observations = groups[paper_id]
        reference = min(observations, key=_representation_key)
        reference_key = _observation_key(reference)
        if any(_observation_key(record) != reference_key for record in observations):
            raise PaperDeduplicationError("DUPLICATE_PAPER_ID_CONFLICT")
        retrieval_paths = sorted({path for record in observations for path in record.retrieval_paths})
        coalesced.append(reference.model_copy(update={"retrieval_paths": retrieval_paths}, deep=True))
    return coalesced


def _representation_key(record: PaperRecord) -> str:
    """Return a stable raw-record key without paths, which are merged separately."""

    return json.dumps(
        record.model_dump(mode="json", exclude={"retrieval_paths"}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _observation_key(record: PaperRecord) -> tuple[object, ...]:
    normalized = normalize_paper(record)
    return (
        record.paper_id,
        _normalize_scalar(record.source),
        _canonical_source_id(record, normalized),
        normalized.normalized_title,
        _normalize_scalar(record.abstract),
        normalized.normalized_authors,
        record.year,
        normalized.canonical_doi,
        _normalize_url(record.url),
        _normalize_scalar(record.language),
        record.user_visible,
    )


def _canonical_source_id(record: PaperRecord, normalized: NormalizedPaper) -> str:
    if normalized.canonical_arxiv_id is not None:
        return normalized.canonical_arxiv_id
    return _normalize_url(record.source_id)


def _normalize_scalar(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(unicode_normalize("NFKC", value).split()).casefold()


def _normalize_url(value: str) -> str:
    return unicode_normalize("NFKC", value).strip()


def _requires_complete_link(decision: DedupDecision) -> bool:
    return decision.reason in {
        DedupReason.EXACT_NORMALIZED_TITLE_WITH_AUTHOR,
        DedupReason.HIGH_TITLE_SIMILARITY_WITH_AUTHOR,
    }


def _has_complete_auto_merge_link(
    parent: dict[str, str],
    base_decisions: Mapping[tuple[str, str], DedupDecision],
    left: NormalizedPaper,
    right: NormalizedPaper,
) -> bool:
    left_root = _root(parent, left.record.paper_id)
    right_root = _root(parent, right.record.paper_id)
    if left_root == right_root:
        return True
    left_members = sorted(
        paper_id for paper_id in parent if _root(parent, paper_id) == left_root
    )
    right_members = sorted(
        paper_id for paper_id in parent if _root(parent, paper_id) == right_root
    )
    for left_id in left_members:
        for right_id in right_members:
            if {left_id, right_id} == {left.record.paper_id, right.record.paper_id}:
                continue
            pair_key = (left_id, right_id) if left_id < right_id else (right_id, left_id)
            decision = base_decisions.get(pair_key)
            if decision is None or decision.action != "auto_merge":
                return False
            if decision.reason is DedupReason.IDENTITY_CONFLICT:
                return False
    return True


def _root(parent: dict[str, str], paper_id: str) -> str:
    if parent[paper_id] != paper_id:
        parent[paper_id] = _root(parent, parent[paper_id])
    return parent[paper_id]


def _union(parent: dict[str, str], left_id: str, right_id: str) -> None:
    left_root = _root(parent, left_id)
    right_root = _root(parent, right_id)
    if left_root != right_root:
        parent[max(left_root, right_root)] = min(left_root, right_root)


def _would_conflict(
    parent: dict[str, str],
    normalized: dict[str, NormalizedPaper],
    left: NormalizedPaper,
    right: NormalizedPaper,
) -> bool:
    candidate_ids = {
        paper_id
        for paper_id in parent
        if _root(parent, paper_id) in {_root(parent, left.record.paper_id), _root(parent, right.record.paper_id)}
    }
    dois = {normalized[paper_id].canonical_doi for paper_id in candidate_ids} - {None}
    arxiv_ids = {normalized[paper_id].canonical_arxiv_id for paper_id in candidate_ids} - {None}
    return len(dois) > 1 or len(arxiv_ids) > 1


def _materialize_clusters(
    ordered: Sequence[PaperRecord],
    normalized: dict[str, NormalizedPaper],
    parent: dict[str, str],
    decisions: Sequence[DedupDecision],
) -> list[DedupCluster]:
    components: dict[str, list[PaperRecord]] = defaultdict(list)
    for record in ordered:
        components[_root(parent, record.paper_id)].append(record)

    contested_paper_ids = {
        paper_id
        for decision in decisions
        if decision.reason is DedupReason.IDENTITY_CONFLICT
        for paper_id in (decision.left_paper_id, decision.right_paper_id)
    }
    clusters: list[DedupCluster] = []
    for members in components.values():
        members.sort(key=lambda record: record.paper_id)
        member_ids = {record.paper_id for record in members}
        merge_reasons = [
            decision
            for decision in decisions
            if decision.action == "auto_merge"
            and {decision.left_paper_id, decision.right_paper_id}.issubset(member_ids)
        ]
        canonical = min(members, key=lambda record: _canonical_key(normalized[record.paper_id]))
        clusters.append(
            DedupCluster(
                cluster_id=_cluster_id(members, normalized, contested_paper_ids),
                canonical_record=canonical,
                member_records=members,
                retrieval_paths=sorted(
                    {path for record in members for path in record.retrieval_paths}
                ),
                source_identities=sorted(
                    {
                        SourceIdentity(
                            source=record.source, source_id=record.source_id, url=record.url
                        )
                        for record in members
                    },
                    key=lambda identity: (identity.source, identity.source_id, identity.url),
                ),
                merge_reasons=merge_reasons,
            )
        )
    return sorted(clusters, key=lambda cluster: cluster.cluster_id)


def _canonical_key(normalized: NormalizedPaper) -> tuple[bool, bool, bool, bool, int, bool, int, str]:
    record = normalized.record
    return (
        normalized.canonical_doi is None,
        not record.source_id.strip(),
        not record.url.startswith(("http://", "https://")),
        not bool(record.abstract and record.abstract.strip()),
        -len(record.authors),
        record.year is None,
        -len(normalized.normalized_title),
        record.paper_id,
    )


def _cluster_id(
    members: Sequence[PaperRecord],
    normalized: dict[str, NormalizedPaper],
    contested_paper_ids: set[str],
) -> str:
    if any(record.paper_id in contested_paper_ids for record in members):
        return f"paper:{min(record.paper_id for record in members)}"
    dois = sorted(
        value
        for record in members
        if (value := normalized[record.paper_id].canonical_doi) is not None
    )
    if dois:
        return f"doi:{dois[0]}"
    arxiv_ids = sorted(
        value
        for record in members
        if (value := normalized[record.paper_id].canonical_arxiv_id) is not None
    )
    if arxiv_ids:
        return f"arxiv:{arxiv_ids[0]}"
    return f"paper:{min(record.paper_id for record in members)}"
