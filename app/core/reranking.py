"""Deterministic, fail-closed orchestration for M2-T02 reranking."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from app.adapters.reranking import RerankerProvider
from app.models.embedding import FrozenCandidate
from app.models.reranking import (
    ProviderRawScore,
    RerankerInput,
    RerankerModelDescriptor,
    RerankerRunState,
    RerankerTaskError,
    RerankRecord,
    RerankRun,
)

_CACHE_KEYS = frozenset(
    {"cache_key", "paper_id", "query_sha256", "input_sha256", "descriptor", "raw_score"}
)


def build_reranker_input(candidate: FrozenCandidate) -> RerankerInput:
    """Serialize one frozen candidate without creating or inferring an abstract."""

    title = candidate.title.replace("\r\n", "\n").strip()
    if not title:
        raise RerankerTaskError("INVALID_INPUT")
    abstract = (candidate.abstract or "").replace("\r\n", "\n").strip()
    text = f"title:\n{title}"
    if abstract:
        text += f"\n\nabstract:\n{abstract}"
    return RerankerInput(
        paper_id=candidate.paper_id,
        text=text,
        input_sha256=_sha256_text(text),
    )


def effective_top_k(*, configured_top_k: int, available_count: int) -> int:
    """Validate the fixed M2 reranker window and cap it to available inputs."""

    if (
        isinstance(configured_top_k, bool)
        or not isinstance(configured_top_k, int)
        or not 50 <= configured_top_k <= 100
        or isinstance(available_count, bool)
        or not isinstance(available_count, int)
        or available_count < 0
    ):
        raise RerankerTaskError("INVALID_INPUT")
    return min(configured_top_k, available_count)


def rerank_candidates(
    query: str,
    candidates: Sequence[FrozenCandidate],
    provider: RerankerProvider,
    cache_dir: Path | str,
    *,
    batch_size: int,
    configured_top_k: int,
) -> RerankRun:
    """Score the embedding-ranked top-K with cache-safe batch orchestration."""

    normalized_query = _normalize_query(query)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise RerankerTaskError("INVALID_INPUT")
    _require_unique_paper_ids(candidates)
    selected = list(candidates[:effective_top_k(
        configured_top_k=configured_top_k, available_count=len(candidates)
    )])
    if not selected:
        return RerankRun(state=RerankerRunState.NOT_RUN, records=[])

    try:
        descriptor = _validated_descriptor(provider.descriptor)
    except Exception as error:
        raise RerankerTaskError("PROVIDER_UNAVAILABLE") from error

    query_sha256 = _sha256_text(normalized_query)
    destination = Path(cache_dir)
    input_by_paper_id = {candidate.paper_id: build_reranker_input(candidate) for candidate in selected}
    cached_scores: dict[str, float] = {}
    misses: list[tuple[RerankerInput, str]] = []

    for candidate in selected:
        reranker_input = input_by_paper_id[candidate.paper_id]
        cache_key = _cache_key(query_sha256, reranker_input, descriptor)
        raw_score = _read_cache_entry(
            destination / f"{cache_key}.json",
            cache_key=cache_key,
            query_sha256=query_sha256,
            reranker_input=reranker_input,
            descriptor=descriptor,
        )
        if raw_score is None:
            misses.append((reranker_input, cache_key))
        else:
            cached_scores[reranker_input.paper_id] = raw_score

    new_scores: dict[str, float] = {}
    for start in range(0, len(misses), batch_size):
        chunk = misses[start : start + batch_size]
        try:
            provider_output = provider.score(
                normalized_query,
                [reranker_input for reranker_input, _ in chunk],
                batch_size=batch_size,
            )
        except Exception as error:
            raise RerankerTaskError("PROVIDER_UNAVAILABLE") from error
        validated_scores = _validate_provider_output(provider_output, chunk)
        new_scores.update(validated_scores)

    # Provider calls and validation are complete before any new cache entry is
    # published.  Provider failures therefore cannot leave partial cache output.
    try:
        for reranker_input, cache_key in misses:
            _write_cache_entry(
                destination / f"{cache_key}.json",
                cache_key=cache_key,
                query_sha256=query_sha256,
                reranker_input=reranker_input,
                descriptor=descriptor,
                raw_score=new_scores[reranker_input.paper_id],
            )
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise RerankerTaskError("INVALID_OUTPUT") from error

    raw_scores = {**cached_scores, **new_scores}
    normalized_scores = _normalize_scores(raw_scores)
    records = [
        RerankRecord(
            state=RerankerRunState.SCORED,
            paper_id=reranker_input.paper_id,
            raw_score=raw_scores[reranker_input.paper_id],
            normalized_score=normalized_scores[reranker_input.paper_id],
            descriptor=descriptor,
            input_sha256=reranker_input.input_sha256,
        )
        for reranker_input in input_by_paper_id.values()
    ]
    records.sort(key=lambda record: (-_scored_raw(record), record.paper_id))
    return RerankRun(state=RerankerRunState.SCORED, records=records)


def _normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise RerankerTaskError("INVALID_INPUT")
    normalized_query = query.replace("\r\n", "\n").strip()
    if not normalized_query:
        raise RerankerTaskError("INVALID_INPUT")
    return normalized_query


def _require_unique_paper_ids(candidates: Sequence[FrozenCandidate]) -> None:
    paper_ids = [candidate.paper_id for candidate in candidates]
    if len(paper_ids) != len(set(paper_ids)):
        raise RerankerTaskError("INVALID_INPUT")


def _validated_descriptor(value: object) -> RerankerModelDescriptor:
    if isinstance(value, RerankerModelDescriptor):
        return RerankerModelDescriptor.model_validate(value.model_dump(mode="python"))
    return RerankerModelDescriptor.model_validate(value)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cache_key(
    query_sha256: str,
    reranker_input: RerankerInput,
    descriptor: RerankerModelDescriptor,
) -> str:
    identity = {
        "query_sha256": query_sha256,
        "paper_id": reranker_input.paper_id,
        "input_sha256": reranker_input.input_sha256,
        "descriptor": descriptor.model_dump(mode="json"),
    }
    canonical_json = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(canonical_json)


def _read_cache_entry(
    path: Path,
    *,
    cache_key: str,
    query_sha256: str,
    reranker_input: RerankerInput,
    descriptor: RerankerModelDescriptor,
) -> float | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload: object = json.load(handle)
        if not isinstance(payload, dict) or set(payload) != _CACHE_KEYS:
            return None
        payload_descriptor = RerankerModelDescriptor.model_validate(payload["descriptor"])
        validated_score = ProviderRawScore.model_validate(
            {"paper_id": payload["paper_id"], "raw_score": payload["raw_score"]}
        )
        if (
            payload["cache_key"] != cache_key
            or payload["paper_id"] != reranker_input.paper_id
            or payload["query_sha256"] != query_sha256
            or payload["input_sha256"] != reranker_input.input_sha256
            or payload_descriptor != descriptor
            or _cache_key(query_sha256, reranker_input, descriptor) != cache_key
        ):
            return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValidationError):
        return None
    return validated_score.raw_score


def _validate_provider_output(
    output: object,
    chunk: Sequence[tuple[RerankerInput, str]],
) -> dict[str, float]:
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        raise RerankerTaskError("INVALID_OUTPUT")
    if len(output) != len(chunk):
        raise RerankerTaskError("INVALID_OUTPUT")
    expected_ids = {reranker_input.paper_id for reranker_input, _ in chunk}
    scores: dict[str, float] = {}
    for item in output:
        try:
            payload = item.model_dump(mode="python") if isinstance(item, ProviderRawScore) else item
            validated = ProviderRawScore.model_validate(payload)
        except (AttributeError, TypeError, ValidationError, ValueError) as error:
            raise RerankerTaskError("INVALID_OUTPUT") from error
        if validated.paper_id in scores:
            raise RerankerTaskError("INVALID_OUTPUT")
        scores[validated.paper_id] = validated.raw_score
    if set(scores) != expected_ids:
        raise RerankerTaskError("INVALID_OUTPUT")
    return scores


def _write_cache_entry(
    path: Path,
    *,
    cache_key: str,
    query_sha256: str,
    reranker_input: RerankerInput,
    descriptor: RerankerModelDescriptor,
    raw_score: float,
) -> None:
    payload = {
        "cache_key": cache_key,
        "paper_id": reranker_input.paper_id,
        "query_sha256": query_sha256,
        "input_sha256": reranker_input.input_sha256,
        "descriptor": descriptor.model_dump(mode="json"),
        "raw_score": raw_score,
    }
    temporary_path = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError):
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _normalize_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    values = list(raw_scores.values())
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return {paper_id: 1.0 for paper_id in raw_scores}
    return {paper_id: (raw_score - minimum) / (maximum - minimum) for paper_id, raw_score in raw_scores.items()}


def _scored_raw(record: RerankRecord) -> float:
    assert record.raw_score is not None
    return record.raw_score
