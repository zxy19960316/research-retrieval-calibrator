"""Deterministic construction of the texts sent to M2 embedding providers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.adapters.embedding import EmbeddingProvider
from app.models.embedding import (
    EmbeddingCacheEntry,
    EmbeddingInput,
    EmbeddingModelDescriptor,
    EmbeddingRunStats,
    EmbeddingTaskError,
    EmbeddingVectorRecord,
    FrozenCandidate,
)

_INPUT_FORMAT_VERSION = "m2-title-abstract-v1"


def build_embedding_text(
    record: FrozenCandidate, source_snapshot_sha256: str
) -> EmbeddingInput:
    """Create the exact, source-backed text representation for one paper."""
    return build_embedding_input_from_fields(
        paper_id=record.paper_id,
        title=record.title,
        abstract=record.abstract,
        source_snapshot_sha256=source_snapshot_sha256,
    )


def build_embedding_input_from_fields(
    *, paper_id: str, title: str, abstract: str | None, source_snapshot_sha256: str
) -> EmbeddingInput:
    """Build the paper input from snapshot fields for generation and validation."""
    normalized_title = title.replace("\r\n", "\n").strip()
    normalized_abstract = (abstract or "").replace("\r\n", "\n").strip()
    if not normalized_title or normalized_abstract.casefold() in {
        "no abstract",
        "not provided",
        "no abstract available",
    }:
        raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT")

    text = f"title:\n{normalized_title}"
    if normalized_abstract:
        text += f"\n\nabstract:\n{normalized_abstract}"
    return EmbeddingInput(
        input_id=paper_id,
        input_kind="paper",
        paper_id=paper_id,
        query_id=None,
        input_format_version=_INPUT_FORMAT_VERSION,
        text=text,
        text_sha256=_sha256_text(text),
        source_snapshot_sha256=source_snapshot_sha256,
    )


def build_query_embedding_input(question: str, source_snapshot_sha256: str) -> EmbeddingInput:
    """Create one deterministic input for the original frozen M1 question."""
    normalized_question = question.replace("\r\n", "\n").strip()
    if not normalized_question:
        raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT")

    text = f"query:\n{normalized_question}"
    text_sha256 = _sha256_text(text)
    return EmbeddingInput(
        input_id=f"query:{text_sha256}",
        input_kind="query",
        paper_id=None,
        query_id=f"query:{text_sha256}",
        input_format_version=_INPUT_FORMAT_VERSION,
        text=text,
        text_sha256=text_sha256,
        source_snapshot_sha256=source_snapshot_sha256,
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_inputs(
    inputs: Sequence[EmbeddingInput],
    provider: EmbeddingProvider,
    cache_dir: Path | str,
    *,
    batch_size: int,
) -> tuple[list[EmbeddingVectorRecord], EmbeddingRunStats]:
    """Embed inputs with a descriptor-scoped, per-text local cache.

    Cache entries deliberately have no input identifier: the same normalized text
    and provider identity can safely be reused for distinct callers, while each
    returned record retains the caller's original ``input_id`` and order.
    """
    if batch_size < 1:
        raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT")

    _require_unique_input_ids(inputs)
    destination = Path(cache_dir)
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise EmbeddingTaskError("EMBEDDING_OUTPUT_WRITE_FAILED") from error

    descriptor = provider.descriptor
    cached_records: dict[str, EmbeddingVectorRecord] = {}
    misses: list[tuple[EmbeddingInput, str]] = []
    pending_inputs: dict[str, list[EmbeddingInput]] = {}
    cache_hits = 0
    cache_misses = 0
    cache_corrupt_count = 0

    # Iterate strictly in caller order.  This also fixes the ordering of each
    # provider chunk, regardless of which input happened to be cached.
    for item in inputs:
        cache_key = _cache_key(descriptor, item.text_sha256)
        cached = _read_cache_entry(destination / f"{cache_key}.json", cache_key, item, descriptor)
        if cached is None:
            cache_misses += 1
            _queue_cache_miss(misses, pending_inputs, item, cache_key)
        elif isinstance(cached, _CorruptCacheEntry):
            cache_corrupt_count += 1
            cache_misses += 1
            _queue_cache_miss(misses, pending_inputs, item, cache_key)
        else:
            cache_hits += 1
            cached_records[item.input_id] = _record_for_input(item, cached, descriptor)

    provider_call_count = 0
    provider_input_count = 0
    for chunk_start in range(0, len(misses), batch_size):
        chunk = misses[chunk_start : chunk_start + batch_size]
        texts = [item.text for item, _ in chunk]
        try:
            raw_vectors = provider.embed(texts, batch_size=batch_size)
        except EmbeddingTaskError:
            raise
        except Exception as error:
            raise EmbeddingTaskError("EMBEDDING_PROVIDER_FAILED") from error

        provider_call_count += 1
        provider_input_count += len(chunk)
        vectors = _validate_provider_vectors(raw_vectors, expected_count=len(chunk), descriptor=descriptor)
        for (item, cache_key), vector in zip(chunk, vectors, strict=True):
            entry = EmbeddingCacheEntry(
                cache_key=cache_key,
                text_sha256=item.text_sha256,
                descriptor=descriptor,
                dimension=descriptor.dimension,
                vector=vector,
            )
            _write_cache_entry(destination / f"{cache_key}.json", entry)
            for pending_item in pending_inputs[cache_key]:
                cached_records[pending_item.input_id] = _record_for_input(
                    pending_item, entry, descriptor
                )

    # Every input was either a validated hit or a newly validated miss.  Using
    # the original input sequence restores caller order even for mixed hits.
    return (
        [cached_records[item.input_id] for item in inputs],
        EmbeddingRunStats(
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_corrupt_count=cache_corrupt_count,
            provider_call_count=provider_call_count,
            provider_input_count=provider_input_count,
        ),
    )


class _CorruptCacheEntry:
    """Internal sentinel that distinguishes a cache miss from a bad cache file."""


def _require_unique_input_ids(inputs: Sequence[EmbeddingInput]) -> None:
    input_ids = [item.input_id for item in inputs]
    if len(input_ids) != len(set(input_ids)):
        raise EmbeddingTaskError("DUPLICATE_EMBEDDING_ID")


def _queue_cache_miss(
    misses: list[tuple[EmbeddingInput, str]],
    pending_inputs: dict[str, list[EmbeddingInput]],
    item: EmbeddingInput,
    cache_key: str,
) -> None:
    """Keep one provider input for each canonical cache identity."""
    if cache_key not in pending_inputs:
        misses.append((item, cache_key))
        pending_inputs[cache_key] = []
    pending_inputs[cache_key].append(item)


def _cache_key(descriptor: EmbeddingModelDescriptor, text_sha256: str) -> str:
    """Return the stable identity for one descriptor/text cache entry.

    ``input_id``, source snapshot provenance, and invocation batch size are
    deliberately excluded so they cannot alter an otherwise identical vector.
    """
    identity = {
        "descriptor": descriptor.model_dump(mode="json"),
        "text_sha256": text_sha256,
    }
    canonical_json = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _read_cache_entry(
    path: Path,
    cache_key: str,
    item: EmbeddingInput,
    descriptor: EmbeddingModelDescriptor,
) -> EmbeddingCacheEntry | _CorruptCacheEntry | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            entry = EmbeddingCacheEntry.model_validate(json.load(handle))
        if (
            entry.cache_key != cache_key
            or entry.text_sha256 != item.text_sha256
            or entry.descriptor != descriptor
            or entry.dimension != descriptor.dimension
            or _cache_key(entry.descriptor, entry.text_sha256) != cache_key
        ):
            return _CorruptCacheEntry()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValidationError):
        return _CorruptCacheEntry()
    return entry


def _write_cache_entry(path: Path, entry: EmbeddingCacheEntry) -> None:
    temporary_path = path.with_suffix(".tmp")
    try:
        payload = entry.model_dump(mode="json")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise EmbeddingTaskError("EMBEDDING_OUTPUT_WRITE_FAILED") from error


def _validate_provider_vectors(
    raw_vectors: Any,
    *,
    expected_count: int,
    descriptor: EmbeddingModelDescriptor,
) -> list[list[float]]:
    if not isinstance(raw_vectors, Sequence) or isinstance(raw_vectors, (str, bytes)):
        raise EmbeddingTaskError("EMBEDDING_COUNT_MISMATCH")
    if len(raw_vectors) != expected_count:
        raise EmbeddingTaskError("EMBEDDING_COUNT_MISMATCH")

    validated: list[list[float]] = []
    for raw_vector in raw_vectors:
        if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, (str, bytes)):
            raise EmbeddingTaskError("EMBEDDING_DIMENSION_MISMATCH")
        if len(raw_vector) != descriptor.dimension:
            raise EmbeddingTaskError("EMBEDDING_DIMENSION_MISMATCH")
        vector: list[float] = []
        for value in raw_vector:
            if isinstance(value, bool):
                raise EmbeddingTaskError("EMBEDDING_NON_FINITE")
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise EmbeddingTaskError("EMBEDDING_NON_FINITE") from error
            if not math.isfinite(number):
                raise EmbeddingTaskError("EMBEDDING_NON_FINITE")
            vector.append(number)
        validated.append(vector)
    return validated


def _record_for_input(
    item: EmbeddingInput,
    entry: EmbeddingCacheEntry,
    descriptor: EmbeddingModelDescriptor,
) -> EmbeddingVectorRecord:
    return EmbeddingVectorRecord(
        input_id=item.input_id,
        text_sha256=item.text_sha256,
        descriptor=descriptor,
        dimension=entry.dimension,
        vector=entry.vector,
    )
