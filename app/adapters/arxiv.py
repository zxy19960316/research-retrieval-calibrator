"""Bounded, source-backed arXiv Atom retrieval for M1-T02."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
import xml.etree.ElementTree as element_tree
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.models.paper import PaperRecord
from app.models.query import Query

_ATOM_NAMESPACE = "{http://www.w3.org/2005/Atom}"
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
_MODERN_ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}(?:v[1-9]\d*)?$")
_LEGACY_ARXIV_ID = re.compile(r"[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v[1-9]\d*)?$")
_ARXIV_ID_VERSION = re.compile(r"v\d+$")


class ArxivAdapterError(ValueError):
    """Stable failure surface for an unavailable or invalid source response."""

    def __init__(self, code: str, reason: str | None = None) -> None:
        self.code = code
        self.reason = reason
        super().__init__(code)


class ArxivTransportFailure(OSError):
    """A retryable failure to complete an HTTP transaction."""


class ArxivResponse(NamedTuple):
    status_code: int
    body: bytes
    headers: Mapping[str, str]


class ArxivRequestObservation(NamedTuple):
    attempt_count: int
    http_status: int | None
    retry_after_seconds: float | None
    retry_after_was_capped: bool
    final_error_code: str | None
    elapsed_seconds: float
    cache_hit: bool = False


class ArxivRateLimitObservation(NamedTuple):
    """Safe aggregate timing facts; request URLs are deliberately excluded."""

    configured_min_request_interval_seconds: float
    request_start_offsets_seconds: tuple[float, ...]
    minimum_observed_request_start_delta_seconds: float | None
    rate_limit_wait_count: int
    rate_limit_wait_seconds: float


class ArxivTransport(Protocol):
    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> ArxivResponse: ...


class UrllibArxivTransport:
    """Translate urllib's network exceptions without deciding adapter policy."""

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> ArxivResponse:
        request = Request(url, headers=dict(headers), method="GET")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return ArxivResponse(response.status, response.read(), dict(response.headers.items()))
        except HTTPError as error:
            return ArxivResponse(error.code, error.read(), dict(error.headers.items()))
        except (URLError, OSError) as error:
            raise ArxivTransportFailure(str(error)) from error


class ArxivAdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str = "https://export.arxiv.org/api/query"
    user_agent: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0, le=120)
    page_size: int = Field(ge=1, le=100)
    min_request_interval_seconds: float = Field(ge=0, le=60)
    max_attempts: int = Field(ge=1, le=5)
    max_total_results: int = Field(default=100, ge=1, le=1000)
    max_total_attempts: int = Field(default=20, ge=1, le=100)
    max_retry_after_seconds: float = Field(default=60.0, ge=0, le=300)
    initial_backoff_seconds: float = Field(ge=0, le=60)
    cache_dir: Path | None = None
    cache_schema_version: str = Field(default="m1-t02.v1", min_length=1, max_length=100)
    cache_namespace: str = Field(default="default", min_length=1, max_length=100)


class ArxivAdapter:
    """Fetch bounded first-round arXiv records without inventing metadata."""

    def __init__(
        self,
        config: ArxivAdapterConfig,
        *,
        transport: ArxivTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config
        self._transport = transport or UrllibArxivTransport()
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._utc_now = utc_now
        self._cache: dict[tuple[str, int], list[PaperRecord]] = {}
        self._next_allowed_at: float | None = None
        self._remaining_attempts = config.max_total_attempts
        self._search_started_at = 0.0
        self._last_observation = ArxivRequestObservation(0, None, None, False, None, 0.0)
        self._rate_limit_started_at = monotonic()
        self._request_start_offsets: list[float] = []
        self._rate_limit_wait_count = 0
        self._rate_limit_wait_seconds = 0.0

    @property
    def last_observation(self) -> ArxivRequestObservation:
        """Return safe, URL-free facts from the most recent search call."""
        return self._last_observation

    @property
    def rate_limit_observation(self) -> ArxivRateLimitObservation:
        """Return non-sensitive cumulative timing facts for this adapter instance."""
        offsets = tuple(self._request_start_offsets)
        deltas = [right - left for left, right in pairwise(offsets)]
        return ArxivRateLimitObservation(
            configured_min_request_interval_seconds=self._config.min_request_interval_seconds,
            request_start_offsets_seconds=offsets,
            minimum_observed_request_start_delta_seconds=min(deltas) if deltas else None,
            rate_limit_wait_count=self._rate_limit_wait_count,
            rate_limit_wait_seconds=self._rate_limit_wait_seconds,
        )

    def search_attempt_bound(self, *, max_results: int) -> int:
        """Return the maximum transport attempts one ``search`` call can consume."""
        if max_results < 1 or max_results > self._config.max_total_results:
            raise ArxivAdapterError("INVALID_ARXIV_MAX_RESULTS")
        page_count = (max_results + self._config.page_size - 1) // self._config.page_size
        return min(self._config.max_total_attempts, page_count * self._config.max_attempts)

    def search(self, query: Query, *, max_results: int) -> list[PaperRecord]:
        if max_results < 1 or max_results > self._config.max_total_results:
            raise ArxivAdapterError("INVALID_ARXIV_MAX_RESULTS")
        self._search_started_at = self._monotonic()
        self._remaining_attempts = self._config.max_total_attempts
        self._last_observation = ArxivRequestObservation(0, None, None, False, None, 0.0)
        key = (query.query_text, max_results)
        cached = self._cache.get(key)
        if cached is not None:
            return self._cached_result(cached, query.query_id)
        persistent_cached = self._load_persistent_cache(query.query_text, max_results)
        if persistent_cached is not None:
            self._cache[key] = [paper.model_copy(deep=True) for paper in persistent_cached]
            return self._cached_result(persistent_cached, query.query_id)

        try:
            papers: list[PaperRecord] = []
            seen_ids: set[str] = set()
            start = 0
            while len(papers) < max_results:
                page_size = min(self._config.page_size, max_results - len(papers))
                page = self._parse_atom(
                    self._request(query.query_text, start=start, max_results=page_size).body,
                    query_id=query.query_id,
                )
                if not page:
                    break
                new_records = [paper for paper in page if paper.source_id not in seen_ids]
                if not new_records:
                    break
                for paper in new_records:
                    seen_ids.add(paper.source_id)
                    papers.append(paper)
                    if len(papers) == max_results:
                        break
                if len(page) < page_size:
                    break
                start += page_size
        except ArxivAdapterError as error:
            self._set_final_error(error.code)
            raise
        self._cache[key] = [paper.model_copy(deep=True) for paper in papers]
        self._store_persistent_cache(query.query_text, max_results, papers)
        self._set_elapsed()
        return self._for_query(papers, query.query_id)

    def _cached_result(self, papers: list[PaperRecord], query_id: str) -> list[PaperRecord]:
        self._last_observation = self._last_observation._replace(cache_hit=True)
        self._set_elapsed()
        return self._for_query(papers, query_id)

    def _persistent_cache_path(self, query_text: str, max_results: int) -> Path | None:
        if self._config.cache_dir is None:
            return None
        manifest = self._cache_manifest(query_text, max_results)
        encoded = json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        namespace = re.sub(r"[^A-Za-z0-9._-]+", "_", self._config.cache_namespace).strip("._")
        return self._config.cache_dir / (namespace or "default") / f"{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}.json"

    def _cache_manifest(self, query_text: str, max_results: int) -> dict[str, object]:
        return {
            "adapter_schema_version": self._config.cache_schema_version,
            "cache_namespace": self._config.cache_namespace,
            "endpoint": self._config.endpoint,
            "max_results": max_results,
            "query_text": query_text,
        }

    def _load_persistent_cache(self, query_text: str, max_results: int) -> list[PaperRecord] | None:
        path = self._persistent_cache_path(query_text, max_results)
        if path is None:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("manifest") != self._cache_manifest(
                query_text, max_results
            ):
                return None
            records = payload.get("records")
            if not isinstance(records, list):
                return None
            return [PaperRecord.model_validate(item) for item in records]
        except (OSError, ValueError, ValidationError):
            return None

    def _store_persistent_cache(
        self, query_text: str, max_results: int, papers: list[PaperRecord]
    ) -> None:
        path = self._persistent_cache_path(query_text, max_results)
        if path is None:
            return
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(
                    {
                        "manifest": self._cache_manifest(query_text, max_results),
                        "records": [paper.model_dump(mode="json") for paper in papers],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _request(self, search_query: str, *, start: int, max_results: int) -> ArxivResponse:
        url = f"{self._config.endpoint}?{urlencode({'search_query': search_query, 'start': start, 'max_results': max_results})}"
        retry_not_before: float | None = None
        for attempt in range(self._config.max_attempts):
            if self._remaining_attempts == 0:
                raise ArxivAdapterError("ARXIV_REQUEST_BUDGET_EXHAUSTED")
            self._wait_for_attempt(retry_not_before)
            self._request_start_offsets.append(self._monotonic() - self._rate_limit_started_at)
            self._remaining_attempts -= 1
            try:
                response = self._transport.get(
                    url,
                    headers={"User-Agent": self._config.user_agent},
                    timeout_seconds=self._config.timeout_seconds,
                )
            except ArxivTransportFailure as error:
                self._record_attempt(http_status=None)
                if attempt == self._config.max_attempts - 1:
                    raise ArxivAdapterError("ARXIV_TRANSPORT_ERROR") from error
                retry_not_before = self._monotonic() + self._backoff_seconds(attempt)
                continue

            self._record_attempt(http_status=response.status_code)
            if response.status_code == 200:
                return response
            if response.status_code not in _TRANSIENT_STATUSES or attempt == self._config.max_attempts - 1:
                raise ArxivAdapterError(f"ARXIV_HTTP_{response.status_code}")
            retry_after, retry_after_was_capped = self._retry_after_seconds(response.headers)
            self._last_observation = self._last_observation._replace(
                retry_after_seconds=retry_after,
                retry_after_was_capped=retry_after_was_capped,
            )
            retry_not_before = self._monotonic() + (
                retry_after if retry_after is not None else self._backoff_seconds(attempt)
            )
        raise ArxivAdapterError("ARXIV_TRANSPORT_ERROR")

    def _wait_for_attempt(self, retry_not_before: float | None) -> None:
        targets = [target for target in (self._next_allowed_at, retry_not_before) if target is not None]
        if not targets:
            return
        delay = max(targets) - self._monotonic()
        if delay > 0:
            self._rate_limit_wait_count += 1
            self._rate_limit_wait_seconds += delay
            self._sleeper(delay)

    def _record_attempt(self, *, http_status: int | None) -> None:
        now = self._monotonic()
        self._next_allowed_at = now + self._config.min_request_interval_seconds
        self._last_observation = self._last_observation._replace(
            attempt_count=self._last_observation.attempt_count + 1,
            http_status=http_status,
        )

    def _set_final_error(self, code: str) -> None:
        self._last_observation = self._last_observation._replace(final_error_code=code)
        self._set_elapsed()

    def _set_elapsed(self) -> None:
        self._last_observation = self._last_observation._replace(
            elapsed_seconds=max(0.0, self._monotonic() - self._search_started_at)
        )

    def _backoff_seconds(self, attempt: int) -> float:
        return float(self._config.initial_backoff_seconds * (2**attempt))

    def _retry_after_seconds(self, headers: Mapping[str, str]) -> tuple[float | None, bool]:
        value = next((value for key, value in headers.items() if key.lower() == "retry-after"), None)
        if value is None:
            return None, False
        try:
            parsed_retry_after = max(0.0, float(value))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None, False
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            parsed_retry_after = max(
                0.0, float((parsed - self._utc_now()).total_seconds())
            )
        retry_after_was_capped = parsed_retry_after > self._config.max_retry_after_seconds
        return (
            min(parsed_retry_after, self._config.max_retry_after_seconds),
            retry_after_was_capped,
        )

    @staticmethod
    def _for_query(papers: list[PaperRecord], query_id: str) -> list[PaperRecord]:
        return [paper.model_copy(update={"retrieval_paths": [query_id]}, deep=True) for paper in papers]

    @staticmethod
    def _parse_atom(body: bytes, *, query_id: str) -> list[PaperRecord]:
        try:
            root = element_tree.fromstring(body)
        except element_tree.ParseError as error:
            raise ArxivAdapterError("MALFORMED_ARXIV_ATOM") from error
        if root.tag != f"{_ATOM_NAMESPACE}feed":
            raise ArxivAdapterError("INVALID_ARXIV_ATOM")
        papers: list[PaperRecord] = []
        for entry in root.findall(f"{_ATOM_NAMESPACE}entry"):
            source_url = ArxivAdapter._element_text(entry, "id")
            title = ArxivAdapter._element_text(entry, "title")
            abstract = ArxivAdapter._element_text(entry, "summary")
            if ArxivAdapter._is_error_atom(source_url, title):
                raise ArxivAdapterError("ARXIV_API_ERROR", abstract or None)
            source_id = ArxivAdapter._source_id(source_url)
            if not source_id or not title or not abstract:
                raise ArxivAdapterError("INVALID_ARXIV_ENTRY")
            authors = [
                name
                for author in entry.findall(f"{_ATOM_NAMESPACE}author")
                if (name := ArxivAdapter._element_text(author, "name"))
            ]
            try:
                papers.append(
                    PaperRecord(
                        paper_id=f"arxiv:{source_id}",
                        source="arxiv",
                        source_id=source_id,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        year=ArxivAdapter._year(
                            ArxivAdapter._element_text(entry, "published")
                        ),
                        url=source_url,
                        language="en",
                        retrieval_paths=[query_id],
                    )
                )
            except ValidationError as error:
                raise ArxivAdapterError("INVALID_ARXIV_ENTRY") from error
        return papers

    @staticmethod
    def _is_error_atom(source_url: str, title: str) -> bool:
        parsed = urlparse(source_url)
        return (
            parsed.hostname == "arxiv.org" and parsed.path.startswith("/api/errors")
        ) or title.casefold() == "error"

    @staticmethod
    def _source_id(source_url: str) -> str | None:
        parsed = urlparse(source_url)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname != "arxiv.org"
            or parsed.query
            or parsed.fragment
            or not parsed.path.startswith("/abs/")
        ):
            return None
        raw_id = parsed.path.removeprefix("/abs/")
        if not (_MODERN_ARXIV_ID.fullmatch(raw_id) or _LEGACY_ARXIV_ID.fullmatch(raw_id)):
            return None
        return _ARXIV_ID_VERSION.sub("", raw_id)

    @staticmethod
    def _element_text(element: element_tree.Element, name: str) -> str:
        child = element.find(f"{_ATOM_NAMESPACE}{name}")
        return " ".join(child.text.split()) if child is not None and child.text else ""

    @staticmethod
    def _year(published: str) -> int | None:
        try:
            return int(published[:4])
        except ValueError:
            return None
