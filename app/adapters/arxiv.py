"""Bounded, source-backed arXiv Atom retrieval for M1-T02."""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as element_tree
from collections.abc import Callable, Mapping
from typing import NamedTuple, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from app.models.paper import PaperRecord
from app.models.query import Query

_ATOM_NAMESPACE = "{http://www.w3.org/2005/Atom}"
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
_ARXIV_ID_VERSION = re.compile(r"v\d+$")


class ArxivAdapterError(ValueError):
    """Stable failure surface for an unavailable or invalid source response."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ArxivResponse(NamedTuple):
    status_code: int
    body: bytes
    headers: Mapping[str, str]


class ArxivTransport(Protocol):
    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> ArxivResponse: ...


class UrllibArxivTransport:
    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> ArxivResponse:
        request = Request(url, headers=dict(headers), method="GET")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return ArxivResponse(response.status, response.read(), dict(response.headers.items()))
        except HTTPError as error:
            return ArxivResponse(error.code, error.read(), dict(error.headers.items()))
        except URLError as error:
            raise ArxivAdapterError("ARXIV_TRANSPORT_ERROR") from error


class ArxivAdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str = "https://export.arxiv.org/api/query"
    user_agent: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0, le=120)
    page_size: int = Field(ge=1, le=100)
    min_request_interval_seconds: float = Field(ge=0, le=60)
    max_attempts: int = Field(ge=1, le=5)
    initial_backoff_seconds: float = Field(ge=0, le=60)


class ArxivAdapter:
    """Fetch bounded first-round arXiv records without inventing any metadata."""

    def __init__(
        self,
        config: ArxivAdapterConfig,
        *,
        transport: ArxivTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._transport = transport or UrllibArxivTransport()
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._cache: dict[tuple[str, int], list[PaperRecord]] = {}
        self._last_request_at: float | None = None

    def search(self, query: Query, *, max_results: int) -> list[PaperRecord]:
        if max_results < 1:
            raise ArxivAdapterError("INVALID_ARXIV_MAX_RESULTS")
        key = (query.query_text, max_results)
        cached = self._cache.get(key)
        if cached is not None:
            return [paper.model_copy(deep=True) for paper in cached]

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
        self._cache[key] = [paper.model_copy(deep=True) for paper in papers]
        return papers

    def _request(self, search_query: str, *, start: int, max_results: int) -> ArxivResponse:
        url = f"{self._config.endpoint}?{urlencode({'search_query': search_query, 'start': start, 'max_results': max_results})}"
        for attempt in range(self._config.max_attempts):
            if attempt == 0:
                self._wait_for_rate_limit()
            try:
                response = self._transport.get(
                    url,
                    headers={"User-Agent": self._config.user_agent},
                    timeout_seconds=self._config.timeout_seconds,
                )
            except ArxivAdapterError:
                raise
            except (OSError, TimeoutError) as error:
                if attempt == self._config.max_attempts - 1:
                    raise ArxivAdapterError("ARXIV_TRANSPORT_ERROR") from error
                self._backoff(attempt)
                continue
            self._last_request_at = self._monotonic()
            if response.status_code == 200:
                return response
            if response.status_code not in _TRANSIENT_STATUSES or attempt == self._config.max_attempts - 1:
                raise ArxivAdapterError(f"ARXIV_HTTP_{response.status_code}")
            self._backoff(attempt)
        raise ArxivAdapterError("ARXIV_TRANSPORT_ERROR")

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_at is not None:
            delay = self._config.min_request_interval_seconds - (
                self._monotonic() - self._last_request_at
            )
            if delay > 0:
                self._sleeper(delay)

    def _backoff(self, attempt: int) -> None:
        delay = self._config.initial_backoff_seconds * (2**attempt)
        if delay > 0:
            self._sleeper(delay)

    @staticmethod
    def _parse_atom(body: bytes, *, query_id: str) -> list[PaperRecord]:
        try:
            root = element_tree.fromstring(body)
        except element_tree.ParseError as error:
            raise ArxivAdapterError("MALFORMED_ARXIV_ATOM") from error
        papers: list[PaperRecord] = []
        for entry in root.findall(f"{_ATOM_NAMESPACE}entry"):
            source_url = ArxivAdapter._element_text(entry, "id")
            title = ArxivAdapter._element_text(entry, "title")
            abstract = ArxivAdapter._element_text(entry, "summary")
            if not source_url or not title or not abstract or not source_url.startswith(("http://", "https://")):
                raise ArxivAdapterError("INVALID_ARXIV_ENTRY")
            source_id = _ARXIV_ID_VERSION.sub("", source_url.rstrip("/").rsplit("/", 1)[-1])
            if not source_id:
                raise ArxivAdapterError("INVALID_ARXIV_ENTRY")
            authors = [
                name
                for author in entry.findall(f"{_ATOM_NAMESPACE}author")
                if (name := ArxivAdapter._element_text(author, "name"))
            ]
            papers.append(
                PaperRecord(
                    paper_id=f"arxiv:{source_id}", source="arxiv", source_id=source_id,
                    title=title, abstract=abstract, authors=authors,
                    year=ArxivAdapter._year(ArxivAdapter._element_text(entry, "published")),
                    url=source_url, language="en", retrieval_paths=[query_id],
                )
            )
        return papers

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
