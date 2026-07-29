"""Replaceable reranker-provider protocol for M2-T02."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.models.reranking import ProviderRawScore, RerankerInput, RerankerModelDescriptor


class RerankerProvider(Protocol):
    """A provider that scores already-serialized candidate inputs."""

    @property
    def descriptor(self) -> RerankerModelDescriptor: ...

    def score(
        self,
        query: str,
        inputs: Sequence[RerankerInput],
        *,
        batch_size: int,
    ) -> list[ProviderRawScore]: ...
