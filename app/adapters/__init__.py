"""External-provider boundaries used by core logic."""

from app.adapters.arxiv import ArxivAdapter, ArxivAdapterConfig, ArxivAdapterError
from app.adapters.llm import LLMProvider, StructuredRequest
from app.adapters.evidence_classification import (
    DeterministicFakeEvidenceClassifier,
    EvidenceClassifierProvider,
)

__all__ = [
    "ArxivAdapter",
    "ArxivAdapterConfig",
    "ArxivAdapterError",
    "DeterministicFakeEvidenceClassifier",
    "EvidenceClassifierProvider",
    "LLMProvider",
    "StructuredRequest",
]
