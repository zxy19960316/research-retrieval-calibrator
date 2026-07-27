"""External-provider boundaries used by core logic."""

from app.adapters.arxiv import ArxivAdapter, ArxivAdapterConfig, ArxivAdapterError
from app.adapters.llm import LLMProvider, StructuredRequest

__all__ = [
    "ArxivAdapter",
    "ArxivAdapterConfig",
    "ArxivAdapterError",
    "LLMProvider",
    "StructuredRequest",
]
