"""Run one non-persistent, pinned BGE-M3 construction and dense-vector probe."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.adapters.embedding import BgeM3DenseProvider
from app.models.embedding import BGE_M3_MODEL_ID, BGE_M3_MODEL_REVISION, EmbeddingTaskError
from scripts.embed_frozen_candidates import _validated_model_cache_dir


def run_preflight(provider: BgeM3DenseProvider, *, device: str | None) -> dict[str, object]:
    """Load the pinned model and encode one short text without writing B2 output."""

    vector = provider.embed(["Pinned BGE-M3 preflight."], batch_size=1)[0]
    if len(vector) != 1024:
        raise EmbeddingTaskError("EMBEDDING_DIMENSION_MISMATCH")
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingTaskError("EMBEDDING_NON_FINITE")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise EmbeddingTaskError("EMBEDDING_PROVIDER_FAILED")
    return {
        "status": "success",
        "model_id": BGE_M3_MODEL_ID,
        "model_revision": BGE_M3_MODEL_REVISION,
        "dimension": 1024,
        "provider_library_version": provider.descriptor.provider_library_version,
        "runtime": provider.runtime,
        "device": device,
    }


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--device")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        provider = BgeM3DenseProvider(
            model_id=BGE_M3_MODEL_ID,
            model_revision=BGE_M3_MODEL_REVISION,
            cache_namespace="embedding:bge-m3",
            model_cache_dir=_validated_model_cache_dir(arguments.model_cache_dir),
            device=arguments.device,
        )
        result: dict[str, Any] = run_preflight(provider, device=arguments.device)
    except EmbeddingTaskError as error:
        result = {"status": "failed", "error_code": error.code}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
