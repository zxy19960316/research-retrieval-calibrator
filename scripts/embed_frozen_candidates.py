"""Embed one validated M2 candidate snapshot without ranking or retrieval.

The command deliberately keeps deterministic fake vectors separate from the
real BGE-M3 evidence path.  It is useful for local integration tests, while a
real run remains conditional on the provenance-preserving M1 freeze gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.adapters.embedding import (
    BgeM3DenseProvider,
    DeterministicFakeEmbeddingProvider,
    EmbeddingProvider,
)
from app.core.embedding import build_embedding_text, build_query_embedding_input, embed_inputs
from app.models.embedding import (
    EmbeddingTaskError,
    EmbeddingVectorRecord,
    FrozenCandidate,
    FrozenCandidateSnapshot,
)
from scripts.freeze_m2_candidates import _canonical_json_sha256, validate_frozen_snapshot_bytes

_BGE_VECTOR_SNAPSHOT = "bge-m3-dense-v1.json"
_BGE_VECTOR_MANIFEST = "bge-m3-dense-v1.manifest.json"
_FAKE_VECTOR_SNAPSHOT = "deterministic-fake-dense-v1.json"
_FAKE_VECTOR_MANIFEST = "deterministic-fake-dense-v1.manifest.json"


@dataclass(frozen=True)
class ValidatedFrozenInputs:
    """Snapshot material accepted by the M1 freeze contract."""

    snapshot_sha256: str
    question: str
    candidates: list[FrozenCandidate]


def load_validated_frozen_inputs(
    snapshot_path: Path,
    manifest_path: Path,
    *,
    expected_candidate_count: int = 33,
) -> ValidatedFrozenInputs:
    """Load only a closed, manifest-validated frozen candidate snapshot."""

    try:
        snapshot_bytes = snapshot_path.read_bytes()
        snapshot_payload = json.loads(snapshot_bytes)
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EmbeddingTaskError("FROZEN_SNAPSHOT_MISSING") from error
    if not isinstance(snapshot_payload, dict) or not isinstance(manifest_payload, dict):
        raise EmbeddingTaskError("FROZEN_SNAPSHOT_HASH_MISMATCH")
    if (
        validate_frozen_snapshot_bytes(
            snapshot_bytes,
            manifest_payload,
            expected_candidate_count=expected_candidate_count,
        )
        is not None
    ):
        raise EmbeddingTaskError("FROZEN_SNAPSHOT_HASH_MISMATCH")
    try:
        closed_snapshot = FrozenCandidateSnapshot(
            question=str(snapshot_payload["question"]),
            candidates=[FrozenCandidate.model_validate(value) for value in snapshot_payload["candidates"]],
        )
    except (KeyError, TypeError, ValidationError) as error:
        raise EmbeddingTaskError("FROZEN_SNAPSHOT_HASH_MISMATCH") from error
    return ValidatedFrozenInputs(
        snapshot_sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
        question=closed_snapshot.question,
        candidates=closed_snapshot.candidates,
    )


def embed_frozen_candidates(
    *,
    provider: EmbeddingProvider,
    snapshot_path: Path,
    manifest_path: Path,
    cache_dir: Path,
    output_dir: Path,
    batch_size: int,
    provider_mode: str,
) -> dict[str, Any]:
    """Embed the original question plus every frozen candidate and write JSON.

    This function is intentionally free of provider construction so integration
    tests can exercise the live/replay cache gates entirely with the standard
    library fake implementation.
    """

    if provider_mode not in {"fake", "bge-m3"} or batch_size < 1:
        raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT")
    vector_name, manifest_name = _artifact_names(provider_mode)
    if provider_mode == "fake" and (vector_name == _BGE_VECTOR_SNAPSHOT or manifest_name == _BGE_VECTOR_MANIFEST):
        raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT")

    if provider_mode == "bge-m3":
        _, cache_dir, output_dir = _validate_bge_runtime_paths(
            provider._model_cache_dir if isinstance(provider, BgeM3DenseProvider) else Path("."),
            cache_dir,
            output_dir,
        )
    frozen = load_validated_frozen_inputs(snapshot_path, manifest_path)
    inputs = [build_query_embedding_input(frozen.question, frozen.snapshot_sha256)]
    inputs.extend(build_embedding_text(candidate, frozen.snapshot_sha256) for candidate in frozen.candidates)
    records, stats = embed_inputs(inputs, provider, cache_dir, batch_size=batch_size)

    expected_count = len(frozen.candidates) + 1
    if len(records) != expected_count or stats.cache_misses + stats.cache_hits != expected_count:
        raise EmbeddingTaskError("EMBEDDING_COUNT_MISMATCH")
    payload = {
        "candidate_snapshot_sha256": frozen.snapshot_sha256,
        "input_format_version": provider.descriptor.input_format_version,
        "provider": provider.descriptor.model_dump(mode="json"),
        "records": [record.model_dump(mode="json") for record in records],
        "snapshot_version": "m2-embedding-v1",
    }
    vector_sha256 = _canonical_json_sha256(payload)
    result_manifest = {
        "candidate_count": len(frozen.candidates),
        "candidate_snapshot_sha256": frozen.snapshot_sha256,
        "evidence_type": "deterministic_fake" if provider_mode == "fake" else "real",
        "input_format_version": provider.descriptor.input_format_version,
        "provider": provider.descriptor.model_dump(mode="json"),
        "query_count": 1,
        "stats": stats.model_dump(mode="json"),
        "vector_snapshot_sha256": vector_sha256,
    }
    if provider_mode == "bge-m3":
        runtime = getattr(provider, "runtime", None)
        if not isinstance(runtime, dict):
            raise EmbeddingTaskError("EMBEDDING_PROVIDER_FAILED")
        result_manifest["runtime"] = runtime
        result_manifest["vector_norm_audit"] = _vector_norm_audit(records)
    _write_json(output_dir / vector_name, payload)
    _write_json(output_dir / manifest_name, result_manifest)
    return {
        "manifest_path": str(output_dir / manifest_name),
        "provider": provider_mode,
        "stats": stats.model_dump(mode="json"),
        "status": "success",
        "vector_path": str(output_dir / vector_name),
        "vector_snapshot_sha256": vector_sha256,
    }


def _artifact_names(provider_mode: str) -> tuple[str, str]:
    if provider_mode == "fake":
        return _FAKE_VECTOR_SNAPSHOT, _FAKE_VECTOR_MANIFEST
    if provider_mode == "bge-m3":
        return _BGE_VECTOR_SNAPSHOT, _BGE_VECTOR_MANIFEST
    raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT")


def _vector_norm_audit(records: Sequence[EmbeddingVectorRecord]) -> dict[str, object]:
    norms = [math.sqrt(sum(value * value for value in record.vector)) for record in records]
    return {
        "checked_count": len(norms),
        "non_unit_norm_count": sum(
            not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3) for norm in norms
        ),
        "relative_tolerance": 0.001,
        "absolute_tolerance": 0.001,
    }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise EmbeddingTaskError("EMBEDDING_OUTPUT_WRITE_FAILED") from error


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("fake", "bge-m3"), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--model-revision", "--revision", dest="model_revision")
    parser.add_argument("--cache-namespace")
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--device")
    return parser.parse_args(argv)


def _provider_for_arguments(arguments: argparse.Namespace) -> EmbeddingProvider:
    if arguments.provider == "fake":
        if (
            arguments.model_revision is not None
            or arguments.cache_namespace is not None
            or arguments.model_cache_dir is not None
            or arguments.device is not None
        ):
            raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT")
        return DeterministicFakeEmbeddingProvider()
    if not arguments.model_revision:
        raise EmbeddingTaskError("MODEL_REVISION_UNPINNED")
    if not arguments.cache_namespace or arguments.model_cache_dir is None:
        raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT")
    model_cache_dir, vector_cache_dir, output_dir = _validate_bge_runtime_paths(
        arguments.model_cache_dir,
        arguments.cache_dir,
        arguments.output_dir,
    )
    arguments.model_cache_dir = model_cache_dir
    arguments.cache_dir = vector_cache_dir
    arguments.output_dir = output_dir
    return BgeM3DenseProvider(
        model_id="BAAI/bge-m3",
        model_revision=arguments.model_revision,
        cache_namespace=arguments.cache_namespace,
        model_cache_dir=model_cache_dir,
        device=arguments.device,
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either normalized directory contains the other."""

    try:
        left.relative_to(right)
        return True
    except ValueError:
        try:
            right.relative_to(left)
            return True
        except ValueError:
            return False


def _validated_runtime_cache_dir(path: Path, *, repository_root: Path) -> Path:
    """Allow cache only below .runtime/ when it is inside this repository."""

    resolved = path.resolve()
    root = repository_root.resolve()
    runtime_root = root / ".runtime"
    try:
        resolved.relative_to(root)
    except ValueError:
        return resolved
    try:
        resolved.relative_to(runtime_root)
    except ValueError as error:
        raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT") from error
    return resolved


def _validated_model_cache_dir(path: Path) -> Path:
    """Backward-compatible model-cache validation for the preflight command."""

    return _validated_runtime_cache_dir(path, repository_root=ROOT)


def _validate_bge_runtime_paths(
    model_cache_dir: Path,
    vector_cache_dir: Path,
    output_dir: Path,
    *,
    repository_root: Path = ROOT,
) -> tuple[Path, Path, Path]:
    """Normalize and isolate real-model cache and output destinations."""

    model_cache = _validated_runtime_cache_dir(model_cache_dir, repository_root=repository_root)
    vector_cache = _validated_runtime_cache_dir(vector_cache_dir, repository_root=repository_root)
    output = output_dir.resolve()
    if _paths_overlap(model_cache, vector_cache) or _paths_overlap(output, model_cache) or _paths_overlap(output, vector_cache):
        raise EmbeddingTaskError("INVALID_EMBEDDING_INPUT")
    return model_cache, vector_cache, output


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        provider = _provider_for_arguments(arguments)
        result = embed_frozen_candidates(
            provider=provider,
            snapshot_path=arguments.snapshot,
            manifest_path=arguments.manifest,
            cache_dir=arguments.cache_dir,
            output_dir=arguments.output_dir,
            batch_size=arguments.batch_size,
            provider_mode=arguments.provider,
        )
    except EmbeddingTaskError as error:
        result = {"error_code": error.code, "status": "failed"}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
