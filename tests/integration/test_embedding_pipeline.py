"""Red-first fake-only integration expectations for M2 embedding orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.adapters import embedding as embedding_adapter
from app.adapters.embedding import BgeM3DenseProvider, DeterministicFakeEmbeddingProvider
from app.core.embedding import embed_inputs
from app.models.dedup import SourceIdentity
from app.models.embedding import (
    EmbeddingInput,
    EmbeddingModelDescriptor,
    EmbeddingTaskError,
    FrozenCandidate,
)
from scripts.embed_frozen_candidates import embed_frozen_candidates, main
from scripts.freeze_m2_candidates import _canonical_json_sha256, _render_json_bytes


@pytest.fixture
def fake_descriptor() -> EmbeddingModelDescriptor:
    return EmbeddingModelDescriptor(provider_name="deterministic_fake", model_id="sha256-vector", model_revision="fake-v1", provider_library="stdlib", provider_library_version="3.12", embedding_mode="dense", input_format_version="m2-title-abstract-v1", normalized=True, dimension=16, cache_namespace="embedding:fake")


@pytest.fixture
def mixed_inputs() -> list[EmbeddingInput]:
    texts = ["title:\nQuery input", "title:\nSource-backed paper", "title:\nTitle-only paper"]
    return [EmbeddingInput(input_id=f"input-{index}", input_kind="query" if index == 0 else "paper", paper_id=None if index == 0 else f"arxiv:2401.0000{index}", query_id="Q1" if index == 0 else None, input_format_version="m2-title-abstract-v1", text=text, text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), source_snapshot_sha256="a" * 64) for index, text in enumerate(texts)]


@pytest.mark.parametrize("batch_size", [1, 2, 3, 10])
def test_mixed_cache_hit_miss_restores_input_order(tmp_path: Path, fake_descriptor: EmbeddingModelDescriptor, mixed_inputs: list[EmbeddingInput], batch_size: int) -> None:
    provider = DeterministicFakeEmbeddingProvider(fake_descriptor)
    _, _ = embed_inputs(mixed_inputs[:2], provider, tmp_path, batch_size=batch_size)
    vectors, stats = embed_inputs(mixed_inputs, provider, tmp_path, batch_size=batch_size)
    assert [vector.input_id for vector in vectors] == [item.input_id for item in mixed_inputs]
    assert stats.cache_hits == 2 and stats.cache_misses == 1 and stats.provider_call_count == 1


def _write_validated_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    candidates: list[dict[str, object]] = []
    for index in range(33):
        source_id = f"2401.{index:05d}"
        candidate = FrozenCandidate(
            paper_id=f"arxiv:{source_id}",
            source="arxiv",
            source_id=source_id,
            title=f"Source-backed candidate {index}",
            abstract=None if index == 0 else f"Source-backed abstract {index}",
            authors=["A. Author"],
            year=2024,
            doi=None,
            url=f"https://arxiv.org/abs/{source_id}",
            language="en",
            categories=[],
            retrieval_paths=["Q1"],
            cluster_id=f"cluster:{source_id}",
            member_source_identities=[SourceIdentity(source="arxiv", source_id=source_id, url=f"https://arxiv.org/abs/{source_id}")],
        )
        candidates.append(candidate.model_dump(mode="json"))
    snapshot: dict[str, object] = {
        "candidates": candidates,
        "count": 33,
        "question": "How can graph-based retrieval support scientific literature discovery?",
        "snapshot_version": "m2-candidates.v1",
        "source_evidence_report": "evaluation/reports/m1-validation.json",
        "source_evidence_report_sha256": "a" * 64,
        "source_evidence_baseline_commit": "b" * 40,
        "source_candidate_array_sha256": "c" * 64,
        "source_merge_commit": "d" * 40,
        "source_output_sha256": "e" * 64,
        "validated_implementation_commit": "f" * 40,
    }
    manifest = {
        "candidate_count": 33,
        "implementation_ancestry": ["f" * 40],
        "m1_completion_merge_commit": "d" * 40,
        "m1_evidence_baseline_commit": "b" * 40,
        "metadata_mismatch_count": 0,
        "source_evidence_report": "evaluation/reports/m1-validation.json",
        "source_evidence_report_sha256": "a" * 64,
        "source_id_coverage": 1.0,
        "url_coverage": 1.0,
        "validated_implementation_commit": "f" * 40,
        "candidate_identity_sha256": _canonical_json_sha256(
            [(item["paper_id"], item["source"], item["source_id"]) for item in candidates]
        ),
        "source_identity_set_sha256": _canonical_json_sha256(
            sorted((item["source"], item["source_id"]) for item in candidates)
        ),
        "zero_transport_replay": {
            "cache_hits": 12,
            "empty_cache_entry_count": 0,
            "query_count": 12,
            "transport_requests": 0,
        },
        "artifact_source_classification": "EXACT_HISTORICAL_ARTIFACTS_RECOVERED",
        "source_bundle": None,
    }
    snapshot_path = tmp_path / "m1-candidates.v1.json"
    manifest_path = tmp_path / "m1-candidates.v1.manifest.json"
    snapshot_bytes = _render_json_bytes(snapshot)
    manifest["snapshot_sha256"] = hashlib.sha256(snapshot_bytes).hexdigest()
    snapshot_path.write_bytes(snapshot_bytes)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot_path, manifest_path


def test_fake_cli_live_then_replay_has_required_cache_stats_and_never_writes_bge_artifacts(
    tmp_path: Path,
) -> None:
    snapshot_path, manifest_path = _write_validated_snapshot(tmp_path)
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    provider = DeterministicFakeEmbeddingProvider()

    live = embed_frozen_candidates(
        provider=provider,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        batch_size=3,
        provider_mode="fake",
    )
    replay = embed_frozen_candidates(
        provider=provider,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        batch_size=1,
        provider_mode="fake",
    )

    assert live["stats"] == {
        "cache_corrupt_count": 0,
        "cache_hits": 0,
        "cache_misses": 34,
        "provider_call_count": 12,
        "provider_input_count": 34,
    }
    assert replay["stats"] == {
        "cache_corrupt_count": 0,
        "cache_hits": 34,
        "cache_misses": 0,
        "provider_call_count": 0,
        "provider_input_count": 0,
    }
    assert live["vector_snapshot_sha256"] == replay["vector_snapshot_sha256"]
    assert (output_dir / "deterministic-fake-dense-v1.json").is_file()
    assert not (output_dir / "bge-m3-dense-v1.json").exists()
    assert not (output_dir / "bge-m3-dense-v1.manifest.json").exists()


def test_bge_cli_requires_an_immutable_revision_and_distinct_namespace(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    snapshot_path, manifest_path = _write_validated_snapshot(tmp_path)
    base_arguments = [
        "--provider", "bge-m3", "--snapshot", str(snapshot_path), "--manifest", str(manifest_path),
        "--cache-dir", str(tmp_path / "cache"), "--output-dir", str(tmp_path / "outputs"), "--batch-size", "1",
    ]
    assert main(base_arguments) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "MODEL_REVISION_UNPINNED"

    assert main(base_arguments + ["--model-revision", "main", "--cache-namespace", "embedding:real", "--model-cache-dir", str(tmp_path / "model")]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "MODEL_REVISION_UNPINNED"

    assert main(base_arguments + ["--model-revision", "0123456789abcdef", "--cache-namespace", "embedding:fake", "--model-cache-dir", str(tmp_path / "model")]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "MODEL_REVISION_UNPINNED"


def test_cli_separates_model_cache_from_vector_cache_and_validates_devices(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot_path, manifest_path = _write_validated_snapshot(tmp_path)
    base = [
        "--snapshot", str(snapshot_path), "--manifest", str(manifest_path),
        "--cache-dir", str(tmp_path / "vectors"), "--output-dir", str(tmp_path / "outputs"), "--batch-size", "1",
    ]
    assert main(["--provider", "fake", *base, "--model-cache-dir", str(tmp_path / "model")]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_EMBEDDING_INPUT"

    assert main(["--provider", "bge-m3", *base, "--model-revision", "5617a9f61b028005a4858fdac845db406aefb181", "--cache-namespace", "embedding:bge-m3"]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_EMBEDDING_INPUT"

    assert main(["--provider", "bge-m3", *base, "--model-revision", "5617a9f61b028005a4858fdac845db406aefb181", "--cache-namespace", "embedding:bge-m3", "--model-cache-dir", str(tmp_path / "model"), "--device", "cuda:1"]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_EMBEDDING_INPUT"

    assert main(["--provider", "bge-m3", *base, "--model-revision", "5617a9f61b028005a4858fdac845db406aefb181", "--cache-namespace", "embedding:bge-m3", "--model-cache-dir", "evaluation/snapshots/m2/model-cache"]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_EMBEDDING_INPUT"


def test_bge_manifest_records_complete_runtime_identity_with_a_stubbed_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_path, manifest_path = _write_validated_snapshot(tmp_path)
    versions = {
        "FlagEmbedding": "1.3.5",
        "torch": "2.4.1",
        "transformers": "4.45.2",
        "huggingface-hub": "0.25.2",
        "numpy": "2.1.1",
    }
    monkeypatch.setattr(embedding_adapter, "version", versions.__getitem__)
    provider = BgeM3DenseProvider(
        model_id="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        cache_namespace="embedding:bge-m3",
        model_cache_dir=tmp_path / "model-cache",
    )
    provider._model = type(
        "Wrapper",
        (),
        {"encode": lambda _self, texts, **_kwargs: {"dense_vecs": [[1.0 / 32] * 1024 for _ in texts]}},
    )()
    provider._torch = type("Torch", (), {"inference_mode": lambda _self: _NoOpContext()})()

    result = embed_frozen_candidates(
        provider=provider,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        cache_dir=tmp_path / "vector-cache",
        output_dir=tmp_path / "outputs",
        batch_size=8,
        provider_mode="bge-m3",
    )

    manifest = json.loads(Path(str(result["manifest_path"])).read_text(encoding="utf-8"))
    assert manifest["evidence_type"] == "real"
    assert manifest["runtime"] == provider.runtime


class _NoOpContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def test_bge_cli_reports_a_lazy_optional_provider_failure_without_writing_a_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_path, manifest_path = _write_validated_snapshot(tmp_path)

    def unavailable(_: BgeM3DenseProvider) -> None:
        raise EmbeddingTaskError("EMBEDDING_PROVIDER_UNAVAILABLE")

    monkeypatch.setattr(BgeM3DenseProvider, "_load_model_if_needed", unavailable)
    monkeypatch.setattr(
        "app.adapters.embedding.version",
        lambda package: {
            "FlagEmbedding": "1.3.5",
            "torch": "2.4.1",
            "transformers": "4.45.2",
            "huggingface-hub": "0.25.2",
            "numpy": "2.1.1",
        }[package],
    )
    output_dir = tmp_path / "outputs"
    assert main([
        "--provider", "bge-m3", "--snapshot", str(snapshot_path), "--manifest", str(manifest_path),
        "--cache-dir", str(tmp_path / "cache"), "--output-dir", str(output_dir), "--batch-size", "1",
        "--model-revision", "5617a9f61b028005a4858fdac845db406aefb181", "--cache-namespace", "embedding:real", "--model-cache-dir", str(tmp_path / "model"),
    ]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "EMBEDDING_PROVIDER_UNAVAILABLE"
    assert not (output_dir / "bge-m3-dense-v1.json").exists()
