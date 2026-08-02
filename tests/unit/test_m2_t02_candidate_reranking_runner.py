"""Fake-only contracts for the future fixed-candidate B3-I runner."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

import scripts.run_m2_t02_candidate_reranking as runner
from app.models.reranking import ProviderRawScore, RerankerModelDescriptor

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOT = _REPO_ROOT / "evaluation/snapshots/m2/m1-candidates.v1.json"
_MANIFEST = _REPO_ROOT / "evaluation/snapshots/m2/m1-candidates.v1.manifest.json"
_PREFLIGHT = _REPO_ROOT / "evaluation/source-artifacts/m2-t02-reranker-preflight.json"
_RECEIPT = _REPO_ROOT / "evaluation/source-artifacts/m2-t02-reranker-preflight-run-receipt.json"


def _paths(tmp_path: Path) -> runner.CandidateRunPaths:
    snapshot = tmp_path / "m1-candidates.v1.json"
    manifest = tmp_path / "m1-candidates.v1.manifest.json"
    preflight = tmp_path / "m2-t02-reranker-preflight.json"
    receipt = tmp_path / "m2-t02-reranker-preflight-run-receipt.json"
    shutil.copyfile(_SNAPSHOT, snapshot)
    shutil.copyfile(_MANIFEST, manifest)
    shutil.copyfile(_PREFLIGHT, preflight)
    shutil.copyfile(_RECEIPT, receipt)
    model_snapshot = tmp_path / "model-snapshot"
    model_snapshot.mkdir()
    return runner.CandidateRunPaths(
        candidate_snapshot_path=snapshot,
        candidate_manifest_path=manifest,
        preflight_evidence_path=preflight,
        preflight_receipt_path=receipt,
        model_snapshot_path=model_snapshot,
        result_path=tmp_path / "m2-t02-reranker-candidate-run.json",
    )


class _FakeProvider:
    instances: ClassVar[int] = 0

    def __init__(self, *, fail_on_call: int | None = None, equal_scores: bool = False) -> None:
        type(self).instances += 1
        self.descriptor = RerankerModelDescriptor(
            provider_name="synthetic",
            model_id="synthetic-cross-encoder",
            model_revision="a" * 40,
            provider_library="tests",
            provider_library_version="1",
            input_format_version="m2-reranker-title-abstract-v1",
            cache_namespace="reranker:synthetic-b3-i",
        )
        self.fail_on_call = fail_on_call
        self.equal_scores = equal_scores
        self.calls = 0

    def score(self, query: str, inputs: list[Any], *, batch_size: int) -> list[ProviderRawScore]:
        assert query == runner.FIXED_QUERY
        assert batch_size == 2
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("fake provider failure")
        return [
            ProviderRawScore(
                paper_id=item.paper_id,
                raw_score=1.0 if self.equal_scores else float(100 - index),
            )
            for index, item in enumerate(inputs)
        ]


def _run(
    paths: runner.CandidateRunPaths,
    provider: _FakeProvider,
    *,
    cache_factory: Any | None = None,
) -> dict[str, object]:
    return runner.run_candidate_reranking(
        execute_local_reranking=True,
        paths=paths,
        provider_factory=lambda _snapshot: provider,
        cache_factory=cache_factory,
        publish_result=False,
    )


def test_importing_runner_does_not_import_runtime_packages() -> None:
    script = (
        "import scripts.run_m2_t02_candidate_reranking; import sys; "
        "blocked = {'torch', 'transformers', 'huggingface_hub', 'safetensors', 'tokenizers'}; "
        "raise SystemExit(bool({name.split('.', 1)[0] for name in sys.modules} & blocked))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--execute-local-reranking", "--execute-local-reranking"],
        ["--query", runner.FIXED_QUERY],
        ["--execute-local-reranking", "--batch-size", "2"],
        ["--unknown"],
    ],
)
def test_invalid_cli_vectors_have_zero_side_effects(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    called = False

    def fail_if_called(**_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("invalid CLI vector reached the runner")

    monkeypatch.setattr(runner, "run_candidate_reranking", fail_if_called)
    assert runner.main(argv) == 2
    assert called is False


def test_fixed_run_has_exact_batches_single_provider_zero_hits_and_no_formal_result(
    tmp_path: Path,
) -> None:
    _FakeProvider.instances = 0
    paths = _paths(tmp_path)
    provider = _FakeProvider()
    report = _run(paths, provider)

    assert _FakeProvider.instances == 1
    assert provider.calls == 17
    assert report["input"]["candidate_count"] == 33  # type: ignore[index]
    assert report["policy"]["configured_top_k"] == 50  # type: ignore[index]
    assert report["policy"]["effective_top_k"] == 33  # type: ignore[index]
    assert report["execution"]["provider_call_count"] == 17  # type: ignore[index]
    assert report["execution"]["provider_scored_count"] == 33  # type: ignore[index]
    assert report["execution"]["batch_sizes"] == [2] * 16 + [1]  # type: ignore[index]
    assert report["execution"]["provider_instance_count"] == 1  # type: ignore[index]
    assert report["execution"]["runtime_load_count"] == 1  # type: ignore[index]
    assert report["execution"]["cache_hits"] == 0  # type: ignore[index]
    assert not paths.result_path.exists()


def test_equal_scores_normalize_to_one_and_report_contains_only_closed_record_fields(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    report = _run(paths, _FakeProvider(equal_scores=True))

    runner.validate_candidate_run_report(report)
    records = report["records"]
    assert isinstance(records, list)
    assert len(records) == 33
    assert all(record["normalized_score"] == 1.0 for record in records)
    assert all(
        set(record) == {"rank", "paper_id", "input_sha256", "raw_score", "normalized_score"}
        for record in records
    )
    serialized = json.dumps(report, ensure_ascii=False)
    assert all("title" not in record and "abstract" not in record for record in records)
    assert "https://" not in serialized


def test_provider_failure_has_no_partial_report_or_cache(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    owned_cache = tmp_path / "owned-cache"

    def cache_factory() -> Path:
        owned_cache.mkdir()
        return owned_cache

    with pytest.raises(runner.CandidateRerankingError) as captured:
        _run(paths, _FakeProvider(fail_on_call=5), cache_factory=cache_factory)
    assert captured.value.code == "PROVIDER_UNAVAILABLE"
    assert not owned_cache.exists()
    assert not paths.result_path.exists()


@pytest.mark.parametrize("mutation", ["snapshot", "manifest-count", "manifest-order", "query"])
def test_fixed_input_drift_fails_before_provider_construction(
    tmp_path: Path, mutation: str
) -> None:
    paths = _paths(tmp_path)
    if mutation == "snapshot":
        paths.candidate_snapshot_path.write_bytes(paths.candidate_snapshot_path.read_bytes() + b" ")
    elif mutation.startswith("manifest"):
        payload = json.loads(paths.candidate_manifest_path.read_text(encoding="utf-8"))
        if mutation == "manifest-count":
            payload["candidate_count"] = 32
        else:
            payload["paper_id_order"] = list(reversed(payload["paper_id_order"]))
        paths.candidate_manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        payload = json.loads(paths.candidate_snapshot_path.read_text(encoding="utf-8"))
        payload["question"] = "different query"
        paths.candidate_snapshot_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    constructed = False

    def provider_factory(_snapshot: Path) -> _FakeProvider:
        nonlocal constructed
        constructed = True
        raise AssertionError("provider was constructed after fixed-input drift")

    with pytest.raises(runner.CandidateRerankingError):
        runner.run_candidate_reranking(
            execute_local_reranking=True,
            paths=paths,
            provider_factory=provider_factory,
            publish_result=False,
        )
    assert constructed is False


def test_candidate_input_sha_is_built_by_existing_core_serializer(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    report = _run(paths, _FakeProvider())
    ids = [record["paper_id"] for record in report["records"]]  # type: ignore[index]
    hashes = [record["input_sha256"] for record in report["records"]]  # type: ignore[index]
    assert len(ids) == len(set(ids)) == 33
    assert all(isinstance(value, str) and len(value) == 64 for value in hashes)
    assert all(value == value.lower() for value in hashes)
    assert hashlib.sha256(b"title:\n") .hexdigest() not in hashes


def test_runner_module_exports_fixed_paths_and_public_validator() -> None:
    imported = importlib.import_module("scripts.run_m2_t02_candidate_reranking")
    assert imported.FORMAL_RESULT_PATH.as_posix() == (
        "evaluation/source-artifacts/m2-t02-reranker-candidate-run.json"
    )
    assert callable(imported.validate_candidate_run_report)
