"""Fake-only fixed-input runner tests for M2-T03."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import scripts.run_m2_t03_evidence_classification as runner
from app.adapters.evidence_classification import (
    DeterministicFakeEvidenceClassifier,
    EvidenceClassifierProvider,
)
from app.models.evidence_classification import EvidenceClassificationError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOT = _REPO_ROOT / runner.CANDIDATE_SNAPSHOT_PATH
_MANIFEST = _REPO_ROOT / runner.CANDIDATE_MANIFEST_PATH
_RERANK_RESULT = _REPO_ROOT / runner.RERANKER_RESULT_PATH
_RERANK_RECEIPT = _REPO_ROOT / runner.RERANKER_RECEIPT_PATH


def _paths(tmp_path: Path) -> runner.EvidenceClassificationRunPaths:
    snapshot = tmp_path / "m1-candidates.v1.json"
    manifest = tmp_path / "m1-candidates.v1.manifest.json"
    rerank_result = tmp_path / "m2-t02-reranker-candidate-run.json"
    rerank_receipt = tmp_path / "m2-t02-reranker-candidate-run-receipt.json"
    shutil.copyfile(_SNAPSHOT, snapshot)
    shutil.copyfile(_MANIFEST, manifest)
    shutil.copyfile(_RERANK_RESULT, rerank_result)
    shutil.copyfile(_RERANK_RECEIPT, rerank_receipt)
    return runner.EvidenceClassificationRunPaths(
        candidate_snapshot_path=snapshot,
        candidate_manifest_path=manifest,
        reranker_result_path=rerank_result,
        reranker_receipt_path=rerank_receipt,
        result_path=tmp_path / "m2-t03-evidence-classification-run.json",
        receipt_path=tmp_path / "m2-t03-evidence-classification-run-receipt.json",
    )


def _run(
    paths: runner.EvidenceClassificationRunPaths,
    *,
    provider_factory: Any | None = None,
    publish_result: bool = False,
) -> dict[str, object]:
    return runner.run_m2_t03_evidence_classification(
        execute_deterministic_fake=True,
        paths=paths,
        provider_factory=provider_factory,
        publish_result=publish_result,
    )


def test_fixed_run_classifies_all_33_candidates_without_scores_or_selection_fields(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    report = _run(paths)
    runner.validate_classification_run_report(report)
    assert report["task_id"] == "M2-T03"
    assert report["input"]["candidate_count"] == 33  # type: ignore[index]
    assert report["execution"]["record_count"] == 33  # type: ignore[index]
    assert report["execution"]["rejected_count"] == 0  # type: ignore[index]
    serialized = json.dumps(report, ensure_ascii=False)
    assert all(
        forbidden not in serialized
        for forbidden in ('"score"', '"weight"', '"total_score"', '"selection_rank"', '"rank"')
    )
    assert not paths.result_path.exists()
    assert not paths.receipt_path.exists()


def test_publish_is_byte_idempotent_and_conflicts_do_not_replace(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    report = _run(paths, publish_result=True)
    assert paths.result_path.is_file()
    assert paths.receipt_path.is_file()
    result_bytes = paths.result_path.read_bytes()
    receipt_bytes = paths.receipt_path.read_bytes()
    result_mtime = paths.result_path.stat().st_mtime_ns
    receipt_mtime = paths.receipt_path.stat().st_mtime_ns

    _run(paths, publish_result=True)
    assert paths.result_path.read_bytes() == result_bytes
    assert paths.receipt_path.read_bytes() == receipt_bytes
    assert paths.result_path.stat().st_mtime_ns == result_mtime
    assert paths.receipt_path.stat().st_mtime_ns == receipt_mtime
    assert not list(tmp_path.glob("*.tmp-*"))

    paths.result_path.write_bytes(b"different\n")
    with pytest.raises(EvidenceClassificationError) as captured:
        runner.publish_classification_run(report, destination=paths.result_path)
    assert captured.value.code == "RESULT_CONFLICT"
    assert paths.result_path.read_bytes() == b"different\n"


def test_provider_failure_leaves_no_formal_artifact_or_temp_file(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    class _FailingProvider:
        descriptor = DeterministicFakeEvidenceClassifier().descriptor

        def classify(self, inputs: list[object]) -> list[object]:
            del inputs
            raise RuntimeError("raw provider detail must not be persisted")

    with pytest.raises(EvidenceClassificationError) as captured:
        _run(paths, provider_factory=lambda _descriptor: _FailingProvider(), publish_result=True)
    assert captured.value.code == "PROVIDER_UNAVAILABLE"
    assert not paths.result_path.exists()
    assert not paths.receipt_path.exists()
    assert not list(tmp_path.glob("*.tmp-*"))


def test_fixed_input_drift_fails_before_provider_construction(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.candidate_snapshot_path.write_bytes(paths.candidate_snapshot_path.read_bytes() + b" ")
    constructed = False

    def provider_factory(_descriptor: object) -> EvidenceClassifierProvider:
        nonlocal constructed
        constructed = True
        raise AssertionError("provider must not be constructed after input drift")

    with pytest.raises(EvidenceClassificationError) as captured:
        _run(paths, provider_factory=provider_factory)
    assert captured.value.code == "FIXED_INPUT_INVALID"
    assert constructed is False


def test_provider_receives_only_title_abstract_input_not_reranker_data(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    observed: list[object] = []

    class _ObservingProvider:
        descriptor = DeterministicFakeEvidenceClassifier().descriptor

        def classify(self, inputs: Any) -> list[object]:
            observed.extend(inputs)
            return DeterministicFakeEvidenceClassifier().classify(inputs)

    _run(paths, provider_factory=lambda _descriptor: _ObservingProvider())
    assert observed
    assert all(
        not hasattr(item, field)
        for item in observed
        for field in ("raw_score", "normalized_score", "rank", "selection_rank", "total_score")
    )


def test_runner_module_import_does_not_load_optional_model_runtime() -> None:
    script = (
        "import scripts.run_m2_t03_evidence_classification; import sys; "
        "blocked = {'torch', 'transformers', 'huggingface_hub', 'safetensors', 'tokenizers'}; "
        "raise SystemExit(bool({name.split('.', 1)[0] for name in sys.modules} & blocked))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_invalid_cli_has_no_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fail_if_called(**_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("invalid CLI vector reached the runner")

    monkeypatch.setattr(runner, "run_m2_t03_evidence_classification", fail_if_called)
    assert runner.main([]) == 2
    assert runner.main(["--execute-deterministic-fake", "--unknown"]) == 2
    assert called is False
