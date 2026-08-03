"""Unit tests for the M2-T03 human-pending bundle boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import scripts.build_m2_t03_human_adjudication_bundle as builder
from app.models.m2_t03_human_adjudication import HumanAdjudicationFields


def test_human_adjudication_fields_start_empty() -> None:
    fields = HumanAdjudicationFields()

    assert fields.model_dump(mode="json") == {
        "evidence_slot": None,
        "grounded_reason": None,
        "notes": None,
        "reviewed_at_utc": None,
        "reviewer_id": None,
        "support_level": None,
        "supporting_excerpt": None,
        "verdict": None,
    }


def test_bundle_builder_requires_explicit_offline_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert builder.main([]) == 2
    output = capsys.readouterr().out
    assert json.loads(output)["error_code"] == "BUNDLE_EXECUTION_NOT_AUTHORIZED"


def _publish_targets() -> dict[Path, bytes]:
    return {
        Path("evaluation/source-artifacts/bundle.json"): b"bundle\n",
        Path("evaluation/source-artifacts/receipt.json"): b"receipt\n",
        Path("evaluation/reports/report.json"): b"report\n",
    }


@pytest.mark.parametrize("failure_at", [1, 2, 3])
def test_publish_rolls_back_each_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: int
) -> None:
    real_replace = builder.os.replace
    calls = 0

    def fail_at_selected_call(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("injected failure")
        real_replace(source, destination)

    monkeypatch.setattr(builder.os, "replace", fail_at_selected_call)

    with pytest.raises(builder.BundleError):
        builder._publish_conflict_safe(tmp_path, _publish_targets())

    assert not any(
        path.exists() for path in (
            tmp_path / "evaluation/source-artifacts/bundle.json",
            tmp_path / "evaluation/source-artifacts/receipt.json",
            tmp_path / "evaluation/reports/report.json",
        )
    )
    assert not list(tmp_path.rglob("*.m2-t03-tmp-*"))


def test_publish_reuses_same_bytes_without_modification(tmp_path: Path) -> None:
    targets = _publish_targets()
    for relative_path, data in targets.items():
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    before = {
        relative_path: (tmp_path / relative_path).stat().st_mtime_ns
        for relative_path in targets
    }
    builder._publish_conflict_safe(tmp_path, targets)

    assert all(
        (tmp_path / relative_path).stat().st_mtime_ns == before[relative_path]
        for relative_path in targets
    )


def test_publish_conflicts_without_modifying_existing_different_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation/source-artifacts/bundle.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old\n")
    before = destination.stat().st_mtime_ns

    with pytest.raises(builder.BundleError):
        builder._publish_conflict_safe(
            tmp_path,
            {
                Path("evaluation/source-artifacts/bundle.json"): b"new\n",
                Path("evaluation/reports/report.json"): b"report\n",
            },
        )

    assert destination.read_bytes() == b"old\n"
    assert destination.stat().st_mtime_ns == before
    assert not (tmp_path / "evaluation/reports/report.json").exists()


def test_publish_rejects_parent_file(tmp_path: Path) -> None:
    parent = tmp_path / "evaluation"
    parent.write_bytes(b"not a directory")

    with pytest.raises(builder.BundleError):
        builder._publish_conflict_safe(
            tmp_path, {Path("evaluation/report.json"): b"report\n"}
        )


def test_publish_rejects_symlink_target(tmp_path: Path) -> None:
    target = tmp_path / "bundle.json"
    target.write_bytes(b"target\n")
    link = tmp_path / "published.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable in this Windows environment")

    with pytest.raises(builder.BundleError):
        builder._publish_conflict_safe(tmp_path, {Path("published.json"): b"target\n"})
