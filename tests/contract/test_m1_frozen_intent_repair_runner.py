"""Contract tests for the append-only M1 frozen-intent repair runner."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts import repair_m1_frozen_intent_replay as repair

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_CANDIDATE_SNAPSHOT = Path("evaluation/snapshots/m2/m1-candidates.v1.json")
EXPECTED_PROTECTED_CANDIDATE_SNAPSHOT_SHA256 = (
    "4a2aec0fd0a1d22adc801fd3bc506e5da89895d1276cd572e2ac64014c162448"
)


def test_runner_refuses_without_explicit_offline_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert repair.main([]) == 2
    output = capsys.readouterr().out
    assert "REPAIR_EXECUTION_NOT_AUTHORIZED" in output


@pytest.mark.parametrize(
    "relative_path",
    [repair.SOURCE_MANIFEST, repair.SOURCE_FIRST_RUN, repair.SOURCE_REPLAY],
)
def test_mutated_fixed_source_artifact_fails_closed_without_repair_artifact(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    repository = _copy_runner_fixture(tmp_path)
    target = repository / relative_path
    target.write_bytes(target.read_bytes() + b"\nmutation")

    with pytest.raises(repair.RepairError) as raised:
        repair.execute_offline_repair(repository_root=repository)

    assert raised.value.code == "FIXED_HISTORICAL_HASH_MISMATCH"
    assert not (repository / repair.REPAIR_BUNDLE).exists()


def test_non_timestamp_historical_intent_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _copy_runner_fixture(tmp_path)
    original_read_fixed = repair._read_fixed

    def read_with_drift(path: Path, expected_sha256: str) -> bytes:
        data = original_read_fixed(path, expected_sha256)
        if path == repository / repair.SOURCE_REPLAY:
            payload = json.loads(data.decode("utf-8"))
            assert isinstance(payload["intent"], dict)
            payload["intent"]["revision"] += 1
            return json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return data

    monkeypatch.setattr(repair, "_read_fixed", read_with_drift)

    with pytest.raises(repair.RepairError) as raised:
        repair.execute_offline_repair(repository_root=repository)

    assert raised.value.code == "HISTORICAL_INTENT_DRIFT_NOT_EXACT"
    assert not (repository / repair.REPAIR_BUNDLE).exists()


def test_cache_miss_calls_forbidden_transport_and_publishes_nothing(tmp_path: Path) -> None:
    repository = _copy_runner_fixture(tmp_path)

    with pytest.raises(repair.RepairError) as raised:
        repair.execute_offline_repair(
            repository_root=repository,
            cache_dir=tmp_path / "empty-cache",
        )

    assert raised.value.code == "OFFLINE_REPLAY_FAILED"
    assert not (repository / repair.REPAIR_BUNDLE).exists()


def test_first_run_publishes_exactly_three_artifacts_and_second_run_reuses_bytes(
    tmp_path: Path,
) -> None:
    repository = _copy_runner_fixture(tmp_path)

    first_summary = repair.execute_offline_repair(repository_root=repository)
    bundle = repository / repair.REPAIR_BUNDLE
    first_bytes = _bundle_bytes(bundle)

    second_summary = repair.execute_offline_repair(repository_root=repository)
    second_bytes = _bundle_bytes(bundle)

    assert set(first_bytes) == {
        "first-run/first-round.json",
        "replay/first-round.json",
        "repair-manifest.json",
    }
    assert second_bytes == first_bytes
    assert second_summary == first_summary
    assert not [
        path
        for path in bundle.parent.iterdir()
        if path.name.startswith(".") or ".tmp" in path.name.casefold()
    ]


def test_existing_target_conflict_does_not_leave_partial_bundle(tmp_path: Path) -> None:
    repository = _copy_runner_fixture(tmp_path)
    bundle = repository / repair.REPAIR_BUNDLE
    first_target = bundle / "first-run/first-round.json"
    replay_target = bundle / "replay/first-round.json"
    first_target.parent.mkdir(parents=True)
    replay_target.parent.mkdir(parents=True)
    first_target.write_bytes((repository / repair.SOURCE_FIRST_RUN).read_bytes())
    replay_target.write_bytes(b"conflicting target")

    with pytest.raises(repair.RepairError) as raised:
        repair.execute_offline_repair(repository_root=repository)

    assert raised.value.code == "REPAIR_TARGET_CONFLICT"
    assert first_target.read_bytes() == (repository / repair.SOURCE_FIRST_RUN).read_bytes()
    assert replay_target.read_bytes() == b"conflicting target"
    assert not (bundle / "repair-manifest.json").exists()
    assert not [
        path
        for path in bundle.parent.iterdir()
        if path.name.startswith(".") or ".tmp" in path.name.casefold()
    ]


def test_raw_replay_candidate_audit_preserves_abstract_and_protected_snapshot_binding(
    tmp_path: Path,
) -> None:
    repository = _copy_runner_fixture(tmp_path)
    repair.execute_offline_repair(repository_root=repository)

    manifest = _read_json(repository / repair.REPAIR_MANIFEST)
    replay = _read_json(repository / repair.REPAIR_REPLAY)
    snapshot = _read_json(repository / PROTECTED_CANDIDATE_SNAPSHOT)
    audit = manifest["candidate_audit"]

    assert audit["candidate_count"] == 33
    assert audit["candidate_order_equal"] is True
    assert audit["candidate_identity_equal"] is True
    assert audit["candidate_payload_equal"] is False
    assert audit["candidate_delta_fields"] == ["abstract"]
    assert audit["protected_candidate_snapshot_path"] == PROTECTED_CANDIDATE_SNAPSHOT.as_posix()
    assert audit["protected_candidate_snapshot_sha256"] == EXPECTED_PROTECTED_CANDIDATE_SNAPSHOT_SHA256
    assert all(candidate["abstract"] for candidate in replay["candidates"])

    expected = {candidate["paper_id"]: candidate for candidate in snapshot["candidates"]}
    for candidate in replay["candidates"]:
        protected = expected[candidate["paper_id"]]
        assert candidate["title"] == protected["title"]
        assert candidate["abstract"] == protected["abstract"]
        assert candidate["source"] == protected["source"]
        assert candidate["source_id"] == protected["source_id"]


def _copy_runner_fixture(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    source_bundle = REPOSITORY_ROOT / repair.SOURCE_BUNDLE
    target_bundle = repository / repair.SOURCE_BUNDLE
    target_bundle.parent.mkdir(parents=True)
    shutil.copytree(source_bundle, target_bundle)

    snapshot = REPOSITORY_ROOT / PROTECTED_CANDIDATE_SNAPSHOT
    target_snapshot = repository / PROTECTED_CANDIDATE_SNAPSHOT
    target_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot, target_snapshot)
    return repository


def _bundle_bytes(bundle: Path) -> dict[str, bytes]:
    return {
        path.relative_to(bundle).as_posix(): path.read_bytes()
        for path in bundle.rglob("*")
        if path.is_file()
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
