"""Contract tests for alternate-root M1 replay repair validation."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts import validate_m1_frozen_intent_replay_repair as validator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = validator.REPORT_PATH


def test_source_bundle_inventory_is_public_and_complete(tmp_path: Path) -> None:
    repository = _copy_validation_fixture(tmp_path)

    audit = validator.validate_source_bundle_inventory(repository)

    assert audit.valid is True
    assert audit.verified_file_count == 16
    assert audit.payload_file_count == 16
    assert audit.cache_file_count == 12


def test_alternate_root_baseline_is_valid(tmp_path: Path) -> None:
    repository = _copy_validation_fixture(tmp_path)

    result = validator.validate_m1_frozen_intent_replay_repair(
        report_path=REPORT_PATH,
        repository_root=repository,
    )

    assert result.valid is True, result.errors


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        (
            "missing manifest field",
            lambda repository: _pop_json(repository / validator.REPAIR_MANIFEST, "candidate_audit"),
        ),
        (
            "extra manifest field",
            lambda repository: _add_json(repository / validator.REPAIR_MANIFEST, "unexpected", True),
        ),
        (
            "missing report field",
            lambda repository: _pop_json(repository / REPORT_PATH, "candidate_audit"),
        ),
        (
            "extra report field",
            lambda repository: _add_json(repository / REPORT_PATH, "unexpected", True),
        ),
        (
            "canonical intent hash drift",
            lambda repository: _set_nested_json(
                repository / validator.REPAIR_MANIFEST,
                ("canonical_intent_sha256",),
                "0" * 64,
            ),
        ),
        (
            "corrected replay hash drift",
            lambda repository: _set_nested_json(
                repository / validator.REPAIR_MANIFEST,
                ("corrected_replay_sha256",),
                "0" * 64,
            ),
        ),
        (
            "zero transport value drift",
            lambda repository: _set_nested_json(
                repository / validator.REPAIR_MANIFEST,
                ("zero_transport_replay", "cache_hits"),
                11,
            ),
        ),
        (
            "unknown nested candidate audit field",
            lambda repository: _set_nested_json(
                repository / validator.REPAIR_MANIFEST,
                ("candidate_audit", "unexpected"),
                True,
            ),
        ),
        (
            "unknown nested zero transport field",
            lambda repository: _set_nested_json(
                repository / validator.REPAIR_MANIFEST,
                ("zero_transport_replay", "unexpected"),
                True,
            ),
        ),
    ],
)
def test_alternate_root_closed_schema_and_hash_mutations_fail_closed(
    tmp_path: Path,
    label: str,
    mutate: Callable[[Path], None],
) -> None:
    repository = _copy_validation_fixture(tmp_path)
    mutate(repository)

    result = validator.validate_m1_frozen_intent_replay_repair(
        report_path=REPORT_PATH,
        repository_root=repository,
    )

    assert result.valid is False, label


def test_status_m2_progress_drift_fails_closed(tmp_path: Path) -> None:
    repository = _copy_validation_fixture(tmp_path)
    status_path = repository / "STATUS.md"
    status_path.write_text(
        status_path.read_text(encoding="utf-8").replace("| IN_PROGRESS | 2/5 |", "| IN_PROGRESS | 3/5 |"),
        encoding="utf-8",
    )

    result = validator.validate_m1_frozen_intent_replay_repair(
        report_path=REPORT_PATH,
        repository_root=repository,
    )

    assert result.valid is False
    assert any("M2 IN_PROGRESS 2/5" in error for error in result.errors)


@pytest.mark.parametrize("relative_path", sorted(validator.PROTECTED_HASHES))
def test_protected_m2_artifact_drift_fails_closed(tmp_path: Path, relative_path: str) -> None:
    repository = _copy_validation_fixture(tmp_path)
    target = repository / relative_path
    target.write_bytes(target.read_bytes() + b"\nmutation")

    result = validator.validate_m1_frozen_intent_replay_repair(
        report_path=REPORT_PATH,
        repository_root=repository,
    )

    assert result.valid is False
    assert any("hash mismatch" in error for error in result.errors)


@pytest.mark.parametrize(
    ("label", "replacement", "expected"),
    [
        ("absolute path", "C:\\Users\\secret\\token.txt", "absolute path"),
        ("credential-shaped text", "Bearer secret-token", "forbidden credential"),
    ],
)
def test_report_safe_text_gate_rejects_sensitive_text(
    tmp_path: Path,
    label: str,
    replacement: str,
    expected: str,
) -> None:
    repository = _copy_validation_fixture(tmp_path)
    _set_nested_json(repository / REPORT_PATH, ("root_cause",), replacement)

    result = validator.validate_m1_frozen_intent_replay_repair(
        report_path=REPORT_PATH,
        repository_root=repository,
    )

    assert result.valid is False, label
    assert any(expected in error for error in result.errors)


def test_extra_cache_file_in_repair_bundle_fails_closed(tmp_path: Path) -> None:
    repository = _copy_validation_fixture(tmp_path)
    extra = repository / validator.REPAIR_BUNDLE / "cache/forbidden.json"
    extra.parent.mkdir(parents=True)
    extra.write_text("{}", encoding="utf-8")

    result = validator.validate_m1_frozen_intent_replay_repair(
        report_path=REPORT_PATH,
        repository_root=repository,
    )

    assert result.valid is False


def test_missing_replay_fails_closed(tmp_path: Path) -> None:
    repository = _copy_validation_fixture(tmp_path)
    (repository / validator.REPAIR_REPLAY).unlink()

    result = validator.validate_m1_frozen_intent_replay_repair(
        report_path=REPORT_PATH,
        repository_root=repository,
    )

    assert result.valid is False


def test_implementation_commit_not_ancestor_of_head_fails_closed(tmp_path: Path) -> None:
    repository = _copy_validation_fixture(tmp_path)
    _set_nested_json(repository / REPORT_PATH, ("implementation_commit",), "0" * 40)

    result = validator.validate_m1_frozen_intent_replay_repair(
        report_path=REPORT_PATH,
        repository_root=repository,
    )

    assert result.valid is False
    assert any("not an ancestor" in error for error in result.errors)


def _copy_validation_fixture(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    (repository / ".git").write_text(
        f"gitdir: {(REPOSITORY_ROOT / '.git').as_posix()}\n",
        encoding="utf-8",
    )

    directories = (
        validator.SOURCE_BUNDLE,
        validator.REPAIR_BUNDLE,
    )
    for relative in directories:
        source = REPOSITORY_ROOT / relative
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)

    files = (
        validator.REPORT_PATH,
        Path("STATUS.md"),
        *tuple(Path(path) for path in validator.PROTECTED_HASHES),
        *tuple(Path(path) for path in validator.PROTECTED_REPORT_HASHES),
    )
    for relative in files:
        source = REPOSITORY_ROOT / relative
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return repository


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _add_json(path: Path, key: str, value: object) -> None:
    payload = _read_json(path)
    payload[key] = value
    _write_json(path, payload)


def _pop_json(path: Path, key: str) -> None:
    payload = _read_json(path)
    payload.pop(key, None)
    _write_json(path, payload)


def _set_nested_json(path: Path, keys: tuple[str, ...], value: object) -> None:
    payload = _read_json(path)
    current: dict[str, object] = payload
    for key in keys[:-1]:
        nested = current.setdefault(key, {})
        assert isinstance(nested, dict)
        current = nested
    current[keys[-1]] = value
    _write_json(path, payload)
