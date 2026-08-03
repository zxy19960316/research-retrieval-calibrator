"""Contract tests for the first-phase M2-T03 human review bundle."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import scripts.validate_m2_t03_human_adjudication_bundle as validator
from scripts.validate_m2_t03_human_adjudication_bundle import (
    BUNDLE_PATH,
    validate_bundle,
)

_ALTERNATE_REPOSITORY_FILES = (
    Path("STATUS.md"),
    Path("docs/reviews/m2-t03-human-adjudication-protocol.md"),
    Path("scripts/build_m2_t03_human_adjudication_bundle.py"),
    Path("scripts/validate_m2_t03_human_adjudication_bundle.py"),
    Path("evaluation/source-artifacts/m1-intent-replay-repair-2026-08-03/first-run/first-round.json"),
    Path("evaluation/source-artifacts/m1-intent-replay-repair-2026-08-03/repair-manifest.json"),
    Path("evaluation/snapshots/m2/m1-candidates.v1.json"),
    Path("evaluation/source-artifacts/m2-t03-evidence-classification-run.json"),
    Path("evaluation/source-artifacts/m2-t03-human-adjudication-bundle.json"),
    Path("evaluation/source-artifacts/m2-t03-human-adjudication-bundle-receipt.json"),
    Path("evaluation/source-artifacts/m2-t03-human-adjudication-review-template.json"),
    Path("evaluation/reports/m2-t03-human-adjudication.json"),
)


def _bundle_payload() -> dict[str, object]:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def _run_git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=repository_root, capture_output=True, check=True, text=True
    )
    return completed.stdout.strip()


def _commit_alternate_repository(repository_root: Path, message: str) -> str:
    _run_git(repository_root, "add", "--all")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=M2-T03 test",
            "-c",
            "user.email=m2-t03@example.invalid",
            "commit",
            "--quiet",
            "-m",
            message,
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return _run_git(repository_root, "rev-parse", "HEAD")


def _write_json(path: Path, payload: dict[str, object]) -> bytes:
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(raw)
    return raw


def _copy_alternate_repository(source_root: Path, target_root: Path) -> None:
    for relative_path in _ALTERNATE_REPOSITORY_FILES:
        target = target_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative_path, target)


def _make_alternate_pending_repository(source_root: Path, target_root: Path) -> None:
    _copy_alternate_repository(source_root, target_root)
    _run_git(target_root, "init", "--quiet")
    _run_git(target_root, "config", "user.name", "M2-T03 test")
    _run_git(target_root, "config", "user.email", "m2-t03@example.invalid")
    historical_commit = _commit_alternate_repository(target_root, "test: record historical pending STATUS")

    bundle_path = target_root / BUNDLE_PATH
    receipt_path = target_root / "evaluation/source-artifacts/m2-t03-human-adjudication-bundle-receipt.json"
    template_path = target_root / "evaluation/source-artifacts/m2-t03-human-adjudication-review-template.json"
    report_path = target_root / "evaluation/reports/m2-t03-human-adjudication.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    template = json.loads(template_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    bundle["generated_from_commit"] = historical_commit
    bundle_raw = _write_json(bundle_path, bundle)
    bundle_sha = hashlib.sha256(bundle_raw).hexdigest()
    template["pending_bundle"]["sha256"] = bundle_sha  # type: ignore[index]
    template_raw = _write_json(template_path, template)
    template_sha = hashlib.sha256(template_raw).hexdigest()
    receipt["generated_from_commit"] = historical_commit
    receipt["bundle_sha256"] = bundle_sha
    receipt["review_template"]["sha256"] = template_sha  # type: ignore[index]
    receipt_raw = _write_json(receipt_path, receipt)
    report["generated_from_commit"] = historical_commit
    report["artifact"] = [
        {"path": BUNDLE_PATH.as_posix(), "sha256": bundle_sha},
        {
            "path": "evaluation/source-artifacts/m2-t03-human-adjudication-bundle-receipt.json",
            "sha256": hashlib.sha256(receipt_raw).hexdigest(),
        },
        {
            "path": "evaluation/source-artifacts/m2-t03-human-adjudication-review-template.json",
            "sha256": template_sha,
        },
    ]
    historical_status = (target_root / "STATUS.md").read_bytes()
    report["input_artifacts"][-1]["sha256"] = hashlib.sha256(historical_status).hexdigest()  # type: ignore[index]
    _write_json(report_path, report)
    _commit_alternate_repository(target_root, "test: bind pending artifacts to historical commit")


def test_bundle_is_closed_and_stops_at_the_human_work_point() -> None:
    payload = _bundle_payload()
    assert set(payload) == {
        "bundle_version",
        "candidate_count",
        "candidate_snapshot",
        "classification_result",
        "generated_at_utc",
        "generated_from_commit",
        "intent_context",
        "items",
        "paper_id_order",
        "phase",
        "review_policy",
        "status_boundary",
        "task_id",
    }
    assert payload["candidate_count"] == 33
    assert payload["status_boundary"] == {
        "human_review": "pending",
        "m2_progress": "IN_PROGRESS 2/5",
        "m2_t04": "not_started",
        "m3_progress": "BLOCKED_BY_M2 0/5",
        "scoring_eligible": False,
    }
    assert payload["review_policy"]["machine_labels_are_advisory"] is True  # type: ignore[index]
    assert payload["review_policy"]["human_fields_initially_empty"] is True  # type: ignore[index]

    items = payload["items"]
    assert isinstance(items, list)
    assert len(items) == 33
    assert all(
        set(item) == {"context", "machine_advisory", "human_adjudication"} for item in items
    )
    assert all(
        all(value is None for value in item["human_adjudication"].values()) for item in items
    )
    serialized = json.dumps(payload, ensure_ascii=False)
    assert all(token not in serialized.casefold() for token in ('"score"', '"rank"', '"weight"'))


def test_validator_accepts_the_committed_bundle() -> None:
    result = validate_bundle()
    assert result.valid, result.errors


def test_validator_rejects_any_prefilled_human_decision(tmp_path: Path) -> None:
    payload = copy.deepcopy(_bundle_payload())
    payload["items"][0]["human_adjudication"]["verdict"] = "SUPPORTED"  # type: ignore[index]
    mutated = tmp_path / "bundle.json"
    mutated.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = validate_bundle(mutated)

    assert not result.valid
    assert any("context or advisory" in error for error in result.errors)


def test_validator_rejects_unknown_bundle_fields(tmp_path: Path) -> None:
    payload = _bundle_payload()
    payload["unexpected"] = True
    mutated = tmp_path / "bundle.json"
    mutated.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = validate_bundle(mutated)

    assert not result.valid
    assert result.errors == ["cannot load closed M2-T03 human adjudication artifacts"]


def test_protocol_is_frozen_and_declares_blind_review_boundary() -> None:
    protocol = Path("docs/reviews/m2-t03-human-adjudication-protocol.md").read_text(
        encoding="utf-8"
    )

    for value in (
        "paper_id",
        "title",
        "abstract",
        "source",
        "source_id",
        "url",
        "ResearchIntent",
        "PROBLEM_EXISTENCE",
        "CURRENT_METHODS",
        "METHOD_TRANSFERABILITY",
        "IMPLEMENTATION_PATH",
        "EVALUATION_BASIS",
        "DIRECT",
        "INDIRECT",
        "HYPOTHETICAL",
        "CONFIRM",
        "REVISE",
        "REJECT",
    ):
        assert value in protocol


def test_current_gate_is_separate_from_pending_bundle_validation() -> None:
    assert hasattr(validator, "validate_current_m2_gate")


def test_pending_validation_uses_historical_status_in_an_alternate_root(tmp_path: Path) -> None:
    source_root = Path(__file__).parents[2]
    alternate_root = tmp_path / "alternate-repository"
    alternate_root.mkdir()
    _make_alternate_pending_repository(source_root, alternate_root)

    status_path = alternate_root / "STATUS.md"
    current_status = status_path.read_text(encoding="utf-8")
    status_path.write_text(current_status.replace("| 2/5 |", "| 3/5 |", 1), encoding="utf-8")

    pending = validate_bundle(repository_root=alternate_root)
    assert pending.valid, pending.errors

    current = validator.validate_current_m2_gate(alternate_root)
    assert not current.valid
    assert any("completed human-result" in error for error in current.errors)

    for relative_path in (
        "evaluation/source-artifacts/m2-t03-human-adjudication-result.json",
        "evaluation/source-artifacts/m2-t03-human-adjudication-result-receipt.json",
        "evaluation/reports/m2-t03-human-adjudication-result.json",
    ):
        path = alternate_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    completed = validator.validate_current_m2_gate(
        alternate_root, completed_result_validator=lambda _: True
    )
    assert completed.valid, completed.errors


def test_validator_rejects_policy_hash_command_and_blind_template_drift(tmp_path: Path) -> None:
    source_root = Path(__file__).parents[2]
    mutations = {
        "source_fields": lambda bundle, receipt, template, report: bundle["review_policy"].update(
            {"source_fields": ["title", "paper_id", "abstract", "source", "source_id", "url", "ResearchIntent"]}
        ),
        "disallowed_sources": lambda bundle, receipt, template, report: bundle["review_policy"].update(
            {"disallowed_sources": ["paper_full_text", "paper_full_text"]}
        ),
        "report_command": lambda bundle, receipt, template, report: report.update(
            {"commands": ["python unexpected-command.py"]}
        ),
        "receipt_runner": lambda bundle, receipt, template, report: receipt.update(
            {"runner_sha256": "0" * 64}
        ),
        "template_advisory": lambda bundle, receipt, template, report: template.update(
            {"machine_advisory": {"score": 1}}
        ),
        "credential_text": lambda bundle, receipt, template, report: bundle["items"][0]["context"].update(
            {"title": "credentials"}
        ),
    }

    for name, mutate in mutations.items():
        alternate_root = tmp_path / name
        alternate_root.mkdir()
        _make_alternate_pending_repository(source_root, alternate_root)
        paths = {
            "bundle": alternate_root / BUNDLE_PATH,
            "receipt": alternate_root / "evaluation/source-artifacts/m2-t03-human-adjudication-bundle-receipt.json",
            "template": alternate_root / "evaluation/source-artifacts/m2-t03-human-adjudication-review-template.json",
            "report": alternate_root / "evaluation/reports/m2-t03-human-adjudication.json",
        }
        payloads = {
            key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()
        }
        mutate(
            payloads["bundle"],
            payloads["receipt"],
            payloads["template"],
            payloads["report"],
        )
        for key, path in paths.items():
            _write_json(path, payloads[key])
        invalid = validate_bundle(repository_root=alternate_root)
        assert not invalid.valid, name
