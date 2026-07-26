import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "validate_project_docs.py"
SPEC = importlib.util.spec_from_file_location("validate_project_docs", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _status(
    rows: list[tuple[str, str, int, int]],
    *,
    current_phase: str = "M0",
    current_status: str = "IN_PROGRESS",
) -> str:
    table = "\n".join(
        f"| {phase} phase | {status} | {completed}/{total} | evidence |"
        for phase, status, completed, total in rows
    )
    return (
        f"- 当前阶段：`{current_phase}`\n"
        f"- 当前状态：`{current_status}`\n\n"
        "| phase | status | tasks | evidence |\n"
        "|---|---|---:|---|\n"
        f"{table}\n"
    )


def test_status_validation_allows_in_progress_task_counts() -> None:
    errors: list[str] = []

    validator.validate_status_text(
        _status(
            [
                ("M0", "IN_PROGRESS", 1, 4),
                ("M1", "BLOCKED_BY_M0", 0, 4),
                ("M2", "BLOCKED_BY_M1", 0, 5),
                ("M3", "BLOCKED_BY_M2", 0, 5),
                ("M4", "BLOCKED_BY_M3", 0, 4),
                ("M5", "BLOCKED_BY_M4", 0, 4),
                ("M6", "BLOCKED_BY_M5", 0, 5),
            ]
        ),
        errors,
    )

    assert errors == []


def test_status_validation_rejects_another_active_phase_and_bad_dependency() -> None:
    errors: list[str] = []

    validator.validate_status_text(
        _status(
            [
                ("M0", "IN_PROGRESS", 1, 4),
                ("M1", "READY", 0, 4),
                ("M2", "BLOCKED_BY_M6", 0, 5),
                ("M3", "BLOCKED_BY_M2", 0, 5),
                ("M4", "BLOCKED_BY_M3", 0, 4),
                ("M5", "BLOCKED_BY_M4", 0, 4),
                ("M6", "BLOCKED_BY_M5", 0, 5),
            ]
        ),
        errors,
    )

    assert any("must be BLOCKED_BY_M0" in error for error in errors)
    assert any("must be BLOCKED_BY_M1" in error for error in errors)
    assert any("exactly one active phase" in error for error in errors)


def test_status_validation_allows_m4_no_go_after_all_tasks_complete() -> None:
    errors: list[str] = []

    validator.validate_status_text(
        _status(
            [
                ("M0", "COMPLETE", 4, 4),
                ("M1", "COMPLETE", 4, 4),
                ("M2", "COMPLETE", 5, 5),
                ("M3", "COMPLETE", 5, 5),
                ("M4", "NO_GO", 4, 4),
                ("M5", "BLOCKED_BY_M4", 0, 4),
                ("M6", "BLOCKED_BY_M5", 0, 5),
            ],
            current_phase="M4",
            current_status="NO_GO",
        ),
        errors,
    )

    assert errors == []


def test_status_validation_rejects_no_go_outside_m4() -> None:
    errors: list[str] = []

    validator.validate_status_text(
        _status(
            [
                ("M0", "NO_GO", 4, 4),
                ("M1", "BLOCKED_BY_M0", 0, 4),
                ("M2", "BLOCKED_BY_M1", 0, 5),
                ("M3", "BLOCKED_BY_M2", 0, 5),
                ("M4", "BLOCKED_BY_M3", 0, 4),
                ("M5", "BLOCKED_BY_M4", 0, 4),
                ("M6", "BLOCKED_BY_M5", 0, 5),
            ],
            current_status="NO_GO",
        ),
        errors,
    )

    assert any("only M4 may be NO_GO" in error for error in errors)


def test_status_validation_rejects_incomplete_m4_no_go() -> None:
    errors: list[str] = []

    validator.validate_status_text(
        _status(
            [
                ("M0", "COMPLETE", 4, 4),
                ("M1", "COMPLETE", 4, 4),
                ("M2", "COMPLETE", 5, 5),
                ("M3", "COMPLETE", 5, 5),
                ("M4", "NO_GO", 3, 4),
                ("M5", "BLOCKED_BY_M4", 0, 4),
                ("M6", "BLOCKED_BY_M5", 0, 5),
            ],
            current_phase="M4",
            current_status="NO_GO",
        ),
        errors,
    )

    assert "M4 is NO_GO but must declare 4/4 tasks; observed 3/4" in errors


@pytest.mark.parametrize(
    ("status", "completed", "expected_error"),
    [
        ("READY", 1, "M1 is READY but declares 1/4 tasks"),
        ("BLOCKED_BY_M0", 1, "M1 is BLOCKED_BY_M0 but declares 1/4 tasks"),
        ("IN_PROGRESS", 4, "M1 is IN_PROGRESS but declares all 4/4 tasks"),
        ("COMPLETE", 3, "M1 is COMPLETE but declares only 3/4 tasks"),
    ],
)
def test_status_validation_enforces_status_task_counts(
    status: str, completed: int, expected_error: str
) -> None:
    errors: list[str] = []
    rows = [
        ("M0", "COMPLETE", 4, 4),
        ("M1", status, completed, 4),
        ("M2", "BLOCKED_BY_M1", 0, 5),
        ("M3", "BLOCKED_BY_M2", 0, 5),
        ("M4", "BLOCKED_BY_M3", 0, 4),
        ("M5", "BLOCKED_BY_M4", 0, 4),
        ("M6", "BLOCKED_BY_M5", 0, 5),
    ]

    validator.validate_status_text(
        _status(rows, current_phase="M1", current_status=status), errors
    )

    assert expected_error in errors
