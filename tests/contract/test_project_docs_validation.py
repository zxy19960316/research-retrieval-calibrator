import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "validate_project_docs.py"
SPEC = importlib.util.spec_from_file_location("validate_project_docs", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _status(rows: list[tuple[str, str, int, int]]) -> str:
    table = "\n".join(
        f"| {phase} phase | {status} | {completed}/{total} | evidence |"
        for phase, status, completed, total in rows
    )
    return (
        "- 当前阶段：`M0`\n"
        "- 当前状态：`IN_PROGRESS`\n\n"
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
