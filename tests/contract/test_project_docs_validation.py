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


@pytest.mark.parametrize("completed", (0, 1, 2, 3))
def test_status_validation_allows_every_legal_m1_precompletion_state(completed: int) -> None:
    errors: list[str] = []
    state = "READY" if completed == 0 else "IN_PROGRESS"
    rows = [
        ("M0", "COMPLETE", 4, 4),
        ("M1", state, completed, 4),
        ("M2", "BLOCKED_BY_M1", 0, 5),
        ("M3", "BLOCKED_BY_M2", 0, 5),
        ("M4", "BLOCKED_BY_M3", 0, 4),
        ("M5", "BLOCKED_BY_M4", 0, 4),
        ("M6", "BLOCKED_BY_M5", 0, 5),
    ]

    validator.validate_status_text(
        _status(rows, current_phase="M1", current_status=state), errors
    )

    assert errors == []


def test_status_validation_allows_m1_completion_then_m2_ready() -> None:
    errors: list[str] = []
    rows = [
        ("M0", "COMPLETE", 4, 4),
        ("M1", "COMPLETE", 4, 4),
        ("M2", "READY", 0, 5),
        ("M3", "BLOCKED_BY_M2", 0, 5),
        ("M4", "BLOCKED_BY_M3", 0, 4),
        ("M5", "BLOCKED_BY_M4", 0, 4),
        ("M6", "BLOCKED_BY_M5", 0, 5),
    ]

    validator.validate_status_text(
        _status(rows, current_phase="M2", current_status="READY"), errors
    )

    assert errors == []


def test_dynamic_status_delegation_rejects_future_hard_coded_phase() -> None:
    root = Path(__file__).parents[2]
    readme = (root / "README.md").read_text(encoding="utf-8")
    agent = (root / "agent.md").read_text(encoding="utf-8")

    errors: list[str] = []
    validator.validate_dynamic_status_delegation(readme, agent, errors)
    assert errors == []

    stale_agent = agent.replace(
        "当前活动阶段、当前任务、允许执行范围和阶段门禁，\n均以 STATUS.md 为唯一权威来源。",
        "当前活动阶段为 M1。",
    )
    errors = []
    validator.validate_dynamic_status_delegation(readme, stale_agent, errors)
    assert any("agent.md" in error and "dynamic phase" in error for error in errors)

    stale_readme = readme + "\n当前阶段为 M1。\n"
    errors = []
    validator.validate_dynamic_status_delegation(stale_readme, agent, errors)
    assert any("README.md" in error and "dynamic phase" in error for error in errors)


def test_status_handoff_requires_next_task_in_current_phase() -> None:
    root = Path(__file__).parents[2]
    status = (root / "STATUS.md").read_text(encoding="utf-8")
    phase = (root / "docs" / "phases" / "M2-ranking-and-round1-selection.md").read_text(
        encoding="utf-8"
    )

    errors: list[str] = []
    validator.validate_status_handoff(status, phase, errors)
    assert errors == []

    stale_status = status.replace(
        "下一动作：`M2-T03 质量收口`",
        "下一动作：`M1-T01`",
    )
    errors = []
    validator.validate_status_handoff(stale_status, phase, errors)
    assert any("next task" in error for error in errors)


def test_m2_task_contract_requires_order_dependency_and_two_of_five_baseline() -> None:
    root = Path(__file__).parents[2]
    status = (root / "STATUS.md").read_text(encoding="utf-8")
    phase = (root / "docs" / "phases" / "M2-ranking-and-round1-selection.md").read_text(
        encoding="utf-8"
    )

    errors: list[str] = []
    validator.validate_m2_task_contract(status, phase, errors)
    assert errors == []

    swapped = phase.replace(
        "### M2-T03：证据槽位分类", "### M2-T03：六分项评分"
    ).replace("### M2-T04：六分项评分", "### M2-T04：证据槽位分类")
    errors = []
    validator.validate_m2_task_contract(status, swapped, errors)
    assert any("M2 task IDs" in error for error in errors)

    without_dependency = phase.replace(
        "- evidence slot score（由更早的 M2-T03 提供）",
        "- evidence slot score",
    )
    errors = []
    validator.validate_m2_task_contract(status, without_dependency, errors)
    assert any("evidence slot score" in error for error in errors)

    advanced_status = status.replace(
        "| M2 首轮排序与选择 | IN_PROGRESS | 2/5 |",
        "| M2 首轮排序与选择 | IN_PROGRESS | 3/5 |",
    )
    errors = []
    validator.validate_m2_task_contract(advanced_status, phase, errors)
    assert any("M2 baseline" in error for error in errors)


def test_docs_workflow_validates_m2_t03_evidence() -> None:
    root = Path(__file__).parents[2]
    workflow = (root / ".github" / "workflows" / "docs-validation.yml").read_text(
        encoding="utf-8"
    )

    assert "- name: Validate completed M2-T03 evidence" in workflow
    assert "run: python scripts/validate_m2_t03_evidence.py" in workflow
