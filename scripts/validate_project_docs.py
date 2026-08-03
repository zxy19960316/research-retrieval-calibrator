from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Model snapshots are external artifacts. Their model-card links may point to
# assets that are intentionally not part of the repository documentation set.
EXTERNAL_MARKDOWN_ROOTS = (ROOT / "models",)
EXPECTED_TASKS = {
    "M0": 4,
    "M1": 4,
    "M2": 5,
    "M3": 5,
    "M4": 4,
    "M5": 4,
    "M6": 5,
}
PHASE_FILES = {
    "M0": "M0-product-and-evaluation-contracts.md",
    "M1": "M1-real-first-round-retrieval.md",
    "M2": "M2-ranking-and-round1-selection.md",
    "M3": "M3-feedback-and-round2-calibration.md",
    "M4": "M4-ten-question-evaluation.md",
    "M5": "M5-openai-compatible-service.md",
    "M6": "M6-cnki-deployment-and-demo.md",
}
REQUIRED_FILES = [
    "AGENTS.md",
    "agent.md",
    "README.md",
    "PRODUCT_SPEC.md",
    "PROJECT_PLAN.md",
    "STATUS.md",
    "docs/architecture.md",
    "docs/data_model.md",
    "docs/evaluation.md",
    "docs/git_workflow.md",
    "docs/phases/README.md",
    "docs/superpowers/plans/2026-07-26-research-retrieval-calibrator-m0.md",
]
PHASE_ORDER = tuple(EXPECTED_TASKS)
ACTIVE_STATUSES = {"READY", "IN_PROGRESS"}
VALID_STATUSES = ACTIVE_STATUSES | {"NO_GO", "COMPLETE"} | {
    f"BLOCKED_BY_{phase}" for phase in PHASE_ORDER
}
CURRENT_PHASE_PATTERN = re.compile(r"^- \u5f53\u524d\u9636\u6bb5\uff1a`(M\d+)`$", re.MULTILINE)
CURRENT_STATUS_PATTERN = re.compile(r"^- \u5f53\u524d\u72b6\u6001\uff1a`([A-Z0-9_]+)`$", re.MULTILINE)
STATUS_ROW_PATTERN = re.compile(
    r"^\|\s*(M\d+)\s+[^|]*\|\s*([A-Z0-9_]+)\s*\|\s*(\d+)/(\d+)\s*\|",
    re.MULTILINE,
)
NEXT_TASK_PATTERN = re.compile(
    r"下一动作：\s*\W*(M\d+-T\d{2})",
    re.MULTILINE,
)


def extract_phase_task_ids(phase: str, text: str) -> list[str]:
    return re.findall(
        rf"^### ({re.escape(phase)}-T\d{{2}})：",
        text,
        flags=re.MULTILINE,
    )


def validate_dynamic_status_delegation(
    readme: str, agent: str, errors: list[str]
) -> None:
    if "STATUS.md" not in readme or "唯一权威来源" not in readme:
        errors.append("README.md must delegate dynamic status to STATUS.md")
    if re.search(r"当前(?:活动)?阶段\s*(?:为|是)\s*M\d+", readme):
        errors.append("README.md contains a hard-coded dynamic phase")
    if "STATUS.md" not in agent or "唯一权威来源" not in agent:
        errors.append("agent.md must delegate dynamic status to STATUS.md")
    if re.search(r"当前(?:活动)?阶段\s*(?:为|是)\s*M\d+", agent):
        errors.append("agent.md contains a hard-coded dynamic phase")
    if re.search(r"M\d+\s*(?:尚未完成|已完成|为当前)", agent):
        errors.append("agent.md contains a hard-coded dynamic phase status")
    if "不重复维护动态阶段编号" not in agent:
        errors.append("agent.md must not duplicate dynamic phase numbering")


def validate_status_handoff(status: str, phase_text: str, errors: list[str]) -> None:
    current_phases = CURRENT_PHASE_PATTERN.findall(status)
    if len(current_phases) != 1 or current_phases[0] not in EXPECTED_TASKS:
        errors.append("STATUS.md current phase is unavailable for next-task validation")
        return

    next_task_match = NEXT_TASK_PATTERN.search(status)
    if next_task_match is None:
        errors.append("STATUS.md must declare an explicit next task")
        return

    current_phase = current_phases[0]
    next_task = next_task_match.group(1)
    if next_task not in extract_phase_task_ids(current_phase, phase_text):
        errors.append(
            f"next task {next_task} does not belong to current phase {current_phase}"
        )


def validate_m2_task_contract(status: str, phase_text: str, errors: list[str]) -> None:
    expected_headers = [
        "### M2-T01：EmbeddingProvider 与冻结向量",
        "### M2-T02：RerankerProvider",
        "### M2-T03：证据槽位分类",
        "### M2-T04：六分项评分",
        "### M2-T05：多样性选择与首轮报告",
    ]
    observed_headers = re.findall(
        r"^### M2-T\d{2}：.*$",
        phase_text,
        flags=re.MULTILINE,
    )
    if observed_headers != expected_headers:
        errors.append(
            f"M2 task IDs/order mismatch: expected {expected_headers}, observed {observed_headers}"
        )

    t04_match = re.search(
        r"^### M2-T04：.*?(?=^### M2-T05：|\Z)",
        phase_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    t04_section = t04_match.group(0) if t04_match else ""
    if not re.search(
        r"evidence slot score.*M2-T03.*提供",
        t04_section,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        errors.append(
            "M2-T04 evidence slot score must be supplied by earlier M2-T03"
        )

    rows = {
        phase: (phase_status, int(completed), int(total))
        for phase, phase_status, completed, total in STATUS_ROW_PATTERN.findall(status)
    }
    if rows.get("M2") != ("IN_PROGRESS", 2, 5):
        errors.append("M2 baseline must remain IN_PROGRESS 2/5")
    if rows.get("M3") != ("BLOCKED_BY_M2", 0, 5):
        errors.append("M3 baseline must remain BLOCKED_BY_M2 0/5")


def validate_required_files(errors: list[str]) -> None:
    for relative_path in REQUIRED_FILES:
        if not (ROOT / relative_path).is_file():
            errors.append(f"missing required file: {relative_path}")


def validate_phase_tasks(errors: list[str]) -> None:
    for phase, expected_count in EXPECTED_TASKS.items():
        path = ROOT / "docs" / "phases" / PHASE_FILES[phase]
        if not path.is_file():
            errors.append(f"missing phase file: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        task_ids = extract_phase_task_ids(phase, text)
        expected_ids = [f"{phase}-T{index:02d}" for index in range(1, expected_count + 1)]
        if task_ids != expected_ids:
            errors.append(
                f"{phase} task IDs mismatch: expected {expected_ids}, observed {task_ids}"
            )
        if "## Codex \u6267\u884c\u6307\u4ee4" not in text:
            errors.append(f"{phase} has no Codex execution instruction")


def validate_local_links(errors: list[str]) -> None:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in ROOT.rglob("*.md"):
        if any(path.is_relative_to(root) for root in EXTERNAL_MARKDOWN_ROOTS):
            continue
        text = path.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            if target.startswith(("http://", "https://", "#")):
                continue
            relative_target = target.split("#", maxsplit=1)[0]
            if relative_target and not (path.parent / relative_target).resolve().exists():
                errors.append(
                    f"broken local link: {path.relative_to(ROOT)} -> {relative_target}"
                )


def validate_status_text(status: str, errors: list[str]) -> None:
    current_phases = CURRENT_PHASE_PATTERN.findall(status)
    current_statuses = CURRENT_STATUS_PATTERN.findall(status)
    if len(current_phases) != 1 or current_phases[0] not in EXPECTED_TASKS:
        errors.append("STATUS.md must declare exactly one known current phase")
        return
    if len(current_statuses) != 1 or current_statuses[0] not in VALID_STATUSES:
        errors.append("STATUS.md must declare exactly one valid current status")
        return

    rows = STATUS_ROW_PATTERN.findall(status)
    by_phase: dict[str, tuple[str, int, int]] = {}
    for phase, phase_status, completed, total in rows:
        if phase in EXPECTED_TASKS:
            if phase in by_phase:
                errors.append(f"STATUS.md has duplicate row for {phase}")
                continue
            by_phase[phase] = (phase_status, int(completed), int(total))

    if set(by_phase) != set(PHASE_ORDER):
        errors.append(f"STATUS.md phase rows mismatch: observed {sorted(by_phase)}")
        return

    active_phases: list[str] = []
    for index, phase in enumerate(PHASE_ORDER):
        phase_status, completed, total = by_phase[phase]
        expected_total = EXPECTED_TASKS[phase]
        if phase_status not in VALID_STATUSES:
            errors.append(f"{phase} has invalid status: {phase_status}")
        if total != expected_total or not 0 <= completed <= total:
            errors.append(
                f"{phase} task count must be within 0/{expected_total} to {expected_total}/{expected_total}; "
                f"observed {completed}/{total}"
            )
        if phase_status == "READY" and completed != 0:
            errors.append(f"{phase} is READY but declares {completed}/{total} tasks")
        if phase_status.startswith("BLOCKED_BY_") and completed != 0:
            errors.append(f"{phase} is {phase_status} but declares {completed}/{total} tasks")
        if phase_status == "IN_PROGRESS" and completed >= expected_total:
            errors.append(f"{phase} is IN_PROGRESS but declares all {completed}/{total} tasks")
        if phase_status == "COMPLETE" and completed != total:
            errors.append(f"{phase} is COMPLETE but declares only {completed}/{total} tasks")
        if phase_status == "NO_GO":
            if phase != "M4":
                errors.append(f"{phase} is NO_GO but only M4 may be NO_GO")
            if (completed, total) != (expected_total, expected_total):
                errors.append(
                    f"{phase} is NO_GO but must declare {expected_total}/{expected_total} tasks; "
                    f"observed {completed}/{total}"
                )
        if phase_status in ACTIVE_STATUSES:
            active_phases.append(phase)
        if index:
            predecessor = PHASE_ORDER[index - 1]
            predecessor_status = by_phase[predecessor][0]
            expected_blocker = f"BLOCKED_BY_{predecessor}"
            if predecessor_status == "COMPLETE":
                if phase_status == expected_blocker:
                    errors.append(f"{phase} remains blocked although {predecessor} is COMPLETE")
            elif phase_status != expected_blocker:
                errors.append(
                    f"{phase} must be {expected_blocker} until {predecessor} is COMPLETE; "
                    f"observed {phase_status}"
                )

    current_phase = current_phases[0]
    current_status = current_statuses[0]
    if by_phase[current_phase][0] != current_status:
        errors.append(
            f"current status mismatch for {current_phase}: "
            f"header is {current_status}, table is {by_phase[current_phase][0]}"
        )
    if current_status == "NO_GO":
        if current_phase != "M4":
            errors.append("STATUS.md may declare NO_GO only for current phase M4")
        if active_phases:
            errors.append(
                "STATUS.md with current NO_GO must not have an active phase; "
                f"observed {active_phases}"
            )
    elif active_phases != [current_phase]:
        errors.append(
            f"STATUS.md must have exactly one active phase matching its header; "
            f"observed {active_phases}"
        )


def validate_status(errors: list[str]) -> None:
    validate_status_text((ROOT / "STATUS.md").read_text(encoding="utf-8"), errors)


def main() -> int:
    errors: list[str] = []
    validate_required_files(errors)
    validate_phase_tasks(errors)
    validate_local_links(errors)
    validate_status(errors)
    readme_path = ROOT / "README.md"
    agent_path = ROOT / "agent.md"
    status_path = ROOT / "STATUS.md"
    m2_phase_path = ROOT / "docs" / "phases" / PHASE_FILES["M2"]
    if all(path.is_file() for path in (readme_path, agent_path, status_path, m2_phase_path)):
        readme = readme_path.read_text(encoding="utf-8")
        agent = agent_path.read_text(encoding="utf-8")
        status = status_path.read_text(encoding="utf-8")
        m2_phase = m2_phase_path.read_text(encoding="utf-8")
        validate_dynamic_status_delegation(readme, agent, errors)
        validate_status_handoff(status, m2_phase, errors)
        validate_m2_task_contract(status, m2_phase, errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "PASS: project docs are complete; "
        f"{sum(EXPECTED_TASKS.values())} phase tasks are indexed; local links resolve."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
