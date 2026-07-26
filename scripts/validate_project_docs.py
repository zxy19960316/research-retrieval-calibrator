from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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
        task_ids = re.findall(rf"^### ({phase}-T\d{{2}})：", text, flags=re.MULTILINE)
        expected_ids = [f"{phase}-T{index:02d}" for index in range(1, expected_count + 1)]
        if task_ids != expected_ids:
            errors.append(
                f"{phase} task IDs mismatch: expected {expected_ids}, observed {task_ids}"
            )
        if "## Codex 执行指令" not in text:
            errors.append(f"{phase} has no Codex execution instruction")


def validate_local_links(errors: list[str]) -> None:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            if target.startswith(("http://", "https://", "#")):
                continue
            relative_target = target.split("#", maxsplit=1)[0]
            if relative_target and not (path.parent / relative_target).resolve().exists():
                errors.append(
                    f"broken local link: {path.relative_to(ROOT)} -> {relative_target}"
                )


def validate_status(errors: list[str]) -> None:
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    if "- 当前阶段：`M0`" not in status or "- 当前状态：`READY`" not in status:
        errors.append("baseline STATUS.md must start at M0 READY")
    for phase, expected_count in EXPECTED_TASKS.items():
        if f"| {phase} " not in status or f"0/{expected_count}" not in status:
            errors.append(f"STATUS.md does not declare {phase} 0/{expected_count}")


def main() -> int:
    errors: list[str] = []
    validate_required_files(errors)
    validate_phase_tasks(errors)
    validate_local_links(errors)
    validate_status(errors)
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
