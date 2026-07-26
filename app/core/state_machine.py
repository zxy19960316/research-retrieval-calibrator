"""Dependency-free guards for M0 project-stage transitions.

``transition_if_expected`` detects a stale request before applying the pure
transition rule. It is deliberately not an atomic concurrency mechanism: a
future persistence layer must enforce the same expected-stage condition in one
database update, such as ``WHERE stage = :expected_stage``.
"""

from enum import StrEnum

from app.models.enums import ProjectStage


class TransitionErrorCode(StrEnum):
    """Stable machine-readable reasons for a rejected stage transition."""

    INVALID_TRANSITION = "INVALID_TRANSITION"
    INSUFFICIENT_FEEDBACK = "INSUFFICIENT_FEEDBACK"
    FINALIZED_TERMINAL = "FINALIZED_TERMINAL"
    ALREADY_APPLIED = "ALREADY_APPLIED"
    STALE_STATE = "STALE_STATE"


class InvalidTransition(ValueError):
    """Raised when a stage transition violates a stable transition contract."""

    def __init__(self, code: TransitionErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


_NEXT_STAGE: dict[ProjectStage, ProjectStage] = {
    ProjectStage.INIT: ProjectStage.CLARIFYING,
    ProjectStage.CLARIFYING: ProjectStage.ROUND1_SEARCHING,
    ProjectStage.ROUND1_SEARCHING: ProjectStage.WAITING_FOR_FEEDBACK,
    ProjectStage.WAITING_FOR_FEEDBACK: ProjectStage.ROUND2_SEARCHING,
    ProjectStage.ROUND2_SEARCHING: ProjectStage.WAITING_FOR_DIRECTION_CONFIRMATION,
    ProjectStage.WAITING_FOR_DIRECTION_CONFIRMATION: ProjectStage.FINALIZED,
}


def transition(
    current: ProjectStage,
    target: ProjectStage,
    valid_feedback_count: int,
) -> ProjectStage:
    """Return one legal next stage or raise an error with a stable code."""
    if current is ProjectStage.FINALIZED:
        raise InvalidTransition(TransitionErrorCode.FINALIZED_TERMINAL)
    if target is current:
        raise InvalidTransition(TransitionErrorCode.ALREADY_APPLIED)
    if _NEXT_STAGE.get(current) is not target:
        raise InvalidTransition(TransitionErrorCode.INVALID_TRANSITION)
    if target is ProjectStage.ROUND2_SEARCHING and valid_feedback_count < 6:
        raise InvalidTransition(TransitionErrorCode.INSUFFICIENT_FEEDBACK)
    return target


def transition_if_expected(
    *,
    actual_stage: ProjectStage,
    expected_stage: ProjectStage,
    target: ProjectStage,
    valid_feedback_count: int,
) -> ProjectStage:
    """Reject a stale request before applying the pure stage transition.

    This guard is network- and database-independent. Callers with persistent
    state must still perform an atomic compare-and-set update in that layer.
    """
    if actual_stage is not expected_stage:
        raise InvalidTransition(TransitionErrorCode.STALE_STATE)
    return transition(actual_stage, target, valid_feedback_count)
