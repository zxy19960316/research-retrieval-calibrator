"""Contract tests for the M0-T02 project-stage transition guard."""

import pytest

from app.core.state_machine import (
    InvalidTransition,
    TransitionErrorCode,
    transition,
    transition_if_expected,
)
from app.models.enums import ProjectStage


@pytest.mark.parametrize(
    ("current", "target", "valid_feedback_count"),
    [
        (ProjectStage.INIT, ProjectStage.CLARIFYING, 0),
        (ProjectStage.CLARIFYING, ProjectStage.ROUND1_SEARCHING, 0),
        (ProjectStage.ROUND1_SEARCHING, ProjectStage.WAITING_FOR_FEEDBACK, 0),
        (ProjectStage.WAITING_FOR_FEEDBACK, ProjectStage.ROUND2_SEARCHING, 6),
        (
            ProjectStage.ROUND2_SEARCHING,
            ProjectStage.WAITING_FOR_DIRECTION_CONFIRMATION,
            6,
        ),
        (
            ProjectStage.WAITING_FOR_DIRECTION_CONFIRMATION,
            ProjectStage.FINALIZED,
            6,
        ),
    ],
)
def test_all_legal_stepwise_transitions_return_the_target(
    current: ProjectStage,
    target: ProjectStage,
    valid_feedback_count: int,
) -> None:
    assert transition(current, target, valid_feedback_count) is target


def test_cannot_skip_clarification() -> None:
    with pytest.raises(InvalidTransition) as error:
        transition(ProjectStage.INIT, ProjectStage.ROUND1_SEARCHING, 0)

    assert error.value.code is TransitionErrorCode.INVALID_TRANSITION


def test_cannot_move_backward() -> None:
    with pytest.raises(InvalidTransition) as error:
        transition(ProjectStage.ROUND1_SEARCHING, ProjectStage.CLARIFYING, 0)

    assert error.value.code is TransitionErrorCode.INVALID_TRANSITION


def test_five_feedback_items_cannot_start_round_two() -> None:
    with pytest.raises(InvalidTransition) as error:
        transition(
            ProjectStage.WAITING_FOR_FEEDBACK,
            ProjectStage.ROUND2_SEARCHING,
            5,
        )

    assert error.value.code is TransitionErrorCode.INSUFFICIENT_FEEDBACK


def test_six_feedback_items_can_start_round_two() -> None:
    assert (
        transition(
            ProjectStage.WAITING_FOR_FEEDBACK,
            ProjectStage.ROUND2_SEARCHING,
            6,
        )
        is ProjectStage.ROUND2_SEARCHING
    )


def test_finalized_cannot_roll_back() -> None:
    with pytest.raises(InvalidTransition) as error:
        transition(ProjectStage.FINALIZED, ProjectStage.CLARIFYING, 6)

    assert error.value.code is TransitionErrorCode.FINALIZED_TERMINAL


def test_reapplying_finalized_is_reported_as_already_applied() -> None:
    with pytest.raises(InvalidTransition) as error:
        transition(ProjectStage.FINALIZED, ProjectStage.FINALIZED, 6)

    assert error.value.code is TransitionErrorCode.ALREADY_APPLIED


def test_reapplying_an_already_applied_transition_is_rejected() -> None:
    with pytest.raises(InvalidTransition) as error:
        transition(ProjectStage.CLARIFYING, ProjectStage.CLARIFYING, 0)

    assert error.value.code is TransitionErrorCode.ALREADY_APPLIED


def test_illegal_transition_does_not_change_the_callers_state() -> None:
    current = ProjectStage.INIT

    with pytest.raises(InvalidTransition):
        transition(current, ProjectStage.ROUND1_SEARCHING, 0)

    assert current is ProjectStage.INIT


@pytest.mark.parametrize(
    ("current", "target", "valid_feedback_count", "expected_code"),
    [
        (
            ProjectStage.INIT,
            ProjectStage.ROUND1_SEARCHING,
            0,
            TransitionErrorCode.INVALID_TRANSITION,
        ),
        (
            ProjectStage.WAITING_FOR_FEEDBACK,
            ProjectStage.ROUND2_SEARCHING,
            5,
            TransitionErrorCode.INSUFFICIENT_FEEDBACK,
        ),
        (
            ProjectStage.FINALIZED,
            ProjectStage.CLARIFYING,
            6,
            TransitionErrorCode.FINALIZED_TERMINAL,
        ),
        (
            ProjectStage.CLARIFYING,
            ProjectStage.CLARIFYING,
            0,
            TransitionErrorCode.ALREADY_APPLIED,
        ),
    ],
)
def test_error_codes_are_stable(
    current: ProjectStage,
    target: ProjectStage,
    valid_feedback_count: int,
    expected_code: TransitionErrorCode,
) -> None:
    with pytest.raises(InvalidTransition) as error:
        transition(current, target, valid_feedback_count)

    assert error.value.code is expected_code
    assert str(error.value) == expected_code.value


def test_expected_state_guard_rejects_a_concurrent_duplicate_request() -> None:
    actual_stage = ProjectStage.CLARIFYING
    request_expected_stage = ProjectStage.INIT

    with pytest.raises(InvalidTransition) as error:
        transition_if_expected(
            actual_stage=actual_stage,
            expected_stage=request_expected_stage,
            target=ProjectStage.CLARIFYING,
            valid_feedback_count=0,
        )

    assert error.value.code is TransitionErrorCode.STALE_STATE
    assert actual_stage is ProjectStage.CLARIFYING


def test_expected_state_guard_delegates_when_expected_state_matches_actual_state() -> None:
    assert (
        transition_if_expected(
            actual_stage=ProjectStage.INIT,
            expected_stage=ProjectStage.INIT,
            target=ProjectStage.CLARIFYING,
            valid_feedback_count=0,
        )
        is ProjectStage.CLARIFYING
    )


def test_sequential_duplicate_request_with_stale_expected_state_is_rejected() -> None:
    actual_stage = ProjectStage.INIT

    actual_stage = transition_if_expected(
        actual_stage=actual_stage,
        expected_stage=ProjectStage.INIT,
        target=ProjectStage.CLARIFYING,
        valid_feedback_count=0,
    )

    with pytest.raises(InvalidTransition) as error:
        transition_if_expected(
            actual_stage=actual_stage,
            expected_stage=ProjectStage.INIT,
            target=ProjectStage.CLARIFYING,
            valid_feedback_count=0,
        )

    assert error.value.code is TransitionErrorCode.STALE_STATE
