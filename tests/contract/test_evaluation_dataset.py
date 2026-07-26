from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "evaluation" / "datasets" / "questions.schema.json"
DATASET_PATH = ROOT / "evaluation" / "datasets" / "questions.v0.1.yaml"
EXPECTED_IDS = [f"RRC-Q{number:02d}" for number in range(1, 11)]


def _load_schema() -> dict[str, object]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _load_dataset() -> dict[str, object]:
    payload = yaml.safe_load(DATASET_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _validation_errors(instance: object) -> list[jsonschema.ValidationError]:
    validator = jsonschema.Draft202012Validator(_load_schema())
    return list(validator.iter_errors(instance))


def _assert_template_invariants(payload: dict[str, object]) -> None:
    questions = payload["questions"]
    assert isinstance(questions, list)
    ids = [question["question_id"] for question in questions]
    domains = [question["domain"] for question in questions]
    assert ids == EXPECTED_IDS
    assert len(ids) == len(set(ids)) == 10
    assert domains[:5] == ["nuclear_engineering"] * 5
    assert domains[5:] == ["cross_domain_stem"] * 5


def test_dataset_and_schema_are_present() -> None:
    assert SCHEMA_PATH.is_file()
    assert DATASET_PATH.is_file()


def test_schema_is_valid_json_schema() -> None:
    jsonschema.Draft202012Validator.check_schema(_load_schema())


def test_yaml_loads_safely_and_satisfies_schema() -> None:
    assert _validation_errors(_load_dataset()) == []


def test_template_has_exactly_ten_stable_unique_ids_and_five_by_five_domains() -> None:
    _assert_template_invariants(_load_dataset())


def test_current_m0_template_is_entirely_unjudged() -> None:
    questions = _load_dataset()["questions"]
    assert isinstance(questions, list)
    assert [question["label_status"] for question in questions] == ["unjudged"] * 10


def test_schema_rejects_duplicate_ids_and_wrong_domain_split() -> None:
    duplicate_ids = copy.deepcopy(_load_dataset())
    duplicate_ids["questions"][1]["question_id"] = "RRC-Q01"
    with pytest.raises(AssertionError):
        _assert_template_invariants(duplicate_ids)

    wrong_split = copy.deepcopy(_load_dataset())
    wrong_split["questions"][0]["domain"] = "cross_domain_stem"
    with pytest.raises(AssertionError):
        _assert_template_invariants(wrong_split)


def test_schema_rejects_empty_questions_missing_frozen_metadata_and_extra_fields() -> None:
    empty_question = copy.deepcopy(_load_dataset())
    empty_question["questions"][0]["question_zh"] = "   "
    empty_question["questions"][0]["question_en"] = ""
    assert _validation_errors(empty_question)

    frozen_without_metadata = copy.deepcopy(_load_dataset())
    frozen_without_metadata["questions"][0]["label_status"] = "frozen"
    assert _validation_errors(frozen_without_metadata)

    extra_field = copy.deepcopy(_load_dataset())
    extra_field["questions"][0]["undeclared"] = "forbidden"
    assert _validation_errors(extra_field)
