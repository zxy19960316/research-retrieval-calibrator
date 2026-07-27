"""Unit coverage for the safe M1-T04 command-line boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.adapters import arxiv
from app.cli.first_round import (
    _adapter_from_arguments,
    _config_from_arguments,
    _parse_arguments,
    main,
)

QUESTION = "How can graph-based retrieval support scientific literature discovery?"


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--question", "   ", "--output-dir", "out"],
        ["--question", QUESTION, "--output-dir", "out", "--max-results-per-query", "6"],
    ],
)
def test_invalid_cli_arguments_emit_only_structured_json(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "failed"
    assert payload["error_code"] == "INVALID_CLI_ARGUMENT"
    assert "Traceback" not in captured.err


def test_recorded_cli_writes_json_and_markdown_without_invented_metadata(tmp_path: Path) -> None:
    assert main(["--question", QUESTION, "--output-dir", str(tmp_path), "--mode", "recorded"]) == 0

    payload = json.loads((tmp_path / "first-round.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "first-round.md").read_text(encoding="utf-8")
    assert payload["metrics"]["source_id_coverage"] == 1.0
    assert "Abstract:" not in markdown
    assert "Translation:" not in markdown
    assert "Ranking:" not in markdown
    assert "DOI:" not in markdown or all(item["doi"] for item in payload["candidates"])


def test_recorded_cli_creates_a_missing_nested_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "missing" / "nested" / "output"

    assert main(["--question", QUESTION, "--output-dir", str(output_dir), "--mode", "recorded"]) == 0

    assert (output_dir / "first-round.json").is_file()
    assert (output_dir / "first-round.md").is_file()


def test_existing_output_file_is_an_invalid_cli_argument(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("not a directory", encoding="utf-8")

    assert main(["--question", QUESTION, "--output-dir", str(output_file)]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "INVALID_CLI_ARGUMENT"


def test_output_write_failure_emits_structured_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_writer(_: Path, __: str) -> None:
        raise OSError("recorded write failure")

    assert (
        main(
            ["--question", QUESTION, "--output-dir", str(tmp_path), "--mode", "recorded"],
            writer=failing_writer,
        )
        == 1
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "failed"
    assert payload["error_code"] == "OUTPUT_WRITE_FAILED"
    assert "Traceback" not in captured.err


def test_recorded_cache_replay_has_no_transport_requests_and_preserves_candidates(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    arguments = [
        "--question",
        QUESTION,
        "--mode",
        "recorded",
        "--cache-dir",
        str(cache_dir),
    ]

    assert main([*arguments, "--output-dir", str(first_output)]) == 0
    assert main([*arguments, "--output-dir", str(second_output)]) == 0

    first = json.loads((first_output / "first-round.json").read_text(encoding="utf-8"))
    second = json.loads((second_output / "first-round.json").read_text(encoding="utf-8"))
    assert first["metrics"]["transport_requests"] > 0
    assert second["metrics"]["transport_requests"] == 0
    assert second["metrics"]["cache_hits"] > 0
    assert all(result["cache_hit"] for result in second["query_results"])
    assert first["candidates"] == second["candidates"]


def test_real_cli_uses_the_adapter_default_sleeper_and_a_positive_interval(tmp_path: Path) -> None:
    arguments = _parse_arguments(
        ["--question", QUESTION, "--output-dir", str(tmp_path), "--mode", "real"]
    )
    config = _config_from_arguments(arguments)
    adapter = _adapter_from_arguments(arguments, config)

    assert config.min_request_interval_seconds == 3.0
    assert adapter._sleeper is arxiv.time.sleep
