"""Safe, deterministic command-line rendering for M1-T04 first-round retrieval."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NoReturn
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from app.adapters.arxiv import (
    ArxivAdapter,
    ArxivAdapterConfig,
    ArxivResponse,
    UrllibArxivTransport,
)
from app.core.first_round import run_first_round
from app.models.first_round import FirstRoundConfig, FirstRoundRun, FirstRoundStatus

_DEFAULT_USER_AGENT = "research-retrieval-calibrator/0.1 (M1-T04)"
_FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "arxiv"
Writer = Callable[[Path, str], None]


class _CliArgumentError(ValueError):
    """Avoid argparse's human text and retain the stable JSON failure surface."""


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _CliArgumentError(message)


class RecordedArxivTransport:
    """A fixture-only transport used by recorded CLI replays and tests."""

    def __init__(self, fixtures_dir: Path = _FIXTURES) -> None:
        self._graph = (fixtures_dir / "first_round_graph.xml").read_bytes()
        self._empty = (fixtures_dir / "first_round_empty.xml").read_bytes()
        self.response_count = 0

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> ArxivResponse:
        del headers, timeout_seconds
        self.response_count += 1
        query = parse_qs(urlparse(url).query).get("search_query", [""])[0]
        body = self._empty if "recorded-empty" in query else self._graph
        return ArxivResponse(200, body, {})


def render_json(run: FirstRoundRun) -> str:
    """Serialize the closed run contract in a stable, machine-readable form."""

    return json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_markdown(run: FirstRoundRun) -> str:
    """Render only source-backed candidate fields; never infer paper metadata."""

    lines = ["# First-round retrieval", "", f"Status: `{run.status.value}`", ""]
    if run.failure is not None:
        lines.extend([f"Failure: `{run.failure.error_code}`", ""])
    lines.extend(["## Candidates", ""])
    for index, candidate in enumerate(run.candidates, start=1):
        lines.extend(
            [
                f"### {index}. {candidate.title}",
                "",
                f"- Authors: {', '.join(candidate.authors) if candidate.authors else 'Not provided'}",
                f"- Year: {candidate.year if candidate.year is not None else 'Not provided'}",
                f"- Source: {candidate.source}",
                f"- Source ID: `{candidate.source_id}`",
                f"- URL: {candidate.url}",
                f"- Retrieval paths: {', '.join(candidate.retrieval_paths)}",
                "- Merge reasons: "
                + (
                    ", ".join(decision.reason.value for decision in candidate.merge_reasons)
                    if candidate.merge_reasons
                    else "None"
                ),
                "- Manual review required: "
                + ("yes" if any(item.action == "manual_review" for item in candidate.merge_reasons) else "no"),
                "",
            ]
        )
    if not run.candidates:
        lines.extend(["No source-backed candidates were produced.", ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, *, writer: Writer | None = None) -> int:
    """Run the bounded CLI, reserving stdout for structured failure JSON only."""

    try:
        arguments = _parse_arguments(argv)
        config = _config_from_arguments(arguments)
        adapter = _adapter_from_arguments(arguments, config)
    except (_CliArgumentError, ValidationError, ValueError) as error:
        _print_failure("INVALID_CLI_ARGUMENT", str(error))
        return 2

    try:
        run = run_first_round(arguments.question, config=config, adapter=adapter)
    except Exception as error:  # noqa: BLE001 - the CLI must not emit tracebacks.
        _print_failure("FIRST_ROUND_RUN_FAILED", str(error))
        return 1

    destination = arguments.output_dir
    write = writer or _write_text
    try:
        write(destination / "first-round.json", render_json(run))
        write(destination / "first-round.md", render_markdown(run))
    except OSError as error:
        _print_failure("OUTPUT_WRITE_FAILED", str(error))
        return 1
    return 0 if run.status is FirstRoundStatus.SUCCESS else 1


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _JsonArgumentParser(add_help=False)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--user-agent", default=_DEFAULT_USER_AGENT)
    parser.add_argument("--max-results-per-query", type=int, default=5)
    parser.add_argument("--max-total-candidates", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--mode", choices=("recorded", "real"), default="recorded")
    arguments = parser.parse_args(argv)
    if not arguments.question.strip() or not arguments.user_agent.strip():
        raise _CliArgumentError("question and user-agent must be non-blank")
    if not arguments.output_dir.parent.is_dir():
        raise _CliArgumentError("output directory parent must already exist")
    if arguments.output_dir.exists() and not arguments.output_dir.is_dir():
        raise _CliArgumentError("output-dir must be a directory")
    return arguments


def _config_from_arguments(arguments: argparse.Namespace) -> FirstRoundConfig:
    cache_dir = arguments.cache_dir or arguments.output_dir.parent / ".first-round-cache"
    return FirstRoundConfig(
        max_results_per_query=arguments.max_results_per_query,
        max_total_candidates=arguments.max_total_candidates,
        timeout_seconds=arguments.timeout_seconds,
        cache_dir=cache_dir,
        mode=arguments.mode,
    )


def _adapter_from_arguments(arguments: argparse.Namespace, config: FirstRoundConfig) -> ArxivAdapter:
    transport = RecordedArxivTransport() if config.mode == "recorded" else UrllibArxivTransport()
    return ArxivAdapter(
        ArxivAdapterConfig(
            user_agent=arguments.user_agent,
            timeout_seconds=config.timeout_seconds,
            page_size=config.max_results_per_query,
            min_request_interval_seconds=0.0,
            max_attempts=1,
            max_total_results=config.max_results_per_query,
            max_total_attempts=config.max_total_attempts,
            initial_backoff_seconds=0.0,
            cache_dir=config.cache_dir,
            cache_schema_version=config.adapter_schema_version,
        ),
        transport=transport,
        sleeper=lambda _: None,
    )


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _print_failure(error_code: str, reason: str) -> None:
    print(
        json.dumps(
            {
                "error_code": error_code,
                "failure": {"error_code": error_code, "reason": reason, "scope": "run"},
                "status": "failed",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
