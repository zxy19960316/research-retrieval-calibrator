"""Run one explicitly requested, bounded real-network arXiv adapter smoke check."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.adapters.arxiv import ArxivAdapter, ArxivAdapterConfig, ArxivAdapterError
from app.models.enums import QueryBranch, QueryBreadth
from app.models.query import Query


def _invalid_argument() -> int:
    print(
        json.dumps(
            {
                "evidence_type": "real_external",
                "status": "failed",
                "error_code": "INVALID_SMOKE_ARGUMENT",
            },
            sort_keys=True,
        )
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default='all:"radiation shielding"')
    parser.add_argument("--max-results", type=int, default=1)
    parser.add_argument("--user-agent", default="research-retrieval-calibrator/0.1 (M1-T02 smoke)")
    arguments = parser.parse_args(argv)
    if arguments.max_results < 1 or not arguments.user_agent.strip() or not arguments.query.strip():
        return _invalid_argument()
    try:
        config = ArxivAdapterConfig(
            user_agent=arguments.user_agent,
            timeout_seconds=20.0,
            page_size=min(arguments.max_results, 100),
            min_request_interval_seconds=3.0,
            max_attempts=3,
            initial_backoff_seconds=3.0,
        )
        if arguments.max_results > config.max_total_results:
            return _invalid_argument()
        query = Query(
            query_id="Q1-m1-t02-smoke",
            round_number=1,
            branch=QueryBranch.DIRECT_INTERSECTION,
            breadth=QueryBreadth.NARROW,
            language="en",
            query_text=arguments.query,
            weight=1.0,
        )
    except ValueError:
        return _invalid_argument()
    adapter = ArxivAdapter(config)
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        papers = adapter.search(query, max_results=arguments.max_results)
    except ArxivAdapterError as error:
        observation = adapter.last_observation
        print(
            json.dumps(
                {
                    "evidence_type": "real_external",
                    "status": "failed",
                    "observed_at_utc": observed_at,
                    "query": arguments.query,
                    "error_code": error.code,
                    "attempt_count": observation.attempt_count,
                    "http_status": observation.http_status,
                    "retry_after_seconds": observation.retry_after_seconds,
                    "elapsed_seconds": observation.elapsed_seconds,
                },
                sort_keys=True,
            )
        )
        return 1
    observation = adapter.last_observation
    error_code = None if papers else "NO_CONFIRMED_ARXIV_RESULT"
    print(
        json.dumps(
            {
                "evidence_type": "real_external",
                "status": "passed" if papers else "failed",
                "observed_at_utc": observed_at,
                "query": arguments.query,
                "result_count": len(papers),
                "source_id_sample": [paper.source_id for paper in papers[:3]],
                "error_code": error_code,
                "attempt_count": observation.attempt_count,
                "http_status": observation.http_status,
                "retry_after_seconds": observation.retry_after_seconds,
                "elapsed_seconds": observation.elapsed_seconds,
            },
            sort_keys=True,
        )
    )
    return 0 if papers else 1


if __name__ == "__main__":
    raise SystemExit(main())
