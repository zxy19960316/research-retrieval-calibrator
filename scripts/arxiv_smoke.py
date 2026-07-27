"""Run one explicitly requested, bounded real-network arXiv adapter smoke check."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.adapters.arxiv import ArxivAdapter, ArxivAdapterConfig, ArxivAdapterError
from app.models.enums import QueryBranch, QueryBreadth
from app.models.query import Query


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default='all:"radiation shielding"')
    parser.add_argument("--max-results", type=int, default=1)
    parser.add_argument("--user-agent", default="research-retrieval-calibrator/0.1 (M1-T02 smoke)")
    arguments = parser.parse_args()
    query = Query(
        query_id="Q1-m1-t02-smoke",
        round_number=1,
        branch=QueryBranch.DIRECT_INTERSECTION,
        breadth=QueryBreadth.NARROW,
        language="en",
        query_text=arguments.query,
        weight=1.0,
    )
    adapter = ArxivAdapter(
        ArxivAdapterConfig(
            user_agent=arguments.user_agent,
            timeout_seconds=20.0,
            page_size=arguments.max_results,
            min_request_interval_seconds=3.0,
            max_attempts=3,
            initial_backoff_seconds=3.0,
        )
    )
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        papers = adapter.search(query, max_results=arguments.max_results)
    except ArxivAdapterError as error:
        print(
            json.dumps(
                {
                    "evidence_type": "real_external",
                    "status": "failed",
                    "observed_at_utc": observed_at,
                    "query": arguments.query,
                    "error_code": error.code,
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "evidence_type": "real_external",
                "status": "passed" if papers else "failed",
                "observed_at_utc": observed_at,
                "query": arguments.query,
                "result_count": len(papers),
                "source_id_sample": [paper.source_id for paper in papers[:3]],
            },
            sort_keys=True,
        )
    )
    return 0 if papers else 1


if __name__ == "__main__":
    raise SystemExit(main())
