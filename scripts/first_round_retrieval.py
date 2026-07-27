"""Repository-root entrypoint for the M1-T04 first-round retrieval CLI."""

from __future__ import annotations

import sys
from pathlib import Path


def _main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app.cli.first_round import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
