"""Dependency-free canonical text comparison for deterministic planning contracts."""

import re
import unicodedata


def normalise_text(value: str) -> str:
    """Apply NFKC, trim, whitespace collapse, and casefold exactly once."""

    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip()).casefold()
