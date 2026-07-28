"""Targeted M2-T01 tests for the pre-replay frozen-input gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.freeze_m2_candidates import _canonical_json_sha256, validate_m1_freeze_input


def _synthetic_m1_json(*, mode: str = "real") -> bytes:
    payload = {
        "config": {"mode": mode},
        "candidates": [{"paper_id": f"arxiv:{index:05d}"} for index in range(33)],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def test_pure_validator_accepts_only_matching_real_33_candidate_payload() -> None:
    raw_output = _synthetic_m1_json()
    payload = json.loads(raw_output)

    assert validate_m1_freeze_input(
        raw_output,
        expected_output_sha256=hashlib.sha256(raw_output).hexdigest(),
        expected_candidate_array_sha256=_canonical_json_sha256(payload["candidates"]),
    ) is None

    non_real = _synthetic_m1_json(mode="recorded")
    assert validate_m1_freeze_input(
        non_real,
        expected_output_sha256=hashlib.sha256(non_real).hexdigest(),
        expected_candidate_array_sha256=_canonical_json_sha256(json.loads(non_real)["candidates"]),
    ) == "M1 output is not from real mode"


def test_missing_m1_source_emits_structured_block_and_writes_no_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "snapshots"
    command = [
        sys.executable,
        "scripts/freeze_m2_candidates.py",
        "--m1-output",
        str(tmp_path / "missing-first-round.json"),
        "--m1-cache-dir",
        str(tmp_path / "missing-real-cache"),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert completed.returncode != 0
    assert json.loads(completed.stdout) == {
        "error_code": "M2_T01_FROZEN_INPUT_MISSING",
        "reason": "accepted M1 output and first-round:real cache must both be present",
        "status": "blocked",
    }
    assert not output_dir.exists()
