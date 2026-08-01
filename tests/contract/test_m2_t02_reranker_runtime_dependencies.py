"""Offline contract for the M2-T02 reranker runtime-dependency selection lock."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

_ARTIFACT_PATH = Path("evaluation/source-artifacts/m2-t02-reranker-runtime-dependencies.json")
_TOP_LEVEL_FIELDS = {
    "report_version",
    "phase",
    "task_id",
    "baseline_commit",
    "decision_status",
    "python",
    "packages",
    "runtime_policy",
    "verification_state",
    "sources",
}
_PYTHON_FIELDS = {"requires_python", "tested_minor"}
_PACKAGE_FIELDS = {
    "name",
    "version",
    "role",
    "python_compatibility",
    "license",
    "source_type",
    "source_reference",
    "selection_reason",
}
_RUNTIME_POLICY_FIELDS = {
    "device",
    "dtype",
    "model_id",
    "model_revision",
    "local_files_only_after_download",
    "trust_remote_code",
    "use_safetensors",
    "network_during_preflight",
}
_VERIFICATION_STATE_FIELDS = {
    "dependencies_installed_in_clean_environment",
    "model_downloaded",
    "tokenizer_loaded",
    "model_loaded",
    "inference_run",
    "real_scores_generated",
}
_SOURCE_FIELDS = {"source_type", "source_reference", "queried_on"}
_EXPECTED_PACKAGES = {
    "torch": {
        "version": "2.7.1",
        "source_type": "pypi_project_metadata",
        "source_reference": "https://pypi.org/pypi/torch/2.7.1/json",
    },
    "transformers": {
        "version": "4.53.2",
        "source_type": "pypi_project_metadata",
        "source_reference": "https://pypi.org/pypi/transformers/4.53.2/json",
    },
    "huggingface-hub": {
        "version": "0.34.3",
        "source_type": "pypi_project_metadata",
        "source_reference": "https://pypi.org/pypi/huggingface-hub/0.34.3/json",
    },
    "safetensors": {
        "version": "0.5.3",
        "source_type": "pypi_project_metadata",
        "source_reference": "https://pypi.org/pypi/safetensors/0.5.3/json",
    },
    "tokenizers": {
        "version": "0.21.2",
        "source_type": "pypi_project_metadata",
        "source_reference": "https://pypi.org/pypi/tokenizers/0.21.2/json",
    },
}
_EXPECTED_SOURCES = {
    (
        "pypi_project_metadata",
        "https://pypi.org/pypi/torch/2.7.1/json",
    ),
    (
        "pytorch_installation_compatibility_documentation",
        "https://pytorch.org/get-started/previous-versions/",
    ),
    (
        "pypi_project_metadata",
        "https://pypi.org/pypi/transformers/4.53.2/json",
    ),
    (
        "transformers_installation_documentation",
        "https://huggingface.co/docs/transformers/v4.53.2/en/installation",
    ),
    (
        "transformers_official_release_notes",
        "https://github.com/huggingface/transformers/releases/tag/v4.53.2",
    ),
    (
        "pypi_project_metadata",
        "https://pypi.org/pypi/huggingface-hub/0.34.3/json",
    ),
    (
        "huggingface_hub_file_download_documentation",
        "https://huggingface.co/docs/huggingface_hub/v0.34.3/en/package_reference/file_download",
    ),
    (
        "huggingface_hub_official_release_notes",
        "https://github.com/huggingface/huggingface_hub/releases/tag/v0.34.3",
    ),
    (
        "pypi_project_metadata",
        "https://pypi.org/pypi/safetensors/0.5.3/json",
    ),
    (
        "pypi_project_metadata",
        "https://pypi.org/pypi/tokenizers/0.21.2/json",
    ),
}
_RUNTIME_POLICY = {
    "device": "cpu",
    "dtype": "float32",
    "model_id": "BAAI/bge-reranker-v2-m3",
    "model_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "local_files_only_after_download": True,
    "trust_remote_code": False,
    "use_safetensors": True,
    "network_during_preflight": "forbidden",
}
_FORBIDDEN_RESULT_KEYS = {
    "score",
    "scores",
    "real_score",
    "real_scores",
    "duration",
    "duration_seconds",
    "elapsed",
    "elapsed_seconds",
    "runtime_seconds",
    "benchmark",
    "benchmarks",
}


def _mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_mapping_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_mapping_keys(item) for item in value))
    return set()


def test_m2_t02_runtime_dependency_lock_is_closed_pinned_and_not_run() -> None:
    artifact = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert set(artifact) == _TOP_LEVEL_FIELDS
    assert artifact["report_version"] == "m2-t02-reranker-runtime-dependencies.v1"
    assert artifact["phase"] == "M2"
    assert artifact["task_id"] == "M2-T02"
    assert artifact["baseline_commit"] == "902ccea0eae1475db2143a68021cee70d2139b68"
    assert artifact["decision_status"] == "selected_not_installed"

    python = artifact["python"]
    assert set(python) == _PYTHON_FIELDS
    assert python == {"requires_python": ">=3.12,<3.13", "tested_minor": "3.12"}

    packages = artifact["packages"]
    assert [package["name"] for package in packages] == list(_EXPECTED_PACKAGES)
    assert {package["name"] for package in packages} == _EXPECTED_PACKAGES.keys()
    for package in packages:
        assert set(package) == _PACKAGE_FIELDS
        expected = _EXPECTED_PACKAGES[package["name"]]
        assert package["version"] == expected["version"]
        assert re.fullmatch(r"\d+\.\d+\.\d+", package["version"])
        assert not re.search(r"[<>=~*]|\b(?:latest|main)\b", package["version"], re.IGNORECASE)
        assert package["source_type"] == expected["source_type"]
        assert package["source_reference"] == expected["source_reference"]
        assert isinstance(package["license"], str) and package["license"].strip()
        assert isinstance(package["role"], str) and package["role"].strip()
        assert isinstance(package["python_compatibility"], str) and "3.12" in package[
            "python_compatibility"
        ]
        assert isinstance(package["selection_reason"], str) and package["selection_reason"].strip()

    runtime_policy = artifact["runtime_policy"]
    assert set(runtime_policy) == _RUNTIME_POLICY_FIELDS
    assert runtime_policy == _RUNTIME_POLICY

    verification_state = artifact["verification_state"]
    assert set(verification_state) == _VERIFICATION_STATE_FIELDS
    assert verification_state == dict.fromkeys(_VERIFICATION_STATE_FIELDS, False)

    sources = artifact["sources"]
    assert {
        (source["source_type"], source["source_reference"])
        for source in sources
    } == _EXPECTED_SOURCES
    for source in sources:
        assert set(source) == _SOURCE_FIELDS
        assert source["queried_on"] == "2026-08-01"
        parsed = urlparse(source["source_reference"])
        assert parsed.scheme == "https" and parsed.netloc and not parsed.username
        assert not re.search(
            r"(?:authorization|cookie|bearer\s|access[_-]?token)",
            " ".join(str(value) for value in source.values()),
            re.IGNORECASE,
        )

    source_pairs = {(source["source_type"], source["source_reference"]) for source in sources}
    assert {
        (package["source_type"], package["source_reference"]) for package in packages
    } <= source_pairs
    assert not _mapping_keys(artifact) & _FORBIDDEN_RESULT_KEYS
