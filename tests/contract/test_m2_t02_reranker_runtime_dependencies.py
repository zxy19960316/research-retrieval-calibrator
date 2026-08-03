"""Offline contract for the M2-T02 shared reranker runtime-dependency lock."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from urllib.parse import urlparse

from packaging.version import Version

_ARTIFACT_PATH = Path("evaluation/source-artifacts/m2-t02-reranker-runtime-dependencies.json")
_PYPROJECT_PATH = Path("pyproject.toml")
_TOP_LEVEL_FIELDS = {
    "report_version",
    "phase",
    "task_id",
    "baseline_commit",
    "decision_status",
    "python",
    "packages",
    "environment_policy",
    "compatibility_policy",
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
_ENVIRONMENT_POLICY_FIELDS = {
    "environment_strategy",
    "python_version",
    "torch_version",
    "torch_distribution",
    "torch_install_source",
    "combined_extras_install_required",
    "separate_runtime_environments",
}
_COMPATIBILITY_POLICY_FIELDS = {
    "flagembedding_version",
    "flagembedding_torch_requirement",
    "flagembedding_transformers_requirement",
    "transformers_torch_requirement",
    "transformers_hub_requirement",
    "transformers_tokenizers_requirement",
    "transformers_safetensors_requirement",
    "declared_requirements_satisfied",
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
        "version": "2.4.1",
        "license": "BSD-3-Clause",
        "source_type": "pypi_project_metadata",
        "source_reference": "https://pypi.org/pypi/torch/2.4.1/json",
        "selection_reason": (
            "Reuses the project's embedding torch==2.4.1 lock for a shared Python 3.12 "
            "environment: FlagEmbedding 1.3.5 declares torch>=1.6.0 and transformers "
            "4.53.2 declares torch>=2.1; 2.4.1 satisfies both, supports Python 3.12 and "
            "Windows, and PyTorch publishes an official CPU-only wheel index. Reusing the "
            "shared pin avoids an unnecessary shared-Torch upgrade risk."
        ),
    },
    "transformers": {
        "version": "4.53.2",
        "license": "Apache-2.0",
        "source_type": "pypi_project_metadata",
        "source_reference": "https://pypi.org/pypi/transformers/4.53.2/json",
    },
    "huggingface-hub": {
        "version": "0.34.3",
        "license": "Apache-2.0",
        "source_type": "pypi_project_metadata",
        "source_reference": "https://pypi.org/pypi/huggingface-hub/0.34.3/json",
    },
    "safetensors": {
        "version": "0.5.3",
        "license": "Apache-2.0",
        "source_type": "pypi_project_metadata",
        "source_reference": "https://pypi.org/pypi/safetensors/0.5.3/json",
    },
    "tokenizers": {
        "version": "0.21.2",
        "license": "Apache-2.0",
        "source_type": "pypi_project_metadata",
        "source_reference": "https://pypi.org/pypi/tokenizers/0.21.2/json",
    },
}
_EXPECTED_ENVIRONMENT_POLICY = {
    "environment_strategy": "shared_embedding_and_reranker",
    "python_version": "3.12",
    "torch_version": "2.4.1",
    "torch_distribution": "cpu_only",
    "torch_install_source": "official_pytorch_cpu_index",
    "combined_extras_install_required": True,
    "separate_runtime_environments": False,
}
_EXPECTED_COMPATIBILITY_POLICY = {
    "flagembedding_version": "1.3.5",
    "flagembedding_torch_requirement": ">=1.6.0",
    "flagembedding_transformers_requirement": ">=4.44.2",
    "transformers_torch_requirement": ">=2.1",
    "transformers_hub_requirement": ">=0.30.0,<1.0",
    "transformers_tokenizers_requirement": ">=0.21,<0.22",
    "transformers_safetensors_requirement": ">=0.4.3",
    "declared_requirements_satisfied": True,
}
_EXPECTED_RERANKER_REQUIREMENTS = [
    "torch==2.4.1",
    "transformers==4.53.2",
    "huggingface-hub==0.34.3",
    "safetensors==0.5.3",
    "tokenizers==0.21.2",
]
_EXPECTED_SOURCES = {
    ("pypi_project_metadata", "https://pypi.org/pypi/torch/2.4.1/json"),
    (
        "pytorch_previous_versions_cpu_installation_documentation",
        "https://pytorch.org/get-started/previous-versions/",
    ),
    ("pypi_project_metadata", "https://pypi.org/pypi/FlagEmbedding/1.3.5/json"),
    ("pypi_project_metadata", "https://pypi.org/pypi/transformers/4.53.2/json"),
    (
        "transformers_installation_documentation",
        "https://huggingface.co/docs/transformers/v4.53.2/en/installation",
    ),
    (
        "transformers_official_release_notes",
        "https://github.com/huggingface/transformers/releases/tag/v4.53.2",
    ),
    ("pypi_project_metadata", "https://pypi.org/pypi/huggingface-hub/0.34.3/json"),
    (
        "huggingface_hub_file_download_documentation",
        "https://huggingface.co/docs/huggingface_hub/v0.34.3/en/package_reference/file_download",
    ),
    (
        "huggingface_hub_official_release_notes",
        "https://github.com/huggingface/huggingface_hub/releases/tag/v0.34.3",
    ),
    ("pypi_project_metadata", "https://pypi.org/pypi/safetensors/0.5.3/json"),
    ("pypi_project_metadata", "https://pypi.org/pypi/tokenizers/0.21.2/json"),
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
_LOCAL_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^\\\\|/(?:Users|home|tmp)/)")


def _mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_mapping_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_mapping_keys(item) for item in value))
    return set()


def _embedding_torch_version(requirements: list[object]) -> str:
    torch_requirements = [
        requirement for requirement in requirements if isinstance(requirement, str) and requirement.startswith("torch==")
    ]
    assert len(torch_requirements) == 1
    return torch_requirements[0].removeprefix("torch==")


def test_m2_t02_runtime_dependency_lock_is_closed_shared_and_clean_verified() -> None:
    artifact = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
    pyproject = tomllib.loads(_PYPROJECT_PATH.read_text(encoding="utf-8"))

    assert set(artifact) == _TOP_LEVEL_FIELDS
    assert artifact["report_version"] == "m2-t02-reranker-runtime-dependencies.v1.2"
    assert artifact["phase"] == "M2"
    assert artifact["task_id"] == "M2-T02"
    assert artifact["baseline_commit"] == "00a323d89bbca08d21e13322da3a9b7c25e3ffa5"
    assert artifact["decision_status"] == "selected_and_clean_environment_verified"

    python = artifact["python"]
    assert set(python) == _PYTHON_FIELDS
    assert python == {"requires_python": ">=3.12,<3.13", "tested_minor": "3.12"}

    packages = artifact["packages"]
    assert [package["name"] for package in packages] == list(_EXPECTED_PACKAGES)
    assert {package["name"] for package in packages} == _EXPECTED_PACKAGES.keys()
    selected_packages = {package["name"]: package for package in packages}
    for package in packages:
        assert set(package) == _PACKAGE_FIELDS
        expected = _EXPECTED_PACKAGES[package["name"]]
        assert package["version"] == expected["version"]
        assert package["license"] == expected["license"]
        assert re.fullmatch(r"\d+\.\d+\.\d+", package["version"])
        assert not re.search(r"[<>=~*]|\b(?:latest|main)\b", package["version"], re.IGNORECASE)
        assert package["source_type"] == expected["source_type"]
        assert package["source_reference"] == expected["source_reference"]
        assert isinstance(package["role"], str) and package["role"].strip()
        assert isinstance(package["python_compatibility"], str) and "3.12" in package[
            "python_compatibility"
        ]
        assert isinstance(package["selection_reason"], str) and package["selection_reason"].strip()
    assert selected_packages["torch"]["selection_reason"] == _EXPECTED_PACKAGES["torch"][
        "selection_reason"
    ]

    environment_policy = artifact["environment_policy"]
    assert set(environment_policy) == _ENVIRONMENT_POLICY_FIELDS
    assert environment_policy == _EXPECTED_ENVIRONMENT_POLICY

    embedding_requirements = pyproject["project"]["optional-dependencies"]["embedding"]
    assert "FlagEmbedding==1.3.5" in embedding_requirements
    embedding_torch_version = _embedding_torch_version(embedding_requirements)
    assert embedding_torch_version == "2.4.1"
    assert pyproject["project"]["optional-dependencies"]["reranker"] == _EXPECTED_RERANKER_REQUIREMENTS
    assert selected_packages["torch"]["version"] == embedding_torch_version
    assert environment_policy["torch_version"] == embedding_torch_version

    compatibility_policy = artifact["compatibility_policy"]
    assert set(compatibility_policy) == _COMPATIBILITY_POLICY_FIELDS
    assert compatibility_policy == _EXPECTED_COMPATIBILITY_POLICY
    assert Version(selected_packages["torch"]["version"]) >= Version("1.6.0")
    assert Version(selected_packages["transformers"]["version"]) >= Version("4.44.2")
    assert Version(selected_packages["torch"]["version"]) >= Version("2.1")
    assert Version(selected_packages["huggingface-hub"]["version"]) >= Version("0.30.0")
    assert Version(selected_packages["huggingface-hub"]["version"]) < Version("1.0")
    assert Version(selected_packages["tokenizers"]["version"]) >= Version("0.21")
    assert Version(selected_packages["tokenizers"]["version"]) < Version("0.22")
    assert Version(selected_packages["safetensors"]["version"]) >= Version("0.4.3")

    runtime_policy = artifact["runtime_policy"]
    assert set(runtime_policy) == _RUNTIME_POLICY_FIELDS
    assert runtime_policy == _RUNTIME_POLICY

    verification_state = artifact["verification_state"]
    assert set(verification_state) == _VERIFICATION_STATE_FIELDS
    assert verification_state == {
        "dependencies_installed_in_clean_environment": True,
        "model_downloaded": False,
        "tokenizer_loaded": False,
        "model_loaded": False,
        "inference_run": False,
        "real_scores_generated": False,
    }

    sources = artifact["sources"]
    source_pairs = [(source["source_type"], source["source_reference"]) for source in sources]
    assert set(source_pairs) == _EXPECTED_SOURCES
    assert len(source_pairs) == len(set(source_pairs))
    for source in sources:
        assert set(source) == _SOURCE_FIELDS
        assert source["queried_on"] == "2026-08-01"
        parsed = urlparse(source["source_reference"])
        assert parsed.scheme == "https" and parsed.netloc
        assert not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
        assert not _LOCAL_ABSOLUTE_PATH.search(source["source_reference"])
        assert not re.search(
            r"(?:authorization|cookie|bearer\s|access[_-]?token)",
            " ".join(str(value) for value in source.values()),
            re.IGNORECASE,
        )

    assert {
        (package["source_type"], package["source_reference"]) for package in packages
    } <= set(source_pairs)
    assert not _mapping_keys(artifact) & _FORBIDDEN_RESULT_KEYS
