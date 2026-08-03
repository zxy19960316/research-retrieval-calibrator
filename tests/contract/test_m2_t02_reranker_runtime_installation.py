"""Offline contract for M2-T02 clean shared-runtime installation evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path

_ARTIFACT_PATH = Path("evaluation/source-artifacts/m2-t02-reranker-runtime-installation.json")
_TOP_LEVEL_FIELDS = {
    "report_version",
    "phase",
    "task_id",
    "baseline_commit",
    "decision_status",
    "platform",
    "installation_policy",
    "installed_packages",
    "verification",
    "execution_state",
}
_PLATFORM_FIELDS = {"system", "machine", "python_version", "environment_type"}
_INSTALLATION_POLICY_FIELDS = {
    "torch_index_url",
    "torch_requested_version",
    "torch_distribution",
    "combined_extras",
    "editable_install",
    "hf_hub_disable_implicit_token",
    "hf_hub_disable_telemetry",
    "preinstalled_runtime_packages",
    "network_access",
}
_PACKAGE_FIELDS = {"name", "expected_version", "installed_version", "matched"}
_VERIFICATION_FIELDS = {
    "torch_base_version",
    "torch_cuda_version",
    "torch_cuda_available",
    "pip_check",
    "all_imports",
    "imports",
}
_EXECUTION_STATE_FIELDS = {
    "dependencies_installed_in_clean_environment",
    "model_downloaded",
    "tokenizer_loaded",
    "model_loaded",
    "inference_run",
    "real_scores_generated",
}
_EXPECTED_PACKAGES = {
    "FlagEmbedding": "1.3.5",
    "torch": "2.4.1",
    "transformers": "4.53.2",
    "huggingface-hub": "0.34.3",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.2",
}
_EXPECTED_IMPORTS = {
    "torch",
    "transformers",
    "huggingface_hub",
    "safetensors",
    "tokenizers",
    "FlagEmbedding",
}
_LOCAL_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^\\\\|/(?:Users|home|tmp)/)")
_FORBIDDEN_TEXT = re.compile(
    r"(?:authorization|cookie|bearer\\s|access[_-]?token|api[_-]?key|password|"
    r"benchmark|duration|elapsed|runtime_seconds|username)",
    re.IGNORECASE,
)


def _string_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [string for child in value.values() for string in _string_values(child)]
    if isinstance(value, list):
        return [string for child in value for string in _string_values(child)]
    return [value] if isinstance(value, str) else []


def _matches_expected_version(name: str, expected: str, installed: str) -> bool:
    if name == "torch":
        return installed == expected or installed.startswith(f"{expected}+")
    return installed == expected


def test_m2_t02_clean_runtime_installation_evidence_is_closed_and_safe() -> None:
    artifact = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert set(artifact) == _TOP_LEVEL_FIELDS
    assert artifact["report_version"] == "m2-t02-reranker-runtime-installation.v1"
    assert artifact["phase"] == "M2"
    assert artifact["task_id"] == "M2-T02"
    assert artifact["baseline_commit"] == "00a323d89bbca08d21e13322da3a9b7c25e3ffa5"
    assert artifact["decision_status"] == "installed_in_clean_environment"

    platform = artifact["platform"]
    assert set(platform) == _PLATFORM_FIELDS
    assert platform["system"] == "Windows"
    assert re.fullmatch(r"3\.12\.\d+", platform["python_version"])
    assert isinstance(platform["machine"], str) and platform["machine"].strip()
    assert platform["environment_type"] == "clean_temporary_venv"

    policy = artifact["installation_policy"]
    assert set(policy) == _INSTALLATION_POLICY_FIELDS
    assert policy["torch_index_url"] == "https://download.pytorch.org/whl/cpu"
    assert policy["torch_requested_version"] == "2.4.1"
    assert policy["torch_distribution"] == "cpu_only"
    assert policy["combined_extras"] == ["embedding", "reranker"]
    assert policy["editable_install"] is True
    assert policy["hf_hub_disable_implicit_token"] is True
    assert policy["hf_hub_disable_telemetry"] is True
    assert policy["preinstalled_runtime_packages"] == []
    assert policy["network_access"] == "pytorch_cpu_index_and_pypi_package_installation_only"

    packages = artifact["installed_packages"]
    assert isinstance(packages, list)
    assert [package["name"] for package in packages] == list(_EXPECTED_PACKAGES)
    for package in packages:
        assert set(package) == _PACKAGE_FIELDS
        expected = _EXPECTED_PACKAGES[package["name"]]
        assert package["expected_version"] == expected
        assert isinstance(package["installed_version"], str) and package["installed_version"]
        assert package["matched"] is True
        assert _matches_expected_version(package["name"], expected, package["installed_version"])

    verification = artifact["verification"]
    assert set(verification) == _VERIFICATION_FIELDS
    assert verification["torch_base_version"] == "2.4.1"
    assert verification["torch_cuda_version"] is None
    assert verification["torch_cuda_available"] is False
    assert verification["pip_check"] == "passed"
    assert verification["all_imports"] == "passed"
    assert set(verification["imports"]) == _EXPECTED_IMPORTS

    execution_state = artifact["execution_state"]
    assert set(execution_state) == _EXECUTION_STATE_FIELDS
    assert execution_state == {
        "dependencies_installed_in_clean_environment": True,
        "model_downloaded": False,
        "tokenizer_loaded": False,
        "model_loaded": False,
        "inference_run": False,
        "real_scores_generated": False,
    }

    for value in _string_values(artifact):
        assert not _LOCAL_ABSOLUTE_PATH.search(value)
        assert not _FORBIDDEN_TEXT.search(value)
