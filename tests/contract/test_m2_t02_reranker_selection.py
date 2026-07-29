"""Offline contract for the B0 reranker model-selection artifact."""

from __future__ import annotations

import json
import re
from pathlib import Path

_ARTIFACT_PATH = Path("evaluation/source-artifacts/m2-t02-reranker-selection.json")
_TOP_LEVEL_FIELDS = {
    "report_version",
    "phase",
    "task_id",
    "baseline_commit",
    "decision_status",
    "selected_model",
    "alternatives",
    "runtime_policy",
    "source_files",
    "execution_state",
}
_SELECTED_MODEL_FIELDS = {
    "provider_name",
    "model_id",
    "model_revision",
    "license",
    "multilingual",
    "trust_remote_code",
    "runtime_library",
    "raw_score_semantics",
    "normalization_owner",
    "normalization_method",
    "huggingface_used_storage_bytes",
    "pinned_source_files_size_bytes",
    "required_runtime_files_size_bytes",
    "weight_size_bytes",
    "model_parameter_count",
    "standard_transformers_compatibility",
    "tokenizer_model_max_length",
    "model_max_position_embeddings",
    "cpu_feasibility_risk",
    "offline_replay_complexity",
    "commercial_use_constraint",
    "official_metadata_source",
    "official_model_card_url",
}
_OFFICIAL_METADATA_SOURCE_FIELDS = {
    "provider",
    "model_id",
    "resolved_revision",
    "blobs",
}
_RUNTIME_POLICY_FIELDS = {
    "device",
    "dtype",
    "max_length",
    "pair_truncation_strategy",
    "batch_size",
    "network_after_download",
}
_EXECUTION_STATE_FIELDS = {
    "weights_downloaded",
    "tokenizer_downloaded",
    "model_loaded",
    "inference_run",
    "real_scores_generated",
}
_SOURCE_FILE_FIELDS = {
    "path",
    "size_bytes",
    "digest",
    "digest_type",
    "storage_type",
    "required_for_runtime",
}
_ALTERNATIVE_FIELDS = {
    "model_id",
    "model_revision",
    "license",
    "multilingual",
    "trust_remote_code",
    "selection_status",
    "reason",
    "languages",
    "model_parameter_count",
    "weight_size_bytes",
    "standard_transformers_compatibility",
    "raw_score_semantics",
    "maximum_supported_input",
    "cpu_feasibility_risk",
    "offline_replay_complexity",
    "commercial_use_constraint",
}
_EXPECTED_SOURCE_FILES = {
    "README.md": {
        "size_bytes": 17229,
        "digest": "553540879ec61aea21df00434c984c0f760a3fcc",
        "digest_type": "git_blob_sha1",
        "storage_type": "git",
        "required_for_runtime": False,
    },
    "config.json": {
        "size_bytes": 795,
        "digest": "9f62673cb00ec41dcec8947b9ed16f6f2eb23ba2",
        "digest_type": "git_blob_sha1",
        "storage_type": "git",
        "required_for_runtime": True,
    },
    "model.safetensors": {
        "size_bytes": 2271071852,
        "digest": "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286",
        "digest_type": "sha256",
        "storage_type": "lfs",
        "required_for_runtime": True,
    },
    "sentencepiece.bpe.model": {
        "size_bytes": 5069051,
        "digest": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
        "digest_type": "sha256",
        "storage_type": "lfs",
        "required_for_runtime": True,
    },
    "special_tokens_map.json": {
        "size_bytes": 964,
        "digest": "b1879d702821e753ffe4245048eee415d54a9385",
        "digest_type": "git_blob_sha1",
        "storage_type": "git",
        "required_for_runtime": True,
    },
    "tokenizer.json": {
        "size_bytes": 17098273,
        "digest": "69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15",
        "digest_type": "sha256",
        "storage_type": "lfs",
        "required_for_runtime": True,
    },
    "tokenizer_config.json": {
        "size_bytes": 1173,
        "digest": "328a00a9a560aadcf2a3064f917517359eb3cc26",
        "digest_type": "git_blob_sha1",
        "storage_type": "git",
        "required_for_runtime": True,
    },
}
_EXPECTED_ALTERNATIVES = {
    "Alibaba-NLP/gte-multilingual-reranker-base": {
        "model_revision": "8215cf04918ba6f7b6a62bb44238ce2953d8831c",
        "license": "apache-2.0",
        "multilingual": True,
        "trust_remote_code": True,
        "selection_status": "backup_only_due_to_remote_custom_code",
    },
    "cross-encoder/ms-marco-MiniLM-L6-v2": {
        "model_revision": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
        "license": "apache-2.0",
        "multilingual": False,
        "trust_remote_code": False,
        "selection_status": "english_cpu_baseline_only",
    },
    "jinaai/jina-reranker-v2-base-multilingual": {
        "model_revision": "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9",
        "license": "cc-by-nc-4.0",
        "multilingual": True,
        "trust_remote_code": True,
        "selection_status": "not_selected_due_to_noncommercial_license_and_remote_custom_code",
    },
}


def _all_values(value: object) -> list[object]:
    if isinstance(value, dict):
        return [value, *[item for nested in value.values() for item in _all_values(nested)]]
    if isinstance(value, list):
        return [value, *[item for nested in value for item in _all_values(nested)]]
    return [value]


def test_b0_selection_artifact_is_closed_pinned_and_not_run() -> None:
    artifact = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert set(artifact) == _TOP_LEVEL_FIELDS
    assert artifact["report_version"] == "m2-t02-reranker-selection.v1"
    assert artifact["phase"] == "M2"
    assert artifact["task_id"] == "M2-T02"
    assert artifact["baseline_commit"] == "1ebcc5d2c0b0efe2172633ece0a90d7f3d9290fa"
    assert artifact["decision_status"] == "selected_not_downloaded"

    selected = artifact["selected_model"]
    assert set(selected) == _SELECTED_MODEL_FIELDS
    assert selected["provider_name"] == "bge_reranker_v2_m3"
    assert selected["model_id"] == "BAAI/bge-reranker-v2-m3"
    assert selected["model_revision"] == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert selected["license"] == "apache-2.0"
    assert selected["multilingual"] is True
    assert selected["trust_remote_code"] is False
    assert selected["runtime_library"] == "transformers"
    assert selected["raw_score_semantics"] == "higher_logit_is_more_relevant"
    assert selected["normalization_owner"] == "app.core.reranking"
    assert selected["normalization_method"] == "global_min_max"
    assert selected["huggingface_used_storage_bytes"] == 7975340915
    assert selected["pinned_source_files_size_bytes"] == 2293259337
    assert selected["required_runtime_files_size_bytes"] == 2293242108
    assert selected["weight_size_bytes"] == 2271071852
    assert selected["tokenizer_model_max_length"] == 8192
    assert selected["model_max_position_embeddings"] == 8194
    assert selected["official_metadata_source"] == {
        "provider": "huggingface_model_api",
        "model_id": "BAAI/bge-reranker-v2-m3",
        "resolved_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "blobs": True,
    }
    assert set(selected["official_metadata_source"]) == _OFFICIAL_METADATA_SOURCE_FIELDS

    runtime_policy = artifact["runtime_policy"]
    assert set(runtime_policy) == _RUNTIME_POLICY_FIELDS
    assert runtime_policy == {
        "device": "cpu_preflight_required",
        "dtype": "float32_initial_preflight",
        "max_length": 512,
        "pair_truncation_strategy": "longest_first",
        "batch_size": "to_be_determined_by_preflight",
        "network_after_download": "forbidden",
    }
    execution_state = artifact["execution_state"]
    assert set(execution_state) == _EXECUTION_STATE_FIELDS
    assert execution_state == {
        "weights_downloaded": False,
        "tokenizer_downloaded": False,
        "model_loaded": False,
        "inference_run": False,
        "real_scores_generated": False,
    }

    source_files = artifact["source_files"]
    assert {entry["path"] for entry in source_files} == _EXPECTED_SOURCE_FILES.keys()
    assert len({entry["path"] for entry in source_files}) == len(source_files)
    for entry in source_files:
        assert set(entry) == _SOURCE_FILE_FIELDS
        path = entry["path"]
        assert entry == {"path": path, **_EXPECTED_SOURCE_FILES[path]}
        assert (entry["storage_type"] == "git") is (entry["digest_type"] == "git_blob_sha1")
        assert (entry["storage_type"] == "lfs") is (entry["digest_type"] == "sha256")

    assert selected["pinned_source_files_size_bytes"] == sum(
        entry["size_bytes"] for entry in source_files
    )
    assert selected["required_runtime_files_size_bytes"] == sum(
        entry["size_bytes"] for entry in source_files if entry["required_for_runtime"]
    )
    assert selected["weight_size_bytes"] == next(
        entry["size_bytes"] for entry in source_files if entry["path"] == "model.safetensors"
    )

    alternatives = {entry["model_id"]: entry for entry in artifact["alternatives"]}
    assert alternatives.keys() == _EXPECTED_ALTERNATIVES.keys()
    for model_id, expected in _EXPECTED_ALTERNATIVES.items():
        entry = alternatives[model_id]
        assert set(entry) == _ALTERNATIVE_FIELDS
        assert entry["model_id"] == model_id
        assert entry["model_revision"] == expected["model_revision"]
        assert entry["license"] == expected["license"]
        assert entry["multilingual"] is expected["multilingual"]
        assert entry["trust_remote_code"] is expected["trust_remote_code"]
        assert entry["selection_status"] == expected["selection_status"]
        assert re.fullmatch(r"[0-9a-f]{40}", entry["model_revision"])

    serialized = json.dumps(artifact, ensure_ascii=False).lower()
    assert not re.search(r"(?<![a-z])(?:authorization|cookie|token)(?![a-z])", serialized)
    assert not any(
        isinstance(value, str) and (re.match(r"^[a-z]:[\\/]", value.lower()) or value.startswith("/"))
        for value in _all_values(artifact)
    )
    assert not any(
        isinstance(value, str) and value in {"main", "latest"} for value in _all_values(artifact)
    )
    assert not any(
        isinstance(value, dict) and {"real_score", "runtime_seconds", "benchmark"} & set(value)
        for value in _all_values(artifact)
    )
