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
_REQUIRED_SOURCE_FILES = {
    "README.md",
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
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
    assert selected["provider_name"] == "bge_reranker_v2_m3"
    assert selected["model_id"] == "BAAI/bge-reranker-v2-m3"
    assert re.fullmatch(r"[0-9a-f]{40}", selected["model_revision"])
    assert selected["license"] == "apache-2.0"
    assert selected["multilingual"] is True
    assert selected["trust_remote_code"] is False
    assert selected["runtime_library"] == "transformers"
    assert selected["raw_score_semantics"] == "higher_logit_is_more_relevant"
    assert selected["normalization_owner"] == "app.core.reranking.global_min_max"
    assert selected["repository_size_bytes"] > 0
    assert selected["weight_size_bytes"] > 0

    assert artifact["runtime_policy"] == {
        "device": "cpu_preflight_required",
        "dtype": "float32_initial_preflight",
        "max_length": 512,
        "truncation": "only_second_or_pair_truncation_explicitly_recorded",
        "batch_size": "to_be_determined_by_preflight",
        "network_after_download": "forbidden",
    }
    assert artifact["execution_state"] == {
        "weights_downloaded": False,
        "tokenizer_downloaded": False,
        "model_loaded": False,
        "inference_run": False,
        "real_scores_generated": False,
    }

    source_files = artifact["source_files"]
    assert {entry["path"] for entry in source_files} == _REQUIRED_SOURCE_FILES
    assert len({entry["path"] for entry in source_files}) == len(source_files)
    for entry in source_files:
        assert entry["size_bytes"] > 0
        assert entry["storage_type"] in {"git", "lfs"}
        assert entry["required_for_runtime"] is (entry["path"] != "README.md")
        if entry["digest_type"] == "sha256":
            assert re.fullmatch(r"[0-9a-f]{64}", entry["digest"])
        else:
            assert entry["digest_type"] == "git_blob_sha1"
            assert re.fullmatch(r"[0-9a-f]{40}", entry["digest"])

    alternatives = {entry["model_id"]: entry for entry in artifact["alternatives"]}
    assert alternatives.keys() == {
        "Alibaba-NLP/gte-multilingual-reranker-base",
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "jinaai/jina-reranker-v2-base-multilingual",
    }
    assert alternatives["Alibaba-NLP/gte-multilingual-reranker-base"]["selection_status"] == (
        "backup_only_due_to_remote_custom_code"
    )
    assert alternatives["cross-encoder/ms-marco-MiniLM-L6-v2"]["selection_status"] == (
        "english_cpu_baseline_only"
    )
    assert alternatives["jinaai/jina-reranker-v2-base-multilingual"]["selection_status"] == (
        "not_selected_due_to_noncommercial_license_and_remote_custom_code"
    )
    for entry in alternatives.values():
        assert set(entry) >= {
            "model_id",
            "license",
            "multilingual",
            "trust_remote_code",
            "selection_status",
            "reason",
        }

    serialized = json.dumps(artifact, ensure_ascii=False).lower()
    assert not re.search(r"(?<![a-z])(?:authorization|cookie|token)(?![a-z])", serialized)
    assert not any(
        isinstance(value, str) and (re.match(r"^[a-z]:[\\/]", value.lower()) or value.startswith("/"))
        for value in _all_values(artifact)
    )
    assert selected["model_revision"] not in {"main", "latest"}
    assert not any(
        isinstance(value, dict) and {"real_score", "runtime_seconds", "benchmark"} & set(value)
        for value in _all_values(artifact)
    )
