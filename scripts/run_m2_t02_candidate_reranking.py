"""Run the future-only, fixed-candidate M2-T02 reranking contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if __package__ in {None, ""}:
    repository_root_text = str(_REPOSITORY_ROOT)
    if repository_root_text not in sys.path:
        sys.path.insert(0, repository_root_text)

from app.adapters.reranking import RerankerProvider
from app.core.reranking import rerank_candidates
from app.models.embedding import FrozenCandidate
from app.models.reranking import (
    BGE_RERANKER_MODEL_ID,
    BGE_RERANKER_MODEL_REVISION,
    BGE_RERANKER_PROVIDER_NAME,
    ProviderRawScore,
    RerankerModelDescriptor,
    RerankerRunState,
    RerankerTaskError,
    RerankRun,
)
from scripts.run_m2_t02_reranker_preflight import validate_preflight_evidence

CANDIDATE_SNAPSHOT_PATH = Path("evaluation/snapshots/m2/m1-candidates.v1.json")
CANDIDATE_MANIFEST_PATH = Path("evaluation/snapshots/m2/m1-candidates.v1.manifest.json")
PREFLIGHT_EVIDENCE_PATH = Path("evaluation/source-artifacts/m2-t02-reranker-preflight.json")
PREFLIGHT_RECEIPT_PATH = Path(
    "evaluation/source-artifacts/m2-t02-reranker-preflight-run-receipt.json"
)
MODEL_SNAPSHOT_PATH = Path(
    "models/m2-t02/bge-reranker-v2-m3/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
)
FORMAL_RESULT_PATH = Path(
    "evaluation/source-artifacts/m2-t02-reranker-candidate-run.json"
)

FIXED_QUERY = "How can graph-based retrieval support scientific literature discovery?"
BASELINE_COMMIT = "cdadca0ad4b49098aa115a9c05b9728ccd8eb7a0"
PREFLIGHT_EVIDENCE_SHA256 = (
    "103a73cc82674c2c1f1139a72a7a3d1a880029a7c01d292f52de24d6dc781b06"
)
PREFLIGHT_EXECUTION_COMMIT = "a50f1b9f5467c318dbbabe8807df6ede47d79b14"
CONFIGURED_TOP_K = 50
EXPECTED_CANDIDATE_COUNT = 33
BATCH_SIZE = 2
MAX_LENGTH = 512
EXPECTED_BATCH_SIZES = [2] * 16 + [1]

_RECEIPT_FIELDS = {
    "report_version",
    "phase",
    "task_id",
    "execution_commit",
    "evidence_path",
    "evidence_sha256",
    "runner_path",
    "runner_sha256",
    "selection_path",
    "selection_sha256",
    "runtime_installation_path",
    "runtime_installation_sha256",
    "snapshot_download_evidence_path",
    "snapshot_download_evidence_sha256",
    "exit_code",
    "run_count",
    "decision_status",
    "formal_evidence_created",
    "model_rerun_performed",
    "full_candidate_reranking_run",
}
_REPORT_FIELDS = {
    "report_version",
    "phase",
    "task_id",
    "baseline_commit",
    "decision_status",
    "input",
    "model",
    "runtime",
    "policy",
    "execution",
    "records",
}
_INPUT_FIELDS = {
    "candidate_snapshot_path",
    "candidate_manifest_path",
    "candidate_snapshot_sha256",
    "preflight_evidence_sha256",
    "preflight_receipt_execution_commit",
    "candidate_count",
    "abstract_present_count",
    "abstract_missing_count",
    "question",
    "paper_id_order",
}
_MODEL_FIELDS = {
    "provider_name",
    "model_id",
    "revision",
    "snapshot_path",
    "preparation_status",
}
_RUNTIME_FIELDS = {
    "device",
    "dtype",
    "max_length",
    "local_files_only",
    "trust_remote_code",
}
_POLICY_FIELDS = {
    "configured_top_k",
    "effective_top_k",
    "batch_size",
    "parallelism",
    "cache_policy",
    "network_forbidden",
}
_EXECUTION_FIELDS = {
    "provider_instance_count",
    "runtime_load_count",
    "provider_call_count",
    "provider_scored_count",
    "batch_sizes",
    "cache_hits",
}
_RECORD_FIELDS = {
    "rank",
    "paper_id",
    "input_sha256",
    "raw_score",
    "normalized_score",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_LOCAL_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^\\\\|/(?:Users|home|tmp)/)")
_FORBIDDEN_TEXT = re.compile(
    r"(?:https?://|file://|authorization|cookie|bearer\s|access[_-]?token|"
    r"api[_-]?key|password|username|hostname|raw[_-]?logit|tokenizer)",
    re.IGNORECASE,
)

CandidateRerankingErrorCode = Literal[
    "LOCAL_RERANKING_NOT_ENABLED",
    "PREFLIGHT_EVIDENCE_INVALID",
    "PREFLIGHT_RECEIPT_INVALID",
    "CANDIDATE_INPUT_INVALID",
    "MODEL_SNAPSHOT_INVALID",
    "PROVIDER_UNAVAILABLE",
    "INVALID_OUTPUT",
    "RESULT_SCHEMA_INVALID",
    "RESULT_CONFLICT",
    "RESULT_PUBLICATION_FAILED",
]
_ERROR_CODES = frozenset(
    {
        "LOCAL_RERANKING_NOT_ENABLED",
        "PREFLIGHT_EVIDENCE_INVALID",
        "PREFLIGHT_RECEIPT_INVALID",
        "CANDIDATE_INPUT_INVALID",
        "MODEL_SNAPSHOT_INVALID",
        "PROVIDER_UNAVAILABLE",
        "INVALID_OUTPUT",
        "RESULT_SCHEMA_INVALID",
        "RESULT_CONFLICT",
        "RESULT_PUBLICATION_FAILED",
    }
)


class CandidateRerankingError(RuntimeError):
    """A closed, privacy-safe failure at the fixed-candidate runner boundary."""

    def __init__(self, code: CandidateRerankingErrorCode | str) -> None:
        if code not in _ERROR_CODES:
            raise ValueError(f"Unknown candidate reranking error code: {code}")
        self.code: CandidateRerankingErrorCode = cast(CandidateRerankingErrorCode, code)
        super().__init__(code)


@dataclass(frozen=True)
class CandidateRunPaths:
    """Fixed repository paths, injectable only by Python tests."""

    candidate_snapshot_path: Path | str = CANDIDATE_SNAPSHOT_PATH
    candidate_manifest_path: Path | str = CANDIDATE_MANIFEST_PATH
    preflight_evidence_path: Path | str = PREFLIGHT_EVIDENCE_PATH
    preflight_receipt_path: Path | str = PREFLIGHT_RECEIPT_PATH
    model_snapshot_path: Path | str = MODEL_SNAPSHOT_PATH
    result_path: Path | str = FORMAL_RESULT_PATH


@dataclass(frozen=True)
class _FixedInputContext:
    candidates: tuple[FrozenCandidate, ...]
    paper_id_order: tuple[str, ...]
    snapshot_sha256: str
    preflight_evidence_sha256: str
    preflight_execution_commit: str


class CountingRerankerProvider:
    """Count serial provider calls without changing the provider contract."""

    def __init__(self, provider: RerankerProvider) -> None:
        self._provider = provider
        self.provider_instance_count = 1
        self.provider_call_count = 0
        self.provider_scored_count = 0
        self.batch_sizes: list[int] = []

    @property
    def descriptor(self) -> RerankerModelDescriptor:
        return self._provider.descriptor

    @property
    def runtime_load_count(self) -> int:
        value = getattr(self._provider, "runtime_load_count", 1)
        return value if _strict_int(value) else 1

    def score(
        self,
        query: str,
        inputs: Sequence[Any],
        *,
        batch_size: int,
    ) -> list[ProviderRawScore]:
        self.provider_call_count += 1
        self.batch_sizes.append(len(inputs))
        result = self._provider.score(query, inputs, batch_size=batch_size)
        try:
            self.provider_scored_count += len(result)
        except TypeError:
            pass
        return result


def _operational_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _REPOSITORY_ROOT / candidate


def _is_link_or_junction(path: Path, metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    checker = getattr(path, "is_junction", None)
    if callable(checker):
        try:
            return bool(checker())
        except OSError:
            return True
    return False


def _read_regular_file(path: Path, code: CandidateRerankingErrorCode) -> bytes:
    try:
        metadata = path.lstat()
        if _is_link_or_junction(path, metadata) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if _is_link_or_junction(path, opened) or not stat.S_ISREG(opened.st_mode):
                raise ValueError
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise ValueError
            payload = handle.read()
            final = os.fstat(handle.fileno())
            if (final.st_dev, final.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise ValueError
            return payload
    except CandidateRerankingError:
        raise
    except Exception:  # noqa: BLE001
        raise CandidateRerankingError(code) from None


def _read_json_file(path: Path, code: CandidateRerankingErrorCode) -> tuple[bytes, object]:
    raw = _read_regular_file(path, code)
    try:
        return raw, json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001
        raise CandidateRerankingError(code) from None


def _require_ordinary_directory(path: Path, code: CandidateRerankingErrorCode) -> None:
    try:
        metadata = path.lstat()
        if _is_link_or_junction(path, metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError
    except CandidateRerankingError:
        raise
    except Exception:  # noqa: BLE001
        raise CandidateRerankingError(code) from None


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError
    return cast(dict[str, object], value)


def _strict_int(value: object) -> bool:
    return type(value) is int


def _strict_bool(value: object) -> bool:
    return type(value) is bool


def _require_hex(value: object, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError
    return value


def _require_relative_identifier(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError
    path = Path(value)
    if path.is_absolute() or "\\" in value or ".." in path.parts:
        raise ValueError
    if _LOCAL_ABSOLUTE_PATH.search(value):
        raise ValueError
    return value


def _string_values(value: object) -> list[str]:
    if isinstance(value, dict):
        result: list[str] = []
        for item in value.values():
            result.extend(_string_values(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_string_values(item))
        return result
    return [value] if isinstance(value, str) else []


def _validate_safe_strings(value: object) -> None:
    for item in _string_values(value):
        if _LOCAL_ABSOLUTE_PATH.search(item) or _FORBIDDEN_TEXT.search(item):
            raise ValueError


def _validate_preflight_receipt(receipt: object) -> None:
    actual = _as_mapping(receipt)
    if set(actual) != _RECEIPT_FIELDS:
        raise ValueError
    if actual["report_version"] != "m2-t02-reranker-preflight-run-receipt.v1":
        raise ValueError
    if actual["phase"] != "M2" or actual["task_id"] != "M2-T02":
        raise ValueError
    if actual["execution_commit"] != PREFLIGHT_EXECUTION_COMMIT:
        raise ValueError
    if actual["evidence_path"] != PREFLIGHT_EVIDENCE_PATH.as_posix():
        raise ValueError
    if actual["evidence_sha256"] != PREFLIGHT_EVIDENCE_SHA256:
        raise ValueError
    for field in (
        "runner_path",
        "selection_path",
        "runtime_installation_path",
        "snapshot_download_evidence_path",
    ):
        _require_relative_identifier(actual[field])
    for field in (
        "runner_sha256",
        "selection_sha256",
        "runtime_installation_sha256",
        "snapshot_download_evidence_sha256",
    ):
        _require_hex(actual[field], _HEX64)
    if actual["exit_code"] != 0 or actual["run_count"] != 1:
        raise ValueError
    if actual["decision_status"] != "cpu_float32_preflight_passed":
        raise ValueError
    if (
        actual["formal_evidence_created"] is not True
        or actual["model_rerun_performed"] is not False
        or actual["full_candidate_reranking_run"] is not False
    ):
        raise ValueError
    _validate_safe_strings(actual)


def _validate_preflight_inputs(paths: CandidateRunPaths) -> tuple[str, str]:
    evidence_path = _operational_path(paths.preflight_evidence_path)
    receipt_path = _operational_path(paths.preflight_receipt_path)
    evidence_raw, evidence = _read_json_file(evidence_path, "PREFLIGHT_EVIDENCE_INVALID")
    if hashlib.sha256(evidence_raw).hexdigest() != PREFLIGHT_EVIDENCE_SHA256:
        raise CandidateRerankingError("PREFLIGHT_EVIDENCE_INVALID")
    try:
        validate_preflight_evidence(evidence)
        evidence_mapping = _as_mapping(evidence)
        snapshot_mapping = _as_mapping(evidence_mapping["snapshot"])
        if snapshot_mapping["preparation_status"] != "REUSED":
            raise ValueError
    except Exception:  # noqa: BLE001
        raise CandidateRerankingError("PREFLIGHT_EVIDENCE_INVALID") from None
    _, receipt = _read_json_file(receipt_path, "PREFLIGHT_RECEIPT_INVALID")
    try:
        _validate_preflight_receipt(receipt)
    except Exception:  # noqa: BLE001
        raise CandidateRerankingError("PREFLIGHT_RECEIPT_INVALID") from None
    return hashlib.sha256(evidence_raw).hexdigest(), PREFLIGHT_EXECUTION_COMMIT


def _validate_candidate_snapshot(paths: CandidateRunPaths) -> _FixedInputContext:
    evidence_sha256, execution_commit = _validate_preflight_inputs(paths)
    snapshot_path = _operational_path(paths.candidate_snapshot_path)
    manifest_path = _operational_path(paths.candidate_manifest_path)
    snapshot_raw, snapshot_value = _read_json_file(snapshot_path, "CANDIDATE_INPUT_INVALID")
    _, manifest_value = _read_json_file(manifest_path, "CANDIDATE_INPUT_INVALID")
    try:
        snapshot = _as_mapping(snapshot_value)
        manifest = _as_mapping(manifest_value)
        candidate_values = snapshot["candidates"]
        if not isinstance(candidate_values, list) or len(candidate_values) != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        if snapshot["count"] != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        if snapshot["question"] != FIXED_QUERY:
            raise ValueError
        if manifest["candidate_count"] != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        if manifest["abstract_present_count"] != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        if manifest["abstract_missing_count"] != 0:
            raise ValueError
        snapshot_sha256 = hashlib.sha256(snapshot_raw).hexdigest()
        if manifest["snapshot_sha256"] != snapshot_sha256:
            raise ValueError
        paper_id_order_value = manifest["paper_id_order"]
        if not isinstance(paper_id_order_value, list) or len(paper_id_order_value) != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        if any(not isinstance(value, str) or not value.strip() for value in paper_id_order_value):
            raise ValueError
        candidates: list[FrozenCandidate] = []
        paper_ids: list[str] = []
        for raw_candidate in candidate_values:
            candidate_mapping = _as_mapping(raw_candidate)
            title = candidate_mapping.get("title")
            abstract = candidate_mapping.get("abstract")
            paper_id = candidate_mapping.get("paper_id")
            if (
                not isinstance(paper_id, str)
                or not paper_id.strip()
                or not isinstance(title, str)
                or not title.strip()
                or not isinstance(abstract, str)
                or not abstract.strip()
                or candidate_mapping.get("language") != "en"
            ):
                raise ValueError
            candidate = FrozenCandidate.model_validate(candidate_mapping)
            candidates.append(candidate)
            paper_ids.append(candidate.paper_id)
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError
        if paper_ids != paper_id_order_value:
            raise ValueError
        if list(paper_id_order_value) != paper_ids:
            raise ValueError
        return _FixedInputContext(
            candidates=tuple(candidates),
            paper_id_order=tuple(paper_ids),
            snapshot_sha256=snapshot_sha256,
            preflight_evidence_sha256=evidence_sha256,
            preflight_execution_commit=execution_commit,
        )
    except CandidateRerankingError:
        raise
    except Exception:  # noqa: BLE001
        raise CandidateRerankingError("CANDIDATE_INPUT_INVALID") from None


def _validate_model_snapshot(path: Path) -> None:
    _require_ordinary_directory(path, "MODEL_SNAPSHOT_INVALID")


def _default_provider_factory(snapshot_path: Path) -> RerankerProvider:
    from app.adapters.reranking import BgeRerankerProvider

    return BgeRerankerProvider(
        model_id=BGE_RERANKER_MODEL_ID,
        model_revision=BGE_RERANKER_MODEL_REVISION,
        cache_namespace="m2-t02:local-candidate-reranking",
        model_dir=snapshot_path,
        max_length=MAX_LENGTH,
        device="cpu",
    )


def _new_cache_directory() -> Path:
    return Path(tempfile.mkdtemp(prefix="m2-t02-candidate-reranking-cache-"))


def _cache_is_empty(cache_dir: Path) -> None:
    try:
        metadata = cache_dir.lstat()
        if _is_link_or_junction(cache_dir, metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError
        with os.scandir(cache_dir) as iterator:
            if next(iterator, None) is not None:
                raise ValueError
    except Exception:  # noqa: BLE001
        raise CandidateRerankingError("INVALID_OUTPUT") from None


def _cleanup_cache(cache_dir: Path) -> None:
    try:
        metadata = cache_dir.lstat()
        if _is_link_or_junction(cache_dir, metadata) or not stat.S_ISDIR(metadata.st_mode):
            return
        shutil.rmtree(cache_dir)
    except (FileNotFoundError, OSError):
        pass


def _runtime_load_count(provider: object) -> int:
    value = getattr(provider, "runtime_load_count", 1)
    return value if _strict_int(value) and value == 1 else 1


def _build_candidate_run_report(
    context: _FixedInputContext,
    provider: CountingRerankerProvider,
    run: RerankRun,
) -> dict[str, object]:
    if run.state is not RerankerRunState.SCORED or len(run.records) != EXPECTED_CANDIDATE_COUNT:
        raise CandidateRerankingError("INVALID_OUTPUT")
    records = [
        {
            "rank": rank,
            "paper_id": record.paper_id,
            "input_sha256": record.input_sha256,
            "raw_score": record.raw_score,
            "normalized_score": record.normalized_score,
        }
        for rank, record in enumerate(run.records, start=1)
    ]
    report: dict[str, object] = {
        "report_version": "m2-t02-reranker-candidate-run.v1",
        "phase": "M2",
        "task_id": "M2-T02",
        "baseline_commit": BASELINE_COMMIT,
        "decision_status": "candidate_reranking_completed",
        "input": {
            "candidate_snapshot_path": CANDIDATE_SNAPSHOT_PATH.as_posix(),
            "candidate_manifest_path": CANDIDATE_MANIFEST_PATH.as_posix(),
            "candidate_snapshot_sha256": context.snapshot_sha256,
            "preflight_evidence_sha256": context.preflight_evidence_sha256,
            "preflight_receipt_execution_commit": context.preflight_execution_commit,
            "candidate_count": EXPECTED_CANDIDATE_COUNT,
            "abstract_present_count": EXPECTED_CANDIDATE_COUNT,
            "abstract_missing_count": 0,
            "question": FIXED_QUERY,
            "paper_id_order": list(context.paper_id_order),
        },
        "model": {
            "provider_name": BGE_RERANKER_PROVIDER_NAME,
            "model_id": BGE_RERANKER_MODEL_ID,
            "revision": BGE_RERANKER_MODEL_REVISION,
            "snapshot_path": MODEL_SNAPSHOT_PATH.as_posix(),
            "preparation_status": "REUSED",
        },
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "max_length": MAX_LENGTH,
            "local_files_only": True,
            "trust_remote_code": False,
        },
        "policy": {
            "configured_top_k": CONFIGURED_TOP_K,
            "effective_top_k": EXPECTED_CANDIDATE_COUNT,
            "batch_size": BATCH_SIZE,
            "parallelism": "serial",
            "cache_policy": "invocation_owned_empty",
            "network_forbidden": True,
        },
        "execution": {
            "provider_instance_count": provider.provider_instance_count,
            "runtime_load_count": _runtime_load_count(provider),
            "provider_call_count": provider.provider_call_count,
            "provider_scored_count": provider.provider_scored_count,
            "batch_sizes": list(provider.batch_sizes),
            "cache_hits": 0,
        },
        "records": records,
    }
    return report


def _require_finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError
    return numeric


def validate_candidate_run_report(value: object) -> None:
    """Validate the closed, privacy-safe future B3 candidate-run schema."""

    try:
        report = _as_mapping(value)
        if set(report) != _REPORT_FIELDS:
            raise ValueError
        if (
            report["report_version"] != "m2-t02-reranker-candidate-run.v1"
            or report["phase"] != "M2"
            or report["task_id"] != "M2-T02"
            or report["baseline_commit"] != BASELINE_COMMIT
            or report["decision_status"] != "candidate_reranking_completed"
        ):
            raise ValueError
        input_data = _as_mapping(report["input"])
        model_data = _as_mapping(report["model"])
        runtime_data = _as_mapping(report["runtime"])
        policy_data = _as_mapping(report["policy"])
        execution_data = _as_mapping(report["execution"])
        if set(input_data) != _INPUT_FIELDS:
            raise ValueError
        if set(model_data) != _MODEL_FIELDS:
            raise ValueError
        if set(runtime_data) != _RUNTIME_FIELDS:
            raise ValueError
        if set(policy_data) != _POLICY_FIELDS:
            raise ValueError
        if set(execution_data) != _EXECUTION_FIELDS:
            raise ValueError
        if input_data["candidate_snapshot_path"] != CANDIDATE_SNAPSHOT_PATH.as_posix():
            raise ValueError
        if input_data["candidate_manifest_path"] != CANDIDATE_MANIFEST_PATH.as_posix():
            raise ValueError
        _require_hex(input_data["candidate_snapshot_sha256"], _HEX64)
        if input_data["preflight_evidence_sha256"] != PREFLIGHT_EVIDENCE_SHA256:
            raise ValueError
        if input_data["preflight_receipt_execution_commit"] != PREFLIGHT_EXECUTION_COMMIT:
            raise ValueError
        if (
            input_data["candidate_count"] != EXPECTED_CANDIDATE_COUNT
            or input_data["abstract_present_count"] != EXPECTED_CANDIDATE_COUNT
            or input_data["abstract_missing_count"] != 0
            or input_data["question"] != FIXED_QUERY
        ):
            raise ValueError
        paper_id_order = input_data["paper_id_order"]
        if (
            not isinstance(paper_id_order, list)
            or len(paper_id_order) != EXPECTED_CANDIDATE_COUNT
            or any(
                not isinstance(paper_id, str)
                or not paper_id.startswith("arxiv:")
                or "/" in paper_id
                or "\\" in paper_id
                for paper_id in paper_id_order
            )
            or len(set(paper_id_order)) != EXPECTED_CANDIDATE_COUNT
        ):
            raise ValueError
        if model_data != {
            "provider_name": BGE_RERANKER_PROVIDER_NAME,
            "model_id": BGE_RERANKER_MODEL_ID,
            "revision": BGE_RERANKER_MODEL_REVISION,
            "snapshot_path": MODEL_SNAPSHOT_PATH.as_posix(),
            "preparation_status": "REUSED",
        }:
            raise ValueError
        if runtime_data != {
            "device": "cpu",
            "dtype": "float32",
            "max_length": MAX_LENGTH,
            "local_files_only": True,
            "trust_remote_code": False,
        }:
            raise ValueError
        if policy_data != {
            "configured_top_k": CONFIGURED_TOP_K,
            "effective_top_k": EXPECTED_CANDIDATE_COUNT,
            "batch_size": BATCH_SIZE,
            "parallelism": "serial",
            "cache_policy": "invocation_owned_empty",
            "network_forbidden": True,
        }:
            raise ValueError
        if execution_data != {
            "provider_instance_count": 1,
            "runtime_load_count": 1,
            "provider_call_count": 17,
            "provider_scored_count": EXPECTED_CANDIDATE_COUNT,
            "batch_sizes": EXPECTED_BATCH_SIZES,
            "cache_hits": 0,
        }:
            raise ValueError
        records = report["records"]
        if not isinstance(records, list) or len(records) != EXPECTED_CANDIDATE_COUNT:
            raise ValueError
        seen_ids: set[str] = set()
        sortable: list[tuple[float, str]] = []
        for expected_rank, raw_record in enumerate(records, start=1):
            record = _as_mapping(raw_record)
            if set(record) != _RECORD_FIELDS:
                raise ValueError
            if record["rank"] != expected_rank:
                raise ValueError
            paper_id = record["paper_id"]
            if (
                not isinstance(paper_id, str)
                or not paper_id.startswith("arxiv:")
                or "/" in paper_id
                or "\\" in paper_id
                or paper_id in seen_ids
            ):
                raise ValueError
            seen_ids.add(paper_id)
            _require_hex(record["input_sha256"], _HEX64)
            raw_score = _require_finite_number(record["raw_score"])
            normalized_score = _require_finite_number(record["normalized_score"])
            if not 0 <= normalized_score <= 1:
                raise ValueError
            sortable.append((raw_score, paper_id))
        if seen_ids != set(paper_id_order):
            raise ValueError
        expected_order = sorted(sortable, key=lambda item: (-item[0], item[1]))
        if sortable != expected_order:
            raise ValueError
        _validate_safe_strings(report)
    except CandidateRerankingError:
        raise
    except Exception:  # noqa: BLE001
        raise CandidateRerankingError("RESULT_SCHEMA_INVALID") from None


def _serialized_report(report: Mapping[str, object]) -> bytes:
    try:
        validate_candidate_run_report(report)
        return (
            json.dumps(
                report,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except CandidateRerankingError:
        raise
    except Exception:  # noqa: BLE001
        raise CandidateRerankingError("RESULT_SCHEMA_INVALID") from None


def _existing_result_bytes(destination: Path) -> bytes | None:
    try:
        metadata = destination.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise CandidateRerankingError("RESULT_CONFLICT") from None
    if _is_link_or_junction(destination, metadata) or not stat.S_ISREG(metadata.st_mode):
        raise CandidateRerankingError("RESULT_CONFLICT")
    try:
        return destination.read_bytes()
    except OSError:
        raise CandidateRerankingError("RESULT_CONFLICT") from None


def _cleanup_owned_temp(path: Path | None) -> None:
    if path is None:
        return
    try:
        metadata = path.lstat()
        if not _is_link_or_junction(path, metadata) and stat.S_ISREG(metadata.st_mode):
            path.unlink()
    except (FileNotFoundError, OSError):
        pass


def publish_candidate_run(
    report: Mapping[str, object],
    *,
    destination: Path | str = FORMAL_RESULT_PATH,
) -> Literal["PUBLISHED", "REUSED"]:
    """Atomically publish a validated report to a future result path."""

    serialized = _serialized_report(report)
    final_path = _operational_path(destination)
    try:
        parent_metadata = final_path.parent.lstat()
        if (
            _is_link_or_junction(final_path.parent, parent_metadata)
            or not stat.S_ISDIR(parent_metadata.st_mode)
        ):
            raise ValueError
        existing = _existing_result_bytes(final_path)
        if existing is not None:
            if existing == serialized:
                return "REUSED"
            raise CandidateRerankingError("RESULT_CONFLICT")
    except CandidateRerankingError:
        raise
    except (FileNotFoundError, OSError, ValueError):
        raise CandidateRerankingError("RESULT_CONFLICT") from None

    temporary_path: Path | None = None
    try:
        raw_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{final_path.name}.tmp-",
            dir=str(final_path.parent),
        )
        os.close(raw_fd)
        temporary_path = Path(temporary_name)
        with temporary_path.open("wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if _existing_result_bytes(final_path) is not None:
            raise CandidateRerankingError("RESULT_CONFLICT")
        os.replace(temporary_path, final_path)
        temporary_path = None
        return "PUBLISHED"
    except CandidateRerankingError:
        raise
    except (OSError, TypeError, ValueError):
        raise CandidateRerankingError("RESULT_PUBLICATION_FAILED") from None
    finally:
        _cleanup_owned_temp(temporary_path)


def run_candidate_reranking(
    *,
    execute_local_reranking: bool = False,
    paths: CandidateRunPaths | None = None,
    provider_factory: Callable[[Path], RerankerProvider] | None = None,
    cache_factory: Callable[[], Path] | None = None,
    publish_result: bool = False,
) -> dict[str, object]:
    """Run the fixed 33-candidate contract; real CLI publishing is future-only."""

    if execute_local_reranking is not True:
        raise CandidateRerankingError("LOCAL_RERANKING_NOT_ENABLED")
    selected_paths = paths or CandidateRunPaths()
    context = _validate_candidate_snapshot(selected_paths)
    model_snapshot_path = _operational_path(selected_paths.model_snapshot_path)
    _validate_model_snapshot(model_snapshot_path)
    selected_provider_factory = provider_factory or _default_provider_factory
    try:
        provider = selected_provider_factory(model_snapshot_path)
    except CandidateRerankingError:
        raise
    except Exception:  # noqa: BLE001
        raise CandidateRerankingError("PROVIDER_UNAVAILABLE") from None
    counting_provider = CountingRerankerProvider(provider)
    cache_dir = cache_factory() if cache_factory is not None else _new_cache_directory()
    cache_path = Path(cache_dir)
    try:
        _cache_is_empty(cache_path)
        run = rerank_candidates(
            FIXED_QUERY,
            context.candidates,
            counting_provider,
            cache_path,
            batch_size=BATCH_SIZE,
            configured_top_k=CONFIGURED_TOP_K,
        )
        report = _build_candidate_run_report(context, counting_provider, run)
        validate_candidate_run_report(report)
        if publish_result:
            publish_candidate_run(report, destination=selected_paths.result_path)
        return report
    except CandidateRerankingError:
        raise
    except RerankerTaskError as error:
        if error.code == "INVALID_OUTPUT":
            code: CandidateRerankingErrorCode = "INVALID_OUTPUT"
        elif error.code == "PROVIDER_UNAVAILABLE":
            code = "PROVIDER_UNAVAILABLE"
        else:
            code = "CANDIDATE_INPUT_INVALID"
        raise CandidateRerankingError(code) from None
    except Exception:  # noqa: BLE001
        raise CandidateRerankingError("PROVIDER_UNAVAILABLE") from None
    finally:
        _cleanup_cache(cache_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Accept only the one explicit future local-reranking flag."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments != ["--execute-local-reranking"]:
        return 2
    try:
        run_candidate_reranking(execute_local_reranking=True, publish_result=True)
    except CandidateRerankingError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
