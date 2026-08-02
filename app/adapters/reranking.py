"""Replaceable reranker-provider protocol for M2-T02."""

from __future__ import annotations

import importlib.metadata
import os
import stat
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from app.models.reranking import (
    BGE_RERANKER_INPUT_FORMAT_VERSION,
    BGE_RERANKER_MAX_LENGTH,
    BGE_RERANKER_MODEL_ID,
    BGE_RERANKER_MODEL_REVISION,
    BGE_RERANKER_PROVIDER_NAME,
    BGE_RERANKER_REQUIRED_FILES,
    ProviderRawScore,
    RerankerInput,
    RerankerModelDescriptor,
    RerankerTaskError,
)


class RerankerProvider(Protocol):
    """A provider that scores already-serialized candidate inputs."""

    @property
    def descriptor(self) -> RerankerModelDescriptor: ...

    def score(
        self,
        query: str,
        inputs: Sequence[RerankerInput],
        *,
        batch_size: int,
    ) -> list[ProviderRawScore]: ...


class BgeRerankerProvider:
    """Pinned, CPU-only BGE reranker backed by a complete local snapshot."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        cache_namespace: str,
        model_dir: Path | str,
        max_length: int = BGE_RERANKER_MAX_LENGTH,
        device: str = "cpu",
    ) -> None:
        if (
            model_id != BGE_RERANKER_MODEL_ID
            or model_revision != BGE_RERANKER_MODEL_REVISION
            or not isinstance(max_length, int)
            or isinstance(max_length, bool)
            or max_length != BGE_RERANKER_MAX_LENGTH
            or device != "cpu"
            or not isinstance(cache_namespace, str)
            or not cache_namespace.strip()
            or cache_namespace == "synthetic"
        ):
            raise RerankerTaskError("INVALID_INPUT")

        try:
            resolved_model_dir = Path(model_dir)
            if not resolved_model_dir.is_dir():
                raise OSError("model directory is unavailable")
        except (OSError, TypeError, ValueError):
            raise RerankerTaskError("INVALID_INPUT") from None

        for required_file in BGE_RERANKER_REQUIRED_FILES:
            try:
                file_stat = (resolved_model_dir / required_file).stat()
            except OSError:
                raise RerankerTaskError("PROVIDER_UNAVAILABLE") from None
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size <= 0:
                raise RerankerTaskError("PROVIDER_UNAVAILABLE")

        try:
            transformers_version = importlib.metadata.version("transformers")
            torch_version = importlib.metadata.version("torch")
            descriptor = RerankerModelDescriptor(
                provider_name=BGE_RERANKER_PROVIDER_NAME,
                model_id=BGE_RERANKER_MODEL_ID,
                model_revision=BGE_RERANKER_MODEL_REVISION,
                provider_library="transformers",
                provider_library_version=transformers_version,
                input_format_version=BGE_RERANKER_INPUT_FORMAT_VERSION,
                cache_namespace=cache_namespace,
            )
        except Exception:  # noqa: BLE001
            raise RerankerTaskError("PROVIDER_UNAVAILABLE") from None

        self._model_dir = resolved_model_dir
        self._descriptor = descriptor
        self._runtime: dict[str, object] = {
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
            "transformers_version": transformers_version,
            "torch_version": torch_version,
            "device": "cpu",
            "dtype": "float32",
            "max_length": BGE_RERANKER_MAX_LENGTH,
            "local_files_only": True,
            "trust_remote_code": False,
            "model_revision": BGE_RERANKER_MODEL_REVISION,
            "model_dir": str(resolved_model_dir),
        }
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._runtime_load_count = 0

    @property
    def descriptor(self) -> RerankerModelDescriptor:
        """Return the pinned reranker identity without loading runtime packages."""
        return self._descriptor

    @property
    def runtime(self) -> dict[str, object]:
        """Return a copy of the non-secret local runtime configuration."""
        return dict(self._runtime)

    @property
    def runtime_load_count(self) -> int:
        """Return the number of successful local runtime loads."""
        return self._runtime_load_count

    def score(
        self,
        query: str,
        inputs: Sequence[RerankerInput],
        *,
        batch_size: int,
    ) -> list[ProviderRawScore]:
        if not isinstance(query, str) or not query.strip():
            raise RerankerTaskError("INVALID_INPUT")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise RerankerTaskError("INVALID_INPUT")

        try:
            input_count = len(inputs)
        except Exception:  # noqa: BLE001
            raise RerankerTaskError("INVALID_INPUT") from None
        if input_count > batch_size:
            raise RerankerTaskError("INVALID_INPUT")
        if not input_count:
            return []

        validated_inputs: list[RerankerInput] = []
        for item in inputs:
            try:
                payload = {
                    "paper_id": item.paper_id,
                    "text": item.text,
                    "input_sha256": item.input_sha256,
                }
                validated = RerankerInput.model_validate(payload)
            except Exception:  # noqa: BLE001
                raise RerankerTaskError("INVALID_INPUT") from None
            validated_inputs.append(validated)

        paper_ids = [item.paper_id for item in validated_inputs]
        if len(paper_ids) != len(set(paper_ids)):
            raise RerankerTaskError("INVALID_INPUT")

        self._load_runtime_if_needed()
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch
        if tokenizer is None or model is None or torch is None:
            raise RerankerTaskError("PROVIDER_UNAVAILABLE")

        queries = [query for _ in validated_inputs]
        passages = [reranker_input.text for reranker_input in validated_inputs]
        try:
            try:
                encoded = tokenizer(
                    queries,
                    passages,
                    padding=True,
                    truncation=True,
                    max_length=BGE_RERANKER_MAX_LENGTH,
                    return_tensors="pt",
                )
            except TypeError:
                # Keep the old contract fake usable while the real preflight path
                # uses the exact two-sequence tokenizer semantics above.
                pairs = [[query, reranker_input.text] for reranker_input in validated_inputs]
                encoded = tokenizer(
                    pairs,
                    padding=True,
                    truncation="longest_first",
                    max_length=BGE_RERANKER_MAX_LENGTH,
                    return_tensors="pt",
                )
        except Exception:  # noqa: BLE001
            raise RerankerTaskError("PROVIDER_UNAVAILABLE") from None

        try:
            with torch.inference_mode():
                output = model(**encoded, return_dict=True)
        except Exception:  # noqa: BLE001
            raise RerankerTaskError("PROVIDER_UNAVAILABLE") from None

        try:
            values = output.logits.flatten().detach().cpu().tolist()
            if not isinstance(values, list) or len(values) != len(validated_inputs):
                raise ValueError("invalid output length")
            return [
                ProviderRawScore(paper_id=reranker_input.paper_id, raw_score=value)
                for reranker_input, value in zip(validated_inputs, values, strict=True)
            ]
        except Exception:  # noqa: BLE001
            raise RerankerTaskError("INVALID_OUTPUT") from None

    def _load_runtime_if_needed(self) -> None:
        if self._tokenizer is not None and self._model is not None and self._torch is not None:
            return

        self._tokenizer = None
        self._model = None
        self._torch = None
        try:
            with _offline_runtime_environment():
                import torch  # type: ignore[import-not-found]
                from transformers import (  # type: ignore[import-not-found]
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    self._model_dir,
                    local_files_only=True,
                    trust_remote_code=False,
                )
                model_kwargs: dict[str, object] = {
                    "local_files_only": True,
                    "trust_remote_code": False,
                    "use_safetensors": True,
                }
                torch_float32 = getattr(torch, "float32", None)
                if torch_float32 is not None:
                    model_kwargs["torch_dtype"] = torch_float32
                model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_dir,
                    **model_kwargs,
                )
                model.to("cpu")
                model.eval()
                _validate_cpu_float32_runtime(model, torch)
        except Exception:  # noqa: BLE001
            self._tokenizer = None
            self._model = None
            self._torch = None
            raise RerankerTaskError("PROVIDER_UNAVAILABLE") from None

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._runtime_load_count += 1


_OFFLINE_ENVIRONMENT_KEYS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    "HF_HUB_DISABLE_TELEMETRY",
)


@contextmanager
def _offline_runtime_environment() -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in _OFFLINE_ENVIRONMENT_KEYS}
    try:
        for key in _OFFLINE_ENVIRONMENT_KEYS:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _validate_cpu_float32_runtime(model: object, torch: object) -> None:
    training = getattr(model, "training", None)
    if training is not None and training is not False:
        raise ValueError

    torch_version = getattr(torch, "version", None)
    cuda_version = getattr(torch_version, "cuda", None) if torch_version is not None else None
    if torch_version is not None and cuda_version is not None:
        raise ValueError

    cuda_module = getattr(torch, "cuda", None)
    is_available = getattr(cuda_module, "is_available", None) if cuda_module is not None else None
    if callable(is_available) and is_available() is not False:
        raise ValueError

    parameters_method = getattr(model, "parameters", None)
    if not callable(parameters_method):
        return
    parameters = list(parameters_method())
    if not parameters:
        raise ValueError
    torch_float32 = getattr(torch, "float32", None)
    if torch_float32 is None:
        raise ValueError
    for parameter in parameters:
        parameter_device = getattr(parameter, "device", None)
        parameter_dtype = getattr(parameter, "dtype", None)
        if parameter_device is None or str(parameter_device) != "cpu":
            raise ValueError
        if parameter_dtype != torch_float32:
            raise ValueError
