"""Mocked red-first contract for the future pinned BGE reranker provider."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import math
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import pytest

import app.adapters.reranking as reranking_adapters

_MODEL_ID = "BAAI/bge-reranker-v2-m3"
_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_FORBIDDEN_RUNTIME_PACKAGES = (
    "torch",
    "transformers",
    "huggingface_hub",
    "sentence_transformers",
    "FlagEmbedding",
)


def _provider_type() -> type[Any]:
    provider_type = getattr(reranking_adapters, "BgeRerankerProvider", None)
    assert provider_type is not None, "BgeRerankerProvider has not been implemented"
    return provider_type


def _models() -> types.ModuleType:
    return importlib.import_module("app.models.reranking")


def _task_error() -> type[Exception]:
    return _models().RerankerTaskError


def _input(paper_id: str, text: str) -> Any:
    input_type = _models().RerankerInput
    return input_type(
        paper_id=paper_id,
        text=text,
        input_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _snapshot(tmp_path: Path) -> Path:
    model_dir = tmp_path / "pinned-bge"
    model_dir.mkdir()
    for name in _REQUIRED_FILES:
        (model_dir / name).write_bytes(b"mock-local-snapshot")
    return model_dir


def _provider(model_dir: Path, **overrides: object) -> Any:
    kwargs: dict[str, object] = {
        "model_id": _MODEL_ID,
        "model_revision": _REVISION,
        "cache_namespace": "m2-t02:mocked-contract",
        "model_dir": model_dir,
        "max_length": 512,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return _provider_type()(**kwargs)


class _FakeTensor:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def flatten(self) -> _FakeTensor:
        return self

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> list[object]:
        return list(self.values)


class _FakeModelOutput:
    def __init__(self, logits: object | None) -> None:
        if logits is not None:
            self.logits = logits


class _FakeBatchEncoding(dict[str, object]):
    pass


class _FakeParameter:
    def __init__(self, *, device: object = "cpu", dtype: object = "float32") -> None:
        self.device = device
        self.dtype = dtype


class _FakeAutoTokenizer:
    from_pretrained_calls: ClassVar[list[tuple[object, dict[str, object]]]] = []
    calls: ClassVar[list[tuple[object, dict[str, object]]]] = []
    load_error: ClassVar[Exception | None] = None
    call_error: ClassVar[Exception | None] = None

    @classmethod
    def from_pretrained(cls, path: object, **kwargs: object) -> _FakeAutoTokenizer:
        cls.from_pretrained_calls.append((path, kwargs))
        if cls.load_error is not None:
            raise cls.load_error
        return cls()

    def __call__(
        self, queries: object, passages: object, **kwargs: object
    ) -> _FakeBatchEncoding:
        type(self).calls.append(((queries, passages), kwargs))
        if type(self).call_error is not None:
            raise type(self).call_error
        return _FakeBatchEncoding(input_ids=[1, 2])


class _FakeAutoModelForSequenceClassification:
    from_pretrained_calls: ClassVar[list[tuple[object, dict[str, object]]]] = []
    instances: ClassVar[list[_FakeAutoModelForSequenceClassification]] = []
    load_error: ClassVar[Exception | None] = None
    logits: ClassVar[object | None] = _FakeTensor([-3.0, 0.25, 8.5])

    @classmethod
    def from_pretrained(
        cls, path: object, **kwargs: object
    ) -> _FakeAutoModelForSequenceClassification:
        cls.from_pretrained_calls.append((path, kwargs))
        if cls.load_error is not None:
            raise cls.load_error
        instance = cls()
        cls.instances.append(instance)
        return instance

    def __init__(self) -> None:
        self.to_calls: list[object] = []
        self.eval_calls = 0
        self.forward_calls: list[tuple[dict[str, object], dict[str, object]]] = []
        self.to_error: Exception | None = None
        self.eval_error: Exception | None = None
        self.forward_error: Exception | None = None
        self.training = True
        self.parameters_values: list[object] = [_FakeParameter()]
        self.parameters_error: Exception | None = None

    def to(self, device: object) -> _FakeAutoModelForSequenceClassification:
        self.to_calls.append(device)
        if self.to_error is not None:
            raise self.to_error
        return self

    def eval(self) -> _FakeAutoModelForSequenceClassification:
        self.eval_calls += 1
        if self.eval_error is not None:
            raise self.eval_error
        self.training = False
        return self

    def parameters(self) -> list[object]:
        if self.parameters_error is not None:
            raise self.parameters_error
        return list(self.parameters_values)

    def __call__(self, **kwargs: object) -> _FakeModelOutput:
        self.forward_calls.append((dict(kwargs), {}))
        if self.forward_error is not None:
            raise self.forward_error
        return _FakeModelOutput(type(self).logits)


class _FakeTorch:
    inference_mode_calls: ClassVar[int] = 0

    @classmethod
    @contextmanager
    def inference_mode(cls) -> Any:
        cls.inference_mode_calls += 1
        yield


def _install_fake_runtime_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: {"transformers": "4.test", "torch": "2.test"}[name],
    )


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeAutoTokenizer.from_pretrained_calls = []
    _FakeAutoTokenizer.calls = []
    _FakeAutoTokenizer.load_error = None
    _FakeAutoTokenizer.call_error = None
    _FakeAutoModelForSequenceClassification.from_pretrained_calls = []
    _FakeAutoModelForSequenceClassification.instances = []
    _FakeAutoModelForSequenceClassification.load_error = None
    _FakeAutoModelForSequenceClassification.logits = _FakeTensor([-3.0, 0.25, 8.5])
    _FakeTorch.inference_mode_calls = 0
    _install_fake_runtime_versions(monkeypatch)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    fake_transformers.AutoModelForSequenceClassification = _FakeAutoModelForSequenceClassification
    fake_torch = types.ModuleType("torch")
    fake_torch.inference_mode = _FakeTorch.inference_mode
    fake_torch.float32 = "float32"
    fake_torch.version = types.SimpleNamespace(cuda=None)
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _assert_error(code: str, action: Any) -> None:
    with pytest.raises(_task_error()) as raised:
        action()
    assert raised.value.code == code


def _assert_score_error_without_partial_result(provider: Any, inputs: list[Any]) -> None:
    returned_scores: list[Any] | None = None
    with pytest.raises(_task_error()) as raised:
        returned_scores = provider.score("radiation", inputs, batch_size=1)
    assert raised.value.code == "PROVIDER_UNAVAILABLE"
    assert returned_scores is None


def test_importing_reranking_adapter_does_not_import_runtime_packages() -> None:
    script = """import app.adapters.reranking; import sys; blocked = {name for name in sys.modules if name.split('.')[0] in {'torch', 'transformers', 'huggingface_hub', 'sentence_transformers', 'FlagEmbedding'}}; raise SystemExit(bool(blocked))"""
    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("overrides", "label"),
    [
        ({"model_id": "wrong/model"}, "wrong-model-id"),
        ({"model_revision": "0" * 40}, "wrong-revision"),
        ({"model_revision": "main"}, "main-revision"),
        ({"model_revision": "latest"}, "latest-revision"),
        ({"model_revision": "953dc6f"}, "short-revision"),
        ({"cache_namespace": ""}, "empty-cache-namespace"),
        ({"cache_namespace": "synthetic"}, "synthetic-cache-namespace"),
        ({"max_length": 511}, "wrong-max-length"),
        ({"device": "cuda"}, "non-cpu-device"),
        ({"device": "mps"}, "non-cpu-mps-device"),
    ],
)
def test_provider_rejects_unpinned_or_unauthorized_configuration(
    tmp_path: Path, overrides: dict[str, object], label: str
) -> None:
    del label
    _assert_error("INVALID_INPUT", lambda: _provider(_snapshot(tmp_path), **overrides))


def test_provider_rejects_nonexistent_model_dir(tmp_path: Path) -> None:
    _assert_error("INVALID_INPUT", lambda: _provider(tmp_path / "missing"))


@pytest.mark.parametrize("missing", _REQUIRED_FILES)
def test_provider_rejects_missing_required_local_snapshot_file(tmp_path: Path, missing: str) -> None:
    model_dir = _snapshot(tmp_path)
    (model_dir / missing).unlink()
    _assert_error("PROVIDER_UNAVAILABLE", lambda: _provider(model_dir))


@pytest.mark.parametrize("empty", _REQUIRED_FILES)
def test_provider_rejects_empty_required_local_snapshot_file(tmp_path: Path, empty: str) -> None:
    model_dir = _snapshot(tmp_path)
    (model_dir / empty).write_bytes(b"")
    _assert_error("PROVIDER_UNAVAILABLE", lambda: _provider(model_dir))


def test_readme_is_not_a_runtime_required_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_runtime_versions(monkeypatch)
    provider = _provider(_snapshot(tmp_path))
    assert provider.descriptor.model_id == _MODEL_ID


def test_descriptor_and_runtime_are_pinned_and_runtime_is_a_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_runtime(monkeypatch)
    model_dir = _snapshot(tmp_path)
    provider = _provider(model_dir)
    descriptor = provider.descriptor
    assert descriptor.provider_name == "bge_reranker_v2_m3"
    assert descriptor.model_id == _MODEL_ID
    assert descriptor.model_revision == _REVISION
    assert descriptor.provider_library == "transformers"
    assert descriptor.provider_library_version == "4.test"
    assert descriptor.input_format_version == "m2-reranker-title-abstract-v1"
    assert descriptor.cache_namespace == "m2-t02:mocked-contract"
    runtime = provider.runtime
    assert runtime["transformers_version"] == "4.test"
    assert runtime["torch_version"] == "2.test"
    assert isinstance(runtime["python_version"], str)
    assert runtime["device"] == "cpu"
    assert runtime["dtype"] == "float32"
    assert runtime["max_length"] == 512
    assert runtime["local_files_only"] is True
    assert runtime["trust_remote_code"] is False
    assert runtime["model_revision"] == _REVISION
    assert runtime["model_dir"] == str(model_dir)
    assert not {"token", "cookie", "authorization"} & {key.lower() for key in runtime}
    runtime["device"] = "mutated"
    assert provider.runtime["device"] == "cpu"


def test_constructor_descriptor_and_empty_score_are_lazy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for package in ("torch", "transformers"):
        monkeypatch.delitem(sys.modules, package, raising=False)
    _install_fake_runtime_versions(monkeypatch)
    provider = _provider(_snapshot(tmp_path))
    assert provider.descriptor.model_id == _MODEL_ID
    assert not {"torch", "transformers"} & set(sys.modules)
    assert provider.score("radiation shielding", [], batch_size=1) == []


@pytest.mark.parametrize("package", ("transformers", "torch"))
def test_missing_runtime_package_maps_to_provider_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package: str
) -> None:
    _install_fake_runtime_versions(monkeypatch)
    monkeypatch.setitem(sys.modules, package, None)
    provider = _provider(_snapshot(tmp_path))
    _assert_error(
        "PROVIDER_UNAVAILABLE",
        lambda: provider.score("radiation", [_input("a", "input_a")], batch_size=1),
    )


def test_unreadable_runtime_package_version_maps_to_provider_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_runtime(monkeypatch)
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError()),
    )
    _assert_error("PROVIDER_UNAVAILABLE", lambda: _provider(_snapshot(tmp_path)).descriptor)


def test_score_loads_once_uses_local_pair_encoding_and_preserves_raw_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_runtime(monkeypatch)
    provider = _provider(_snapshot(tmp_path))
    inputs = [_input("a", "input_a"), _input("b", "input_b"), _input("c", "input_c")]
    scores = provider.score("radiation shielding", inputs, batch_size=3)
    assert [(score.paper_id, score.raw_score) for score in scores] == [
        ("a", -3.0), ("b", 0.25), ("c", 8.5)
    ]
    tokenizer_path, tokenizer_kwargs = _FakeAutoTokenizer.from_pretrained_calls[0]
    assert tokenizer_path == tmp_path / "pinned-bge"
    assert tokenizer_kwargs == {"local_files_only": True, "trust_remote_code": False}
    assert _FakeAutoTokenizer.calls == [
        (
            (
                ["radiation shielding", "radiation shielding", "radiation shielding"],
                ["input_a", "input_b", "input_c"],
            ),
            {"padding": True, "truncation": True, "max_length": 512, "return_tensors": "pt"},
        )
    ]
    model_path, model_kwargs = _FakeAutoModelForSequenceClassification.from_pretrained_calls[0]
    assert model_path == tmp_path / "pinned-bge"
    assert model_kwargs == {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "torch_dtype": "float32",
    }
    model = _FakeAutoModelForSequenceClassification.instances[0]
    assert model.to_calls == ["cpu"]
    assert model.eval_calls == 1
    assert _FakeTorch.inference_mode_calls == 1
    assert model.forward_calls == [({"input_ids": [1, 2], "return_dict": True}, {})]
    provider.score("radiation shielding", inputs, batch_size=3)
    assert len(_FakeAutoTokenizer.from_pretrained_calls) == 1
    assert len(_FakeAutoModelForSequenceClassification.from_pretrained_calls) == 1


@pytest.mark.parametrize(
    ("query", "input_specs", "batch_size"),
    [
        ("", (("a", "input_a"),), 1),
        ("   ", (("a", "input_a"),), 1),
        (42, (("a", "input_a"),), 1),
        ("radiation", (("a", "input_a"),), 0),
        ("radiation", (("a", "input_a"),), True),
        ("radiation", (("a", "input_a"),), -1),
        ("radiation", (("a", "input_a"),), 1.5),
        ("radiation", (("a", "input_a"),), "1"),
        ("radiation", (("a", "input_a"), ("b", "input_b")), 1),
        ("radiation", (("a", "input_a"), ("a", "input_b")), 2),
    ],
)
def test_score_rejects_invalid_chunk_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    query: Any,
    input_specs: tuple[tuple[str, str], ...],
    batch_size: Any,
) -> None:
    inputs = [_input(*spec) for spec in input_specs]
    _install_fake_runtime(monkeypatch)
    provider = _provider(_snapshot(tmp_path))
    _assert_error(
        "INVALID_INPUT",
        lambda: provider.score(query, inputs, batch_size=batch_size),
    )
    assert _FakeAutoTokenizer.from_pretrained_calls == []
    assert _FakeAutoModelForSequenceClassification.from_pretrained_calls == []


@pytest.mark.parametrize(
    "invalid_input",
    [
        lambda: _models().RerankerInput.model_construct(
            paper_id=" ", text="input_a", input_sha256=hashlib.sha256(b"input_a").hexdigest()
        ),
        lambda: _models().RerankerInput.model_construct(
            paper_id="a", text=" ", input_sha256=hashlib.sha256(b" ").hexdigest()
        ),
        lambda: _models().RerankerInput.model_construct(
            paper_id="a", text="input_a", input_sha256="0" * 63
        ),
        lambda: _models().RerankerInput.model_construct(
            paper_id="a", text="input_a", input_sha256="0" * 64
        ),
    ],
)
def test_score_rejects_constructed_invalid_input_without_runtime_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_input: Any
) -> None:
    _install_fake_runtime(monkeypatch)
    provider = _provider(_snapshot(tmp_path))
    _assert_error(
        "INVALID_INPUT",
        lambda: provider.score("radiation", [invalid_input()], batch_size=1),
    )
    assert _FakeAutoTokenizer.from_pretrained_calls == []
    assert _FakeAutoModelForSequenceClassification.from_pretrained_calls == []


@pytest.mark.parametrize(
    "logits",
    [
        None,
        _FakeTensor([-3.0]),
        _FakeTensor([-3.0, 0.25, 8.5, 9.0]),
        _FakeTensor(["not-a-number", 0.25, 8.5]),
        _FakeTensor([True, 0.25, 8.5]),
        _FakeTensor([math.nan, 0.25, 8.5]),
        _FakeTensor([math.inf, 0.25, 8.5]),
        _FakeTensor([-math.inf, 0.25, 8.5]),
        object(),
    ],
)
def test_score_rejects_invalid_model_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, logits: object
) -> None:
    _install_fake_runtime(monkeypatch)
    _FakeAutoModelForSequenceClassification.logits = logits
    inputs = [_input("a", "input_a"), _input("b", "input_b"), _input("c", "input_c")]
    _assert_error("INVALID_OUTPUT", lambda: _provider(_snapshot(tmp_path)).score("radiation", inputs, batch_size=3))


@pytest.mark.parametrize(
    "failure",
    ["tokenizer-load", "model-load", "tokenizer-call", "model-to", "model-eval", "model-forward"],
)
def test_runtime_failures_map_to_provider_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _install_fake_runtime(monkeypatch)
    if failure == "tokenizer-load":
        _FakeAutoTokenizer.load_error = RuntimeError(failure)
    elif failure == "model-load":
        _FakeAutoModelForSequenceClassification.load_error = RuntimeError(failure)
    provider = _provider(_snapshot(tmp_path))
    if failure == "tokenizer-call":
        _FakeAutoTokenizer.call_error = RuntimeError(failure)
    if failure in {"model-to", "model-eval", "model-forward"}:
        original = _FakeAutoModelForSequenceClassification.from_pretrained

        @classmethod
        def configured_loader(
            cls, path: object, **kwargs: object
        ) -> _FakeAutoModelForSequenceClassification:
            instance = original(path, **kwargs)
            setattr(instance, f"{failure.removeprefix('model-')}_error", RuntimeError(failure))
            return instance

        monkeypatch.setattr(
            _FakeAutoModelForSequenceClassification,
            "from_pretrained",
            configured_loader,
        )
    inputs = [_input("a", "input_a")]
    _assert_score_error_without_partial_result(provider, inputs)
    if failure == "tokenizer-load":
        assert len(_FakeAutoTokenizer.from_pretrained_calls) == 1
        assert _FakeAutoTokenizer.calls == []
        assert _FakeAutoModelForSequenceClassification.from_pretrained_calls == []
    elif failure == "tokenizer-call":
        assert len(_FakeAutoTokenizer.from_pretrained_calls) == 1
        assert len(_FakeAutoModelForSequenceClassification.from_pretrained_calls) == 1
        assert len(_FakeAutoTokenizer.calls) == 1
        assert _FakeAutoModelForSequenceClassification.instances[0].forward_calls == []
    elif failure == "model-to":
        model = _FakeAutoModelForSequenceClassification.instances[0]
        assert model.to_calls == ["cpu"]
        assert model.eval_calls == 0
        assert model.forward_calls == []
    elif failure == "model-eval":
        model = _FakeAutoModelForSequenceClassification.instances[0]
        assert model.to_calls == ["cpu"]
        assert model.eval_calls == 1
        assert model.forward_calls == []
    elif failure == "model-forward":
        model = _FakeAutoModelForSequenceClassification.instances[0]
        assert model.to_calls == ["cpu"]
        assert model.eval_calls == 1
        assert len(model.forward_calls) == 1


def test_tokenizer_type_error_is_not_retried_with_pair_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_runtime(monkeypatch)
    _FakeAutoTokenizer.call_error = TypeError("two-sequence contract failure")
    provider = _provider(_snapshot(tmp_path))
    inputs = [_input("a", "input_a")]

    _assert_score_error_without_partial_result(provider, inputs)
    assert len(_FakeAutoTokenizer.calls) == 1


@pytest.mark.parametrize(
    "runtime_mutation",
    [
        "missing-training",
        "noop-eval",
        "missing-version",
        "cuda-version",
        "missing-cuda",
        "missing-is-available",
        "cuda-available",
        "missing-float32",
        "missing-parameters",
        "empty-parameters",
        "parameters-error",
        "non-cpu-parameter",
        "non-float32-parameter",
    ],
)
def test_missing_exceptional_or_drifted_runtime_observations_are_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_mutation: str,
) -> None:
    _install_fake_runtime(monkeypatch)
    fake_torch = sys.modules["torch"]
    original_loader = _FakeAutoModelForSequenceClassification.from_pretrained

    @classmethod
    def configured_loader(
        cls, path: object, **kwargs: object
    ) -> _FakeAutoModelForSequenceClassification:
        instance = original_loader(path, **kwargs)
        if runtime_mutation == "missing-training":
            del instance.training
            instance.eval = lambda: instance  # type: ignore[method-assign]
        elif runtime_mutation == "noop-eval":
            instance.training = True
            instance.eval = lambda: instance  # type: ignore[method-assign]
        elif runtime_mutation == "missing-parameters":
            instance.parameters = None  # type: ignore[method-assign]
        elif runtime_mutation == "empty-parameters":
            instance.parameters_values = []
        elif runtime_mutation == "parameters-error":
            instance.parameters_error = RuntimeError("parameters failed")
        elif runtime_mutation == "non-cpu-parameter":
            instance.parameters_values = [_FakeParameter(device="cuda")]
        elif runtime_mutation == "non-float32-parameter":
            instance.parameters_values = [_FakeParameter(dtype="float64")]
        return instance

    if runtime_mutation in {
        "missing-training",
        "noop-eval",
        "missing-parameters",
        "empty-parameters",
        "parameters-error",
        "non-cpu-parameter",
        "non-float32-parameter",
    }:
        monkeypatch.setattr(
            _FakeAutoModelForSequenceClassification,
            "from_pretrained",
            configured_loader,
        )
    elif runtime_mutation == "missing-version":
        del fake_torch.version
    elif runtime_mutation == "cuda-version":
        fake_torch.version.cuda = "12.1"
    elif runtime_mutation == "missing-cuda":
        del fake_torch.cuda
    elif runtime_mutation == "missing-is-available":
        del fake_torch.cuda.is_available
    elif runtime_mutation == "cuda-available":
        fake_torch.cuda.is_available = lambda: True
    elif runtime_mutation == "missing-float32":
        del fake_torch.float32

    provider = _provider(_snapshot(tmp_path))
    _assert_score_error_without_partial_result(provider, [_input("a", "input_a")])
