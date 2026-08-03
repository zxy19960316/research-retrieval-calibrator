"""Validate wheel and editable installs expose all repository packages."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

pytestmark = pytest.mark.packaging

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_IMPORT_PROBE = """\
import importlib
module_names = (
    'app',
    'scripts',
    'scripts.run_m2_t02_candidate_reranking',
    'scripts.validate_m2_t02_evidence',
)
for module_name in module_names:
    module = importlib.import_module(module_name)
    print(f'{module_name}={module.__file__}')
"""


def _ignore_repository_artifacts(directory: str, names: list[str]) -> set[str]:
    """Copy source files without bringing model/cache/build artifacts into the test."""

    directory_path = Path(directory)
    ignored = {
        name
        for name in names
        if name in {
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".runtime",
            ".venv",
            "build",
            "dist",
            "venv",
            "__pycache__",
        }
        or name.endswith(".egg-info")
    }
    if directory_path == _REPOSITORY_ROOT:
        ignored.add("models")
    return ignored


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        text=True,
    )


def _assert_success(completed: subprocess.CompletedProcess[str], *, action: str) -> None:
    assert completed.returncode == 0, (
        f"{action} failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _venv_python(venv_root: Path) -> Path:
    return (
        venv_root / "Scripts" / "python.exe"
        if os.name == "nt"
        else venv_root / "bin" / "python"
    )


def _create_venv(venv_root: Path) -> Path:
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_root)
    python = _venv_python(venv_root)
    assert python.is_file()
    return python


def _clean_install_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment


def _install_and_probe(
    python: Path,
    install_target: Path,
    *,
    cwd: Path,
    editable: bool,
) -> str:
    target = [str(install_target)]
    if editable:
        target = ["-e", str(install_target)]
    environment = _clean_install_environment()
    install = _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            *target,
        ],
        cwd=cwd,
        env=environment,
    )
    _assert_success(install, action="package installation")

    probe = _run(
        [str(python), "-c", _IMPORT_PROBE],
        cwd=cwd,
        env=environment,
    )
    _assert_success(probe, action="installed package import probe")
    return probe.stdout


def test_wheel_and_editable_install_expose_scripts(tmp_path: Path) -> None:
    """Build a wheel and verify wheel/editable imports in isolated environments."""

    source_root = tmp_path / "source"
    shutil.copytree(
        _REPOSITORY_ROOT,
        source_root,
        ignore=_ignore_repository_artifacts,
    )
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()

    build = _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_dir),
        ],
        cwd=source_root,
        env=_clean_install_environment(),
    )
    _assert_success(build, action="wheel build")

    wheels = sorted(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found {wheels!r}"

    wheel_python = _create_venv(tmp_path / "wheel-venv")
    wheel_output = _install_and_probe(
        wheel_python,
        wheels[0],
        cwd=tmp_path,
        editable=False,
    )
    assert str(source_root) not in wheel_output

    editable_python = _create_venv(tmp_path / "editable-venv")
    editable_output = _install_and_probe(
        editable_python,
        source_root,
        cwd=tmp_path,
        editable=True,
    )
    assert "scripts.run_m2_t02_candidate_reranking=" in editable_output
    assert "scripts.validate_m2_t02_evidence=" in editable_output
