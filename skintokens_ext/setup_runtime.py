from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import venv
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .events import EventWriter
from .paths import EXTENSION_ID, ModlyLayout, extension_root, resolve_modly_layout, runtime_root
from .readiness import (
    MODEL_SENTINELS,
    MODEL_LOGICAL_ROOT,
    model_root_for_layout,
    planned_state,
    upstream_dir_for_layout,
    write_state_for_layout,
)


TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
TORCH_PACKAGES = ["torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0"]
PIP_FLAGS = ["--no-cache-dir", "--retries", "5", "--timeout", "60"]
GENERIC_REQUIREMENTS = [
    "transformers>=4.57.0",
    "diffusers>=0.35.0",
    "python-box",
    "einops",
    "omegaconf",
    "lightning",
    "addict",
    "fast-simplification",
    "scipy",
    "trimesh",
    "huggingface_hub",
    "numpy>=1.26.0",
    "gradio",
    "bottle",
    "tornado",
    "requests",
    "tqdm",
]
BPy_PACKAGE = "bpy>=4.2"
FLASH_ATTN_PACKAGE = "flash-attn"
FLASH_ATTN_BUILD_REQUIREMENTS = ["psutil", "ninja"]
OPTIONAL_REQUIREMENTS = ["open3d"]
NATIVE_PROBES = {
    "torch": "torch",
    "flash_attn": "flash_attn",
    "fast_simplification": "fast_simplification",
}
OPTIONAL_PROBES = {
    "open3d": "open3d",
}
BPY_PROBE_MARKER = "SKINTOKENS_BPY_PROBE_RESULT="
FLASH_ATTN_WHEELHOUSE_RELATIVE = "wheelhouse/flash-attn"
UPSTREAM_ARCHIVE_URL = "https://github.com/VAST-AI-Research/SkinTokens/archive/{ref}.zip"


class SetupError(RuntimeError):
    def __init__(self, message: str, *, code: str, stage: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.detail = detail or {}


@dataclass(frozen=True)
class SetupAction:
    id: str
    description: str
    command: list[str] | None = None


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout_tail: str
    stderr_tail: str
    ok: bool


class Runner(Protocol):
    def run(self, args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
        ...


class SubprocessRunner:
    def run(self, args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
        run_env = {**os.environ, **env} if env else None
        completed = subprocess.run(args, cwd=cwd, env=run_env, text=True, capture_output=True, check=False)
        return CommandResult(
            args=list(args),
            returncode=completed.returncode,
            stdout_tail=completed.stdout[-4000:],
            stderr_tail=completed.stderr[-4000:],
            ok=completed.returncode == 0,
        )


def build_setup_plan(python_exe: str = sys.executable) -> list[SetupAction]:
    return [
        SetupAction("create-venv", "Create extension-owned Python virtual environment."),
        SetupAction("upgrade-pip", "Upgrade pip/setuptools/wheel in the extension venv."),
        SetupAction("install-torch-cu128", "Install exact PyTorch CUDA 12.8 lane.", [python_exe, "-m", "pip", "install", *PIP_FLAGS, *TORCH_PACKAGES, "--index-url", TORCH_INDEX_URL]),
        SetupAction("install-requirements", "Install SkinTokens Python requirements.", [python_exe, "-m", "pip", "install", *PIP_FLAGS, *GENERIC_REQUIREMENTS]),
        SetupAction("install-flash-attn-build-prereqs", "Install flash-attn build prerequisites.", [python_exe, "-m", "pip", "install", *PIP_FLAGS, *FLASH_ATTN_BUILD_REQUIREMENTS]),
        SetupAction("install-flash-attn", "Install flash-attn from a local wheelhouse, binary wheel, or explicit source-build opt-in."),
        SetupAction("install-optional-open3d", "Try installing optional open3d for voxel postprocess support."),
        SetupAction("resolve-bpy", "Install bpy from PyPI or fall back to a probed Blender executable."),
        SetupAction("download-upstream-source", "Download SkinTokens source snapshot into the extension runtime vendor directory."),
        SetupAction("download-models", f"Download public SkinTokens checkpoints into {MODEL_LOGICAL_ROOT}."),
        SetupAction("pip-check", "Run pip check for dependency conflicts.", [python_exe, "-m", "pip", "check"]),
        SetupAction("import-probes", "Probe critical native/runtime modules: " + ", ".join(NATIVE_PROBES)),
    ]


def _coerce_payload_arg(explicit_payload: str | None, positional_payload: str | None, unknown_args: list[str]) -> str | None:
    if explicit_payload:
        return explicit_payload
    if positional_payload:
        return positional_payload
    if unknown_args and unknown_args[0].lstrip().startswith("{"):
        return " ".join(unknown_args)
    return None


def _load_payload(raw_payload: str | None) -> dict[str, Any]:
    if not raw_payload:
        return {}
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise SetupError("Modly setup payload must be a JSON object.", code="payload-invalid", stage="payload")
    return payload


def _payload_value(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _emit_command_tail(writer: EventWriter | None, result: CommandResult, *, stage: str) -> None:
    if writer is None:
        return
    if result.stdout_tail.strip():
        writer.log("stdout tail:\n" + result.stdout_tail.strip(), stage)
    if result.stderr_tail.strip():
        writer.log("stderr tail:\n" + result.stderr_tail.strip(), stage)


def _run_checked(
    runner: Runner,
    args: list[str],
    *,
    stage: str,
    cwd: Path | None = None,
    writer: EventWriter | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    result = runner.run(args, cwd=cwd, env=env)
    _emit_command_tail(writer, result, stage=stage)
    if not result.ok:
        raise SetupError(f"Setup command failed at {stage}.", code=f"{stage}-failed", stage=stage, detail=asdict(result))
    return result


def _create_venv(venv_dir: Path, *, base_python: str | None, runner: Runner, writer: EventWriter | None = None) -> dict[str, Any]:
    if (venv_dir / "pyvenv.cfg").exists():
        return {"status": "skipped", "reason": "already-exists", "venv_python": str(_venv_python(venv_dir))}
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    if base_python and Path(base_python).expanduser().exists() and Path(base_python).resolve() != Path(sys.executable).resolve():
        result = _run_checked(runner, [base_python, "-m", "venv", str(venv_dir)], stage="create-venv", writer=writer)
        return {"status": "created", "method": "payload-python", "result": asdict(result), "venv_python": str(_venv_python(venv_dir))}
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(venv_dir)
    return {"status": "created", "method": "stdlib-venv", "venv_python": str(_venv_python(venv_dir))}


def _download_upstream_source(*, layout: ModlyLayout, source_ref: str, writer: EventWriter) -> dict[str, Any]:
    runtime = runtime_root(layout.ext_dir)
    vendor_dir = upstream_dir_for_layout(layout)
    cache_dir = runtime / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"skintokens-{source_ref}.zip"
    url = UPSTREAM_ARCHIVE_URL.format(ref=source_ref)
    writer.log(f"Downloading SkinTokens source snapshot: {url}", "download-upstream-source")
    urllib.request.urlretrieve(url, archive_path)
    extract_root = runtime / "vendor" / f"skintokens-{source_ref}.extracting"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_root)
    candidates = [path for path in extract_root.iterdir() if path.is_dir()]
    if len(candidates) != 1:
        raise SetupError("SkinTokens source archive did not contain exactly one root directory.", code="source-archive-invalid", stage="download-upstream-source")
    if vendor_dir.exists():
        shutil.rmtree(vendor_dir)
    vendor_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(candidates[0]), str(vendor_dir))
    shutil.rmtree(extract_root, ignore_errors=True)
    return {"status": "downloaded", "source_ref": source_ref, "archive": str(archive_path), "vendor_dir": str(vendor_dir)}


def _sync_runtime_resources(layout: ModlyLayout) -> dict[str, Any]:
    """Copy non-weight upstream runtime resources into the model cwd.

    SkinTokens loads weights from `experiments/...` and Qwen config from
    `models/Qwen3-0.6B/...`, so the runtime cwd is the Modly model root. Some
    upstream configs are also resolved relative to cwd, e.g.
    `configs/skeleton/vroid.yaml`. Keep those small resources next to the model
    cwd while leaving the actual source code vendored under the extension
    runtime directory.
    """

    source_root = upstream_dir_for_layout(layout)
    model_root = model_root_for_layout(layout)
    copied: list[dict[str, Any]] = []
    for relative in ["configs"]:
        source = source_root / relative
        target = model_root / relative
        if not source.exists():
            continue
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        file_count = sum(1 for path in target.rglob("*") if path.is_file())
        copied.append({"relative": relative, "source": str(source), "target": str(target), "file_count": file_count})
    return {"status": "synced", "resources": copied}


def _download_model_assets(*, venv_python: Path, layout: ModlyLayout, runner: Runner, writer: EventWriter | None = None) -> dict[str, Any]:
    model_root = model_root_for_layout(layout)
    model_root.mkdir(parents=True, exist_ok=True)
    script = f"""
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path
root = Path({str(model_root)!r})
root.mkdir(parents=True, exist_ok=True)
hf_repo = 'VAST-AI/SkinTokens'
files = {[item['hf_path'] for item in MODEL_SENTINELS if item.get('hf_path')]!r}
for filename in files:
    hf_hub_download(repo_id=hf_repo, filename=filename, local_dir=str(root))
snapshot_download(repo_id='Qwen/Qwen3-0.6B', local_dir=str(root / 'models' / 'Qwen3-0.6B'), ignore_patterns=['*.bin', '*.safetensors'])
print('downloaded')
"""
    result = _run_checked(runner, [str(venv_python), "-c", script], stage="download-models", writer=writer)
    return {"status": "downloaded", "model_root": str(model_root), "result": asdict(result)}


def _find_blender_candidate(payload: dict[str, Any]) -> str | None:
    explicit = payload.get("blender_exe") or payload.get("blenderExe") or os.environ.get("MODLY_SKINTOKENS_BLENDER_EXE")
    if explicit:
        path = Path(str(explicit)).expanduser()
        return str(path) if path.exists() else str(explicit)
    return shutil.which("blender")


def _probe_python_bpy(venv_python: Path, runner: Runner, writer: EventWriter | None = None) -> dict[str, Any]:
    script = """
import importlib, json
payload = {'provider': 'python-bpy', 'status': 'failed'}
try:
    bpy = importlib.import_module('bpy')
    payload.update({'status': 'ok', 'version': getattr(bpy.app, 'version_string', getattr(bpy, '__version__', None))})
except Exception as exc:
    payload.update({'error': f'{type(exc).__name__}: {exc}'})
print(json.dumps(payload, sort_keys=True))
"""
    result = runner.run([str(venv_python), "-c", script])
    _emit_command_tail(writer, result, stage="probe-python-bpy")
    try:
        payload = json.loads(result.stdout_tail.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        payload = {"provider": "python-bpy", "status": "failed", "error": "probe did not emit JSON"}
    if result.ok and payload.get("status") == "ok":
        return {"kind": "python-bpy", "status": "ok", "probe": payload}
    return {"kind": "python-bpy", "status": "failed", "probe": payload, "result": asdict(result)}


def _probe_external_blender(blender_exe: str, runner: Runner, writer: EventWriter | None = None) -> dict[str, Any]:
    expr = (
        "import json, sys; "
        "import bpy; "
        "payload={'provider':'external-blender','status':'ok','blender_version':getattr(bpy.app,'version_string',''),"
        "'python_version':sys.version,'python_executable':sys.executable}; "
        f"print({BPY_PROBE_MARKER!r} + json.dumps(payload, sort_keys=True))"
    )
    result = runner.run([blender_exe, "--background", "--factory-startup", "--python-expr", expr])
    _emit_command_tail(writer, result, stage="probe-external-blender")
    payload: dict[str, Any] = {"provider": "external-blender", "status": "failed", "executable": blender_exe}
    for line in result.stdout_tail.splitlines() + result.stderr_tail.splitlines():
        if line.startswith(BPY_PROBE_MARKER):
            try:
                payload = json.loads(line[len(BPY_PROBE_MARKER) :])
                payload["executable"] = blender_exe
            except json.JSONDecodeError:
                payload = {"provider": "external-blender", "status": "failed", "executable": blender_exe, "error": "probe JSON marker was invalid"}
            break
    if result.ok and payload.get("status") == "ok":
        return {"kind": "external-blender", "status": "ok", "executable": blender_exe, "probe": payload}
    return {"kind": "external-blender", "status": "failed", "executable": blender_exe, "probe": payload, "result": asdict(result)}


def _resolve_bpy_provider(*, venv_python: Path, payload: dict[str, Any], runner: Runner, writer: EventWriter) -> dict[str, Any]:
    writer.progress(55, "Installing/probing bpy provider", "resolve-bpy")
    install_result = runner.run([str(venv_python), "-m", "pip", "install", *PIP_FLAGS, BPy_PACKAGE])
    _emit_command_tail(writer, install_result, stage="install-bpy")
    if install_result.ok:
        python_probe = _probe_python_bpy(venv_python, runner, writer)
        if python_probe.get("status") == "ok":
            writer.log("Using bpy installed in the extension venv.", "resolve-bpy")
            return {"kind": "python-bpy", "status": "ok", "install": asdict(install_result), "probe": python_probe}
        writer.log("bpy installed but import probe failed; trying Blender fallback.", "resolve-bpy")
    else:
        writer.log("pip install bpy failed; trying Blender fallback.", "resolve-bpy")

    candidate = _find_blender_candidate(payload)
    if candidate:
        blender_probe = _probe_external_blender(candidate, runner, writer)
        if blender_probe.get("status") == "ok":
            writer.log(f"Using external Blender fallback: {candidate}", "resolve-bpy")
            return {"kind": "external-blender", "status": "ok", "pip_install": asdict(install_result), "probe": blender_probe}
        detail = {"pip_install": asdict(install_result), "blender_probe": blender_probe}
    else:
        detail = {"pip_install": asdict(install_result), "blender_probe": {"status": "missing", "message": "No Blender executable found in payload, env, or PATH."}}

    raise SetupError(
        "No usable bpy provider found. pip install bpy failed and Blender fallback was unavailable or failed its probe.",
        code="bpy-provider-unavailable",
        stage="resolve-bpy",
        detail=detail,
    )


def _install_optional_packages(*, venv_python: Path, runner: Runner, writer: EventWriter) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for package in OPTIONAL_REQUIREMENTS:
        stage = f"install-optional-{package}"
        result = runner.run([str(venv_python), "-m", "pip", "install", *PIP_FLAGS, package])
        _emit_command_tail(writer, result, stage=stage)
        if result.ok:
            writer.log(f"Optional package installed: {package}", stage)
            results[package] = {"status": "installed", "result": asdict(result)}
        else:
            writer.log(f"Optional package unavailable; related features will be disabled unless installed manually: {package}", stage)
            results[package] = {"status": "unavailable", "result": asdict(result)}
    return results


def _flash_attn_wheelhouse(layout: ModlyLayout) -> Path:
    return runtime_root(layout.ext_dir) / FLASH_ATTN_WHEELHOUSE_RELATIVE


def _local_flash_attn_wheels(layout: ModlyLayout) -> list[Path]:
    wheelhouse = _flash_attn_wheelhouse(layout)
    if not wheelhouse.exists():
        return []
    return sorted([*wheelhouse.glob("flash_attn-*.whl"), *wheelhouse.glob("flash-attn-*.whl")])


def _install_flash_attn(
    *,
    venv_python: Path,
    layout: ModlyLayout,
    runner: Runner,
    writer: EventWriter,
    allow_source_build: bool,
) -> dict[str, Any]:
    wheelhouse = _flash_attn_wheelhouse(layout)
    wheels = _local_flash_attn_wheels(layout)
    if wheels:
        writer.log(f"Installing flash-attn from local wheelhouse: {wheelhouse}", "install-flash-attn")
        command = [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            FLASH_ATTN_PACKAGE,
        ]
        return {"mode": "local-wheelhouse", "wheelhouse": str(wheelhouse), "wheels": [path.name for path in wheels], "result": asdict(_run_checked(runner, command, stage="install-flash-attn", writer=writer))}

    writer.log("No local flash-attn wheel found; trying binary-only pip install before any source build.", "install-flash-attn")
    binary_command = [str(venv_python), "-m", "pip", "install", *PIP_FLAGS, "--only-binary", ":all:", FLASH_ATTN_PACKAGE]
    binary_result = runner.run(binary_command)
    _emit_command_tail(writer, binary_result, stage="install-flash-attn")
    if binary_result.ok:
        return {"mode": "binary-pip", "result": asdict(binary_result)}

    if not allow_source_build:
        raise SetupError(
            "No compatible flash-attn wheel was available. Build a local flash-attn wheel with setup.py --build-flash-attn-wheel, or rerun setup with --allow-flash-attn-source-build if an explicit source build is acceptable.",
            code="flash-attn-wheel-unavailable",
            stage="install-flash-attn",
            detail={
                "local_wheelhouse": str(wheelhouse),
                "binary_attempt": asdict(binary_result),
                "next_steps": [
                    "Run setup.py --build-flash-attn-wheel with the same Modly payload to build a reusable local wheel.",
                    "Then rerun normal setup so it installs from the local wheelhouse.",
                    "Do not leave source build enabled by default for end users.",
                ],
            },
        )

    flash_env = _cuda_build_env("12.8")
    if flash_env:
        writer.log(f"Using CUDA_HOME={flash_env['CUDA_HOME']} for explicit flash-attn source build", "install-flash-attn")
    writer.log("Explicit source build is enabled; this can take a long time.", "install-flash-attn")
    source_command = [str(venv_python), "-m", "pip", "install", *PIP_FLAGS, FLASH_ATTN_PACKAGE, "--no-build-isolation"]
    return {"mode": "source-build", "result": asdict(_run_checked(runner, source_command, stage="install-flash-attn", writer=writer, env=flash_env or None))}


def _build_flash_attn_wheel(
    *,
    layout: ModlyLayout,
    base_python: str,
    runner: Runner,
    writer: EventWriter,
    max_build_jobs: str | None = None,
) -> int:
    venv_dir = layout.ext_dir / "venv"
    venv_python = _venv_python(venv_dir)
    wheelhouse = _flash_attn_wheelhouse(layout)
    wheelhouse.mkdir(parents=True, exist_ok=True)
    writer.progress(5, "Preparing venv for flash-attn wheel build", "flash-attn-build-venv")
    _create_venv(venv_dir, base_python=base_python, runner=runner, writer=writer)
    writer.progress(15, "Installing flash-attn wheel build tooling", "flash-attn-build-tools")
    _run_checked(runner, [str(venv_python), "-m", "pip", "install", *PIP_FLAGS, "--upgrade", "pip", "setuptools", "wheel", *FLASH_ATTN_BUILD_REQUIREMENTS], stage="flash-attn-build-tools", writer=writer)
    writer.progress(30, "Installing PyTorch CUDA 12.8 lane for wheel build", "flash-attn-build-torch")
    _run_checked(runner, [str(venv_python), "-m", "pip", "install", *PIP_FLAGS, *TORCH_PACKAGES, "--index-url", TORCH_INDEX_URL], stage="flash-attn-build-torch", writer=writer)
    writer.progress(50, "Building flash-attn wheel into local wheelhouse", "flash-attn-build-wheel")
    env = _cuda_build_env("12.8")
    if max_build_jobs:
        env = {**env, "MAX_JOBS": str(max_build_jobs)}
        writer.log(f"Using MAX_JOBS={max_build_jobs} for flash-attn wheel build", "flash-attn-build-wheel")
    if env.get("CUDA_HOME"):
        writer.log(f"Using CUDA_HOME={env['CUDA_HOME']} for flash-attn wheel build", "flash-attn-build-wheel")
    command = [str(venv_python), "-m", "pip", "wheel", *PIP_FLAGS, FLASH_ATTN_PACKAGE, "--no-build-isolation", "--no-deps", "--wheel-dir", str(wheelhouse)]
    result = _run_checked(runner, command, stage="flash-attn-build-wheel", writer=writer, env=env or None)
    wheels = _local_flash_attn_wheels(layout)
    if not wheels:
        raise SetupError("flash-attn wheel build completed but no wheel was found in the local wheelhouse.", code="flash-attn-wheel-missing-after-build", stage="flash-attn-build-wheel", detail={"result": asdict(result), "wheelhouse": str(wheelhouse)})
    writer.progress(100, "flash-attn wheel build complete", "flash-attn-build-done")
    writer.send({"type": "setup_done", "status": "flash_attn_wheel_ready", "wheelhouse": str(wheelhouse), "wheels": [path.name for path in wheels]})
    return 0


def _cuda_build_env(torch_cuda_version: str = "12.8") -> dict[str, str]:
    candidates = [
        Path(f"/usr/local/cuda-{torch_cuda_version}"),
        Path(f"/usr/local/cuda-{torch_cuda_version.split('.', 1)[0]}"),
    ]
    for cuda_home in candidates:
        if (cuda_home / "bin" / "nvcc").exists():
            path = str(cuda_home / "bin") + os.pathsep + os.environ.get("PATH", "")
            ld_library_path = str(cuda_home / "lib64") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
            return {
                "CUDA_HOME": str(cuda_home),
                "CUDA_PATH": str(cuda_home),
                "PATH": path,
                "LD_LIBRARY_PATH": ld_library_path,
            }
    return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model_assets(layout: ModlyLayout) -> dict[str, Any]:
    root = model_root_for_layout(layout)
    checks: dict[str, Any] = {}
    for item in MODEL_SENTINELS:
        path = root / str(item["path"])
        record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            checks[str(item["id"])] = {**record, "status": "missing"}
            continue
        if "size_bytes" in item:
            record["size_bytes"] = path.stat().st_size
            record["size_ok"] = path.stat().st_size == int(item["size_bytes"])
        if "sha256" in item:
            actual = _sha256(path)
            record["sha256"] = actual
            record["sha256_ok"] = actual == item["sha256"]
        ok = bool(record.get("exists")) and record.get("size_ok", True) and record.get("sha256_ok", True)
        checks[str(item["id"])] = {**record, "status": "ok" if ok else "failed"}
    failed = {key: value for key, value in checks.items() if value.get("status") != "ok"}
    if failed:
        raise SetupError("One or more SkinTokens model asset checks failed.", code="model-asset-check-failed", stage="verify-models", detail=failed)
    return checks


def _run_import_probes(venv_python: Path, runner: Runner, *, bpy_provider: dict[str, Any], writer: EventWriter | None = None) -> dict[str, Any]:
    script = f"""
import importlib, json, sys
probes = {NATIVE_PROBES!r}
result = {{}}
for label, module_name in probes.items():
    try:
        module = importlib.import_module(module_name)
        result[label] = {{'status': 'ok', 'version': getattr(module, '__version__', None)}}
    except Exception as exc:
        result[label] = {{'status': 'failed', 'error': f'{{type(exc).__name__}}: {{exc}}'}}
try:
    import torch
    result['torch_cuda'] = {{'status': 'ok' if torch.cuda.is_available() else 'failed', 'torch_version': torch.__version__, 'torch_cuda_version': torch.version.cuda, 'cuda_available': bool(torch.cuda.is_available())}}
except Exception as exc:
    result['torch_cuda'] = {{'status': 'failed', 'error': f'{{type(exc).__name__}}: {{exc}}'}}
print(json.dumps(result, sort_keys=True))
failed = [key for key, value in result.items() if value.get('status') != 'ok']
sys.exit(1 if failed else 0)
"""
    result = runner.run([str(venv_python), "-c", script])
    _emit_command_tail(writer, result, stage="import-probes")
    try:
        payload = json.loads(result.stdout_tail.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        payload = {"status": "failed", "error": "probe did not emit JSON"}
    if not result.ok:
        raise SetupError("SkinTokens import probes failed.", code="import-probes-failed", stage="import-probes", detail={"result": asdict(result), "probes": payload})
    optional_payload: dict[str, Any] = {}
    for label, module_name in OPTIONAL_PROBES.items():
        optional_result = runner.run([str(venv_python), "-c", f"import importlib, json\ntry:\n    module = importlib.import_module({module_name!r})\n    payload = {{'status': 'ok', 'version': getattr(module, '__version__', None)}}\nexcept Exception as exc:\n    payload = {{'status': 'unavailable', 'error': f'{{type(exc).__name__}}: {{exc}}'}}\nprint(json.dumps(payload, sort_keys=True))"])
        try:
            optional_payload[label] = json.loads(optional_result.stdout_tail.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            optional_payload[label] = {"status": "unavailable", "error": "optional probe did not emit JSON"}
    payload["bpy_provider"] = bpy_provider
    payload["optional"] = optional_payload
    return payload


def _write_failed_state(*, layout: ModlyLayout, error: SetupError, installs_started: bool, downloads_started: bool) -> Path:
    state = planned_state(status="failed", dry_run=False, layout=layout)
    state.update(
        {
            "failure_code": error.code,
            "failure_stage": error.stage,
            "failure_detail": error.detail,
            "downloads_started": downloads_started,
            "installs_started": installs_started,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "next_steps": [
                "Review the setup failure stage and stderr_tail/detail payload.",
                "Fix the platform-specific dependency or model download issue.",
                "Rerun Modly extension setup before generation.",
            ],
        }
    )
    return write_state_for_layout(state, layout)


def _write_ready_state(*, layout: ModlyLayout, setup_summary: dict[str, Any]) -> Path:
    state = planned_state(status="ready", dry_run=False, layout=layout)
    state.update(
        {
            "failure_code": None,
            "downloads_started": True,
            "installs_started": True,
            "setup_summary": setup_summary,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "next_steps": ["Run a first SkinTokens generation smoke test in Modly.", "Only mark the platform supported after a successful rigged GLB output."],
        }
    )
    return write_state_for_layout(state, layout)


def run_setup(
    *,
    dry_run: bool,
    prepare: bool = False,
    payload: dict[str, Any] | None = None,
    workspace_root: Path | str | None = None,
    source_ref: str = "main",
    skip_install: bool = False,
    skip_download: bool = False,
    root: Path | None = None,
    writer: EventWriter | None = None,
    runner: Runner | None = None,
    allow_flash_attn_source_build: bool = False,
) -> int:
    payload = payload or {}
    root = root or extension_root()
    layout = resolve_modly_layout(
        workspace_root or root,
        ext_dir=_payload_value(payload, "ext_dir", "extension_dir", "extensionDir") or root,
        modly_home=_payload_value(payload, "modly_home", "modlyHome"),
        models_root=_payload_value(payload, "models_root", "modelsRoot", "models_dir", "modelsDir"),
    )
    writer = writer or EventWriter()
    runner = runner or SubprocessRunner()
    execute_real = prepare or bool(payload)
    base_python = str(_payload_value(payload, "python_exe", "python", "pythonExe") or sys.executable)
    venv_dir = layout.ext_dir / "venv"
    venv_python = _venv_python(venv_dir)
    actions = build_setup_plan(str(venv_python))

    writer.progress(1, "Preparing SkinTokens setup", "setup-start")
    runtime_root(layout.ext_dir).mkdir(parents=True, exist_ok=True)
    model_root_for_layout(layout).mkdir(parents=True, exist_ok=True)
    writer.log(f"Extension directory: {layout.ext_dir}", "setup-start")
    writer.log(f"Model root: {model_root_for_layout(layout)}", "setup-start")

    if dry_run or not execute_real:
        for index, action in enumerate(actions, start=1):
            percent = min(95, 5 + int(index / len(actions) * 85))
            writer.progress(percent, action.description, action.id)
            if action.command:
                writer.log("planned command: " + " ".join(action.command), action.id)
        state = planned_state(status="dry_run", dry_run=True, layout=layout)
        state["planned_actions"] = [asdict(action) for action in actions]
        state["probes"] = {name: {"status": "planned"} for name in NATIVE_PROBES}
        state["downloads"] = {item["id"]: {"status": "planned", "logical_path": item.get("logical_path"), "path": item["path"]} for item in MODEL_SENTINELS}
        path = write_state_for_layout(state, layout)
        writer.progress(100, "SkinTokens setup dry-run recorded", "setup-done")
        writer.send({"type": "setup_done", "status": "dry_run", "statePath": str(path), "resolved_paths": layout.as_dict()})
        return 0

    installs_started = False
    downloads_started = False
    summary: dict[str, Any] = {"resolved_paths": layout.as_dict(), "source_ref": source_ref, "commands": {}}
    try:
        writer.progress(5, "Creating extension virtual environment", "create-venv")
        summary["venv"] = _create_venv(venv_dir, base_python=base_python, runner=runner, writer=writer)

        if not skip_install:
            installs_started = True
            writer.progress(12, "Upgrading pip/setuptools/wheel", "upgrade-pip")
            summary["commands"]["upgrade-pip"] = asdict(_run_checked(runner, [str(venv_python), "-m", "pip", "install", *PIP_FLAGS, "--upgrade", "pip", "setuptools", "wheel"], stage="upgrade-pip", writer=writer))
            writer.progress(22, "Installing PyTorch CUDA 12.8 lane", "install-torch-cu128")
            summary["commands"]["install-torch-cu128"] = asdict(_run_checked(runner, [str(venv_python), "-m", "pip", "install", *PIP_FLAGS, *TORCH_PACKAGES, "--index-url", TORCH_INDEX_URL], stage="install-torch-cu128", writer=writer))
            writer.progress(38, "Installing SkinTokens requirements", "install-requirements")
            summary["commands"]["install-requirements"] = asdict(_run_checked(runner, [str(venv_python), "-m", "pip", "install", *PIP_FLAGS, *GENERIC_REQUIREMENTS], stage="install-requirements", writer=writer))
            writer.progress(45, "Installing flash-attn build prerequisites", "install-flash-attn-build-prereqs")
            summary["commands"]["install-flash-attn-build-prereqs"] = asdict(_run_checked(runner, [str(venv_python), "-m", "pip", "install", *PIP_FLAGS, *FLASH_ATTN_BUILD_REQUIREMENTS], stage="install-flash-attn-build-prereqs", writer=writer))
            writer.progress(50, "Installing flash-attn", "install-flash-attn")
            summary["commands"]["install-flash-attn"] = _install_flash_attn(venv_python=venv_python, layout=layout, runner=runner, writer=writer, allow_source_build=allow_flash_attn_source_build)
            writer.progress(53, "Installing optional open3d", "install-optional-open3d")
            summary["optional_packages"] = _install_optional_packages(venv_python=venv_python, runner=runner, writer=writer)
            summary["bpy_provider"] = _resolve_bpy_provider(venv_python=venv_python, payload=payload, runner=runner, writer=writer)
        else:
            writer.log("Dependency installation skipped by explicit flag.", "install-skip")
            summary["install_skipped"] = True
            summary["bpy_provider"] = {"kind": "skipped", "status": "skipped"}

        downloads_started = True
        writer.progress(60, "Downloading SkinTokens source snapshot", "download-upstream-source")
        summary["upstream_source"] = _download_upstream_source(layout=layout, source_ref=source_ref, writer=writer)
        writer.progress(66, "Syncing SkinTokens runtime resources", "sync-runtime-resources")
        summary["runtime_resources"] = _sync_runtime_resources(layout)

        if not skip_download:
            writer.progress(72, "Downloading SkinTokens checkpoints to Modly models", "download-models")
            summary["model_downloads"] = _download_model_assets(venv_python=venv_python, layout=layout, runner=runner, writer=writer)
        else:
            writer.log("Model download skipped by explicit flag; existing sentinels must already be present.", "download-skip")
            summary["download_skipped"] = True

        writer.progress(84, "Verifying model sentinels and checksums", "verify-models")
        summary["model_checks"] = _verify_model_assets(layout)

        if not skip_install:
            writer.progress(90, "Running pip check", "pip-check")
            summary["commands"]["pip-check"] = asdict(_run_checked(runner, [str(venv_python), "-m", "pip", "check"], stage="pip-check", writer=writer))
            writer.progress(95, "Running import probes", "import-probes")
            summary["probes"] = _run_import_probes(venv_python, runner, bpy_provider=summary["bpy_provider"], writer=writer)

        path = _write_ready_state(layout=layout, setup_summary=summary)
        writer.progress(100, "SkinTokens setup ready", "setup-done")
        writer.send({"type": "setup_done", "status": "ready", "statePath": str(path), "resolved_paths": layout.as_dict()})
        return 0
    except SetupError as exc:
        path = _write_failed_state(layout=layout, error=exc, installs_started=installs_started, downloads_started=downloads_started)
        writer.error(str(exc), code=exc.code, stage=exc.stage)
        writer.send({"type": "setup_done", "status": "failed", "failure_code": exc.code, "statePath": str(path), "resolved_paths": layout.as_dict()})
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("SkinTokens Modly setup")
    parser.add_argument("--dry-run", action="store_true", help="Record setup plan without installing or downloading anything.")
    parser.add_argument("--prepare", action="store_true", help="Execute real setup even without a Modly payload.")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--payload-json")
    parser.add_argument("--source-ref", default="main")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip installation; useful only for controlled diagnostics.")
    parser.add_argument("--skip-download", action="store_true", help="Skip HF model downloads; existing sentinels must already exist.")
    parser.add_argument("--allow-flash-attn-source-build", action="store_true", help="Allow normal setup to build flash-attn from source when no wheel is available. Disabled by default to avoid unbounded user installs.")
    parser.add_argument("--build-flash-attn-wheel", action="store_true", help="Build a reusable local flash-attn wheel into .skintokens-runtime/wheelhouse/flash-attn, then exit.")
    parser.add_argument("--max-build-jobs", default=None, help="Optional MAX_JOBS value used only with --build-flash-attn-wheel.")
    parser.add_argument("positional_payload_json", nargs="?", help="Modly install payload JSON. Modly may pass this as a positional argument.")
    args, unknown_args = parser.parse_known_args(argv)
    writer = EventWriter()
    try:
        raw_payload = _coerce_payload_arg(args.payload_json, args.positional_payload_json, unknown_args)
        payload = _load_payload(raw_payload)
        if args.build_flash_attn_wheel:
            root = extension_root()
            layout = resolve_modly_layout(
                args.workspace_root or root,
                ext_dir=_payload_value(payload, "ext_dir", "extension_dir", "extensionDir") or root,
                modly_home=_payload_value(payload, "modly_home", "modlyHome"),
                models_root=_payload_value(payload, "models_root", "modelsRoot", "models_dir", "modelsDir"),
            )
            base_python = str(_payload_value(payload, "python_exe", "python", "pythonExe") or sys.executable)
            return _build_flash_attn_wheel(layout=layout, base_python=base_python, runner=SubprocessRunner(), writer=writer, max_build_jobs=args.max_build_jobs)
        return run_setup(
            dry_run=args.dry_run,
            prepare=args.prepare,
            payload=payload,
            workspace_root=args.workspace_root,
            source_ref=args.source_ref,
            skip_install=args.skip_install,
            skip_download=args.skip_download,
            writer=writer,
            allow_flash_attn_source_build=args.allow_flash_attn_source_build or os.environ.get("MODLY_SKINTOKENS_ALLOW_FLASH_ATTN_SOURCE_BUILD") == "1",
        )
    except (json.JSONDecodeError, SetupError) as exc:
        code = getattr(exc, "code", "payload-json-invalid")
        stage = getattr(exc, "stage", "payload")
        writer.error(str(exc), code=code, stage=stage)
        writer.send({"type": "setup_done", "status": "failed", "failure_code": code})
        return 1
