from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .events import LineRelay
from .paths import create_run_dir, extension_root, resolve_modly_layout, runtime_root, validate_mesh_input
from .readiness import check_ready_for_layout, model_root_for_layout, upstream_dir_for_layout
from .validation import OutputValidation, validate_output


ProgressFn = Callable[[int, str, str | None], None]
LogFn = Callable[[str, str | None], None]


STAGES = [
    (5, "validate-input", "Validating mesh input"),
    (10, "readiness", "Checking SkinTokens runtime assets"),
    (15, "bpy-server", "Starting Blender Python export server"),
    (25, "load-model", "Loading TokenRig and SkinTokens checkpoints"),
    (35, "prepare-mesh", "Preparing mesh dataset"),
    (45, "encode-mesh", "Encoding mesh conditioning"),
    (65, "generate-tokens", "Generating skeleton and SkinTokens"),
    (75, "decode-skin", "Decoding skin weights"),
    (82, "postprocess", "Applying optional voxel skin postprocess"),
    (92, "export-glb", "Exporting rigged GLB"),
    (98, "validate-output", "Validating generated GLB"),
]


class PipelineError(RuntimeError):
    def __init__(self, message: str, *, code: str = "pipeline", stage: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage


@dataclass(frozen=True)
class RuntimeParams:
    top_k: int = 5
    top_p: float = 0.95
    temperature: float = 1.0
    repetition_penalty: float = 2.0
    num_beams: int = 10
    use_skeleton: bool = False
    use_transfer: bool = True
    use_postprocess: bool = False


@dataclass(frozen=True)
class RunResult:
    file_path: Path
    validation: OutputValidation


class Backend(Protocol):
    requires_readiness: bool

    def run(self, *, mesh_path: Path, output_path: Path, params: RuntimeParams, progress: ProgressFn, log: LogFn) -> Path:
        ...


def parse_bool_select(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise PipelineError(f"Invalid boolean select value: {value!r}", code="params")


def normalize_params(raw: dict[str, Any] | None) -> RuntimeParams:
    raw = raw or {}
    try:
        return RuntimeParams(
            top_k=int(raw.get("top_k", 5)),
            top_p=float(raw.get("top_p", 0.95)),
            temperature=float(raw.get("temperature", 1.0)),
            repetition_penalty=float(raw.get("repetition_penalty", 2.0)),
            num_beams=int(raw.get("num_beams", 10)),
            use_skeleton=parse_bool_select(raw.get("use_skeleton"), default=False),
            use_transfer=parse_bool_select(raw.get("use_transfer"), default=True),
            use_postprocess=parse_bool_select(raw.get("use_postprocess"), default=False),
        )
    except ValueError as exc:
        raise PipelineError(f"Invalid numeric generation parameter: {exc}", code="params") from exc


def run_pipeline(
    *,
    mesh_path: Path,
    params: dict[str, Any] | None,
    progress: ProgressFn,
    log: LogFn,
    workspace_dir: Path | None = None,
    run_id: str | None = None,
    backend: Backend | None = None,
) -> RunResult:
    progress(5, "Validating mesh input", "validate-input")
    mesh = validate_mesh_input(mesh_path)
    parsed = normalize_params(params)

    root = extension_root()
    layout = resolve_modly_layout(root, ext_dir=root)
    rr = runtime_root(root)
    rr.mkdir(parents=True, exist_ok=True)
    run_dir = create_run_dir(rr, run_id=run_id)
    output_path = _select_output_path(mesh, run_dir, workspace_dir)

    backend = backend or _select_backend()
    if backend.requires_readiness:
        progress(10, "Checking SkinTokens runtime assets", "readiness")
        readiness = check_ready_for_layout(layout)
        if not readiness.ready:
            raise PipelineError(readiness.public_message(), code="not-ready", stage="readiness")
        log("SkinTokens runtime assets are ready", "readiness")

    produced = backend.run(mesh_path=mesh, output_path=output_path, params=parsed, progress=progress, log=log)
    progress(98, "Validating generated GLB", "validate-output")
    validation = validate_output(produced)
    if validation.warnings:
        log("Output validation warnings: " + "; ".join(validation.warnings), "validate-output")
    return RunResult(file_path=produced, validation=validation)


def _select_output_path(mesh: Path, run_dir: Path, workspace_dir: Path | None) -> Path:
    if workspace_dir is None:
        target_dir = run_dir
    elif workspace_dir.name == "Workflows":
        target_dir = workspace_dir
    else:
        target_dir = workspace_dir / "Workflows"
    return (target_dir / f"{mesh.stem}_skintokens.glb").resolve()


def _select_backend() -> Backend:
    if os.environ.get("MODLY_SKINTOKENS_FAKE_RUNTIME") == "1":
        return FakeBackend()
    return SkinTokensBackend()


class FakeBackend:
    """Lightweight backend for protocol tests; does not import SkinTokens or torch."""

    requires_readiness = False

    def run(self, *, mesh_path: Path, output_path: Path, params: RuntimeParams, progress: ProgressFn, log: LogFn) -> Path:
        for percent, stage, label in STAGES[2:-1]:
            progress(percent, label, stage)
            if stage == "load-model":
                log("Fake backend skipped model load", stage)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mesh_path, output_path)
        return output_path


class SkinTokensBackend:
    """Instrumented wrapper around upstream SkinTokens internals.

    Heavy imports happen only inside `run`, after setup/readiness has passed.
    Tests use `FakeBackend`, so this class is intentionally not imported/executed by
    the lightweight test suite.
    """

    requires_readiness = True

    def run(self, *, mesh_path: Path, output_path: Path, params: RuntimeParams, progress: ProgressFn, log: LogFn) -> Path:
        layout = resolve_modly_layout(extension_root(), ext_dir=extension_root())
        upstream = upstream_dir_for_layout(layout)
        model_root = model_root_for_layout(layout)
        if not upstream.exists():
            raise PipelineError("SkinTokens upstream source is not provisioned.", code="upstream-missing", stage="readiness")
        if not model_root.exists():
            raise PipelineError("SkinTokens model root is not provisioned.", code="models-missing", stage="readiness")

        old_cwd = Path.cwd()
        old_sys_path = list(sys.path)
        sys.path.insert(0, str(upstream))
        os.chdir(model_root)
        relay_out = LineRelay(log, stage="upstream-stdout", prefix="upstream: ")
        relay_err = LineRelay(log, stage="upstream-stderr", prefix="upstream stderr: ")
        try:
            with contextlib.redirect_stdout(relay_out), contextlib.redirect_stderr(relay_err):
                return self._run_upstream(mesh_path=mesh_path, output_path=output_path, params=params, progress=progress, log=log)
        finally:
            relay_out.flush()
            relay_err.flush()
            os.chdir(old_cwd)
            sys.path[:] = old_sys_path

    def _run_upstream(self, *, mesh_path: Path, output_path: Path, params: RuntimeParams, progress: ProgressFn, log: LogFn) -> Path:
        import tempfile

        import requests
        from torch import Tensor

        from src.data.dataset import DatasetConfig, RigDatasetModule
        from src.data.transform import Transform
        from src.data.vertex_group import voxel_skin
        from src.model.tokenrig import TokenRigResult
        from src.server.spec import BPY_SERVER, bytes_to_object, get_model, object_to_bytes
        from src.tokenizer.parse import get_tokenizer

        progress(15, "Starting Blender Python export server", "bpy-server")
        server_proc = self._start_bpy_server(log)
        try:
            self._wait_for_bpy_server(BPY_SERVER)
            log("bpy_server is ready", "bpy-server")

            progress(25, "Loading TokenRig and SkinTokens checkpoints", "load-model")
            model_ckpt = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
            model = get_model(model_ckpt, hf_path=None)
            if model.tokenizer_config is None:
                raise PipelineError("SkinTokens checkpoint did not provide tokenizer_config.", code="model-load", stage="load-model")
            tokenizer = get_tokenizer(**model.tokenizer_config)
            transform = Transform.parse(**model.transform_config["predict_transform"])

            progress(35, "Preparing mesh dataset", "prepare-mesh")
            dataset_config = DatasetConfig.parse(
                shuffle=False,
                batch_size=1,
                num_workers=1,
                pin_memory=True,
                persistent_workers=False,
                datapath={
                    "data_name": None,
                    "loader": "bpy_server",
                    "filepaths": {"articulation": [str(mesh_path)]},
                },
            ).split_by_cls()
            module = RigDatasetModule(
                predict_dataset_config=dataset_config,
                predict_transform=transform,
                tokenizer=tokenizer,
                process_fn=model._process_fn,
            )
            dataloader = module.predict_dataloader()["articulation"]

            result_path: Path | None = None
            for batch in dataloader:
                progress(45, "Encoding mesh conditioning", "encode-mesh")
                batch = {key: value.to("cuda") if isinstance(value, Tensor) else value for key, value in batch.items()}
                if not params.use_skeleton:
                    batch.pop("skeleton_tokens", None)
                    batch.pop("skeleton_mask", None)
                batch["generate_kwargs"] = {
                    "max_length": 2048,
                    "top_k": params.top_k,
                    "top_p": params.top_p,
                    "temperature": params.temperature,
                    "repetition_penalty": params.repetition_penalty,
                    "num_return_sequences": 1,
                    "num_beams": params.num_beams,
                    "do_sample": True,
                }
                skeleton_tokens = None
                if "skeleton_tokens" in batch and "skeleton_mask" in batch:
                    mask = batch["skeleton_mask"][0] == 1
                    skeleton_tokens = batch["skeleton_tokens"][0][mask].cpu().numpy()

                progress(65, "Generating skeleton and SkinTokens", "generate-tokens")
                preds: list[TokenRigResult] = model.predict_step(
                    batch,
                    skeleton_tokens=[skeleton_tokens] if skeleton_tokens is not None else None,
                    make_asset=True,
                )["results"]
                asset = preds[0].asset
                if asset is None:
                    raise PipelineError("SkinTokens returned no asset.", code="empty-asset", stage="decode-skin")
                progress(75, "Decoding skin weights", "decode-skin")

                if params.use_postprocess:
                    progress(82, "Applying optional voxel skin postprocess", "postprocess")
                    try:
                        import open3d  # noqa: F401
                    except Exception as exc:
                        raise PipelineError(
                            "Voxel Skin Postprocess requires open3d, but open3d is not available in this extension venv. Disable Voxel Skin Postprocess or install a platform-compatible open3d build.",
                            code="open3d-unavailable",
                            stage="postprocess",
                        ) from exc
                    voxel = asset.voxel(resolution=196)
                    asset.skin *= voxel_skin(
                        grid=0,
                        grid_coords=voxel.coords,
                        joints=asset.joints,
                        vertices=asset.vertices,
                        faces=asset.faces,
                        mode="square",
                        voxel_size=voxel.voxel_size,
                    )
                    asset.normalize_skin()
                else:
                    progress(82, "Skipping voxel skin postprocess", "postprocess")

                progress(92, "Exporting rigged GLB", "export-glb")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if params.use_transfer:
                    payload = {"source_asset": asset, "target_path": asset.path, "export_path": str(output_path), "group_per_vertex": 4}
                    response = self._post_bpy_payload("transfer", payload, object_to_bytes, bytes_to_object, requests, tempfile)
                else:
                    payload = {"asset": asset, "filepath": str(output_path), "group_per_vertex": 4}
                    response = self._post_bpy_payload("export", payload, object_to_bytes, bytes_to_object, requests, tempfile)
                if response != "ok":
                    raise PipelineError(f"SkinTokens export failed: {response}", code="export", stage="export-glb")
                result_path = output_path
                break

            if result_path is None:
                raise PipelineError("SkinTokens dataloader produced no batches.", code="empty-input", stage="prepare-mesh")
            return result_path
        finally:
            self._terminate_process(server_proc, log)

    def _start_bpy_server(self, log: LogFn) -> subprocess.Popen:
        layout = resolve_modly_layout(extension_root(), ext_dir=extension_root())
        upstream = upstream_dir_for_layout(layout)
        provider = self._bpy_provider(layout)
        env = {**os.environ, "PYTHONPATH": self._pythonpath_for_bpy(layout, upstream)}
        if provider.get("kind") == "external-blender":
            executable = str(provider.get("executable") or provider.get("probe", {}).get("executable") or "").strip()
            if not executable:
                raise PipelineError("Readiness selected external Blender but did not record an executable.", code="bpy-provider-invalid", stage="bpy-server")
            command = [executable, "--background", "--factory-startup", "--python", str(upstream / "bpy_server.py")]
            log(f"Starting bpy_server with external Blender: {executable}", "bpy-server")
        else:
            command = [sys.executable, str(upstream / "bpy_server.py")]
            log("Starting bpy_server with venv Python bpy provider", "bpy-server")
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(model_root_for_layout(layout)),
            env=env,
        )
        self._relay_pipe(proc.stdout, log, stage="bpy-server", prefix="bpy stdout: ")
        self._relay_pipe(proc.stderr, log, stage="bpy-server", prefix="bpy stderr: ")
        return proc

    def _bpy_provider(self, layout) -> dict:
        state = check_ready_for_layout(layout).state
        summary = state.get("setup_summary") if isinstance(state.get("setup_summary"), dict) else {}
        provider = summary.get("bpy_provider") if isinstance(summary.get("bpy_provider"), dict) else None
        return provider or {"kind": "python-bpy", "status": "assumed"}

    def _pythonpath_for_bpy(self, layout, upstream: Path) -> str:
        paths = [str(upstream)]
        venv_dir = layout.ext_dir / "venv"
        candidates = list((venv_dir / "lib").glob("python*/site-packages")) if (venv_dir / "lib").exists() else []
        candidates.append(venv_dir / "Lib" / "site-packages")
        paths.extend(str(path) for path in candidates if path.exists())
        existing = os.environ.get("PYTHONPATH")
        if existing:
            paths.append(existing)
        return os.pathsep.join(paths)

    def _relay_pipe(self, pipe, log: LogFn, *, stage: str, prefix: str) -> None:
        if pipe is None:
            return

        def worker() -> None:
            for line in pipe:
                text = line.strip()
                if text:
                    log(prefix + text, stage)

        threading.Thread(target=worker, daemon=True).start()

    def _wait_for_bpy_server(self, server_url: str, timeout: int = 30) -> None:
        import requests

        started = time.time()
        while True:
            try:
                requests.get(f"{server_url}/ping", timeout=1)
                return
            except Exception as exc:
                if time.time() - started > timeout:
                    raise PipelineError("bpy_server failed to start within timeout.", code="bpy-timeout", stage="bpy-server") from exc
                time.sleep(0.5)

    def _post_bpy_payload(self, endpoint: str, payload, object_to_bytes, bytes_to_object, requests_module, tempfile_module):
        from src.server.spec import BPY_SERVER

        payload_path = None
        try:
            with tempfile_module.NamedTemporaryFile(prefix=f"skintokens_{endpoint}_", suffix=".pt", delete=False) as handle:
                handle.write(object_to_bytes(payload))
                payload_path = handle.name
            response = requests_module.post(f"{BPY_SERVER}/{endpoint}", data=object_to_bytes({"payload_path": payload_path}))
            response.raise_for_status()
            result = bytes_to_object(response.content)
            if isinstance(result, dict) and result.get("error") is not None:
                raise PipelineError(str(result.get("traceback") or result["error"]), code="bpy-error", stage="export-glb")
            return result
        finally:
            if payload_path:
                try:
                    os.remove(payload_path)
                except OSError:
                    pass

    def _terminate_process(self, proc: subprocess.Popen, log: LogFn) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            log("Forced bpy_server shutdown after timeout", "bpy-server")
