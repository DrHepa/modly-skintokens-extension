from __future__ import annotations

import json
import subprocess
import sys
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from skintokens_ext.events import EventWriter
from skintokens_ext.paths import ModlyLayout
from skintokens_ext.setup_runtime import BPY_PROBE_MARKER, CommandResult, _build_flash_attn_wheel, _sync_runtime_resources, run_setup


ROOT = Path(__file__).resolve().parents[1]


class SetupTests(unittest.TestCase):
    def test_setup_dry_run_emits_events(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "setup.py"), "--dry-run"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        events = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertTrue(any(event.get("stage") == "install-torch-cu128" for event in events))
        self.assertEqual(events[-1]["type"], "setup_done")
        self.assertEqual(events[-1]["status"], "dry_run")
        state_path = Path(events[-1]["statePath"])
        self.assertTrue(state_path.exists())
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "dry_run")
        self.assertFalse(state["downloads_started"])
        self.assertFalse(state["installs_started"])

    def test_setup_invalid_payload_emits_json_error(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "setup.py"), "{not-json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        events = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual(events[0]["type"], "error")
        self.assertEqual(events[-1]["type"], "setup_done")

    def test_real_setup_payload_uses_modly_models_root(self) -> None:
        with TemporaryDirectory() as tmp:
            modly_home = Path(tmp) / "Modly"
            ext_dir = modly_home / "extensions" / "skintokens-process-extension"
            ext_dir.mkdir(parents=True)
            stream = StringIO()

            with mock.patch("skintokens_ext.setup_runtime._create_venv", return_value={"status": "created", "venv_python": str(ext_dir / "venv" / "bin" / "python")}), \
                mock.patch("skintokens_ext.setup_runtime._download_upstream_source", return_value={"status": "downloaded"}), \
                mock.patch("skintokens_ext.setup_runtime._sync_runtime_resources", return_value={"status": "synced", "resources": []}), \
                mock.patch("skintokens_ext.setup_runtime._verify_model_assets", return_value={"tokenrig-grpo": {"status": "ok"}, "skintokens-vae": {"status": "ok"}, "qwen3-config": {"status": "ok"}}):
                code = run_setup(
                    dry_run=False,
                    payload={"ext_dir": str(ext_dir), "python_exe": sys.executable},
                    skip_install=True,
                    skip_download=True,
                    writer=EventWriter(stream),
                )

            self.assertEqual(code, 0, stream.getvalue())
            events = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(events[-1]["type"], "setup_done")
            self.assertEqual(events[-1]["status"], "ready")
            state_path = Path(events[-1]["statePath"])
            self.assertTrue(state_path.is_relative_to(ext_dir))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "ready")
            self.assertEqual(Path(state["model_root"]), (modly_home / "models" / "skintokens-process-extension" / "tokenrig").resolve())
            self.assertEqual(state["logical_model_root"], "models/skintokens-process-extension/tokenrig")

    def test_failed_install_command_emits_stderr_tail_log(self) -> None:
        class FailingRunner:
            def run(self, args, *, cwd=None, env=None):
                return CommandResult(args=list(args), returncode=1, stdout_tail="", stderr_tail="flash-attn build failed", ok=False)

        with TemporaryDirectory() as tmp:
            modly_home = Path(tmp) / "Modly"
            ext_dir = modly_home / "extensions" / "skintokens-process-extension"
            ext_dir.mkdir(parents=True)
            stream = StringIO()
            with mock.patch("skintokens_ext.setup_runtime._create_venv", return_value={"status": "created", "venv_python": str(ext_dir / "venv" / "bin" / "python")}):
                code = run_setup(
                    dry_run=False,
                    payload={"ext_dir": str(ext_dir), "python_exe": sys.executable},
                    writer=EventWriter(stream),
                    runner=FailingRunner(),
                )

            self.assertEqual(code, 1)
            events = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertTrue(any(event.get("type") == "log" and "flash-attn build failed" in event.get("message", "") for event in events))
            self.assertTrue(any(event.get("type") == "error" and event.get("stage") == "upgrade-pip" for event in events))

    def test_bpy_pip_failure_falls_back_to_blender_provider(self) -> None:
        class BpyFallbackRunner:
            def run(self, args, *, cwd=None, env=None):
                args = list(args)
                if "bpy>=4.2" in args:
                    return CommandResult(args=args, returncode=1, stdout_tail="", stderr_tail="No matching distribution found for bpy>=4.2", ok=False)
                if args and str(args[0]) == "/fake/blender":
                    payload = '{"provider":"external-blender","status":"ok","blender_version":"4.0.2","python_version":"3.12.3"}'
                    return CommandResult(args=args, returncode=0, stdout_tail=BPY_PROBE_MARKER + payload + "\n", stderr_tail="", ok=True)
                if "-c" in args:
                    probes = {
                        "torch": {"status": "ok"},
                        "flash_attn": {"status": "ok"},
                        "open3d": {"status": "ok"},
                        "fast_simplification": {"status": "ok"},
                        "torch_cuda": {"status": "ok"},
                    }
                    return CommandResult(args=args, returncode=0, stdout_tail=json.dumps(probes) + "\n", stderr_tail="", ok=True)
                return CommandResult(args=args, returncode=0, stdout_tail="ok\n", stderr_tail="", ok=True)

        with TemporaryDirectory() as tmp:
            modly_home = Path(tmp) / "Modly"
            ext_dir = modly_home / "extensions" / "skintokens-process-extension"
            ext_dir.mkdir(parents=True)
            stream = StringIO()
            with mock.patch("skintokens_ext.setup_runtime._create_venv", return_value={"status": "created", "venv_python": str(ext_dir / "venv" / "bin" / "python")}), \
                mock.patch("skintokens_ext.setup_runtime._download_upstream_source", return_value={"status": "downloaded"}), \
                mock.patch("skintokens_ext.setup_runtime._sync_runtime_resources", return_value={"status": "synced", "resources": []}), \
                mock.patch("skintokens_ext.setup_runtime._verify_model_assets", return_value={"tokenrig-grpo": {"status": "ok"}, "skintokens-vae": {"status": "ok"}, "qwen3-config": {"status": "ok"}}):
                code = run_setup(
                    dry_run=False,
                    payload={"ext_dir": str(ext_dir), "python_exe": sys.executable, "blender_exe": "/fake/blender"},
                    skip_download=True,
                    writer=EventWriter(stream),
                    runner=BpyFallbackRunner(),
                )

            self.assertEqual(code, 0, stream.getvalue())
            events = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertTrue(any("pip install bpy failed" in event.get("message", "") for event in events))
            self.assertTrue(any("Using external Blender fallback" in event.get("message", "") for event in events))
            state_path = Path(events[-1]["statePath"])
            state = json.loads(state_path.read_text(encoding="utf-8"))
            provider = state["setup_summary"]["bpy_provider"]
            self.assertEqual(provider["kind"], "external-blender")
            self.assertEqual(provider["status"], "ok")

    def test_setup_installs_flash_attn_from_local_wheelhouse_when_present(self) -> None:
        class RecordingRunner:
            def __init__(self):
                self.commands = []

            def run(self, args, *, cwd=None, env=None):
                args = list(args)
                self.commands.append(args)
                if "-c" in args:
                    probes = {
                        "torch": {"status": "ok"},
                        "flash_attn": {"status": "ok"},
                        "fast_simplification": {"status": "ok"},
                        "torch_cuda": {"status": "ok"},
                    }
                    return CommandResult(args=args, returncode=0, stdout_tail=json.dumps(probes) + "\n", stderr_tail="", ok=True)
                return CommandResult(args=args, returncode=0, stdout_tail="ok\n", stderr_tail="", ok=True)

        with TemporaryDirectory() as tmp:
            modly_home = Path(tmp) / "Modly"
            ext_dir = modly_home / "extensions" / "skintokens-process-extension"
            ext_dir.mkdir(parents=True)
            wheelhouse = ext_dir / ".skintokens-runtime" / "wheelhouse" / "flash-attn"
            wheelhouse.mkdir(parents=True)
            (wheelhouse / "flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl").write_bytes(b"fake-wheel")
            runner = RecordingRunner()
            stream = StringIO()
            with mock.patch("skintokens_ext.setup_runtime._create_venv", return_value={"status": "created", "venv_python": str(ext_dir / "venv" / "bin" / "python")}), \
                mock.patch("skintokens_ext.setup_runtime._resolve_bpy_provider", return_value={"kind": "python-bpy", "status": "ok"}), \
                mock.patch("skintokens_ext.setup_runtime._download_upstream_source", return_value={"status": "downloaded"}), \
                mock.patch("skintokens_ext.setup_runtime._sync_runtime_resources", return_value={"status": "synced", "resources": []}), \
                mock.patch("skintokens_ext.setup_runtime._verify_model_assets", return_value={"tokenrig-grpo": {"status": "ok"}, "skintokens-vae": {"status": "ok"}, "qwen3-config": {"status": "ok"}}):
                code = run_setup(
                    dry_run=False,
                    payload={"ext_dir": str(ext_dir), "python_exe": sys.executable},
                    skip_download=True,
                    writer=EventWriter(stream),
                    runner=runner,
                )

            self.assertEqual(code, 0, stream.getvalue())
            flat_commands = [" ".join(command) for command in runner.commands]
            self.assertTrue(any("--no-index" in command and "--find-links" in command and "flash-attn" in command for command in flat_commands))
            self.assertFalse(any(" pip wheel " in f" {command} " for command in flat_commands))

    def test_setup_does_not_source_build_flash_attn_by_default(self) -> None:
        class NoWheelRunner:
            def run(self, args, *, cwd=None, env=None):
                args = list(args)
                if "--only-binary" in args and "flash-attn" in args:
                    return CommandResult(args=args, returncode=1, stdout_tail="", stderr_tail="No matching distribution found for flash-attn", ok=False)
                return CommandResult(args=args, returncode=0, stdout_tail="ok\n", stderr_tail="", ok=True)

        with TemporaryDirectory() as tmp:
            modly_home = Path(tmp) / "Modly"
            ext_dir = modly_home / "extensions" / "skintokens-process-extension"
            ext_dir.mkdir(parents=True)
            stream = StringIO()
            with mock.patch("skintokens_ext.setup_runtime._create_venv", return_value={"status": "created", "venv_python": str(ext_dir / "venv" / "bin" / "python")}):
                code = run_setup(
                    dry_run=False,
                    payload={"ext_dir": str(ext_dir), "python_exe": sys.executable},
                    writer=EventWriter(stream),
                    runner=NoWheelRunner(),
                )

            self.assertEqual(code, 1)
            events = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertTrue(any(event.get("type") == "error" and event.get("code") == "flash-attn-wheel-unavailable" for event in events))
            self.assertEqual(events[-1]["failure_code"], "flash-attn-wheel-unavailable")

    def test_build_flash_attn_wheel_command_writes_local_wheel(self) -> None:
        class WheelBuildRunner:
            def __init__(self, wheelhouse: Path):
                self.wheelhouse = wheelhouse
                self.commands = []

            def run(self, args, *, cwd=None, env=None):
                args = list(args)
                self.commands.append({"args": args, "env": env or {}})
                if "wheel" in args and "--wheel-dir" in args:
                    self.wheelhouse.mkdir(parents=True, exist_ok=True)
                    (self.wheelhouse / "flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl").write_bytes(b"fake-wheel")
                return CommandResult(args=args, returncode=0, stdout_tail="ok\n", stderr_tail="", ok=True)

        with TemporaryDirectory() as tmp:
            modly_home = Path(tmp) / "Modly"
            ext_dir = modly_home / "extensions" / "skintokens-process-extension"
            ext_dir.mkdir(parents=True)
            wheelhouse = ext_dir / ".skintokens-runtime" / "wheelhouse" / "flash-attn"
            stream = StringIO()
            runner = WheelBuildRunner(wheelhouse)
            with mock.patch("skintokens_ext.setup_runtime._create_venv", return_value={"status": "created", "venv_python": str(ext_dir / "venv" / "bin" / "python")}), \
                mock.patch("skintokens_ext.setup_runtime._cuda_build_env", return_value={"CUDA_HOME": "/usr/local/cuda-12.8"}):
                code = _build_flash_attn_wheel(
                    layout=mock.Mock(ext_dir=ext_dir),
                    base_python=sys.executable,
                    runner=runner,
                    writer=EventWriter(stream),
                    max_build_jobs="2",
                )

            self.assertEqual(code, 0, stream.getvalue())
            self.assertTrue((wheelhouse / "flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl").exists())
            self.assertTrue(any(command["env"].get("MAX_JOBS") == "2" for command in runner.commands))
            events = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(events[-1]["status"], "flash_attn_wheel_ready")

    def test_sync_runtime_resources_copies_configs_to_model_root(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp) / "Modly"
            ext_dir = home / "extensions" / "skintokens-process-extension"
            source_config = ext_dir / ".skintokens-runtime" / "vendor" / "skintokens" / "configs" / "skeleton"
            source_config.mkdir(parents=True)
            (source_config / "vroid.yaml").write_text("parts_order: []\n", encoding="utf-8")
            layout = ModlyLayout(modly_home=home.resolve(), ext_dir=ext_dir.resolve(), models_root=(home / "models").resolve())

            result = _sync_runtime_resources(layout)

            target = home / "models" / "skintokens-process-extension" / "tokenrig" / "configs" / "skeleton" / "vroid.yaml"
            self.assertEqual(result["status"], "synced")
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "parts_order: []\n")


if __name__ == "__main__":
    unittest.main()
