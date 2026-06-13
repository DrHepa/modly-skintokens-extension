from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from skintokens_ext.runtime import FakeBackend, PipelineError, _postprocess_dependency_guard, _runtime_gpu_requirement_guard, normalize_params, run_pipeline
from tests.helpers import write_minimal_glb


class RuntimeTests(unittest.TestCase):
    def test_normalize_select_booleans(self) -> None:
        params = normalize_params({"use_skeleton": "true", "use_transfer": "false", "use_postprocess": "1"})
        self.assertTrue(params.use_skeleton)
        self.assertFalse(params.use_transfer)
        self.assertTrue(params.use_postprocess)

    def test_fake_backend_stage_order(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mesh = write_minimal_glb(tmp_path / "mesh.glb")
            progress_events = []
            logs = []
            result = run_pipeline(
                mesh_path=mesh,
                params={"use_transfer": "true"},
                progress=lambda percent, label, stage=None: progress_events.append((percent, stage, label)),
                log=lambda message, stage=None: logs.append((stage, message)),
                workspace_dir=tmp_path,
                backend=FakeBackend(),
            )
            self.assertTrue(result.file_path.exists())
            self.assertEqual(result.file_path.parent, tmp_path / "Workflows")
            stages = [stage for _, stage, _ in progress_events]
            self.assertEqual(stages[0], "validate-input")
            self.assertIn("generate-tokens", stages)
            self.assertEqual(stages[-1], "validate-output")

    def test_workspace_workflows_dir_is_not_nested(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workflows_dir = tmp_path / "Workflows"
            workflows_dir.mkdir()
            mesh = write_minimal_glb(tmp_path / "mesh.glb")
            result = run_pipeline(
                mesh_path=mesh,
                params={},
                progress=lambda percent, label, stage=None: None,
                log=lambda message, stage=None: None,
                workspace_dir=workflows_dir,
                backend=FakeBackend(),
            )
            self.assertEqual(result.file_path.parent, workflows_dir)

    def test_runtime_gpu_guard_rejects_pre_ampere(self) -> None:
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_capability=lambda: (7, 5),
                get_device_name=lambda: "RTX 20xx",
            )
        )
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            with self.assertRaises(PipelineError) as raised:
                _runtime_gpu_requirement_guard()
        self.assertEqual(raised.exception.code, "gpu-too-old")

    def test_runtime_gpu_guard_allows_ampere(self) -> None:
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_capability=lambda: (8, 0),
                get_device_name=lambda: "RTX 30xx",
            )
        )
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            _runtime_gpu_requirement_guard()

    def test_postprocess_guard_rejects_missing_open3d(self) -> None:
        with mock.patch("importlib.import_module", side_effect=ModuleNotFoundError("No module named open3d")):
            with self.assertRaises(PipelineError) as raised:
                _postprocess_dependency_guard()
        self.assertEqual(raised.exception.code, "open3d-unavailable")
        self.assertEqual(raised.exception.stage, "postprocess")


if __name__ == "__main__":
    unittest.main()
