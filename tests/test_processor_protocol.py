from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.helpers import write_minimal_glb


ROOT = Path(__file__).resolve().parents[1]
PROCESSOR = ROOT / "processor.py"


class ProcessorProtocolTests(unittest.TestCase):
    def _run_processor(self, payload: object, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        run_env = os.environ.copy()
        run_env["PYTHONPATH"] = str(ROOT)
        run_env["MODLY_SKINTOKENS_FAKE_RUNTIME"] = "1"
        if env:
            run_env.update(env)
        return subprocess.run(
            [sys.executable, str(PROCESSOR)],
            input=json.dumps(payload) + "\n",
            text=True,
            capture_output=True,
            env=run_env,
            check=False,
        )

    def test_success_emits_stages_and_done(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mesh = write_minimal_glb(tmp_path / "input.glb")
            payload = {"input": {"nodeId": "rig-mesh", "filePath": str(mesh)}, "params": {"use_transfer": "true"}, "workspaceDir": str(tmp_path)}
            proc = self._run_processor(payload)
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            events = [json.loads(line) for line in proc.stdout.splitlines()]
            self.assertFalse(proc.stderr.strip())
            self.assertEqual(events[-1]["type"], "done")
            stages = [event.get("stage") for event in events if event.get("type") == "progress"]
            for stage in ["validate-input", "bpy-server", "load-model", "generate-tokens", "decode-skin", "export-glb", "validate-output"]:
                self.assertIn(stage, stages)

    def test_protocol_error_is_json_error_only(self) -> None:
        proc = self._run_processor({"input": {"nodeId": "rig-mesh"}, "params": {}})
        self.assertNotEqual(proc.returncode, 0)
        events = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertEqual(events[0]["code"], "protocol")


if __name__ == "__main__":
    unittest.main()
