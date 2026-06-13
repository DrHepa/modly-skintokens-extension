from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_manifest_process_contract(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["id"], "skintokens-process-extension")
        self.assertEqual(manifest["type"], "process")
        self.assertEqual(manifest["entry"], "processor.py")
        metadata = manifest["metadata"]
        for key in ["resolution", "implementation_profile", "setup_contract", "support_state", "surface_owner", "headless_eligible", "linux_arm64_risk"]:
            self.assertIn(key, metadata)
        self.assertEqual(metadata["logical_model_root"], "models/skintokens-process-extension/tokenrig")
        node = manifest["nodes"][0]
        self.assertEqual(node["id"], "rig-mesh")
        self.assertEqual(node["input"], "mesh")
        self.assertEqual(node["output"], "mesh")
        self.assertNotIn("hf_repo", node)
        self.assertNotIn("download_check", node)
        self.assertNotIn("weight_owner_id", node)

    def test_boolean_params_are_selects(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        params = {item["id"]: item for item in manifest["nodes"][0]["params_schema"]}
        for key in ["use_skeleton", "use_transfer"]:
            self.assertEqual(params[key]["type"], "select")
            self.assertIn(params[key]["default"], {"true", "false"})
        self.assertNotIn("use_postprocess", params)

    def test_asset_requirements_use_modly_models_root(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["setup"]["entrypoint"], "setup.py")
        self.assertTrue(manifest["setup"]["default_downloads"])
        self.assertTrue(manifest["setup"]["default_installs"])
        self.assertEqual(manifest["assets"]["model_roots"], ["models/skintokens-process-extension/tokenrig"])
        for asset in manifest["asset_requirements"]["model_assets"]:
            self.assertTrue(asset["path"].startswith("models/skintokens-process-extension/tokenrig/"))
        for required in manifest["asset_requirements"]["required"]:
            self.assertTrue(required.startswith("models/skintokens-process-extension/tokenrig/"))


if __name__ == "__main__":
    unittest.main()
