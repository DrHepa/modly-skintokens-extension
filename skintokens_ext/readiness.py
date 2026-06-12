from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import EXTENSION_ID, ModlyLayout, resolve_modly_layout, resolve_storage_path, runtime_root


HF_REPO = "VAST-AI/SkinTokens"
QWEN_REPO = "Qwen/Qwen3-0.6B"
STATE_RELATIVE_PATH = "bootstrap_state.json"
UPSTREAM_DIR_RELATIVE = "vendor/skintokens"
MODEL_OWNER_ID = "tokenrig"
MODEL_LOGICAL_ROOT = f"models/{EXTENSION_ID}/{MODEL_OWNER_ID}"

MODEL_SENTINELS = [
    {
        "id": "tokenrig-grpo",
        "repo": HF_REPO,
        "path": "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
        "logical_path": f"{MODEL_LOGICAL_ROOT}/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
        "hf_path": "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
        "size_bytes": 1131603979,
        "sha256": "f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692",
    },
    {
        "id": "skintokens-vae",
        "repo": HF_REPO,
        "path": "experiments/skin_vae_2_10_32768/last.ckpt",
        "logical_path": f"{MODEL_LOGICAL_ROOT}/experiments/skin_vae_2_10_32768/last.ckpt",
        "hf_path": "experiments/skin_vae_2_10_32768/last.ckpt",
        "size_bytes": 487311745,
        "sha256": "4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4",
    },
    {
        "id": "qwen3-config",
        "repo": QWEN_REPO,
        "path": "models/Qwen3-0.6B/config.json",
        "logical_path": f"{MODEL_LOGICAL_ROOT}/models/Qwen3-0.6B/config.json",
        "config_only": True,
    },
]


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    status: str
    state: dict[str, Any]
    missing: list[str]
    path: Path

    def public_message(self) -> str:
        if self.ready:
            return "SkinTokens runtime is ready."
        if self.status == "missing":
            return "SkinTokens setup has not completed yet. Run extension setup first."
        if self.status == "dry_run":
            return "SkinTokens setup is only dry-run planned. Run real setup before generation."
        if self.missing:
            return "SkinTokens runtime is missing required assets: " + ", ".join(self.missing)
        return f"SkinTokens runtime is not ready (status={self.status})."


def state_path(root: Path | None = None) -> Path:
    return runtime_root(root) / STATE_RELATIVE_PATH


def state_path_for_layout(layout: ModlyLayout) -> Path:
    return runtime_root(layout.ext_dir) / STATE_RELATIVE_PATH


def upstream_dir(root: Path | None = None) -> Path:
    return runtime_root(root) / UPSTREAM_DIR_RELATIVE


def upstream_dir_for_layout(layout: ModlyLayout) -> Path:
    return runtime_root(layout.ext_dir) / UPSTREAM_DIR_RELATIVE


def model_root_for_layout(layout: ModlyLayout) -> Path:
    return resolve_storage_path(layout, MODEL_LOGICAL_ROOT)


def read_state(root: Path | None = None) -> dict[str, Any]:
    path = state_path(root)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "corrupt", "failure_code": "state-json-invalid"}
    return raw if isinstance(raw, dict) else {"status": "corrupt", "failure_code": "state-not-object"}


def read_state_for_layout(layout: ModlyLayout) -> dict[str, Any]:
    path = state_path_for_layout(layout)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "corrupt", "failure_code": "state-json-invalid"}
    return raw if isinstance(raw, dict) else {"status": "corrupt", "failure_code": "state-not-object"}


def write_state(state: dict[str, Any], root: Path | None = None) -> Path:
    path = state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_state_for_layout(state: dict[str, Any], layout: ModlyLayout) -> Path:
    path = state_path_for_layout(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return path


def sentinel_paths(root: Path | None = None) -> list[Path]:
    layout = resolve_modly_layout(root or Path.cwd())
    return sentinel_paths_for_layout(layout)


def sentinel_paths_for_layout(layout: ModlyLayout) -> list[Path]:
    root = model_root_for_layout(layout)
    return [root / str(item["path"]) for item in MODEL_SENTINELS]


def check_ready(root: Path | None = None) -> ReadinessResult:
    layout = resolve_modly_layout(root or Path.cwd())
    return check_ready_for_layout(layout)


def check_ready_for_layout(layout: ModlyLayout) -> ReadinessResult:
    state = read_state_for_layout(layout)
    path = state_path_for_layout(layout)
    if not state:
        return ReadinessResult(False, "missing", {}, [str(p) for p in sentinel_paths_for_layout(layout)], path)
    status = str(state.get("status") or "unknown")
    missing = [str(path) for path in sentinel_paths_for_layout(layout) if not path.exists()]
    ready = status == "ready" and not missing
    return ReadinessResult(ready, status, state, missing, path)


def planned_state(*, status: str, dry_run: bool, layout: ModlyLayout | None = None) -> dict[str, Any]:
    layout = layout or resolve_modly_layout(Path.cwd())
    return {
        "status": status,
        "dry_run": dry_run,
        "failure_code": None if dry_run else "real-setup-not-executed-by-this-pass",
        "downloads_started": False,
        "installs_started": False,
        "next_steps": [
            "Validate target OS/Python/CUDA lane.",
            "Provision exact torch/cu128 runtime.",
            "Install and probe flash-attn, bpy, open3d, and fast-simplification.",
            "Download SkinTokens checkpoints and Qwen3 config into the runtime vendor tree.",
            "Run a first SkinTokens generation smoke test before marking any platform supported.",
        ],
        "model_sentinels": MODEL_SENTINELS,
        "logical_model_root": MODEL_LOGICAL_ROOT,
        "model_root": str(model_root_for_layout(layout)),
        "upstream_dir": str(upstream_dir_for_layout(layout)),
    }
