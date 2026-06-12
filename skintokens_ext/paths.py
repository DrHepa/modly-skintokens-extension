from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
DRIVE_RE = re.compile(r"^[A-Za-z]:")
SAFE_MESH_SUFFIXES = {".obj", ".fbx", ".glb", ".gltf"}
EXTENSION_ID = "skintokens-process-extension"
MODELS_PREFIX = "models"


class PathSafetyError(ValueError):
    pass


@dataclass(frozen=True)
class ModlyLayout:
    modly_home: Path
    ext_dir: Path
    models_root: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "modly_home": str(self.modly_home),
            "ext_dir": str(self.ext_dir),
            "models_root": str(self.models_root),
        }


def extension_root() -> Path:
    return Path(__file__).resolve().parents[1]


def runtime_root(root: Path | None = None) -> Path:
    return (root or extension_root()) / ".skintokens-runtime"


def resolve_modly_layout(
    workspace_root: str | Path | None = None,
    *,
    ext_dir: str | Path | None = None,
    modly_home: str | Path | None = None,
    models_root: str | Path | None = None,
) -> ModlyLayout:
    resolved_ext_dir = Path(ext_dir or workspace_root or extension_root()).expanduser().resolve()
    if modly_home is not None:
        resolved_home = Path(modly_home).expanduser().resolve()
    elif resolved_ext_dir.name == EXTENSION_ID and resolved_ext_dir.parent.name == "extensions":
        resolved_home = resolved_ext_dir.parent.parent.resolve()
    else:
        resolved_home = resolved_ext_dir
    resolved_models = Path(models_root).expanduser().resolve() if models_root is not None else (resolved_home / MODELS_PREFIX).resolve()
    return ModlyLayout(modly_home=resolved_home, ext_dir=resolved_ext_dir, models_root=resolved_models)


def validate_logical_path(value: str | os.PathLike[str]) -> str:
    """Validate extension-owned logical paths such as model sentinels.

    User-provided mesh paths are validated separately; this function is for paths
    owned by the extension contract and must never accept host absolute paths.
    """

    raw = os.fspath(value).strip()
    if not raw:
        raise PathSafetyError("logical path must not be empty")
    normalized = raw.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("//") or DRIVE_RE.match(normalized):
        raise PathSafetyError("logical path must be relative and must not contain a drive letter")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        raise PathSafetyError("logical path must contain at least one segment")
    for part in parts:
        if part in {".", ".."} or ".." in part:
            raise PathSafetyError("logical path must not contain traversal")
        if part.startswith(".") or part.startswith("~"):
            raise PathSafetyError("logical path must not contain hidden or backup-prefixed segments")
        base = part.split(".", 1)[0].upper()
        if base in WINDOWS_RESERVED:
            raise PathSafetyError(f"logical path contains Windows reserved segment: {part}")
    return "/".join(parts)


def resolve_runtime_logical(root: Path, logical_path: str) -> Path:
    safe = validate_logical_path(logical_path)
    resolved = (root / safe).resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise PathSafetyError("resolved path escapes runtime root") from exc
    return resolved


def resolve_storage_path(layout: ModlyLayout, logical_path: str) -> Path:
    safe = validate_logical_path(logical_path)
    parts = safe.split("/")
    if parts and parts[0] == MODELS_PREFIX:
        return (layout.modly_home / Path(*parts)).resolve()
    return (layout.ext_dir / Path(*parts)).resolve()


def validate_mesh_input(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if not candidate.exists():
        raise PathSafetyError(f"Input mesh does not exist: {path}")
    if not candidate.is_file():
        raise PathSafetyError(f"Input mesh must be a file: {path}")
    if candidate.suffix.lower() not in SAFE_MESH_SUFFIXES:
        raise PathSafetyError("Input mesh must be .obj, .fbx, .glb, or .gltf")
    return candidate


def create_run_dir(root: Path, run_id: str | None = None) -> Path:
    import time

    token = re.sub(r"[^A-Za-z0-9_.-]", "-", (run_id or "").strip())[:80]
    if not token:
        token = str(int(time.time() * 1000))
    run_dir = root / "runs" / token
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
