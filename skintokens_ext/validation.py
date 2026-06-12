from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OutputValidation:
    ok: bool
    format: str
    warnings: list[str]
    details: dict[str, Any]


class OutputValidationError(RuntimeError):
    pass


def validate_output(path: Path) -> OutputValidation:
    candidate = path.expanduser().resolve()
    if not candidate.exists():
        raise OutputValidationError(f"Generated output does not exist: {path}")
    if not candidate.is_file():
        raise OutputValidationError(f"Generated output is not a file: {path}")
    if candidate.stat().st_size <= 0:
        raise OutputValidationError(f"Generated output is empty: {path}")

    suffix = candidate.suffix.lower()
    if suffix == ".glb":
        return _validate_glb(candidate)
    if suffix == ".gltf":
        return _validate_gltf(candidate)
    raise OutputValidationError("Generated output must be .glb or .gltf")


def _validate_glb(path: Path) -> OutputValidation:
    data = path.read_bytes()
    if len(data) < 20:
        raise OutputValidationError("GLB is too small to contain a valid header and JSON chunk")
    if data[:4] != b"glTF":
        raise OutputValidationError("GLB header magic is not glTF")
    version, total_length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise OutputValidationError(f"Unsupported GLB version: {version}")
    if total_length > len(data):
        raise OutputValidationError("GLB declared length exceeds file size")
    json_payload: dict[str, Any] = {}
    offset = 12
    while offset + 8 <= len(data):
        chunk_length = struct.unpack_from("<I", data, offset)[0]
        chunk_type = data[offset + 4 : offset + 8]
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == b"JSON":
            json_payload = json.loads(chunk.decode("utf-8").rstrip(" \t\r\n\x00"))
            break
    return _summarize_gltf_json(json_payload, fmt="glb", size=path.stat().st_size)


def _validate_gltf(path: Path) -> OutputValidation:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OutputValidationError("glTF root must be an object")
    return _summarize_gltf_json(payload, fmt="gltf", size=path.stat().st_size)


def _summarize_gltf_json(payload: dict[str, Any], *, fmt: str, size: int) -> OutputValidation:
    if not payload.get("asset"):
        raise OutputValidationError("glTF asset metadata is missing")
    warnings: list[str] = []
    meshes = payload.get("meshes") or []
    skins = payload.get("skins") or []
    nodes = payload.get("nodes") or []
    if not meshes:
        warnings.append("generated glTF has no meshes")
    if not skins:
        warnings.append("generated glTF has no skins; rig evidence is not present in the JSON chunk")
    joint_count = 0
    for skin in skins:
        if isinstance(skin, dict) and isinstance(skin.get("joints"), list):
            joint_count += len(skin["joints"])
    if skins and joint_count == 0:
        warnings.append("generated glTF declares skins but no joints")
    return OutputValidation(
        ok=True,
        format=fmt,
        warnings=warnings,
        details={
            "size_bytes": size,
            "mesh_count": len(meshes) if isinstance(meshes, list) else 0,
            "skin_count": len(skins) if isinstance(skins, list) else 0,
            "node_count": len(nodes) if isinstance(nodes, list) else 0,
            "joint_count": joint_count,
        },
    )
