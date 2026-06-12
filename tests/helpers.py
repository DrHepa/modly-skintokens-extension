from __future__ import annotations

import json
import struct
from pathlib import Path


def write_minimal_glb(path: Path, *, with_skin: bool = True) -> Path:
    payload = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 1}, "indices": 0}]}],
        "buffers": [{"byteLength": 44}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 6, "target": 34963},
            {"buffer": 0, "byteOffset": 8, "byteLength": 36, "target": 34962},
        ],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
        ],
    }
    if with_skin:
        payload["skins"] = [{"joints": [0]}]
        payload["nodes"][0]["skin"] = 0
    json_chunk = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    while len(json_chunk) % 4:
        json_chunk += b" "
    binary = struct.pack("<3H", 0, 1, 2) + b"\x00\x00" + struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    blob = bytearray(b"glTF")
    blob += struct.pack("<I", 2)
    blob += struct.pack("<I", 12 + 8 + len(json_chunk) + 8 + len(binary))
    blob += struct.pack("<I", len(json_chunk))
    blob += b"JSON"
    blob += json_chunk
    blob += struct.pack("<I", len(binary))
    blob += b"BIN\x00"
    blob += binary
    path.write_bytes(blob)
    return path
