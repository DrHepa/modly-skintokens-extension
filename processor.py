from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skintokens_ext.events import EventWriter
from skintokens_ext.runtime import PipelineError, run_pipeline


class ProtocolError(ValueError):
    pass


def _read_payload() -> dict:
    raw = sys.stdin.readline()
    if not raw:
        raise ProtocolError("Processor expected one JSON line on stdin.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Processor stdin is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Processor payload must be a JSON object.")
    return payload


def _object(payload: dict, key: str) -> dict:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProtocolError(f"'{key}' must be a JSON object.")
    return value


def _workspace_dir(payload: dict) -> Path | None:
    raw = payload.get("workspaceDir")
    if raw in (None, ""):
        return None
    if not isinstance(raw, str):
        raise ProtocolError("'workspaceDir' must be a string when provided.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ProtocolError("'workspaceDir' must be an absolute path when provided.")
    if path.exists() and not path.is_dir():
        raise ProtocolError("'workspaceDir' must point to a directory when provided.")
    return path.resolve() if path.exists() else None


def main() -> int:
    writer = EventWriter()
    try:
        payload = _read_payload()
        input_payload = _object(payload, "input")
        params = _object(payload, "params")
        node_id = input_payload.get("nodeId") or payload.get("nodeId") or "rig-mesh"
        if node_id != "rig-mesh":
            raise ProtocolError(f"Unsupported nodeId '{node_id}'. This extension exposes only 'rig-mesh'.")
        file_path = input_payload.get("filePath")
        if not isinstance(file_path, str) or not file_path.strip():
            raise ProtocolError("rig-mesh requires input.filePath.")

        result = run_pipeline(
            mesh_path=Path(file_path),
            params=params,
            progress=writer.progress,
            log=writer.log,
            workspace_dir=_workspace_dir(payload),
            run_id=payload.get("runId") if isinstance(payload.get("runId"), str) else None,
        )
        writer.done(str(result.file_path), {"validation": result.validation.details, "warnings": result.validation.warnings})
        return 0
    except ProtocolError as exc:
        writer.error(str(exc), code="protocol")
        return 1
    except PipelineError as exc:
        writer.error(str(exc), code=exc.code, stage=exc.stage)
        return 1
    except Exception as exc:  # pragma: no cover - safety net for Modly UI
        writer.error(str(exc), code="unexpected")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
