from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import IO, Any


@dataclass
class EventWriter:
    """Write Modly process/setup events as JSONL.

    Keep stdout protocol clean: one JSON object per line, flushed immediately.
    """

    stream: IO[str] | None = None

    def __post_init__(self) -> None:
        if self.stream is None:
            self.stream = sys.stdout

    def send(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise TypeError("event payload must be a dict")
        assert self.stream is not None
        self.stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self.stream.flush()

    def progress(self, percent: int, label: str, stage: str | None = None) -> None:
        event: dict[str, Any] = {"type": "progress", "percent": int(percent), "label": str(label)}
        if stage:
            event["stage"] = str(stage)
        self.send(event)

    def log(self, message: str, stage: str | None = None) -> None:
        event: dict[str, Any] = {"type": "log", "message": str(message)}
        if stage:
            event["stage"] = str(stage)
        self.send(event)

    def error(self, message: str, code: str | None = None, stage: str | None = None) -> None:
        event: dict[str, Any] = {"type": "error", "message": str(message)}
        if code:
            event["code"] = str(code)
        if stage:
            event["stage"] = str(stage)
        self.send(event)

    def done(self, file_path: str, extra: dict[str, Any] | None = None) -> None:
        result: dict[str, Any] = {"filePath": str(file_path)}
        if extra:
            result.update(extra)
        self.send({"type": "done", "result": result})


class LineRelay:
    """File-like object that forwards third-party text as JSON log events."""

    def __init__(self, emit_log, *, stage: str, prefix: str = "") -> None:
        self._emit_log = emit_log
        self._stage = stage
        self._prefix = prefix
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""

    def _emit(self, line: str) -> None:
        line = line.strip()
        if line:
            self._emit_log(f"{self._prefix}{line}", self._stage)
