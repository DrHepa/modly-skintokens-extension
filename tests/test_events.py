from __future__ import annotations

import io
import json
import unittest

from skintokens_ext.events import EventWriter, LineRelay


class EventTests(unittest.TestCase):
    def test_event_writer_outputs_jsonl(self) -> None:
        stream = io.StringIO()
        writer = EventWriter(stream)
        writer.progress(15, "Starting", "bpy-server")
        writer.log("ready", "readiness")
        writer.done("/tmp/out.glb", {"warnings": []})
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(events[0]["type"], "progress")
        self.assertEqual(events[0]["stage"], "bpy-server")
        self.assertEqual(events[1]["type"], "log")
        self.assertEqual(events[2]["result"]["filePath"], "/tmp/out.glb")

    def test_line_relay_forwards_complete_lines(self) -> None:
        captured = []
        relay = LineRelay(lambda message, stage: captured.append((stage, message)), stage="upstream", prefix="u: ")
        relay.write("one\ntwo")
        relay.flush()
        self.assertEqual(captured, [("upstream", "u: one"), ("upstream", "u: two")])


if __name__ == "__main__":
    unittest.main()
