"""Command interpretation and transactional chat behavior without model downloads."""

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from chat_commands import CommandError, EditPlan, parse_command

try:
    from fastapi.testclient import TestClient

    from walkedit_app import create_app
except ImportError:
    TestClient = None


class CommandTests(unittest.TestCase):
    def test_absolute_and_relative_speeds(self):
        current = EditPlan(speed=2)
        for text, expected in [
            ("Set speed to 1.5x", 1.5),
            ("Make it 50% faster", 3),
            ("Make it 20% slower", 1.6),
            ("Walk at half speed", 0.5),
            ("Make it twice as fast", 2),
        ]:
            with self.subTest(text=text):
                self.assertAlmostEqual(
                    parse_command(text, current, 2).plan.speed, expected
                )

    def test_interval_and_context(self):
        plan = parse_command(
            "Set speed to 2x from 0.5 to 1.5 seconds with a smooth transition of 0.2 seconds",
            EditPlan(),
            2,
        ).plan
        self.assertEqual(plan, EditPlan(2, 0.5, 1.5, 0.2))
        self.assertEqual(
            parse_command("Set speed to 0.5x", plan, 2).plan,
            EditPlan(0.5, 0.5, 1.5, 0.2),
        )
        self.assertEqual(
            parse_command("Set the whole motion to 1.5x", plan, 2).plan, EditPlan(1.5)
        )
        self.assertEqual(parse_command("Reset", plan, 2).plan, EditPlan())
        self.assertEqual(parse_command("Undo", plan, 2).action, "undo")

    def test_unclear_unsupported_and_unsafe_requests_do_not_become_edits(self):
        for text in [
            "faster",
            "Set speed to -2x",
            "Set speed to 0x",
            "Set speed to 12x",
            "Set speed to 1.5",
            "Jump at 2x",
            "Set speed to 2x and turn 90 degrees",
            "Set speed to 2x from 1 to 4 seconds",
            "Set speed to 2x then 0.5x",
            "Don't set speed to 2x",
            "Set speed to 2x and change color",
            "Set speed to 2x between 0 and 1",
            "Set speed to 2x from .5 to 1.5 seconds with a transition of .6 seconds",
        ]:
            with self.subTest(text=text), self.assertRaises(CommandError):
                parse_command(text, EditPlan(), 2)

    def test_smoothing_validates_short_intervals(self):
        plan = parse_command(
            "Walk at 2x from 0.5 to 0.7 seconds smoothly", EditPlan(), 2
        ).plan
        self.assertAlmostEqual(plan.transition, 0.05)


@unittest.skipIf(
    TestClient is None, "Install requirements-test.txt to run workspace API checks"
)
class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "original.pt"
        self.model = self.root / "model.npz"
        self.source.write_bytes(b"trusted fixture")
        self.model.write_bytes(b"fixture")
        self.calls = []
        self.release = Event()
        self.entered = Event()
        self.block = False

        def renderer(source, model, destination, plan):
            self.calls.append(plan)
            if self.block:
                self.entered.set()
                if not self.release.wait(3):
                    raise RuntimeError("Fixture release timed out")
            if plan["speed"] == 3:
                raise RuntimeError("Deliberate renderer failure")
            result = dict(
                edited_seconds=2 / plan["speed"],
                edited_frames=round(60 / plan["speed"]),
                output_fps=30,
                source_times=[0, 1, 1.9667],
            )
            (destination / "comparison.mp4").write_bytes(b"video fixture")
            (destination / "edited.pt").write_bytes(b"retimed fixture")
            (destination / "edit.json").write_text(json.dumps(result))
            return result

        app = create_app(
            self.source,
            self.model,
            self.root / "jobs",
            renderer=renderer,
            source_info=dict(frames=60, fps=30, seconds=2, span=59 / 30),
        )
        self.client = TestClient(app)
        self.client.__enter__()
        self.session = self.client.post("/api/sessions").json()["id"]

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def send(self, text):
        response = self.client.post(
            f"/api/sessions/{self.session}/messages", json={"text": text}
        )
        self.assertEqual(response.status_code, 200)
        for _ in range(100):
            result = self.client.get(f"/api/sessions/{self.session}").json()
            if result["state"] != "rendering":
                return result
            time.sleep(0.01)
        self.fail("Test render did not finish")

    def test_unsupported_input_keeps_state_and_never_calls_renderer(self):
        data = self.send("Make it jump")
        self.assertEqual(data["plan"]["speed"], 1)
        self.assertEqual(self.calls, [])
        self.assertIsNone(data["result"])

    def test_render_undo_and_failed_edit_preserve_successful_artifacts(self):
        first = self.send("Set speed to 1.5x")
        self.assertEqual(first["plan"]["speed"], 1.5)
        self.assertAlmostEqual(first["result"]["edited_seconds"], 4 / 3)
        second = self.send("Set speed to 2x")
        self.assertEqual(second["plan"]["speed"], 2)
        undone = self.send("Undo")
        self.assertEqual(undone["revision"], first["revision"])
        self.assertEqual(len(self.calls), 2)
        failed = self.send("Set speed to 3x")
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["revision"], first["revision"])
        self.assertEqual(failed["plan"]["speed"], 1.5)
        restored = self.send("Undo")
        self.assertEqual(restored["plan"]["speed"], 1)

    def test_session_isolation_and_artifact_allowlist(self):
        data = self.send("Set speed to 2x")
        other = self.client.post("/api/sessions").json()
        self.assertEqual(other["plan"]["speed"], 1)
        base = f"/api/sessions/{self.session}/renders/{data['revision']}/"
        self.assertEqual(
            self.client.get(
                base + "comparison.mp4", headers={"Range": "bytes=0-4"}
            ).status_code,
            206,
        )
        self.assertEqual(self.client.get(base + "render.log").status_code, 404)
        self.assertEqual(
            self.client.get(
                f"/api/sessions/{other['id']}/renders/{data['revision']}/comparison.mp4"
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.post(
                f"/api/sessions/{self.session}/messages", json={"text": "x" * 501}
            ).status_code,
            422,
        )

    def test_busy_render_rejects_duplicate_commands(self):
        self.block = True
        response = self.client.post(
            f"/api/sessions/{self.session}/messages", json={"text": "Set speed to 2x"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.entered.wait(1))
        try:
            duplicate = self.client.post(
                f"/api/sessions/{self.session}/messages",
                json={"text": "Set speed to 0.5x"},
            )
            self.assertEqual(duplicate.status_code, 409)
            data = self.client.get(f"/api/sessions/{self.session}").json()
            self.assertEqual(data["plan"]["speed"], 1)
            self.assertEqual(len(self.calls), 1)
        finally:
            self.release.set()
        for _ in range(100):
            if (
                self.client.get(f"/api/sessions/{self.session}").json()["state"]
                != "rendering"
            ):
                break
            time.sleep(0.01)

    def test_sessions_cannot_be_reused_for_a_different_motion(self):
        different = self.root / "different.pt"
        different.write_bytes(b"different reconstruction")
        other_app = create_app(
            different,
            self.model,
            self.root / "jobs",
            source_info=dict(frames=60, fps=30, seconds=2, span=59 / 30),
        )
        with TestClient(other_app) as other:
            self.assertEqual(
                other.get(f"/api/sessions/{self.session}").status_code, 409
            )
            self.assertEqual(other.post("/api/sessions").status_code, 201)


if __name__ == "__main__":
    unittest.main()
