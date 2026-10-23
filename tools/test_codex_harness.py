"""Public CLI tests; the fake Codex is a subprocess, never a model call."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import codex_harness as harness

FAKE = '''#!/usr/bin/env python3
import json, pathlib, sys, time
args = sys.argv[1:]
if args == ["exec", "--help"]:
    print("--ignore-user-config --json --output-schema --ephemeral --sandbox")
elif args == ["features", "list"]:
    print("\\n".join(name + " stable true" for name in DISABLED))
elif args == ["login", "status"]:
    pass
elif args == ["--version"]:
    print("fixture Codex")
else:
    prompt = sys.stdin.read()
    out = pathlib.Path(args[args.index("--output-last-message") + 1])
    if "TEST_TIMEOUT" in prompt:
        time.sleep(20)
    report = {"summary": "candidate", "mathematical_status": "candidate_proof",
              "argument": "A proposed argument, not checked.", "open_obligations": [],
              "citations": [], "next_step": "Review it."}
    if "TEST_EXTRA_FIELD" in prompt:
        report["solved"] = True
    out.write_text(json.dumps(report))
    if "TEST_EXIT_FAILURE" in prompt:
        sys.exit(7)
    if "TEST_MISSING_EVENT" not in prompt:
        print(json.dumps({"type": "turn.completed"}))
    if "TEST_TOOL_EVENT" in prompt:
        print(json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}))
'''


class PortableCodex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fake = self.root / "fake-codex"
        self.fake.write_text(FAKE.replace("import json, pathlib, sys, time",
                            "import json, pathlib, sys, time\nDISABLED = " + repr(harness.DISABLED)))
        self.fake.chmod(0o755)
        self.task = self.root / "task.txt"
        self.task.write_text("Prove a precise mathematical identity.")
        self.run = self.root / "run"

    def cli(self, *args):
        return subprocess.run([sys.executable, str(Path(harness.__file__)), *args],
                              capture_output=True, text=True, timeout=15)

    def start(self, *extra):
        return self.cli("run", "--codex", str(self.fake), "--task-file", str(self.task),
                        "--run-dir", str(self.run), *extra)

    def state(self):
        return json.loads((self.run / "state.json").read_text())

    def test_run_resume_uses_snapshots_and_never_promotes_candidate(self):
        context = self.root / "context.txt"
        original = b"Original exact context\r\n"
        context.write_bytes(original)
        result = self.start("--context-file", str(context))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.run / "inputs/1.txt").read_bytes(), original)
        self.task.write_text("Changed task should not enter continuation")
        context.write_text("Changed source should not enter continuation")
        result = self.cli("resume", "--codex", str(self.fake), "--run-dir", str(self.run))
        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.state()
        self.assertEqual(state["status"], "completed_unverified")
        self.assertEqual([n["parent"] for n in state["nodes"]], [None, 0, 1])
        self.assertEqual(state["nodes"][2]["status"], "unverified")
        prompt = (self.run / "attempts/2/prompt.txt").read_text()
        self.assertIn("Original exact context", prompt)
        self.assertNotIn("Changed task", prompt)
        invocation = json.loads((self.run / "attempts/2/invocation.json").read_text())["argv"]
        self.assertIn("read-only", invocation)
        self.assertIn('approval_policy="never"', invocation)
        self.assertIn("--ignore-user-config", invocation)
        self.assertNotIn("danger-full-access", invocation)
        self.assertNotIn("--model", invocation)
        self.assertEqual({invocation[i + 1] for i, v in enumerate(invocation[:-1]) if v == "--disable"}, set(harness.DISABLED))

    def test_changed_snapshots_and_candidates_refuse_resume(self):
        self.assertEqual(self.start().returncode, 0)
        snapshot = self.run / "inputs/0.txt"
        original = snapshot.read_bytes()
        snapshot.write_text("Different target")
        result = self.cli("resume", "--codex", str(self.fake), "--run-dir", str(self.run))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.state()["nodes"]), 2)
        snapshot.write_bytes(original)
        (self.run / "attempts/1/report.json").write_text("{}")
        result = self.cli("resume", "--codex", str(self.fake), "--run-dir", str(self.run))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.state()["nodes"]), 2)

    def test_cli_failure_is_not_overridden_by_output_file(self):
        self.task.write_text("TEST_EXIT_FAILURE")
        result = self.start()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.run / "attempts/1/report.json").exists())
        self.assertEqual(self.state()["status"], "failed")

    def test_unfinished_event_is_not_overridden_by_output_file(self):
        self.task.write_text("TEST_MISSING_EVENT")
        self.assertNotEqual(self.start().returncode, 0)
        self.assertEqual(self.state()["nodes"][1]["status"], "failed")

    def test_unknown_report_fields_cannot_grant_verification(self):
        self.task.write_text("TEST_EXTRA_FIELD")
        self.assertNotEqual(self.start().returncode, 0)
        self.assertEqual(self.state()["status"], "failed")

    def test_unexpected_execution_cannot_count_as_paper_only_success(self):
        self.task.write_text("TEST_TOOL_EVENT")
        self.assertNotEqual(self.start().returncode, 0)
        self.assertEqual(self.state()["status"], "failed")

    def test_timeout_keeps_recoverable_checkpoint(self):
        self.task.write_text("TEST_TIMEOUT")
        self.assertNotEqual(self.start("--timeout", "1").returncode, 0)
        self.assertEqual(self.state()["status"], "interrupted")
        self.assertTrue((self.run / "attempts/1/prompt.txt").exists())

    def test_existing_run_is_not_overwritten_and_lock_refuses_second_writer(self):
        self.assertEqual(self.start().returncode, 0)
        before = (self.run / "state.json").read_bytes()
        self.assertNotEqual(self.start().returncode, 0)
        self.assertEqual((self.run / "state.json").read_bytes(), before)
        with harness.locked(self.run):
            result = self.cli("resume", "--codex", str(self.fake), "--run-dir", str(self.run))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.run / "state.json").read_bytes(), before)

    def test_interrupted_dispatch_is_not_replayed(self):
        self.assertEqual(self.start().returncode, 0)
        state = self.state()
        state["nodes"][1]["status"] = "running"
        harness.save(self.run / "state.json", state)
        result = self.cli("resume", "--codex", str(self.fake), "--run-dir", str(self.run))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.state()["nodes"][1]["status"], "interrupted")
        self.assertEqual(self.state()["nodes"][2]["parent"], 0)

    def test_symlink_context_is_refused_before_dispatch(self):
        link = self.root / "link.txt"
        link.symlink_to(self.task)
        self.assertNotEqual(self.start("--context-file", str(link)).returncode, 0)
        self.assertFalse(self.run.exists())

    def test_upstream_doubling_policy_drives_the_actual_attempts(self):
        result = self.start("--policy", "pi_doubling", "--width", "2", "--rounds", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([n["parent"] for n in self.state()["nodes"]], [None, 0, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
