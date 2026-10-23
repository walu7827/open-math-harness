"""Broker HTTP boundary and real CLI process ownership, without model credentials."""

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import local_model as model


FAKE = r'''
import json, os, pathlib, signal, subprocess, sys, time
ROOT = pathlib.Path(ROOT_VALUE)
args = sys.argv[1:]
assert '--ignore-user-config' in args and '--ephemeral' in args
assert args[args.index('--sandbox') + 1] == 'read-only'
assert set(DISABLED) <= {args[i + 1] for i, x in enumerate(args[:-1]) if x == '--disable'}
assert '--enable' in args and 'skip_host_skill_discovery' in args
assert 'project_doc_max_bytes=0' in args
assert not {'OPENAI_API_KEY', 'TEST_SECRET', 'SSH_AUTH_SOCK'} & os.environ.keys()
assert list(pathlib.Path.cwd().iterdir()) == []
body = json.loads(sys.stdin.read())
(ROOT / 'seen.json').write_text(json.dumps(body))
text = body['messages'][-1]['content']
if text == 'WAIT':
    code = "import signal,time,pathlib; signal.signal(signal.SIGTERM, lambda *_: (pathlib.Path(" + repr(str(ROOT / 'child-stopped')) + ").touch(), exit(0))); pathlib.Path(" + repr(str(ROOT / 'child-ready')) + ").touch(); time.sleep(60)"
    child = subprocess.Popen([sys.executable, '-c', code])
    def stop(*_):
        child.wait(timeout=3)
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    (ROOT / 'pid').write_text(str(os.getpid()))
    time.sleep(60)
report = {'content': 'fixture answer', 'tool_calls': []}
if text in ('CALL', 'FORGED'):
    report['tool_calls'] = [{'name': 'read' if text == 'CALL' else 'host.exec', 'arguments_json': '{"path":"work/a.txt"}'}]
if text == 'BAD_JSON':
    report['tool_calls'] = [{'name': 'read', 'arguments_json': '{"x":1,"x":2}'}]
out = pathlib.Path(args[args.index('--output-last-message') + 1])
out.write_text(json.dumps(report))
if text == 'EXECUTE':
    print(json.dumps({'type': 'item.completed', 'item': {'type': 'command_execution'}}), flush=True)
    time.sleep(60)
if text == 'OVERSIZE':
    sys.stdout.write('x' * (8 * 1024 * 1024 + 2) + '\n')
elif text != 'UNFINISHED':
    print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 42, 'output_tokens': 7}}))
'''


def request(text="hello", **extra):
    return {"model": "codex/default", "messages": [{"role": "user", "content": text}],
            "tools": [{"type": "function", "function": {"name": "read", "description": "Read a file",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}], **extra}


class BrokerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fake = self.root / "codex"
        fake.write_text("#!" + sys.executable + "\n" + FAKE.replace("ROOT_VALUE", repr(str(self.root)))
                        .replace("set(DISABLED)", "set(" + repr(model.DISABLED) + ")"))
        fake.chmod(0o700)
        self.broker = model.Broker(model.Codex(str(fake)), "t" * 32, timeout=5, max_inflight=1)
        self.server = await asyncio.start_server(self.broker.handle, "127.0.0.1", 0, limit=model.HEADER_BYTES)
        self.port = self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        await self.broker.close()
        self.tmp.cleanup()

    async def connect(self, body, token=None):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        raw = json.dumps(body).encode() if not isinstance(body, bytes) else body
        token = self.broker.token if token is None else token
        writer.write((f"POST /chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {token}\r\n"
                      f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n").encode() + raw)
        await writer.drain()
        return reader, writer

    async def call(self, body, token=None):
        reader, writer = await self.connect(body, token)
        raw = await asyncio.wait_for(reader.read(), 8)
        writer.close()
        await writer.wait_closed()
        header, data = raw.split(b"\r\n\r\n", 1)
        return int(header.split()[1]), json.loads(data)

    async def until(self, predicate):
        async with asyncio.timeout(5):
            while not predicate():
                await asyncio.sleep(0.01)

    async def test_text_and_tool_proposals_preserve_history_without_host_authority(self):
        body = request("CALL", stream=True, usage={"include": True}, stream_options={"include_usage": True})
        status, reply = await self.call(body)
        self.assertEqual(status, 200, reply)
        call = reply["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["function"], {"name": "read", "arguments": '{"path":"work/a.txt"}'})
        self.assertEqual(reply["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(reply["usage"], {"prompt_tokens": 42, "completion_tokens": 7})
        self.assertNotIn("cost", reply["usage"])
        body["messages"] += [{"role": "assistant", "content": "", "tool_calls": [call]},
                             {"role": "tool", "content": "untrusted file result", "tool_call_id": call["id"]}]
        status, reply = await self.call(body)
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["choices"][0]["message"]["content"], "fixture answer")
        self.assertEqual(json.loads((self.root / "seen.json").read_text())["messages"], body["messages"])

    async def test_invalid_or_unsupported_input_never_dispatches(self):
        for body, token, status in [(request(), "wrong", 401), (request(max_tokens=2), None, 400),
                (request(temperature=0), None, 400), (request(model="codex/arbitrary"), None, 400),
                (request(extra="config"), None, 400), (request(reasoning={"effort": "wrong"}), None, 400),
                (b'{"model":"codex/default","model":"other","messages":[]}', None, 400),
                ({"model": "codex/default", "messages": [{"role": "user", "content": []}]}, None, 400)]:
            with self.subTest(body=body):
                got, _ = await self.call(body, token)
                self.assertEqual(got, status)
        self.assertFalse((self.root / "seen.json").exists())

    async def test_cli_errors_and_forged_tools_are_not_success(self):
        for text in ("FORGED", "BAD_JSON", "EXECUTE", "UNFINISHED", "OVERSIZE"):
            with self.subTest(text=text):
                status, _ = await self.call(request(text))
                self.assertEqual(status, 502)

    async def test_disconnect_reaps_the_process_group_and_releases_admission(self):
        reader, writer = await self.connect(request("WAIT"))
        await self.until(lambda: (self.root / "child-ready").exists() and (self.root / "pid").exists())
        pid = int((self.root / "pid").read_text())
        other, other_writer = await self.connect(request())
        # Closing a socket with unread request bytes can produce EOF or RST,
        # depending on the host TCP stack. Neither admits another model call.
        try:
            self.assertEqual(await other.read(), b"")
        except ConnectionResetError:
            pass
        self.assertEqual(json.loads((self.root / "seen.json").read_text())["messages"][-1]["content"], "WAIT")
        other_writer.close()
        try:
            await other_writer.wait_closed()
        except ConnectionResetError:
            pass
        writer.close()
        await writer.wait_closed()
        await self.until(lambda: not self.broker.connections)
        self.assertTrue((self.root / "child-stopped").exists())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        status, _ = await self.call(request())
        self.assertEqual(status, 200)

    async def test_deadline_and_shutdown_also_reap_active_children(self):
        self.broker.timeout = 0.5
        status, _ = await self.call(request("WAIT"))
        self.assertEqual(status, 504)
        self.assertTrue((self.root / "child-stopped").exists())
        self.broker.timeout = 5
        (self.root / "child-ready").unlink()
        (self.root / "child-stopped").unlink()
        reader, writer = await self.connect(request("WAIT"))
        await self.until(lambda: (self.root / "child-ready").exists())
        await self.broker.close()
        self.assertEqual(await reader.read(), b"")
        self.assertTrue((self.root / "child-stopped").exists())
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    unittest.main()
