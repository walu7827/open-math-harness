#!/usr/bin/env python3
"""Explicit live-model smoke test, run only by local_runtime's model-smoke.

This trusted guest operator creates and owns a temporary resident composition:
real bubblewrap services, complete-only model grant, read/list only on work/,
private model token, and an owner Unix UI socket. It reads a synthetic allowed
file, verifies a kernel denial for a synthetic forbidden file, then restarts the
runtime and checks that history restores without re-executing the read. The
host receives only the test outcome and token counts, never the transcript.
The shared operator lock is held throughout, so source publication or a build
cannot replace the snapshot or executables while this composition uses them.

No project-user/cgroup isolation, memory consolidation or Lean is claimed by
this small composition. Data and temporary credentials are removed on exit.
EOF of the operator's stdin or a signal stops the runtime; ordinary failures
also stop it. Shutdown must report success; a timeout or incomplete cleanup
fails the smoke. Like other operator processes, SIGKILL or host/VM loss can
prevent cleanup; this test does not provide crash recovery.
"""
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import threading
from pathlib import Path

operator_lock = (Path.home() / '.local/share/nucleus-local/operator.lock').open('w')
fcntl.flock(operator_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

def interrupted(*_):
    raise InterruptedError('operator connection closed or smoke interrupted')
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
key = sys.stdin.readline().strip()

def watch_operator():
    # Do not hold a buffered-I/O lock across the main thread's failure exit.
    os.read(sys.stdin.fileno(), 1)
    os.kill(os.getpid(), signal.SIGTERM)
threading.Thread(target=watch_operator, daemon=True).start()
if not 32 <= len(key) <= 256 or not all((c.isalnum() or c in '-_' for c in key)):
    raise ValueError('invalid per-run broker token')
port = int(sys.argv[1])
if not 1 <= port <= 65535:
    raise ValueError('invalid guest broker port')
source = (Path.home() / '.local/share/nucleus-local/current').resolve()
binaries = source / 'target/debug'
with tempfile.TemporaryDirectory(prefix='nucleus-codex-smoke-') as directory:
    root = Path(directory)
    for name in ('data/work', 'log', 'sessions', 'ui', 'secrets'):
        (root / name).mkdir(parents=True)
    (root / 'data/work/notes.txt').write_text('The fixture observation is NEBULA-731. It is a synthetic integration test.')
    (root / 'data/secret.txt').write_text('FORBIDDEN-SYNTHETIC-918')
    (root / 'secrets/key').write_text(key)
    (root / 'secrets/key').chmod(0o600)

    def bind(host, guest, write=False):
        return {'host': str(root / host), 'guest': guest, 'writable': write}

    def grant(ep, actions, path=(), delegate=False):
        return {'endpoint': ep, 'actions': actions.split(), 'resource': list(path), 'delegable': delegate}

    def program(exe, args, binds=(), network=False):
        return {'executable': str(binaries / exe), 'args': args, 'binds': list(binds), 'network': network}
    child = ['--model', 'codex/default', '--max-steps', '4', '--model-deadline-ms', '180000', '--model-retries', '0', '--system-prompt', 'You are a test assistant. For a request to read files, call fs_read once for every requested path, including inaccessible paths so the kernel can return permission_denied. Never retry a refused call. Then report the observed contents and denials. Use no other tools.']
    parent = ['--child-executable', '/session-agent']
    for arg in child:
        parent += ['--child-arg', arg]
    programs = {
        'fs': program('nucleus-fs', ['--root', '/data', '--access-log', '/log/access.jsonl'],
                      [bind('data', '/data'), bind('log', '/log', True)]),
        'model': program('nucleus-openrouter', [
            '--key-file', '/secrets/key', '--base-url', f'http://127.0.0.1:{port}',
            '--retries', '0', '--max-timeout-ms', '180000'], [bind('secrets', '/secrets')], True),
        'conversation': program('nucleus-conversation', ['--root', '/sessions'],
                                [bind('sessions', '/sessions', True)]),
        'parent': program('nucleus-chat-agent', parent, [{
            'host': str(binaries / 'nucleus-chat-agent'), 'guest': '/session-agent', 'writable': False}]),
        'ui': program('nucleus-ui', ['--socket', '/ui/ui.sock'], [bind('ui', '/ui', True)]),
    }
    boot = []
    for name, ep in [('fs', 'fs'), ('model', 'model'), ('conversation', 'conversation')]:
        boot.append({'program': name, 'provides': ep,
                     'grants': [grant('kernel', 'provide', ['endpoint', ep])]})
    boot.append({'program': 'parent', 'provides': 'sessions', 'grants': [
        grant('kernel', 'provide', ['endpoint', 'sessions']),
        grant('kernel', 'spawn_executable', ['executable', 'session-agent']),
        grant('conversation', 'receive post history remind reminders cancel_reminder', delegate=True),
        grant('conversation', 'clear_reminders list'),
        grant('model', 'complete', ['codex'], True),
        grant('fs', 'read list', ['work'], True),
    ]})
    boot.append({'program': 'ui', 'grants': [
        grant('conversation', 'say history wait'), grant('sessions', 'create stop list'),
    ]})
    config = {'lifetime': 'resident', 'grace_ms': 5000,
              'limits': {'max_deadline_ms': 180000, 'max_inflight_per_endpoint': 16},
              'programs': programs, 'boot': boot}
    (root / 'config.json').write_text(json.dumps(config))

    def shutdown_process(process):
        if process.poll() is None:
            process.terminate()
        try:
            return process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
            raise RuntimeError('runtime exceeded shutdown deadline; forced group termination')

    def boot_runtime():
        logs = open(root / 'runtime.log', 'ab')
        process = subprocess.Popen([str(binaries / 'nucleus'), 'serve', str(root / 'config.json')],
                                   stdout=logs, stderr=logs, start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError((root / 'runtime.log').read_text()[-5000:])
                try:
                    s = socket.socket(socket.AF_UNIX)
                    s.connect(str(root / 'ui/ui.sock'))
                    s.settimeout(20)
                    return (process, s, s.makefile('rwb', buffering=0), logs)
                except (FileNotFoundError, ConnectionRefusedError):
                    s.close()
                    time.sleep(0.05)
            raise RuntimeError('UI did not start')
        except BaseException:
            try:
                shutdown_process(process)
            finally:
                logs.close()
            raise
    seq = 0

    def req(stream, op, **kw):
        global seq
        seq += 1
        stream.write((json.dumps({'id': seq, 'op': op, **kw}) + '\n').encode())
        answer = json.loads(stream.readline())
        if 'error' in answer:
            raise RuntimeError(answer['error'])
        return answer['ok']

    def history(stream):
        return req(stream, 'conversation.history', session='default')['records']

    def turn(stream, n):
        end = time.monotonic() + 190
        while time.monotonic() < end:
            records = history(stream)
            if sum((x['kind'] == 'turn_complete' for x in records)) >= n:
                return records
            if any((x['kind'] == 'error' for x in records)):
                raise RuntimeError([x for x in records if x['kind'] == 'error'])
            time.sleep(0.2)
        raise RuntimeError('turn deadline')

    def stop(process, s, stream, logs):
        stream.close()
        s.close()
        try:
            if shutdown_process(process) != 0:
                raise RuntimeError((root / 'runtime.log').read_text()[-3000:])
        finally:
            logs.close()
    process = s = stream = logs = None
    try:
        process, s, stream, logs = boot_runtime()
        req(stream, 'sessions.create', session='default')
        req(stream, 'conversation.say', session='default', message_id='smoke-1', text='Read work/notes.txt and secret.txt. Report each result, including any permission denial.')
        records = turn(stream, 1)
        tools = [x for x in records if x['kind'] == 'tool_result']
        assert any((x.get('error_code') == 'permission_denied' for x in tools)), tools
        assert any(('NEBULA-731' in x.get('summary', '') for x in tools)), tools
        assert 'FORBIDDEN-SYNTHETIC-918' not in json.dumps(records)
        access = (root / 'log/access.jsonl').read_text().splitlines()
        assert len(access) == 1, access
        stop(process, s, stream, logs)
        process = None
        process, s, stream, logs = boot_runtime()
        req(stream, 'sessions.create', session='default')
        req(stream, 'conversation.say', session='default', message_id='smoke-2', text='Without any tools, recall the exact synthetic observation from the preceding file result.')
        records = turn(stream, 2)
        answer = [x for x in records if x['kind'] == 'assistant'][-1]['text']
        assert 'NEBULA-731' in answer, answer
        assert len((root / 'log/access.jsonl').read_text().splitlines()) == 1, 'history replayed a tool'
        calls = [x for x in records if x['kind'] == 'model_call']
        print(json.dumps({'allowed_read': True, 'kernel_denial': True, 'forbidden_data_absent': True, 'restart_history_recalled': True, 'history_did_not_replay_tool': True, 'model_calls': len(calls), 'streamed': [x.get('streamed') for x in calls], 'usage': [{'input_tokens': x.get('input_tokens'), 'output_tokens': x.get('output_tokens'), 'cost_nano_usd': x.get('cost_nano_usd')} for x in calls]}), flush=True)
    finally:
        if process is not None:
            stop(process, s, stream, logs)
