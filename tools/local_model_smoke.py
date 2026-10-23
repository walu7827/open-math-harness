"""Explicit Codex account integration check for the committed Lima runtime.

Called by local_runtime model-smoke after its VM boundary check. The host owns
a loopback broker and a non-multiplexed SSH reverse forward, scoped by a fresh
in-memory token. Only that token enters the guest model provider. The helper
and runtime come from the guest's selected committed source snapshot, not the
host's dirty files. No guest files or CLI credentials are copied back.

The SSH command owns the guest helper through stdin: EOF terminates its runtime
in a finally block. Closing its connection never reverses a model call. On host
failure the guest may take its shutdown deadline to exit; an unclean shutdown
is an error, never success. Each retry is an explicit new smoke invocation.
"""

import asyncio
import json
import secrets

import codex_harness
import local_model
import local_runtime


async def run(instance, codex, selected_model):
    backend = local_model.Codex(codex_harness.executable(codex), selected_model)
    backend.doctor()
    token = secrets.token_urlsafe(32)
    broker = local_model.Broker(backend, token, timeout=175, max_inflight=1)
    server = await asyncio.start_server(broker.handle, "127.0.0.1", 0, limit=local_model.HEADER_BYTES)
    guest = None
    tasks = []
    try:
        config = local_runtime.lima("list", "--format={{.SSHConfigFile}}", instance,
                                    capture_output=True, text=True).stdout.strip()
        # One SSH process owns both the explicit reverse forward and the test.
        # Disable the Lima control master: exit must release this forward only.
        port = server.sockets[0].getsockname()[1]
        guest_port = 37071
        command = 'exec python3 "$HOME/.local/share/nucleus-local/current/deploy/local/model_smoke.py" ' + str(guest_port)
        guest = await asyncio.create_subprocess_exec(
            "ssh", "-F", config, "-T", "-o", "ControlMaster=no", "-o", "ControlPath=none",
            "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes", "-o", "ForwardAgent=no",
            "-R", f"127.0.0.1:{guest_port}:127.0.0.1:{port}", "lima-" + instance, command,
            env=local_runtime.host_env(), start_new_session=True,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        guest.stdin.write((token + "\n").encode())
        await guest.stdin.drain()
        # Keep stdin open for ownership until the helper exits. communicate()
        # would close it after sending the token and cancel the smoke itself.
        async def bounded(stream):
            result = bytearray()
            while data := await stream.read(8192):
                result.extend(data)
                if len(result) > local_model.MAX_BYTES:
                    raise ValueError("guest smoke output exceeds limit")
            return bytes(result)
        tasks = [asyncio.create_task(bounded(guest.stdout)), asyncio.create_task(bounded(guest.stderr)),
                 asyncio.create_task(guest.wait())]
        out, err, code = await asyncio.wait_for(asyncio.gather(*tasks), 450)
        if code:
            # The guest emits only synthetic test diagnostics; never a secret.
            raise ValueError("guest model smoke failed: " + err.decode(errors="replace")[-4000:])
        result = local_model.decode(out)
        if not isinstance(result, dict) or not all(result.get(key) is True for key in (
                "allowed_read", "kernel_denial", "forbidden_data_absent", "restart_history_recalled",
                "history_did_not_replay_tool")):
            raise ValueError("guest did not confirm the complete model smoke")
        print(json.dumps(result), flush=True)
    finally:
        if guest is not None:
            guest.stdin.close()
            try:
                await asyncio.wait_for(guest.wait(), 15)
            except asyncio.TimeoutError:
                await local_model.terminate(guest)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        server.close()
        await server.wait_closed()
        await broker.close()
