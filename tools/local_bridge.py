"""Own the host model broker and one non-multiplexed guest SSH connection.

Both the live smoke and the resident composition use this lifetime. The fresh
bearer token travels only on stdin, never in argv or the environment. EOF asks
the guest owner to release its runtime; stopping SSH alone does not undo calls.
The resident additionally binds its systemd unit to that SSH login scope.
Cleanup attempts every independent release and reports incomplete cleanup.
"""

import asyncio
from contextlib import asynccontextmanager
import secrets
import shlex

import codex_harness
import local_model
import local_runtime


@asynccontextmanager
async def connection(instance, codex, selected_model, *, resident=False):
    backend = local_model.Codex(codex_harness.executable(codex), selected_model)
    backend.doctor()
    token = secrets.token_urlsafe(32)
    broker = local_model.Broker(backend, token, timeout=175, max_inflight=2 if resident else 1)
    server = await asyncio.start_server(broker.handle, "127.0.0.1", 0, limit=local_model.HEADER_BYTES)
    guest = None
    try:
        config = local_runtime.lima("list", "--format={{.SSHConfigFile}}", instance,
                                    capture_output=True, text=True).stdout.strip()
        port = server.sockets[0].getsockname()[1]
        guest_port = 37071
        if resident:
            helper = (local_runtime.ROOT / "deploy/local/guest.py").read_text()
            command = "exec python3 -c " + shlex.quote(helper) + " serve " + str(guest_port)
        else:
            command = 'exec python3 "$HOME/.local/share/nucleus-local/current/deploy/local/model_smoke.py" ' + str(guest_port)
        guest = await asyncio.create_subprocess_exec(
            "ssh", "-F", config, "-T", "-o", "ControlMaster=no", "-o", "ControlPath=none",
            "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes", "-o", "ForwardAgent=no",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
            "-R", f"127.0.0.1:{guest_port}:127.0.0.1:{port}", "lima-" + instance, command,
            env=local_runtime.host_env(), start_new_session=True,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        guest.stdin.write((token + "\n").encode())
        await guest.stdin.drain()
        yield guest
    finally:
        errors = []
        try:
            if guest is not None:
                guest.stdin.close()
                try:
                    await asyncio.wait_for(guest.wait(), 60 if resident else 15)
                except asyncio.TimeoutError:
                    await local_model.terminate(guest)
                    raise RuntimeError("guest cleanup exceeded its deadline; SSH was terminated")
                if guest.returncode:
                    raise RuntimeError("guest did not exit cleanly (status " + str(guest.returncode) + ")")
        except BaseException as error:
            errors.append(error)
        try:
            server.close()
            await server.wait_closed()
        except BaseException as error:
            errors.append(error)
        try:
            await broker.close()
        except BaseException as error:
            errors.append(error)
        if errors:
            raise RuntimeError("local bridge cleanup failed: " + "; ".join(str(error) for error in errors))


async def bounded(stream):
    result = bytearray()
    while data := await stream.read(8192):
        result.extend(data)
        if len(result) > local_model.MAX_BYTES:
            raise ValueError("guest output exceeds limit")
    return bytes(result)
