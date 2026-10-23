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

import local_bridge
import local_model


async def run(instance, codex, selected_model):
    async with local_bridge.connection(instance, codex, selected_model) as guest:
        tasks = [asyncio.create_task(local_bridge.bounded(guest.stdout)),
                 asyncio.create_task(local_bridge.bounded(guest.stderr)), asyncio.create_task(guest.wait())]
        try:
            out, err, code = await asyncio.wait_for(asyncio.gather(*tasks), 450)
            if code:
                raise ValueError("guest model smoke failed: " + err.decode(errors="replace")[-4000:])
            result = local_model.decode(out)
            if not isinstance(result, dict) or not all(result.get(key) is True for key in (
                    "allowed_read", "kernel_denial", "forbidden_data_absent", "restart_history_recalled",
                    "history_did_not_replay_tool")):
                raise ValueError("guest did not confirm the complete model smoke")
            print(json.dumps(result), flush=True)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
