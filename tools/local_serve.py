"""Run the local composition until interrupted, owning its host broker.

SIGINT/SIGTERM cancel this task and unwind the SSH/broker lifetime. Other host
loss can prevent cleanup; the guest's SSH-scope dependency ends its service
when the connection ends. Persistent runtime data is never deleted here.
Only deployment diagnostics are forwarded, with bounded volume. Conversations
are read through the separate authenticated operator client.
"""

import asyncio
import signal
import sys

import local_bridge
import local_model


async def run(instance, codex, selected_model):
    loop = asyncio.get_running_loop()
    owner = asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, owner.cancel)
    tasks = []
    try:
        async with local_bridge.connection(instance, codex, selected_model, resident=True) as guest:
            async def forward(stream, destination):
                size = 0
                while line := await stream.readline():
                    size += len(line)
                    if size > local_model.MAX_BYTES:
                        raise ValueError("guest diagnostics exceed limit")
                    text = line.decode(errors="replace")
                    # A guest cannot send terminal control sequences to the host.
                    text = "".join(c if c in "\n\t" or c.isprintable() else "?" for c in text)
                    print(text, end="", file=destination, flush=True)
            tasks = [asyncio.create_task(forward(guest.stdout, sys.stdout)),
                     asyncio.create_task(forward(guest.stderr, sys.stderr)), asyncio.create_task(guest.wait())]
            try:
                _, _, code = await asyncio.gather(*tasks)
                if code:
                    raise ValueError("guest runtime failed with status " + str(code))
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
