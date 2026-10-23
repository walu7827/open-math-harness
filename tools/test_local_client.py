"""The local client's conversation workflow over its public JSON socket protocol."""

import importlib.util
import json
from pathlib import Path
import socket
import threading
import unittest


SPEC = importlib.util.spec_from_file_location(
    "local_client", Path(__file__).resolve().parents[1] / "deploy/local/client.py")
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)


class LocalClient(unittest.TestCase):
    def test_chat_continues_a_live_session_on_a_later_page(self):
        local, remote = socket.socketpair()
        local.settimeout(2)
        remote.settimeout(2)
        calls, errors = [], []
        def serve():
            records = []
            try:
                with remote, remote.makefile("rwb", buffering=0) as stream:
                    while line := stream.readline():
                        request = json.loads(line)
                        calls.append(request)
                        op = request["op"]
                        reply = {"id": request["id"]}
                        if op == "sessions.list":
                            if request.get("after") is None:
                                reply["ok"] = {"sessions": [{"project": "a", "name": "s", "live": True}],
                                               "next": "a/s"}
                            else:
                                reply["ok"] = {"sessions": [{"project": "z", "name": "s", "live": True}]}
                        elif op == "sessions.create":
                            reply["error"] = {"code": "invalid_input", "message": "that session already has an agent"}
                        elif op == "conversation.history":
                            reply["ok"] = {"records": records, "has_more": False}
                        elif op == "conversation.say":
                            records = [{"seq": 0, "kind": "delivered", "id": request["message_id"]},
                                       {"seq": 1, "kind": "turn_complete"}]
                            reply["ok"] = {}
                        else:
                            raise AssertionError("unexpected operation: " + op)
                        stream.write(json.dumps(reply).encode() + b"\n")
            except Exception as error:
                errors.append(error)
        server = threading.Thread(target=serve)
        server.start()
        try:
            with local, local.makefile("rwb", buffering=0) as stream:
                CLIENT.Client(local, stream).chat({"project": "z", "session": "s", "text": "next turn"})
        finally:
            server.join(3)
        self.assertFalse(server.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([r["op"] for r in calls], ["sessions.list", "sessions.list", "conversation.history",
                                                  "conversation.say", "conversation.history"])
        self.assertEqual(calls[1]["after"], "a/s")


if __name__ == "__main__":
    unittest.main()
