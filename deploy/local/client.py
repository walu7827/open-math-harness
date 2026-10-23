#!/usr/bin/env python3
"""Text/JSON operator client of the existing local UI socket.

The host invokes this code as nucleus through the guest operator's sudo. Unix
permissions, not a caller-supplied role, admit that operator to the owner UI.
Project processes cannot reach the socket. All requests still pass through the
UI service's ordinary kernel handles. This client has no kernel connection.

One invocation sends one message, or one explicit JSON UI operation. Calls are
never replayed automatically. A lost send acknowledgement has unknown outcome;
inspect conversation history before trying again. The printed message id is
diagnostic, not a durable recovery journal. Disconnecting or timing out ends
only this client's wait, never the session or its actions. Use sessions.stop
explicitly to stop an agent. History is paged without accumulating the session,
and terminal text cannot carry control sequences. JSON mode preserves fields.
Chat first pages the live session list and creates an agent only if none is
running for that project and session. A concurrent creation or termination may
still make the next operation fail; that failure is reported without replay.
"""

import json
import socket
import sys
import time
import uuid


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate input field")
        result[key] = value
    return result


def display(text):
    print("".join(c if c in "\n\t" or c.isprintable() else "?" for c in str(text)), flush=True)


class Client:
    def __init__(self, connection, stream):
        self.connection, self.stream, self.sequence = connection, stream, 0

    def request(self, operation):
        if not isinstance(operation, dict) or "id" in operation or not isinstance(operation.get("op"), str):
            raise ValueError("UI request needs op and must omit id")
        self.sequence += 1
        wire = json.dumps({"id": self.sequence, **operation}, allow_nan=False).encode() + b"\n"
        if len(wire) > 4 * 1024 * 1024:
            raise ValueError("UI request exceeds limit")
        self.connection.sendall(wire)
        answer = self.stream.readline(8 * 1024 * 1024 + 1)
        if len(answer) > 8 * 1024 * 1024 or not answer.endswith(b"\n"):
            raise ValueError("UI answer is missing or exceeds limit; operation outcome may be unknown")
        reply = json.loads(answer)
        if not isinstance(reply, dict) or reply.get("id") != self.sequence:
            raise ValueError("UI answered a different request")
        if "error" in reply:
            raise ValueError(json.dumps(reply["error"], ensure_ascii=True))
        return reply["ok"]

    def page(self, scope, after):
        result = self.request({"op": "conversation.history", **scope, "after_seq": after})
        if not isinstance(result, dict) or not isinstance(result.get("records"), list) or type(result.get("has_more")) is not bool:
            raise ValueError("invalid history page")
        records = result["records"]
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("invalid history record")
            seq = record.get("seq")
            if type(seq) is not int or seq < 0 or (after is not None and seq <= after):
                raise ValueError("history did not advance")
            after = seq
        if result["has_more"] and not records:
            raise ValueError("empty history continuation")
        return records, after, result["has_more"]

    def chat(self, data):
        if (not isinstance(data, dict) or set(data) != {"project", "session", "text"}
                or any(not isinstance(value, str) or not value for value in data.values())):
            raise ValueError("chat needs project, session and text")
        scope = {key: data[key] for key in ("project", "session")}
        deadline = time.monotonic() + 900
        after = None
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("session listing exceeded the client deadline")
            page = self.request({"op": "sessions.list", "after": after})
            if not isinstance(page, dict) or not isinstance(page.get("sessions"), list):
                raise ValueError("invalid session list")
            if any(isinstance(session, dict) and session.get("project") == scope["project"]
                   and session.get("name") == scope["session"] and session.get("live") is True
                   for session in page["sessions"]):
                break
            following = page.get("next")
            if following is None:
                self.request({"op": "sessions.create", **scope})
                break
            if not isinstance(following, str) or not following or (after is not None and following <= after):
                raise ValueError("session list did not advance")
            after = following
        after = None
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("history read exceeded the client deadline")
            _, after, more = self.page(scope, after)
            if not more:
                break
        message_id = "local-" + uuid.uuid4().hex
        print("message_id=" + message_id, file=sys.stderr, flush=True)
        self.request({"op": "conversation.say", **scope, "message_id": message_id, "text": data["text"]})
        delivered = False
        while time.monotonic() < deadline:
            records, after, more = self.page(scope, after)
            for record in records:
                if record.get("kind") == "delivered" and record.get("id") == message_id:
                    delivered = True
                if not delivered:
                    continue
                kind = record.get("kind")
                if kind == "assistant":
                    display(record.get("text", ""))
                elif kind == "tool_result":
                    display("[tool] " + str(record.get("summary", ""))[:600])
                elif kind == "error":
                    raise ValueError(json.dumps(record, ensure_ascii=True))
                elif kind == "interrupted":
                    raise ValueError("session was interrupted")
                elif kind == "turn_complete":
                    return
            if not more:
                self.request({"op": "conversation.wait", **scope, "after_seq": after})
        raise TimeoutError("client wait expired; the session may still be running")


def main(mode):
    raw = sys.stdin.buffer.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("client input exceeds limit")
    data = json.loads(raw, object_pairs_hook=unique)
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(65)
        connection.connect("/run/nucleus-local/ui/ui.sock")
        with connection.makefile("rwb", buffering=0) as stream:
            client = Client(connection, stream)
            if mode == "chat":
                client.chat(data)
            elif mode == "request":
                print(json.dumps(client.request(data), ensure_ascii=True), flush=True)
            else:
                raise ValueError("unknown client mode")


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("expected chat or request")
        main(sys.argv[1])
    except (OSError, ValueError) as error:
        print("local client: " + str(error), file=sys.stderr)
        sys.exit(1)
