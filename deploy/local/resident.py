#!/usr/bin/env python3
"""Own one local resident runtime for one dedicated SSH operator session.

guest.py holds the operator lock and invokes this root deployment process.
Only the fresh broker token is read from stdin; CLI credentials never enter the
VM. The token is bound only into the model provider. Persistent conversations,
memory, approvals and run records remain in the existing runtime data paths.
The configuration is root-owned; each runtime service still runs as nucleus
under bubblewrap, and project processes use the installed identity helper.
The installed bundle must name this exact committed source snapshot; syncing
different source requires prepare-platform before it can start. A mismatch
fails before a broker token is read or any runtime state is acquired.

systemd owns the runtime and its delegated cgroup. BindsTo plus After binds the
unit to this SSH login's session scope: even if this helper is killed, ending
the dedicated SSH session stops the unit and its descendants. A caller outside
a real login scope is refused. EOF and signals request a clean stop; there is
no service restart or call replay. Stop has a deadline and failure is reported.

/run/nucleus-local is exclusively owned ephemeral state, identified by a root
marker. A later start may remove that state only with no local unit present.
It never removes persistent user data. Independent cleanup still removes the
token if stopping fails; that does not undo remote model calls or their usage.
The host must also close its broker and reverse forward when this process ends.
"""

import json
import os
from pathlib import Path
import pwd
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import composition
from prepare_platform import root_directory


SOURCE = Path(__file__).resolve().parents[2]
RUNTIME = Path("/run/nucleus-local")
UNIT = "nucleus-local.service"
MARKER = {"version": 1, "kind": "nucleus-local-ephemeral"}


def property_of(unit, name):
    return subprocess.check_output(["systemctl", "show", unit, "--property=" + name, "--value"],
                                   text=True, timeout=10).strip()


def main(port, lean_path):
    if os.getuid() != 0:
        raise ValueError("local runtime deployment needs root")
    scope = next((part for part in Path("/proc/self/cgroup").read_text().strip().split("/")
                  if re.fullmatch(r"session-[0-9]+\.scope", part)), None)
    if scope is None or property_of(scope, "ActiveState") != "active":
        raise ValueError("a dedicated active SSH login scope must own the runtime")
    if property_of(UNIT, "LoadState") != "not-found":
        raise ValueError("a local runtime unit already exists")
    if property_of("nucleus.service", "ActiveState") not in ("inactive", "failed"):
        raise ValueError("the hosted runtime is active")
    bundle = Path("/opt/nucleus/bin/nucleus").resolve(strict=True).parents[1]
    root_directory(bundle)
    installed = json.loads((bundle / "manifest.json").read_text())
    selected = json.loads((SOURCE / ".git/nucleus-source.json").read_text())
    if installed.get("version") != 1 or installed.get("source") != selected:
        raise ValueError("installed runtime differs from the selected source; run prepare-platform")
    key = sys.stdin.buffer.readline(258).strip()
    if not re.fullmatch(rb"[A-Za-z0-9_-]{32,128}", key):
        raise ValueError("invalid per-run broker token")
    identity = pwd.getpwnam("nucleus")
    if RUNTIME.exists() or RUNTIME.is_symlink():
        root_directory(RUNTIME)
        if json.loads((RUNTIME / "owner.json").read_text()) != MARKER:
            raise ValueError("unrecognized local runtime state")
        shutil.rmtree(RUNTIME)
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    def watch_operator():
        # The process owns this reader. A raw descriptor wait holds no Python
        # buffered-I/O lock when a startup failure exits the main thread.
        os.read(sys.stdin.fileno(), 1)
        stopped.set()
    threading.Thread(target=watch_operator, daemon=True).start()
    attempted = False
    runtime_owned = False
    ownership = "Nucleus local " + uuid.uuid4().hex
    try:
        RUNTIME.mkdir(mode=0o750)
        runtime_owned = True
        os.chown(RUNTIME, 0, identity.pw_gid)
        (RUNTIME / "owner.json").write_text(json.dumps(MARKER))
        model = RUNTIME / "model"
        model.mkdir(mode=0o750)
        os.chown(model, 0, identity.pw_gid)
        secret = model / "key"
        secret.write_bytes(key)
        os.chown(secret, 0, identity.pw_gid)
        secret.chmod(0o640)
        ui = RUNTIME / "ui"
        ui.mkdir(mode=0o700)
        os.chown(ui, identity.pw_uid, identity.pw_gid)
        for path in (Path("/etc"), Path("/etc/nucleus"), Path("/etc/nucleus/policy")):
            root_directory(path)
        policy = Path("/etc/nucleus/policy/approvals.json")
        expected = (SOURCE / "deploy/vm/approvals.json").read_bytes()
        if policy.is_symlink():
            raise ValueError("approval policy must not be a symbolic link")
        if policy.exists():
            info = policy.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid or info.st_mode & 0o022
                    or policy.read_bytes() != expected):
                raise ValueError("existing approval policy differs or is not root-controlled")
        if not policy.exists():
            policy.write_bytes(expected)
            policy.chmod(0o644)
        config = composition.make(SOURCE, lean_path, port)
        if (json.loads(Path("/opt/nucleus/lean-current/manifest.json").read_text())
                != json.loads((SOURCE / "deploy/local/lean.json").read_text())):
            raise ValueError("published Lean dependencies do not match this source")
        target = Path("/etc/nucleus/local.json")
        with tempfile.TemporaryDirectory(prefix=".local-", dir=target.parent) as temporary:
            prepared = Path(temporary) / "config.json"
            prepared.write_text(json.dumps(config))
            os.chown(prepared, 0, identity.pw_gid)
            prepared.chmod(0o640)
            prepared.replace(target)
        states = ("nucleus nucleus/sessions nucleus/memories nucleus/memory-state "
                  "nucleus/approvals-log nucleus/jobs")
        properties = ["Type=exec", "User=nucleus", "Group=nucleus", "Delegate=yes", "KillMode=control-group",
                      "Restart=no", "TimeoutStopSec=30", "OOMPolicy=continue", "UMask=0077",
                      "StateDirectoryMode=0700", "StateDirectory=" + states,
                      "BindsTo=" + scope, "After=" + scope, "Description=" + ownership]
        # Keep a failed transient unit available for its exit status and
        # diagnostics. A successful stopped unit is collected by systemd.
        command = ["systemd-run", "--quiet", "--unit=" + UNIT]
        for prop in properties:
            command += ["--property=" + prop]
        command += ["/opt/nucleus/bin/nucleus", "serve", str(target)]
        if stopped.is_set():
            raise InterruptedError("operator disconnected before runtime start")
        attempted = True
        subprocess.run(command, check=True, timeout=30)
        deadline = time.monotonic() + 330
        ready = False
        while not stopped.wait(1):
            if property_of(UNIT, "ActiveState") not in ("active", "activating"):
                raise RuntimeError("local runtime exited unexpectedly; inspect journalctl -u nucleus-local")
            if not ready:
                try:
                    with socket.socket(socket.AF_UNIX) as connection:
                        connection.settimeout(2)
                        connection.connect(str(ui / "ui.sock"))
                        connection.sendall(b'{"id":1,"op":"whoami"}\n')
                        with connection.makefile("rb") as stream:
                            answer = json.loads(stream.readline(1024 * 1024))
                        if not isinstance(answer, dict) or answer.get("id") != 1 or "ok" not in answer:
                            raise ValueError("local UI did not confirm readiness")
                    ready = True
                    print(json.dumps({"event": "ready", "model": composition.MODEL,
                                      "unit": UNIT, "usd_cost": "unknown"}), flush=True)
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("local runtime did not become ready")
    finally:
        failure = None
        try:
            if attempted and property_of(UNIT, "Description") == ownership:
                subprocess.run(["systemctl", "stop", UNIT], check=True, timeout=45)
                if property_of(UNIT, "ActiveState") != "inactive":
                    raise RuntimeError("local runtime did not shut down cleanly")
        except BaseException as error:
            failure = error
        try:
            if runtime_owned:
                shutil.rmtree(RUNTIME)
        except BaseException as error:
            failure = failure or error
        if failure is not None:
            raise failure


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("expected the broker port and published Lean import path")
    main(int(sys.argv[1]), sys.argv[2])
