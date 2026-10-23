#!/usr/bin/env python3
"""Operate a dedicated Lima VM from macOS; Linux still enforces confinement.

Usage (run from the checkout; commit intended source changes before sync):
    python3 tools/local_runtime.py start
    python3 tools/local_runtime.py sync
    python3 tools/local_runtime.py check --suite core
    python3 tools/local_runtime.py check --suite all
    python3 tools/local_runtime.py slice
    python3 tools/local_runtime.py model-smoke --codex /path/to/codex
    python3 tools/local_runtime.py stop

This operator tool can start/stop its VM, run trusted build scripts there, and
own the local resident composition's host bridge. It is not an agent capability. Host homes,
credentials, SSH agents and proxy environment are never shared. Only the named
commit's tree enters the VM, via git archive and a checked SHA-256; dirty files,
Git history/configuration and ignored local research are excluded. A commit may
itself contain private data: selecting it is the operator's disclosure decision.
The VM needs outbound network for its packages and build dependencies.

Failure is reported without a native fallback or automatic replay. Stopping
retains the VM disk. Build/test cancellation closes the SSH command, not a claim
that every guest descendant stopped; stop the dedicated VM when that is needed.
The guest helper owns snapshot publication and serializes operations. This is a
development and local deployment entry point. The resident path uses Codex;
Claude is not implemented yet.
Plain mode also removes Lima's guest-agent vsock tunnel, which would otherwise
bypass an offline project's network namespace. `harden` migrates the earlier
development profile: disable that guest service, stop the VM, change only its
plain/forwarding settings, then restart and verify. Failure leaves the last
completed step in place and is reported; no rollback or automatic retry.
"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCE = "open-math-harness"


def host_env():
    # Lima otherwise propagates host proxy variables during provisioning.
    return {key: os.environ[key] for key in ("HOME", "PATH", "TMPDIR") if key in os.environ}


def lima(*args, **kwargs):
    return subprocess.run(["limactl", "--tty=false", *args], env=host_env(), check=True, **kwargs)


def instance_info(name):
    result = lima("list", "--json", capture_output=True, text=True)
    return next((item for line in result.stdout.splitlines()
                 if (item := json.loads(line))["name"] == name), None)


def check_boundary(info, migrating=False):
    """Refuse an existing VM with ambient access to the host."""
    config = info["config"]
    ssh = config.get("ssh", {})
    forwards = config.get("portForwards", [])
    if (config.get("mounts") or config.get("copyToHost") or config.get("env")
            or ssh.get("forwardAgent") or ssh.get("loadDotSSHPubKeys")
            or config.get("propagateProxyEnv", True)):
        raise ValueError("VM must have no mounts, SSH agent/keys, proxy environment or automatic port forwards; "
                         "use deploy/local/lima.yaml with a new --instance name")
    if migrating and not config.get("plain"):
        if (len(forwards) == 1 and forwards[0].get("ignore") is True
                and forwards[0].get("guestIP") == "0.0.0.0" and forwards[0].get("proto") == "any"
                and not forwards[0].get("guestIPMustBeZero", False)
                and forwards[0].get("guestPortRange") == [1, 65535]):
            return
    if config.get("plain") is not True or forwards:
        raise ValueError("VM needs plain mode with no port forwards; run harden for the earlier development profile")


def check_guest_boundary(name):
    # Do not infer enforcement from YAML: a previously enabled service can
    # survive a switch to plain mode. Probe the actual guest as root.
    probe = """import subprocess
state = subprocess.run(['systemctl', 'is-active', '--quiet', 'lima-guestagent.service'])
if state.returncode == 0:
    raise SystemExit('Lima guest agent is running; run harden before using the runtime')
lines = subprocess.check_output(['ss', '--vsock', '--listening', '--numeric', '--processes'], text=True).splitlines()
if len(lines) != 1:
    raise SystemExit('unexpected vsock listener; refusing local runtime operations')
"""
    lima("shell", "--workdir=/", name, "sudo", "python3", "-c", probe)


def archive_commit(root, revision, output):
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", revision + "^{commit}"], cwd=root, text=True
    ).strip()
    subprocess.run(["git", "archive", "--format=tar", commit], cwd=root, stdout=output, check=True)
    output.seek(0)
    digest = hashlib.sha256()
    while chunk := output.read(1024 * 1024):
        digest.update(chunk)
    output.seek(0)
    return commit, digest.hexdigest()


def guest(name, *args, **kwargs):
    # A fixed helper, passed as an argument rather than interpolated shell text.
    helper = (ROOT / "deploy/local/guest.py").read_text()
    return lima("shell", "--workdir=/", name, "python3", "-c", helper, *args, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", default=DEFAULT_INSTANCE)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "stop", "status", "build", "slice", "harden", "prepare-lean", "prepare-platform"):
        commands.add_parser(name)
    sync = commands.add_parser("sync", help="send only a committed source tree")
    sync.add_argument("--ref", default="HEAD")
    check = commands.add_parser("check")
    check.add_argument("--suite", choices=("core", "all", "lean"), default="core")
    for name, description in (("model-smoke", "live Codex tool/denial/session test; consumes account usage"),
                              ("serve", "run the persistent composition with the host Codex account")):
        model = commands.add_parser(name, help=description)
        model.add_argument("--codex", default="codex")
        model.add_argument("--model", help="optional operator-selected Codex model")
    chat = commands.add_parser("chat", help="talk to the running local composition")
    chat.add_argument("--project", default="default")
    chat.add_argument("--session", default="default")
    chat.add_argument("--message", help="one message; omission opens a text prompt")
    commands.add_parser("request", help="send one JSON UI operation from stdin")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", args.instance):
        parser.error("--instance must be a lowercase name, at most 63 characters")
    info = instance_info(args.instance)
    if args.command == "status":
        print(json.dumps({"instance": args.instance, "status": info["status"] if info else "Absent"}))
        return 0
    if args.command == "stop":
        if info and info["status"] != "Stopped":
            lima("stop", args.instance)
        return 0
    if info:
        check_boundary(info, migrating=args.command == "harden")
    if args.command == "harden":
        if not info:
            raise ValueError("no existing VM to harden; run start")
        if info["status"] != "Running":
            lima("start", args.instance)
        lima("shell", "--workdir=/", args.instance, "sudo", "sh", "-c",
             "if systemctl cat lima-guestagent.service >/dev/null 2>&1; then systemctl disable --now lima-guestagent.service; fi")
        lima("stop", args.instance)
        lima("edit", args.instance, "--set=.plain = true | .portForwards = []")
        lima("start", args.instance)
        check_boundary(instance_info(args.instance))
        check_guest_boundary(args.instance)
        return 0
    if args.command == "start":
        if info:
            lima("start", args.instance)
        else:
            lima("start", "--name=" + args.instance, str(ROOT / "deploy/local/lima.yaml"))
        check_boundary(instance_info(args.instance))
        check_guest_boundary(args.instance)
        return 0
    if not info or info["status"] != "Running":
        raise ValueError("VM is not running; run start first")
    check_guest_boundary(args.instance)
    if args.command == "sync":
        with tempfile.TemporaryFile() as archive:
            commit, digest = archive_commit(ROOT, args.ref, archive)
            guest(args.instance, "install", digest, commit, stdin=archive)
    elif args.command == "check":
        guest(args.instance, "check", args.suite)
    elif args.command == "model-smoke":
        import local_model_smoke
        asyncio.run(local_model_smoke.run(args.instance, args.codex, args.model))
    elif args.command == "serve":
        import local_serve
        asyncio.run(local_serve.run(args.instance, args.codex, args.model))
    elif args.command in ("chat", "request"):
        client = (ROOT / "deploy/local/client.py").read_text()
        def send(data):
            lima("shell", "--workdir=/", args.instance, "sudo", "-u", "nucleus", "python3", "-c", client,
                 args.command, input=data)
        if args.command == "request":
            data = sys.stdin.buffer.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise ValueError("client input exceeds limit")
            send(data)
        else:
            def say(message):
                send(json.dumps({"project": args.project, "session": args.session, "text": message}).encode())
            if args.message is not None:
                say(args.message)
            else:
                print("Local chat. /exit leaves the client; the runtime stays running.")
                while True:
                    try:
                        message = input("You: ")
                    except EOFError:
                        break
                    if message == "/exit":
                        break
                    if message.strip():
                        say(message)
    else:
        guest(args.instance, args.command)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        # The SSH argv contains the entire guest helper, not a useful error.
        print(f"local runtime: command failed with exit code {error.returncode}; see output above", file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError) as error:
        print(f"local runtime: {error}", file=sys.stderr)
        sys.exit(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("local runtime: command interrupted", file=sys.stderr)
        sys.exit(130)
