#!/usr/bin/env python3
"""Operate a dedicated Lima VM from macOS; Linux still enforces confinement.

Usage (run from the checkout; commit intended source changes before sync):
    python3 tools/local_runtime.py start
    python3 tools/local_runtime.py sync
    python3 tools/local_runtime.py check --suite core
    python3 tools/local_runtime.py check --suite all
    python3 tools/local_runtime.py slice
    python3 tools/local_runtime.py stop

This operator tool can start/stop its VM and run trusted build scripts there. It
is not an agent capability or an interactive runtime composition. Host homes,
credentials, SSH agents and proxy environment are never shared. Only the named
commit's tree enters the VM, via git archive and a checked SHA-256; dirty files,
Git history/configuration and ignored local research are excluded. A commit may
itself contain private data: selecting it is the operator's disclosure decision.
The VM needs outbound network for its packages and build dependencies.

Failure is reported without a native fallback or automatic replay. Stopping
retains the VM disk. Build/test cancellation closes the SSH command, not a claim
that every guest descendant stopped; stop the dedicated VM when that is needed.
The guest helper owns snapshot publication and serializes operations. This is a
development bootstrap, not yet the Codex/Claude-backed resident composition.
"""

import argparse
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


def check_boundary(info):
    """Refuse an existing VM with ambient access to the host."""
    config = info["config"]
    ssh = config.get("ssh", {})
    forwards = config.get("portForwards", [])
    if (config.get("mounts") or config.get("copyToHost") or config.get("env")
            or ssh.get("forwardAgent") or ssh.get("loadDotSSHPubKeys")
            or config.get("propagateProxyEnv", True)
            or len(forwards) != 1 or forwards[0].get("ignore") is not True
            or forwards[0].get("guestIP") != "0.0.0.0" or forwards[0].get("proto") != "any"
            or forwards[0].get("guestIPMustBeZero", False)
            or forwards[0].get("guestPortRange") != [1, 65535]):
        raise ValueError("VM must have no mounts, SSH agent/keys, proxy environment or automatic port forwards; "
                         "use deploy/local/lima.yaml with a new --instance name")


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
    for name in ("start", "stop", "status", "build", "slice"):
        commands.add_parser(name)
    sync = commands.add_parser("sync", help="send only a committed source tree")
    sync.add_argument("--ref", default="HEAD")
    check = commands.add_parser("check")
    check.add_argument("--suite", choices=("core", "all"), default="core")
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
        check_boundary(info)
    if args.command == "start":
        if info:
            lima("start", args.instance)
        else:
            lima("start", "--name=" + args.instance, str(ROOT / "deploy/local/lima.yaml"))
        check_boundary(instance_info(args.instance))
        return 0
    if not info or info["status"] != "Running":
        raise ValueError("VM is not running; run start first")
    if args.command == "sync":
        with tempfile.TemporaryFile() as archive:
            commit, digest = archive_commit(ROOT, args.ref, archive)
            guest(args.instance, "install", digest, commit, stdin=archive)
    elif args.command == "check":
        guest(args.instance, "check", args.suite)
    else:
        guest(args.instance, args.command)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"local runtime: {error}", file=sys.stderr)
        sys.exit(1)
