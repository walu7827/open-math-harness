#!/usr/bin/env python3
"""Guest half of tools/local_runtime.py; run by its operator over Lima SSH.

The current source is an explicit committed snapshot, never a shared host mount.
An install verifies its whole archive before extraction, accepts ordinary files
and directories only, then publishes a new source directory through an atomic
symlink replacement. Failure leaves the prior current pointer intact and removes
the incomplete snapshot. Completed snapshots remain on the VM disk; stop never
deletes them. No session/model state or live authority is migrated by this tool.

A nonblocking operator lock serializes installs and checks, including the shared
Cargo build cache. The synthetic Git index exists only for repository checks;
it contains no history, remotes, hooks or host configuration. The original commit
and transferred archive hash are recorded outside the source tree. This helper
runs builds as the guest operator, who has sudo; source must be trusted for that
purpose. Runtime test programs still enter the real Nucleus sandboxes.
"""

import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path.home() / ".local/share/nucleus-local"


def run(*args, cwd=None):
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def install(data, digest, commit):
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("invalid source identity")
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("source archive hash mismatch")
    sources = ROOT / "sources"
    sources.mkdir(exist_ok=True)
    source = Path(tempfile.mkdtemp(prefix=commit[:12] + "-", dir=sources))
    published = False
    pointer = ROOT / "next"
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for item in archive.getmembers():
                path = PurePosixPath(item.name)
                if (path.is_absolute() or ".." in path.parts or not path.parts
                        or path.parts[0] in (".git", "target")
                        or not (item.isfile() or item.isdir())):
                    raise ValueError(f"unsupported source archive entry: {item.name}")
            archive.extractall(source, filter="data")
        run("git", "-c", "init.templateDir=", "init", "--quiet", cwd=source)
        run("git", "add", "--all", cwd=source)
        (ROOT / "target").mkdir(exist_ok=True)
        (source / "target").symlink_to(ROOT / "target", target_is_directory=True)
        # Manifest and source move together; no fake commit is created.
        (source / ".git/nucleus-source.json").write_text(json.dumps({"commit": commit, "sha256": digest}) + "\n")
        pointer.unlink(missing_ok=True)
        pointer.symlink_to(source, target_is_directory=True)
        pointer.replace(ROOT / "current")
        published = True
        print(json.dumps({"source": str(source), "commit": commit, "sha256": digest}), flush=True)
    finally:
        pointer.unlink(missing_ok=True)
        if not published:
            shutil.rmtree(source)


def main():
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (ROOT / "operator.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        action = sys.argv[1]
        if action == "install":
            install(sys.stdin.buffer.read(), sys.argv[2], sys.argv[3])
            return
        source = (ROOT / "current").resolve(strict=True)
        os.chdir(source)
        os.environ["PATH"] = str(Path.home() / ".cargo/bin") + ":/usr/local/bin:/usr/bin:/bin"
        os.environ["CARGO_TARGET_DIR"] = str(ROOT / "target")
        run("rustup", "show")  # rust-toolchain.toml is the single version source.
        if action == "build":
            run("cargo", "build", "--locked", "--workspace")
        elif action == "check":
            run("cargo", "fmt", "--all", "--check")
            if sys.argv[2] == "all":
                run("cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings")
            run("cargo", "build", "--locked", "--workspace")
            if sys.argv[2] == "all":
                run("cargo", "test", "--workspace")
            else:
                run("cargo", "test", "-p", "nucleus-e2e", "--test", "slice")
            run("python3", "-m", "unittest", "discover", "-s", "tools", "-p", "test_*.py")
            run("python3", "tools/check_markdown.py")
            if sys.argv[2] == "all":
                run("node", "--test", "tests/mcp/mcp.test.js")
        elif action == "slice":
            run("cargo", "run", "--locked", "--manifest-path", "cli/Cargo.toml", "--", "run", "examples/slice/config.json")
        else:
            raise ValueError("unknown guest action")


if __name__ == "__main__":
    main()
