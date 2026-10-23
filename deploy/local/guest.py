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
and transferred archive hash are recorded outside the tracked source tree. This helper
runs builds as the guest operator, who has sudo; source must be trusted for that
purpose. Runtime test programs still enter the real Nucleus sandboxes.
"""

import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request


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


def node_for_ci(source):
    """Install CI's exact Node release in the guest cache, without root.

    The workflow is the version source. HTTPS fetches only nodejs.org release
    files; verify the archive against that release's SHA-256 list before
    extraction. An incomplete installation is never published. This is build
    tooling, not a Node installation for sandboxed runtime programs.
    """
    versions = re.findall(r"node-version:\s*([0-9]+\.[0-9]+\.[0-9]+)",
                          (source / ".github/workflows/ci.yml").read_text())
    if len(versions) != 1:
        raise ValueError("CI must select exactly one explicit Node version")
    arch = {"aarch64": "arm64", "x86_64": "x64"}[platform.machine()]
    name = f"node-v{versions[0]}-linux-{arch}"
    cache = ROOT / "toolchains"
    destination = cache / name
    binary = destination / "bin/node"
    if not destination.exists():
        cache.mkdir(exist_ok=True)
        url = f"https://nodejs.org/dist/v{versions[0]}/"
        filename = name + ".tar.xz"
        with urllib.request.urlopen(url + "SHASUMS256.txt", timeout=60) as response:
            hashes = dict(line.split()[::-1] for line in response.read().decode().splitlines())
        with urllib.request.urlopen(url + filename, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != hashes[filename]:
            raise ValueError("Node release checksum mismatch")
        with tempfile.TemporaryDirectory(dir=cache) as staging:
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                archive.extractall(staging, filter="data")
            (Path(staging) / name).rename(destination)
    actual = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if actual != "v" + versions[0]:
        raise ValueError("cached Node does not match CI")
    print(f"Node for CI: {actual}", flush=True)
    return str(binary)


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
        if action == "prepare-lean":
            run("python3", str(source / "deploy/local/lean.py"))
            return
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
                run(node_for_ci(source), "--test", "tests/mcp/mcp.test.js")
        elif action == "slice":
            run("cargo", "run", "--locked", "--manifest-path", "cli/Cargo.toml", "--", "run", "examples/slice/config.json")
        else:
            raise ValueError("unknown guest action")


if __name__ == "__main__":
    main()
