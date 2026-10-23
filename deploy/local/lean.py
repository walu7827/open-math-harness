#!/usr/bin/env python3
"""Build the pinned public Lean/Mathlib/REPL environment in the dedicated VM.

Run as the guest operator, never as root. lean.json pins an official Lean
release by hash and both source repositories by commit; PutnamBench's unchanged
lockfile pins Mathlib and its transitive dependencies. No project or private
research is read. Only dependencies selected by those manifests are built.
This is trusted build tooling with the operator's authority, not an agent tool.

The official archive is hashed before extraction into a private staging path.
Each completed installation is published by rename. Failure removes the new
staging directory and preserves an existing completed version; it does not
retry network operations or reuse a partially built tree. An operator lock
belongs to guest.py, which invokes this script. Normal mathlib cache files can
remain in the operator's home after failure. These user-owned development
assets are for tests; a resident verifier must receive root-owned copies/binds
that project processes cannot modify. No claim of that deployment is made here.

The printed JSON supplies PB_LEAN, PB_FROZEN, NUCLEUS_REPL and PATH for the
repository's explicit opt-in tests. The conventional ~/.elan/toolchains path
matches their sandbox mount layout; installing elan or changing its default
toolchain is unnecessary. The tests still use the real Lean binary and REPL.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import urllib.request


def run(*args, cwd=None, env=None):
    print("+ " + " ".join(map(str, args)), flush=True)
    subprocess.run(args, cwd=cwd, env=env, check=True)


def checkout(spec, target, env):
    run("git", "-c", "init.templateDir=", "init", "--quiet", str(target), env=env)
    run("git", "remote", "add", "origin", spec["url"], cwd=target, env=env)
    run("git", "fetch", "--depth=1", "origin", spec["commit"], cwd=target, env=env)
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=target, env=env)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, env=env, text=True).strip()
    if actual != spec["commit"]:
        raise ValueError("dependency checkout does not match pin")


def main():
    if os.getuid() == 0:
        raise ValueError("Lean build must run as the unprivileged guest operator")
    pins = json.loads(Path(__file__).with_suffix(".json").read_text())
    if pins["version"] != 1:
        raise ValueError("unsupported toolchain manifest version")
    cache = Path.home() / ".local/share/nucleus-local/toolchains"
    cache.mkdir(parents=True, exist_ok=True)
    # The complete import Mathlib cache is large. Refuse before downloading;
    # the operator can explicitly enlarge the VM disk, never delete their data.
    identity = hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()
    target = cache / ("lean-environment-" + identity[:16])
    version = pins["lean"]
    toolchain = Path.home() / ".elan/toolchains" / ("nucleus-lean-" + version)
    archive = pins["archives"][platform.machine()]
    if not toolchain.exists():
        if shutil.disk_usage(cache).free < 20 * 1024**3:
            raise ValueError("Lean setup needs at least 20 GiB free in the VM; enlarge its disk first")
        toolchain.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".lean-", dir=toolchain.parent) as temporary:
            staging = Path(temporary)
            package = staging / archive["name"]
            url = f"https://github.com/leanprover/lean4/releases/download/v{version}/{archive['name']}"
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response, package.open("wb") as output:
                size = 0
                while data := response.read(1024 * 1024):
                    size += len(data)
                    if size > 1024**3:
                        raise ValueError("Lean archive exceeds 1 GiB")
                    digest.update(data)
                    output.write(data)
            if digest.hexdigest() != archive["sha256"]:
                raise ValueError("Lean release archive hash mismatch")
            run("tar", "--zstd", "-xf", str(package), "-C", str(staging))
            extracted = staging / archive["name"].removesuffix(".tar.zst")
            (extracted / ".nucleus-sha256").write_text(archive["sha256"])
            extracted.rename(toolchain)
    if (toolchain / ".nucleus-sha256").read_text() != archive["sha256"]:
        raise ValueError("existing Lean installation has a different source digest")
    env = {**os.environ, "PATH": str(toolchain / "bin") + ":" + os.environ["PATH"],
           "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    run(str(toolchain / "bin/lean"), "--version", env=env)
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix=".lean-environment-", dir=cache) as temporary:
            staging = Path(temporary)
            for name in ("putnambench", "repl"):
                checkout(pins[name], staging / name, env)
            pb = staging / "putnambench/lean4"
            expected = "leanprover/lean4:v" + version
            for path in (pb / "lean-toolchain", staging / "repl/lean-toolchain"):
                if path.read_text().strip() != expected:
                    raise ValueError("pinned sources require different Lean versions")
            before = (pb / "lake-manifest.json").read_bytes()
            run("lake", "exe", "cache", "get", cwd=pb, env=env)
            if (pb / "lake-manifest.json").read_bytes() != before:
                raise ValueError("dependency resolution changed the pinned benchmark lockfile")
            run("lake", "build", "repl", cwd=staging / "repl", env=env)
            (staging / "manifest.json").write_text(json.dumps(pins, sort_keys=True) + "\n")
            # Move the children as one tree; temporary directory cleanup is
            # harmless after its path has been renamed.
            staging.rename(target)
    if json.loads((target / "manifest.json").read_text()) != pins:
        raise ValueError("existing environment does not match pinned sources")
    result = {"PB_LEAN": str(target / "putnambench/lean4"),
              "PB_FROZEN": str(target / "putnambench/lean4/src"),
              "NUCLEUS_REPL": str(target / "repl/.lake/build/bin/repl"),
              "PATH_PREFIX": str(toolchain / "bin")}
    pointer = cache / "lean-current.next"
    pointer.write_text(json.dumps(result) + "\n")
    pointer.replace(cache / "lean-current.json")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
