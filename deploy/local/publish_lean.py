#!/usr/bin/env python3
"""Publish the selected development toolchain for the independent verifier.

Only lean.py invokes this root installation phase, while guest.py holds the
operator lock. The operator's selected public sources and builds are trusted;
project processes cannot write that operator's home. Root copies them into a
root-owned, read-only tree, refuses escaping links, and publishes one version
pointer. Projects may read these public dependencies, never modify them. No
candidate proof or project directory is an installation input.

Both possible runtime units must be inactive before any link changes. The four
stable aliases refer through one atomic current pointer. An incomplete copy is
removed; existing completed versions remain. Initial alias creation may be
partial on failure, but no runtime starts here and the command fails. Updating
dependencies is an explicit stopped-runtime operation, never in-place recovery.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile

from prepare_platform import root_directory


def publish(environment, toolchain):
    if os.getuid() != 0:
        raise ValueError("Lean publication needs root")
    pins = json.loads(Path(__file__).with_name("lean.json").read_text())
    if json.loads((environment / "manifest.json").read_text()) != pins:
        raise ValueError("development environment does not match the selected pins")
    if (toolchain / ".nucleus-sha256").read_text() != pins["archives"][platform.machine()]["sha256"]:
        raise ValueError("development toolchain does not match the selected release")
    for unit in ("nucleus.service", "nucleus-local.service"):
        state = subprocess.check_output(["systemctl", "show", unit, "--property=ActiveState", "--value"],
                                        text=True).strip()
        if state not in ("inactive", "failed"):
            raise ValueError("stop the runtime before publishing Lean: " + unit)
    base = Path("/opt/nucleus")
    for path in (Path("/opt"), base, base / "lean-environments"):
        root_directory(path)
    encoded = json.dumps(pins, sort_keys=True).encode()
    destination = base / "lean-environments" / hashlib.sha256(encoded).hexdigest()
    aliases = {Path("/opt/lean"): base / "lean-current/lean",
               Path("/opt/mathlib"): base / "lean-current/mathlib",
               base / "repl": base / "lean-current/repl",
               base / "problems": base / "lean-current/mathlib/src"}
    for alias, target in aliases.items():
        if (alias.exists() or alias.is_symlink()) and (not alias.is_symlink() or alias.readlink() != target):
            raise ValueError("refusing to replace an unrelated installation: " + str(alias))
    current = base / "lean-current"
    if current.exists() or current.is_symlink():
        if not current.is_symlink() or current.resolve().parent != destination.parent:
            raise ValueError("refusing an unrelated Lean current pointer")
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix=".install-", dir=destination.parent) as temporary:
            staging = Path(temporary)
            for source, name in ((toolchain, "lean"), (environment / "putnambench/lean4", "mathlib")):
                shutil.copytree(source, staging / name, symlinks=True, ignore=shutil.ignore_patterns(".git"))
            shutil.copyfile(environment / "repl/.lake/build/bin/repl", staging / "repl")
            (staging / "repl").chmod(0o555)
            (staging / "manifest.json").write_bytes(encoded)
            for path in staging.rglob("*"):
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    path.resolve(strict=True).relative_to(staging)
                    os.chown(path, 0, 0, follow_symlinks=False)
                elif stat.S_ISDIR(mode) or stat.S_ISREG(mode):
                    path.chmod(0o555 if stat.S_ISDIR(mode) or mode & 0o111 else 0o444)
                    os.chown(path, 0, 0)
                else:
                    raise ValueError("unsupported toolchain file: " + str(path))
            staging.chmod(0o555)
            staging.rename(destination)
    root_directory(destination)
    if (destination / "manifest.json").read_bytes() != encoded:
        raise ValueError("published toolchain does not match pins")
    for alias, target in aliases.items():
        if not alias.is_symlink():
            alias.symlink_to(target)
    with tempfile.TemporaryDirectory(prefix=".activate-lean-", dir=base) as temporary:
        pointer = Path(temporary) / "next"
        pointer.symlink_to(destination, target_is_directory=True)
        pointer.replace(current)
    print(json.dumps({"published_lean": str(destination), "runtime_started": False}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("expected the prepared environment and toolchain directories")
    publish(Path(sys.argv[1]).resolve(strict=True), Path(sys.argv[2]).resolve(strict=True))
