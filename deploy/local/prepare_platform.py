#!/usr/bin/env python3
"""Install and verify the Linux project-identity boundary in the local VM.

The guest operator explicitly invokes this trusted deployment tool through
guest.py's lock. Builds run unprivileged. A root child copies the selected
source's programs/resources into a root-owned immutable-by-project bundle,
then invokes the ordinary deploy/vm/activate --prepare-only checks. The same
setuid helper and policy serve the local and hosted compositions; no relaxed
project identity or sandbox fallback is introduced. No services are started.

Root trusts the operator's selected source and compiled outputs. A compromised
operator can already use sudo; projects cannot modify that operator's home.
Installation does not grant the runtime's group to the operator or projects.
The runtime can become only the projects user through its helper. Projects are
deliberately not isolated from each other. Existing workspace data is retained.

A bundle publishes by rename only after all copies match the source hashes.
Preparation can leave accounts/directories or switched links after failure,
as activate documents; nothing rolls back or silently retries. Finally, six
existing opt-in tests run as nucleus in a delegated systemd cgroup and verify
real uid boundaries, network cuts and descendant cleanup. Failure is reported;
it never produces a successful environment check.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import struct
import subprocess
import sys
import tempfile


SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / "tools"))
import package_compute


def run(*args):
    print("+ " + " ".join(map(str, args)), flush=True)
    subprocess.run(args, check=True, cwd=SOURCE)


def digest(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("bundle input is not an ordinary file: " + str(path))
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def root_directory(path):
    if not path.exists():
        path.mkdir(mode=0o755)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid or info.st_mode & 0o022:
        raise ValueError("deployment directory must be root-owned and not writable by others: " + str(path))


def install(test_binary):
    if os.getuid() != 0:
        raise ValueError("installation phase needs root")
    for path in (Path("/opt"), Path("/opt/nucleus"), Path("/opt/nucleus/runtimes")):
        root_directory(path)
    inputs = {"bin/" + name: SOURCE / "target/debug" / name for name in package_compute.BINARIES}
    for name in (*package_compute.DEPLOY, package_compute.GUIDE, *package_compute.NOTICES):
        inputs[name] = SOURCE / name
    for kind, directory in package_compute.SHARE.items():
        for path in (SOURCE / directory).rglob("*"):
            if path.is_file():
                inputs["share/" + kind + "/" + str(path.relative_to(SOURCE / directory))] = path
    inputs["tests/platform"] = test_binary
    machine = {"aarch64": 183, "x86_64": 62}[platform.machine()]
    for name, path in inputs.items():
        if name.startswith(("bin/", "tests/")):
            with path.open("rb") as stream:
                header = stream.read(64)
            if len(header) != 64 or header[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", header, 18)[0] != machine:
                raise ValueError("wrong guest executable architecture: " + name)
    source = json.loads((SOURCE / ".git/nucleus-source.json").read_text())
    manifest = {"version": 1, "source": source, "files": {name: digest(path) for name, path in inputs.items()}}
    encoded = json.dumps(manifest, sort_keys=True).encode()
    destination = Path("/opt/nucleus/runtimes") / hashlib.sha256(encoded).hexdigest()
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix=".install-", dir=destination.parent) as temporary:
            staging = Path(temporary)
            for name, path in inputs.items():
                output = staging / name
                output.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                shutil.copyfile(path, output, follow_symlinks=False)
                output.chmod(0o755 if name.startswith(("bin/", "tests/")) else 0o644)
                if digest(output) != manifest["files"][name]:
                    raise ValueError("build output changed during installation: " + name)
            (staging / "manifest.json").write_bytes(encoded)
            staging.chmod(0o755)
            staging.rename(destination)
    if (destination / "manifest.json").read_bytes() != encoded:
        raise ValueError("existing bundle manifest differs")
    run("sh", str(destination / "deploy/vm/activate"), "--prepare-only")
    run("systemd-run", "--wait", "--pipe", "--collect", "-p", "User=nucleus", "-p", "Group=nucleus",
        "-p", "Delegate=yes", "-p", "RuntimeMaxSec=180", "-p", "TimeoutStopSec=15",
        str(destination / "tests/platform"), "--ignored", "--test-threads=1")
    print(json.dumps({"bundle": str(destination), "source": source["commit"],
                      "identity_and_cgroup_tests": "passed", "resident_service": "not installed"}), flush=True)


def main():
    if sys.argv[1:] and sys.argv[1] == "--install":
        if len(sys.argv) != 3:
            raise ValueError("--install needs the compiled platform test binary")
        install(Path(sys.argv[2]))
        return
    if sys.argv[1:] or os.getuid() == 0:
        raise ValueError("invoke the build phase as the unprivileged guest operator")
    run("cargo", "build", "--locked", "--workspace")
    result = subprocess.check_output(["cargo", "test", "--locked", "-p", "nucleus-platform", "--lib",
                                      "--no-run", "--message-format=json"], cwd=SOURCE, text=True)
    binaries = [item["executable"] for line in result.splitlines() if (item := json.loads(line)).get("reason")
                == "compiler-artifact" and item.get("profile", {}).get("test") and item.get("executable")]
    if len(binaries) != 1:
        raise ValueError("Cargo did not identify exactly one platform test binary")
    run("sudo", "python3", str(Path(__file__).resolve()), "--install", binaries[0])


if __name__ == "__main__":
    main()
