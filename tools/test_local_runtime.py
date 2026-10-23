"""Source publication and the host/guest boundary; no model calls or VM needed."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import local_runtime


SPEC = importlib.util.spec_from_file_location("local_guest", local_runtime.ROOT / "deploy/local/guest.py")
GUEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUEST)


class LocalRuntime(unittest.TestCase):
    def test_archive_and_publication_send_only_the_selected_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, guest = root / "repo", root / "guest"
            repo.mkdir()
            guest.mkdir()
            def git(*args):
                subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
            git("init", "--quiet")
            (repo / "source.txt").write_text("committed source")
            (repo / ".gitignore").write_text("private/\n")
            git("add", "--all")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture")
            (repo / "source.txt").write_text("uncommitted research")
            (repo / "untracked.txt").write_text("untracked secret")
            (repo / "private").mkdir()
            (repo / "private/auth.json").write_text("credential fixture")
            with tempfile.TemporaryFile() as archive:
                commit, digest = local_runtime.archive_commit(repo, "HEAD", archive)
                data = archive.read()
            with patch.object(GUEST, "ROOT", guest):
                GUEST.install(data, digest, commit)
            source = (guest / "current").resolve()
            self.assertEqual((source / "source.txt").read_text(), "committed source")
            self.assertEqual({p.name for p in source.iterdir()}, {"source.txt", ".gitignore", ".git", "target"})
            manifest = json.loads((source / ".git/nucleus-source.json").read_text())
            self.assertEqual(manifest, {"commit": commit, "sha256": hashlib.sha256(data).hexdigest()})
            self.assertFalse((source / ".git/objects/info/alternates").exists())

    def test_failed_transfer_keeps_the_prior_snapshot_and_removes_partial_state(self):
        with tempfile.TemporaryDirectory() as directory:
            guest = Path(directory).resolve()
            previous = guest / "previous"
            previous.mkdir()
            (guest / "current").symlink_to(previous)
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode="w") as archive:
                link = tarfile.TarInfo("escape")
                link.type = tarfile.SYMTYPE
                link.linkname = "../../outside"
                archive.addfile(link)
            raw = data.getvalue()
            with patch.object(GUEST, "ROOT", guest):
                for digest in ("0" * 64, hashlib.sha256(raw).hexdigest()):
                    with self.subTest(digest=digest), self.assertRaises(ValueError):
                        GUEST.install(raw, digest, "a" * 40)
                    self.assertEqual((guest / "current").resolve(), previous)
                    self.assertFalse((guest / "next").exists())
                    self.assertEqual(list((guest / "sources").glob("*")), [])

    def test_existing_vm_with_host_access_is_refused(self):
        safe = {"mounts": [], "ssh": {"forwardAgent": False, "loadDotSSHPubKeys": False},
                "propagateProxyEnv": False, "plain": True, "portForwards": []}
        local_runtime.check_boundary({"config": safe})
        for key, value in (("mounts", [{"location": "~"}]), ("ssh", {"forwardAgent": True}),
                           ("ssh", {"loadDotSSHPubKeys": True}), ("propagateProxyEnv", True),
                           ("env", {"HOST_TOKEN": "fixture"}),
                           ("copyToHost", [{"guest": "/tmp/file", "host": "~/file"}]),
                           ("portForwards", [{"static": True, "guestPort": 8080}]),
                           ("plain", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                local_runtime.check_boundary({"config": {**copy.deepcopy(safe), key: value}})


if __name__ == "__main__":
    unittest.main()
