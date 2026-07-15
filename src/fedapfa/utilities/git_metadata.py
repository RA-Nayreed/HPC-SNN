"""Capture enough Git state to reject incompatible run resumption."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def _run(*args: str, binary: bool = False):
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=not binary,
        check=False,
    ).stdout


def git_metadata():
    commit = _run("rev-parse", "HEAD").strip() or None
    status = _run("status", "--porcelain=v1", "-z", binary=True)
    diff = _run("diff", "--binary", "HEAD", binary=True)
    untracked = _run("ls-files", "--others", "--exclude-standard", "-z", binary=True)
    digest = hashlib.sha256()
    digest.update(diff)
    digest.update(untracked)
    for raw_path in sorted(item for item in untracked.split(b"\0") if item):
        path = Path(raw_path.decode(errors="surrogateescape"))
        if path.is_file():
            digest.update(raw_path)
            digest.update(path.read_bytes())
    return {
        "commit": commit,
        "dirty": bool(status),
        "worktree_sha256": digest.hexdigest() if status else None,
    }
