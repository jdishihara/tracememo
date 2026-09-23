"""Read the current git commit and dirty state, tolerating non-repositories."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_info(cwd: Path) -> tuple[str | None, bool | None]:
    """Return ``(commit_sha, dirty)`` for the repository containing ``cwd``.

    Both are ``None`` when ``cwd`` is not inside a git repository or git is unavailable.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    return commit, bool(status.strip())
