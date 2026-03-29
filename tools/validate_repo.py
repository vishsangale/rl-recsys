#!/usr/bin/env python3
"""Validate rl-recsys syntax, imports, and tests."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
IGNORE_DIRS = {".git", ".venv", ".venv-hydra", "__pycache__", ".pytest_cache"}


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(cmd, cwd=cwd, check=False, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def python_files() -> list[str]:
    return sorted(
        str(path)
        for path in ROOT.rglob("*.py")
        if not any(part in IGNORE_DIRS for part in path.parts)
    )


def main() -> int:
    run([sys.executable, "-m", "compileall", "-q", *python_files()])
    run(
        [
            sys.executable,
            "-c",
            (
                "import importlib; "
                "mods=['rl_recsys.data.pipelines.rl4rs','rl_recsys.data.pipelines.movielens','rl_recsys.data.pipelines.lastfm']; "
                "[importlib.import_module(m) for m in mods]"
            ),
        ],
        cwd=ROOT,
    )
    run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT)
    print("rl-recsys validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
