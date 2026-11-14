"""The git helper the prompt-equivalence and audit-equivalence regeneration scripts share."""

import subprocess
import sys
from pathlib import Path


def git_output_or_refuse_to_regenerate(repository: Path, *arguments: str) -> str:
    """The output of a git command in ``repository``; stops the script when it fails, so a recorded commit is never empty."""
    completed = subprocess.run(["git", *arguments], cwd=repository, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.exit(f"Refusing to regenerate: `git {' '.join(arguments)}` failed in {repository} with exit code "
                 f"{completed.returncode}: {completed.stderr.strip() or 'no error message'}")
    return completed.stdout.strip()
