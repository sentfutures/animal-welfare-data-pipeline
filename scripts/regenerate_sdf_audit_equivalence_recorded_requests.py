#!/usr/bin/env python3
"""Regenerate tests/fixtures/sdf_audit_equivalence/*.json (offline, canned replies, no model API).
Run only in a pull request that changes a judge request on purpose, and commit the files with it; a refactor never regenerates, and never to make a failing test pass.
Each file records the commit it was made from, for reviewers; no test checks it."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "tests"))

import sdf_audit_equivalence_fixture_support as support  # noqa: E402


def git(*arguments: str) -> str:
    """The output of a git command; stops the script when it fails, so a recorded commit is never empty."""
    completed = subprocess.run(["git", *arguments], cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.exit(f"Refusing to regenerate: `git {' '.join(arguments)}` failed with exit code "
                 f"{completed.returncode}: {completed.stderr.strip() or 'no error message'}")
    return completed.stdout.strip()


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    dirty = "with uncommitted changes" if git("status", "--porcelain", "--untracked-files=no") else "clean"
    metadata = {
        "recorded_from": f"commit {git('rev-parse', 'HEAD')} on branch {git('rev-parse', '--abbrev-ref', 'HEAD')} ({dirty})",
        "regeneration_rule": support.REGENERATION_RULE,
    }
    for existing in support.FIXTURE_DIRECTORY.glob("recorded_*_on_origin_main.json"):
        existing.unlink()   # drop recordings of cases that are not recorded this time
    count = 0
    with tempfile.TemporaryDirectory() as base_directory_name:
        base_directory = Path(base_directory_name)
        prepared_runs = support.prepare_all_runs(base_directory)
        for audit in support.AUDITS:
            for layout in support.RUN_LAYOUTS:
                name = support.case_name(audit.name, layout.name)
                print(f"=== recording {name}", flush=True)
                outcome = support.run_audit_in_isolation(
                    audit, layout, prepared_runs[layout.name], base_directory / "audits" / name)
                path = support.write_recorded_file(outcome, metadata)
                print(f"    {len(outcome.recorder.requests)} judge requests, ended with {outcome.ended_with}, "
                      f"{path.stat().st_size} bytes", flush=True)
                count += 1
    print(f"Wrote {count} recorded files under {support.FIXTURE_DIRECTORY}")


if __name__ == "__main__":
    main()
