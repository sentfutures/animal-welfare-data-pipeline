#!/usr/bin/env python3
"""Regenerate the files that tests/test_sdf_audit_equivalence.py compares against.

This is the ONLY code that rewrites tests/fixtures/sdf_audit_equivalence/*.json (one file per audit
and run layout); the test never does. Run it in the same pull request as an intended change to a
judge request (an audit prompt, schema, model or setting), and commit the new recorded files with
that change so their diff shows reviewers which requests changed. A pull request that only
refactors never runs it: there, a failing test means the refactor changed a judge request. The
recorded files are meant to be recorded on origin/main behaviour.

No model API is called and no key is read: every provider client is replaced by a recorder that
returns canned replies (tests/sdf_audit_equivalence_fixture_support.py), with fake keys.

    python scripts/regenerate_sdf_audit_equivalence_recorded_requests.py
"""

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
    return subprocess.run(["git", *arguments], cwd=REPOSITORY_ROOT, capture_output=True, text=True).stdout.strip()


def main() -> None:
    dirty = "with uncommitted changes" if git("status", "--porcelain", "--untracked-files=no") else "clean"
    metadata = {
        "recorded_from": f"commit {git('rev-parse', 'HEAD')} on branch {git('rev-parse', '--abbrev-ref', 'HEAD')} ({dirty})",
        "regeneration_rule": support.REGENERATION_RULE,
    }
    for existing in support.FIXTURE_DIRECTORY.glob("recorded_*_on_origin_main.json"):
        existing.unlink()   # a case that no longer exists must not leave a stale file
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
    # A parser with no options, so that --help prints this text and exits instead of rewriting the recordings.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
