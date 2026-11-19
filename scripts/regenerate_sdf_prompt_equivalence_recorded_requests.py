#!/usr/bin/env python3
"""Regenerate tests/fixtures/sdf_prompt_equivalence/*.json (offline, canned replies, no model API).
Run only in a pull request that changes requests on purpose, and commit the files with it; a refactor never regenerates, and never to make a failing test pass.
Each file records the commit it was made from, for reviewers; no test checks it."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "tests"))

import sdf_prompt_equivalence_fixture_support as support  # noqa: E402
import sdf_prompt_equivalence_scenarios as scenarios  # noqa: E402


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
    all_scenarios = scenarios.all_scenarios()
    for existing in support.FIXTURE_DIRECTORY.glob("recorded_*_on_origin_main.json"):
        existing.unlink()   # a removed scenario leaves no stale recording
    for scenario in all_scenarios:
        print(f"=== recording scenario {scenario.name}", flush=True)
        with tempfile.TemporaryDirectory() as working_directory:
            outcome = support.run_scenario_in_isolation(scenario, Path(working_directory))
            support.write_recorded_file(outcome, metadata)
    print(f"Wrote {len(all_scenarios)} recorded files under {support.FIXTURE_DIRECTORY}")


if __name__ == "__main__":
    main()
