#!/usr/bin/env python3
"""Regenerate the files that tests/test_sdf_prompt_equivalence_and_final_corpus.py compares against.

This is the ONLY code that rewrites tests/fixtures/sdf_prompt_equivalence/*.json (one file per
scenario); the test never does. Run it in the same pull request as an intended request change (a
prompt edit, a config change, a model change), and commit the new recorded files with that change
so their diff shows reviewers which requests changed. A pull request that only refactors never runs
it: there, a failing test means the refactor changed a request. The recorded
files are meant to be recorded on origin/main.

Rule: whether to regenerate depends only on intent. A pull request meant to change requests
regenerates and commits the files in that pull request, whether or not it also touches Python; a
pull request not meant to change requests never regenerates, so any difference fails. This script
therefore does not refuse because Python differs from origin/main. Each recorded file still stores
the commit it was recorded from and the Python files under sdf_pipeline/, shared/ and dad_pipeline/
that differed from origin/main, as information for reviewers only; no test checks that list. The
script still stops when a git command fails or `origin/main` cannot be resolved, because the list
would then be wrong.

No model API is called and no key is read: every provider client is replaced by a recorder that
returns canned replies (tests/sdf_prompt_equivalence_fixture_support.py), with fake keys.

    python scripts/regenerate_sdf_prompt_equivalence_recorded_requests.py
"""

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

PYTHON_DIRECTORIES_THE_REQUESTS_DEPEND_ON = ("sdf_pipeline", "shared", "dad_pipeline")


def git(*arguments: str) -> str:
    """The output of a git command; stops the script when the command fails, because a failed
    command prints nothing and an empty answer would read as "no Python file differs"."""
    completed = subprocess.run(["git", *arguments], cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.exit(f"Refusing to regenerate: `git {' '.join(arguments)}` failed with exit code "
                 f"{completed.returncode}: {completed.stderr.strip() or 'no error message'}")
    return completed.stdout.strip()


def python_files_differing_from_origin_main() -> list[str]:
    if subprocess.run(["git", "rev-parse", "--verify", "--quiet", "origin/main^{commit}"], cwd=REPOSITORY_ROOT,
                      capture_output=True, text=True).returncode != 0:
        sys.exit("Refusing to regenerate: `origin/main` cannot be resolved in this repository, so the "
                 "list of differing Python files cannot be recorded. Fetch it (`git fetch origin main`) and run again.")
    committed = git("diff", "--name-only", "origin/main", "--", *PYTHON_DIRECTORIES_THE_REQUESTS_DEPEND_ON).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *PYTHON_DIRECTORIES_THE_REQUESTS_DEPEND_ON).splitlines()
    return sorted(name for name in {*committed, *untracked} if name.endswith(".py"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    differing = python_files_differing_from_origin_main()
    dirty = "with uncommitted changes" if git("status", "--porcelain", "--untracked-files=no") else "clean"
    metadata = {
        "recorded_from": f"commit {git('rev-parse', 'HEAD')} on branch {git('rev-parse', '--abbrev-ref', 'HEAD')} ({dirty})",
        "python_files_differing_from_origin_main_when_recorded": differing,
        "regeneration_rule": support.REGENERATION_RULE,
    }
    all_scenarios = scenarios.all_scenarios()
    for existing in support.FIXTURE_DIRECTORY.glob("recorded_*_on_origin_main.json"):
        existing.unlink()   # a scenario that no longer exists must not leave a stale file
    for scenario in all_scenarios:
        print(f"=== recording scenario {scenario.name}", flush=True)
        with tempfile.TemporaryDirectory() as working_directory:
            outcome = support.run_scenario_in_isolation(scenario, Path(working_directory))
            support.write_recorded_file(outcome, metadata)
    print(f"Wrote {len(all_scenarios)} recorded files under {support.FIXTURE_DIRECTORY}")


if __name__ == "__main__":
    main()
