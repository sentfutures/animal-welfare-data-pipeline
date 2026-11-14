#!/usr/bin/env python3
"""Regenerate the files that tests/test_sdf_resume_of_runs_started_on_main.py compares against.

This is the ONLY code that writes tests/fixtures/sdf_resume_from_main/*; the test never does. The files
describe how origin/main finishes runs that origin/main started and that crashed partway, so a pull
request that only refactors never regenerates them. For each interruption point
(tests/sdf_resume_from_main_interruption_points.py) this records, on origin/main code:

  (a) the interrupted run folder (stage files, checkpoints, manifest, cost log, and the inputs snapshot
      shared by every folder) as zip files;
  (b) the requests sent while resuming that folder, and the stage files, checkpoints and final corpus
      after the resume, in the compact recording format of the prompt-equivalence test.

Guard: the pipeline Python that runs must be origin/main's. Give --origin-main-checkout a detached git
worktree of origin/main (`git worktree add --detach <folder> origin/main`); its code is put first on the
import path and the script refuses unless that checkout is at origin/main's commit with no difference in
sdf_pipeline/, shared/ or dad_pipeline/ Python (committed or untracked). Without the option the checkout
is this repository, which must satisfy the same rule. The recorded files store the commit and the list of
differing files, and the test fails if the list is not empty.

No model API is called and no key is read: every provider client is replaced by a recorder that returns
canned replies, with fake keys.

    git worktree add --detach <scratch folder>/origin_main_checkout origin/main
    python scripts/regenerate_sdf_resume_from_main_recorded_runs.py --origin-main-checkout <that folder>
    git worktree remove <that folder>
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIRECTORIES_THE_RESUME_DEPENDS_ON = ("sdf_pipeline", "shared", "dad_pipeline")


def git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(["git", *arguments], cwd=repository, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.exit(f"Refusing to regenerate: `git {' '.join(arguments)}` failed in {repository} with exit code "
                 f"{completed.returncode}: {completed.stderr.strip() or 'no error message'}")
    return completed.stdout.strip()


def python_files_differing_from_origin_main(checkout: Path) -> list[str]:
    committed = git(checkout, "diff", "--name-only", "origin/main", "--", *PYTHON_DIRECTORIES_THE_RESUME_DEPENDS_ON).splitlines()
    untracked = git(checkout, "ls-files", "--others", "--exclude-standard", "--",
                    *PYTHON_DIRECTORIES_THE_RESUME_DEPENDS_ON).splitlines()
    return sorted(name for name in {*committed, *untracked} if name.endswith(".py"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origin-main-checkout", type=Path, default=REPOSITORY_ROOT,
                        help="a git worktree of origin/main whose pipeline code is used (default: this repository)")
    parser.add_argument("--allow-python-changes", action="store_true",
                        help="record even though Python files differ from origin/main (the test then fails)")
    arguments = parser.parse_args()
    checkout = arguments.origin_main_checkout.resolve()

    if subprocess.run(["git", "rev-parse", "--verify", "--quiet", "origin/main^{commit}"], cwd=checkout,
                      capture_output=True, text=True).returncode != 0:
        sys.exit("Refusing to regenerate: `origin/main` cannot be resolved, so the guard cannot tell which "
                 "Python files differ from it. Run `git fetch origin main` and try again.")
    origin_main_commit = git(checkout, "rev-parse", "origin/main")
    checkout_commit = git(checkout, "rev-parse", "HEAD")
    differing = python_files_differing_from_origin_main(checkout)
    if checkout != REPOSITORY_ROOT and checkout_commit != origin_main_commit and not arguments.allow_python_changes:
        sys.exit(f"Refusing to regenerate: the checkout {checkout} is at {checkout_commit}, not origin/main "
                 f"({origin_main_commit}).")
    if differing and not arguments.allow_python_changes:
        sys.exit("Refusing to regenerate: these Python files differ from origin/main, so the recorded resumes "
                 f"would not be the ones origin/main performs: {differing}. Use a worktree of origin/main, or "
                 "pass --allow-python-changes (the test will then fail until a human reviews the list).")

    # The checkout's pipeline code first, then this repository's test support.
    sys.path[:0] = [str(checkout), str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]
    import shared  # noqa: E402
    import sdf_pipeline  # noqa: E402
    for module in (shared, sdf_pipeline):
        if not Path(module.__file__).resolve().is_relative_to(checkout):
            sys.exit(f"Refusing to regenerate: {module.__name__} was imported from {module.__file__}, "
                     f"not from the checkout {checkout}.")
    import pytest  # noqa: E402,F401
    import sdf_resume_from_main_fixture_support as resume  # noqa: E402
    import sdf_resume_from_main_interruption_points as points_module  # noqa: E402

    dirty = "with uncommitted changes" if git(checkout, "status", "--porcelain", "--untracked-files=no") else "clean"
    metadata = {
        "recorded_from": f"pipeline code at commit {checkout_commit} ({dirty}); origin/main is {origin_main_commit}",
        "python_files_differing_from_origin_main_when_recorded": differing,
        "regeneration_rule": resume.REGENERATION_RULE,
    }
    points = points_module.all_interruption_points()
    resume.FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for existing in list(resume.FIXTURE_DIRECTORY.glob("*.zip")) + list(resume.FIXTURE_DIRECTORY.glob("recorded_*.json")):
        existing.unlink()   # a point that no longer exists must not leave a stale file

    inputs_of_first_run = None
    for point in points:
        print(f"=== interrupting a run {point.description} (crash on call {point.calls_allowed_before_crash + 1} "
              f"of {point.crash_stage}); resume with {point.resume_command_description}", flush=True)
        with tempfile.TemporaryDirectory() as working_directory:
            files_without_inputs, inputs = resume.run_until_crash_and_store_folder(point, Path(working_directory) / "interrupt")
        if inputs_of_first_run is None:
            inputs_of_first_run = inputs
            resume.write_shared_inputs_snapshot(inputs)
        elif inputs != inputs_of_first_run:
            sys.exit(f"Refusing to record: the inputs snapshot of the run interrupted {point.description} differs "
                     "from the first one, so the snapshot cannot be stored once for every point.")
        resume.write_stored_run_folder(point, files_without_inputs)
        with tempfile.TemporaryDirectory() as working_directory:
            with resume.isolated_process_environment(Path(working_directory)) as monkeypatch:
                outcome, run_directory = resume.resume_stored_run_folder(point, Path(working_directory), monkeypatch)
                record = resume.record_for_resume(outcome, run_directory)
        resume.write_recorded_resume(point, record, metadata)
    print(f"Wrote {len(points)} interrupted run folders and recorded resumes under {resume.FIXTURE_DIRECTORY}")


if __name__ == "__main__":
    main()
