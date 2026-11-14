#!/usr/bin/env python3
"""Regenerate the files that tests/test_dad_request_settings_equivalence.py compares against.

This is the ONLY code that writes tests/fixtures/dad_request_settings_equivalence/*; the test never does.
The files hold every model request the DAD pipeline sends (1a scenario plan, 1b draft, 1c gate, 1d refine,
2a scope, 2a.5 select, 2b respond, step 3 rewrite, the no-system-prompt baseline, and the retry variants
with a fallback model or a doubled token cap), recorded at the provider-client boundary on origin/main
code, for the api, claude_code and auto backends (see tests/dad_request_settings_equivalence_fixture_support.py).

Guard: the pipeline Python that runs must be origin/main's. Give --origin-main-checkout a detached git
worktree of origin/main (`git worktree add --detach <folder> origin/main`); its code is put first on the
import path and the script refuses unless that checkout is at origin/main's commit with no difference in
sdf_pipeline/, shared/ or dad_pipeline/ Python (committed or untracked), and unless the imported
`shared` and `dad_pipeline` packages live inside it. Without the option the checkout is this repository,
which must satisfy the same rule.

No model API is called and no key is read: every provider client is replaced by a recorder that returns
canned replies, with fake keys.

    git worktree add --detach <scratch folder>/origin_main_checkout origin/main
    python scripts/regenerate_dad_request_settings_equivalence_recorded_requests.py --origin-main-checkout <that folder>
    git worktree remove <that folder>
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIRECTORIES_THE_RECORDING_DEPENDS_ON = ("sdf_pipeline", "shared", "dad_pipeline")


def git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(["git", *arguments], cwd=repository, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.exit(f"Refusing to regenerate: `git {' '.join(arguments)}` failed in {repository} with exit code "
                 f"{completed.returncode}: {completed.stderr.strip() or 'no error message'}")
    return completed.stdout.strip()


def python_files_differing_from_origin_main(checkout: Path) -> list[str]:
    committed = git(checkout, "diff", "--name-only", "origin/main", "--",
                    *PYTHON_DIRECTORIES_THE_RECORDING_DEPENDS_ON).splitlines()
    untracked = git(checkout, "ls-files", "--others", "--exclude-standard", "--",
                    *PYTHON_DIRECTORIES_THE_RECORDING_DEPENDS_ON).splitlines()
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
        sys.exit("Refusing to regenerate: these Python files differ from origin/main, so the recorded requests "
                 f"would not be the ones origin/main sends: {differing}. Use a worktree of origin/main, or "
                 "pass --allow-python-changes (the test will then fail until a human reviews the list).")

    # The checkout's pipeline code first, then this repository's test support.
    sys.path[:0] = [str(checkout), str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]
    import shared  # noqa: E402
    import dad_pipeline  # noqa: E402
    for module in (shared, dad_pipeline):
        if not Path(module.__file__).resolve().is_relative_to(checkout):
            sys.exit(f"Refusing to regenerate: {module.__name__} was imported from {module.__file__}, "
                     f"not from the checkout {checkout}.")
    import pytest  # noqa: E402
    import dad_request_settings_equivalence_fixture_support as support  # noqa: E402

    dirty = "with uncommitted changes" if git(checkout, "status", "--porcelain", "--untracked-files=no") else "clean"
    metadata = {
        "recorded_from": f"pipeline code at commit {checkout_commit} ({dirty}); origin/main is {origin_main_commit}",
        "python_files_differing_from_origin_main_when_recorded": differing,
        "regeneration_rule": support.REGENERATION_RULE,
    }
    support.FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for existing in support.FIXTURE_DIRECTORY.glob("recorded_*.json"):
        existing.unlink()   # a configuration that no longer exists must not leave a stale file
    for configuration_name in support.CONFIGURATIONS:
        outcomes = []
        for backend in support.BACKENDS:
            print(f"=== recording configuration {configuration_name!r} on backend {backend!r}", flush=True)
            with tempfile.TemporaryDirectory() as working_directory:
                with pytest.MonkeyPatch.context() as monkeypatch:
                    outcomes.append(support.run_case(configuration_name, backend, Path(working_directory), monkeypatch))
            print(f"    {len(outcomes[-1].records)} requests, {outcomes[-1].final_example_count} final examples")
        path = support.write_recorded_file(configuration_name, outcomes, metadata)
        print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
