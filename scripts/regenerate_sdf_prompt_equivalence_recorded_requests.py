#!/usr/bin/env python3
"""Regenerate the files that tests/test_sdf_prompt_equivalence_and_final_corpus.py compares against.

This is the ONLY code that rewrites tests/fixtures/sdf_prompt_equivalence/*.json; the test never
does. Run it deliberately, in a pull request that changes nothing else, when a request change is
intended (a prompt edit, a config default, a model change). The recorded files are meant to be
recorded on origin/main: run it from a checkout of origin/main, or pass --recorded-from to say what
the code was.

No model API is called and no key is read: every provider client is replaced by a recorder that
returns canned replies (see tests/sdf_prompt_equivalence_fixture_support.py), and fake keys are
set for the run.

    python scripts/regenerate_sdf_prompt_equivalence_recorded_requests.py
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "tests"))

import sdf_prompt_equivalence_fixture_support as support  # noqa: E402
from shared import api, api_keys, openai_api  # noqa: E402


def describe_code_being_recorded() -> str:
    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=REPOSITORY_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = "with uncommitted changes" if git("status", "--porcelain", "--untracked-files=no") else "clean"
    return f"commit {git('rev-parse', 'HEAD')} on branch {git('rev-parse', '--abbrev-ref', 'HEAD')} ({dirty})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recorded-from", default=None,
                        help="text stored in the files as what was recorded (default: this checkout's commit)")
    arguments = parser.parse_args()
    recorded_from = arguments.recorded_from or describe_code_being_recorded()

    outcomes = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        # the same isolation tests/conftest.py gives every test
        with tempfile.TemporaryDirectory() as empty_repository_root:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
            monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
            monkeypatch.setattr(api_keys, "REPO_ROOT", Path(empty_repository_root))
            monkeypatch.setattr(api_keys, "_sources_already_announced", set())
            monkeypatch.setattr(api_keys, "_api_key_source_mode", api_keys.SEARCH_UPWARD_THEN_ENVIRONMENT)
            monkeypatch.setattr(api_keys, "_nearest_dotenv_path_searching_upward", lambda: None)
            monkeypatch.setattr(api_keys, "set_api_key_source_mode", lambda mode: None)
            for scenario in support.all_scenarios():
                with tempfile.TemporaryDirectory() as working_directory, pytest.MonkeyPatch.context() as scenario_patch:
                    for name, value in (("_config", {}), ("_client", None), ("_cost_log_path", None),
                                        ("_record_model_calls", True), ("_backend", "api")):
                        scenario_patch.setattr(api, name, value)
                    scenario_patch.setattr(openai_api, "_client", None)
                    print(f"=== recording scenario {scenario.name}")
                    outcomes.append(support.run_scenario(scenario, Path(working_directory), scenario_patch))
    support.write_recorded_files(outcomes, recorded_from)
    print(f"Wrote {support.RECORDED_REQUESTS_PATH} and {support.RECORDED_FINAL_CORPORA_PATH}")


if __name__ == "__main__":
    main()
