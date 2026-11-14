"""A run started on origin/main and interrupted partway is finished by the current pipeline as origin/main would finish it.

Runs started on ``main`` have stage files and checkpoints. For each interruption point
(``tests/sdf_resume_from_main_interruption_points.py``) the run folder that origin/main left when it
crashed there is stored under ``tests/fixtures/sdf_resume_from_main/``, with a recording of how
origin/main resumed it (the requests sent, the stage files, checkpoints and final corpus afterwards).
Each test unpacks the stored folder, resumes it with the current code on canned replies and compares.
The reference is the same interruption resumed on origin/main, not an uninterrupted run (on main a
resumed run can already differ from an uninterrupted one, for example in which near-duplicate twin the
cull keeps).

Offline: every provider client is replaced by canned replies. The recorded files are written only by
``scripts/regenerate_sdf_resume_from_main_recorded_runs.py`` (see the support module's docstring).

This test is transitional: it protects runs started before the document state redesign merged. Delete it,
its support modules, its interruption points, its regenerate script and ``tests/fixtures/sdf_resume_from_main/``
once no such run still needs resuming. Deleting it does not shrink the repository's history.
"""

import pytest

import sdf_resume_from_main_fixture_support as resume
from sdf_resume_from_main_interruption_points import all_interruption_points

POINTS = all_interruption_points()
POINT_NAMES = [point.name for point in POINTS]


def _failure_message(point, kind, differences):
    return (f"{len(differences)} difference(s) in the {kind} when a run started on origin/main and interrupted "
            f"{point.description} is resumed with `{point.resume_command_description}`, compared with the same "
            "resume on origin/main:\n  " + "\n  ".join(differences[:40]) + ("\n  ..." if len(differences) > 40 else "")
            + f"\n{resume.REGENERATION_RULE}")


@pytest.fixture(scope="module")
def resumed_current(tmp_path_factory):
    """{point name: (record of the resume with the current code, run folder)}, each resumed once."""
    cache = {}

    def get(point):
        if point.name not in cache:
            working_directory = tmp_path_factory.mktemp(point.name[:40])
            with resume.isolated_process_environment(working_directory) as monkeypatch:
                outcome, run_directory = resume.resume_stored_run_folder(point, working_directory, monkeypatch)
                cache[point.name] = (resume.record_for_resume(outcome, run_directory), run_directory)
        return cache[point.name]
    return get


def test_recorded_files_cover_exactly_the_interruption_points():
    recorded = sorted(path.name for path in resume.FIXTURE_DIRECTORY.glob("recorded_*.json"))
    assert recorded == sorted(resume.recorded_resume_path(point).name for point in POINTS)
    stored = sorted(path.name for path in resume.FIXTURE_DIRECTORY.glob("interrupted_run_*.zip"))
    assert stored == sorted(resume.stored_run_folder_zip_path(point).name for point in POINTS)
    assert (resume.FIXTURE_DIRECTORY / resume.SHARED_INPUTS_ZIP_NAME).is_file()


def test_recorded_files_were_made_from_unmodified_origin_main_python_code():
    for point in POINTS:
        recorded = resume.load_recorded_resume(point)
        assert recorded["python_files_differing_from_origin_main_when_recorded"] == [], (
            f"{point.name}: recorded from Python code that differed from origin/main "
            f"({recorded['python_files_differing_from_origin_main_when_recorded']}). {resume.REGENERATION_RULE}")


def test_interruption_points_cover_every_stage_that_makes_calls_and_the_gaps_between_stages():
    stages_crashed_in_the_middle = {p.crash_stage for p in POINTS if p.calls_allowed_before_crash > 0}
    assert stages_crashed_in_the_middle == {
        "layer1_screen", "layer1_plan", "layer1_description_rewrite", "layer1_select", "layer2_draft", "layer3_critique",
        "layer3_revise", "layer4_score"}
    assert sum(1 for p in POINTS if p.calls_allowed_before_crash == 0) >= 1
    assert any("--layer" in p.extra_arguments for p in POINTS)


@pytest.mark.parametrize("point", POINTS, ids=POINT_NAMES)
def test_stored_folder_is_a_main_run_with_a_partly_done_stage(point, tmp_path):
    run_directory = resume.unpack_stored_run_folder(point, tmp_path / "outputs")
    assert (run_directory / "run_manifest.json").is_file() and (run_directory / "inputs" / "prompts").is_dir()
    recorded = resume.load_recorded_resume(point)
    calls = recorded["calls_per_stage_per_invocation"].get("1", {})
    if point.calls_allowed_before_crash > 0 and "--layer" not in point.extra_arguments:
        full_count_of_stage = {"layer1_screen": 3, "layer1_plan": 10, "layer1_description_rewrite": 9,
                               "layer1_select": 7, "layer2_draft": 8, "layer3_critique": 8, "layer3_revise": 8,
                               "layer4_score": 8}[point.crash_stage]
        resent_in_crashed_stage = calls.get(point.crash_stage, 0)
        assert 0 < resent_in_crashed_stage < full_count_of_stage, (
            f"the resume of {point.name} should send only the unfinished part of {point.crash_stage}")


@pytest.mark.parametrize("point", POINTS, ids=POINT_NAMES)
def test_resume_sends_the_requests_main_sent_and_writes_the_files_and_corpus_main_wrote(point, resumed_current):
    current, _ = resumed_current(point)
    differences = resume.differences_between_recorded_and_current_resume(
        point, resume.load_recorded_resume(point), current)
    for kind, found in differences.items():
        assert not found, _failure_message(point, kind, found)
