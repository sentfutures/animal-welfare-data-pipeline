"""Engine for the test that a run started on origin/main and interrupted partway is finished by the current pipeline exactly as origin/main would finish it
(the same interruption resumed on origin/main is the reference). It adds only the crash, the zip storage of a run folder and the resume driver; the fakes, comparison and recording format come from
``sdf_prompt_equivalence_fixture_support``."""

import contextlib
import hashlib
import io
import json
import sys
import threading
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import sdf_prompt_equivalence_fixture_support as support
import sdf_prompt_equivalence_scenarios as scenarios
from sdf_resume_from_main_interruption_points import CRASH_MESSAGE, FIXTURE_SCENARIO_NAME, InterruptionPoint
import sdf_pipeline.run as sdf_run

REPOSITORY_ROOT = support.REPOSITORY_ROOT
FIXTURE_DIRECTORY = REPOSITORY_ROOT / "tests" / "fixtures" / "sdf_resume_from_main"
RECORDED_FILE_FORMAT_VERSION = 1
REGENERATION_COMMAND = ("python scripts/regenerate_sdf_resume_from_main_recorded_runs.py "
                        "--origin-main-checkout <a detached git worktree of origin/main>")
REGENERATION_RULE = (
    "The recorded files in tests/fixtures/sdf_resume_from_main/ are written only by "
    f"`{REGENERATION_COMMAND}`, which refuses unless the pipeline Python is exactly origin/main's. They "
    "describe how origin/main resumes runs it started, so a refactor never regenerates them.")
SHARED_INPUTS_ZIP_NAME = "run_inputs_snapshot_shared_by_every_interrupted_run.zip"
FILES_LEFT_OUT_OF_STORED_RUN_FOLDERS = ("model_calls.jsonl",)
DIRECTORIES_LEFT_OUT_OF_STORED_RUN_FOLDERS = ("system_prompts",)


def stored_run_folder_zip_path(point: InterruptionPoint) -> Path:
    return FIXTURE_DIRECTORY / f"interrupted_run_{point.name}_started_on_origin_main.zip"


def recorded_resume_path(point: InterruptionPoint) -> Path:
    return FIXTURE_DIRECTORY / f"recorded_resume_of_run_interrupted_{point.name}_on_origin_main.json"


# ---------------------------------------------------------------------------------- the crash

class SimulatedProcessCrash(SystemExit):
    """Raised by the canned reply source in place of a reply; a ``SystemExit`` so no ``except Exception`` swallows it as a failed call."""


def reply_source_that_crashes_after(behaviour: scenarios.ReplyBehaviour, crash_stage: str, calls_allowed: int):
    """A ``support.canned_reply`` replacement that crashes on the request after ``calls_allowed`` of ``crash_stage``, and on every request of any stage after it (a dead process sends nothing)."""
    lock = threading.Lock()
    answered_by_stage: dict = {}
    crashed = threading.Event()

    def reply_or_crash(behaviour_given, request):
        with lock:
            if crashed.is_set():
                raise SimulatedProcessCrash(CRASH_MESSAGE)
            if request.stage == crash_stage:
                if answered_by_stage.get(crash_stage, 0) >= calls_allowed:
                    crashed.set()
                    raise SimulatedProcessCrash(CRASH_MESSAGE)
                answered_by_stage[crash_stage] = answered_by_stage.get(crash_stage, 0) + 1
        return real_canned_reply(behaviour_given, request)

    real_canned_reply = support.canned_reply
    return reply_or_crash


# ---------------------------------------------------------------------------------- scenarios

def fixture_scenario(invocations: tuple, sdf_config_overrides_for_workers: dict | None = None) -> support.Scenario:
    """The main prompt-equivalence scenario's configuration and replies without its injected failures, with the given invocations."""
    main_scenario = scenarios.scenario_named(FIXTURE_SCENARIO_NAME)
    return replace(
        main_scenario, name="resume_of_run_started_on_origin_main", invocations=invocations,
        description="A run interrupted by a simulated crash, then resumed.",
        reply_behaviour=replace(main_scenario.reply_behaviour, overrides={}),
        top_level_config_overrides={"effort": dict(support.REASONING_EFFORT_BY_STAGE_TAG),
                                    **(sdf_config_overrides_for_workers or {})})


# ---------------------------------------------------------------------------------- zip storage

def _zip_bytes_of_files(files_by_relative_path: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative_path in sorted(files_by_relative_path):
            entry = zipfile.ZipInfo(relative_path, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, files_by_relative_path[relative_path])
    return buffer.getvalue()


def _files_of_zip(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def files_of_run_folder(run_directory: Path) -> dict:
    """{path relative to the run folder: bytes} of every file in it."""
    return {path.relative_to(run_directory).as_posix(): path.read_bytes()
            for path in sorted(run_directory.rglob("*")) if path.is_file()}


def _is_left_out_of_stored_folder(relative_path: str) -> bool:
    parts = relative_path.split("/")
    return parts[0] in DIRECTORIES_LEFT_OUT_OF_STORED_RUN_FOLDERS or relative_path in FILES_LEFT_OUT_OF_STORED_RUN_FOLDERS


def split_run_folder_files_into_inputs_snapshot_and_rest(files: dict) -> tuple[dict, dict]:
    inputs = {name: data for name, data in files.items() if name.startswith("inputs/")}
    rest = {name: data for name, data in files.items()
            if not name.startswith("inputs/") and not _is_left_out_of_stored_folder(name)}
    return inputs, rest


def unpack_stored_run_folder(point: InterruptionPoint, outputs_root: Path) -> Path:
    """Unpack the point's stored run folder, with the shared inputs snapshot, into
    ``outputs_root/sdf/runs/<its run id>`` and return that folder."""
    files = {**_files_of_zip(FIXTURE_DIRECTORY / SHARED_INPUTS_ZIP_NAME),
             **_files_of_zip(stored_run_folder_zip_path(point))}
    run_id = json.loads(files["run_manifest.json"].decode("utf-8"))["run_id"]
    run_directory = outputs_root / "sdf" / "runs" / run_id
    for relative_path, data in files.items():
        target = run_directory / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return run_directory


# ---------------------------------------------------------------------------------- isolation

@contextlib.contextmanager
def isolated_process_environment(working_directory: Path):
    """What ``support.run_scenario_in_isolation`` sets up (fake keys, no .env, ``shared.api`` globals reset), for the regeneration script; pytest's conftest does the same."""
    working_directory.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as monkeypatch:
        support.isolate_from_real_keys_and_global_state(monkeypatch, working_directory)
        yield monkeypatch


# ---------------------------------------------------------------------------------- making a folder

def run_until_crash_and_store_folder(point: InterruptionPoint, working_directory: Path) -> tuple[dict, dict]:
    """Start a run, crash it as the point says, and return (the run folder's files without inputs, the inputs snapshot's files); one worker, so the stored folder does not depend on thread timing."""
    scenario = fixture_scenario(
        (support.Invocation(expected_exit_message_fragment=CRASH_MESSAGE),), {"workers": 1})
    captured = {}

    def capture_folder(invocation_number, run_directory):
        captured["files"] = files_of_run_folder(run_directory)
        captured["working_directory_text"] = str(working_directory)

    with pytest.MonkeyPatch.context() as patch_reply_source:
        patch_reply_source.setattr(support, "canned_reply", reply_source_that_crashes_after(
            scenario.reply_behaviour, point.crash_stage, point.calls_allowed_before_crash))
        support.run_scenario_in_isolation(scenario, working_directory, after_each_invocation=capture_folder)
    inputs, rest = split_run_folder_files_into_inputs_snapshot_and_rest(captured["files"])
    for name, data in rest.items():
        assert captured["working_directory_text"].encode() not in data, (
            f"{name} holds the absolute path of the recording's working directory, which would differ "
            "when the stored folder is unpacked elsewhere")
    return rest, inputs


def write_stored_run_folder(point: InterruptionPoint, files_without_inputs: dict) -> Path:
    path = stored_run_folder_zip_path(point)
    path.write_bytes(_zip_bytes_of_files(files_without_inputs))
    return path


def write_shared_inputs_snapshot(inputs_files: dict) -> Path:
    path = FIXTURE_DIRECTORY / SHARED_INPUTS_ZIP_NAME
    path.write_bytes(_zip_bytes_of_files(inputs_files))
    return path


# ---------------------------------------------------------------------------------- resuming

def resume_stored_run_folder(point: InterruptionPoint, working_directory: Path,
                             monkeypatch) -> tuple[support.ScenarioOutcome, Path]:
    """Unpack the point's stored folder and resume it on the imported pipeline code with requests recorded; returns the outcome (one invocation) and the run folder. Needs an isolated environment."""
    outputs_root = working_directory / "outputs"
    run_directory = unpack_stored_run_folder(point, outputs_root)
    scenario = fixture_scenario((support.Invocation(("--resume", *point.extra_arguments)),))
    monkeypatch.setenv("PIPELINE_OUTPUT_ROOT", str(outputs_root))
    recorder = support.ModelRequestRecorder(scenario.reply_behaviour)
    support.install_request_recording(monkeypatch, recorder)
    support.install_fixed_entity_pools(monkeypatch)
    recorder.start_invocation(1)
    config = support.build_config(scenario, scenario.invocations[0])
    config_path = working_directory / "fixture_config_for_the_resume.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config_path), "--label", "prompt-equivalence",
                                      "--resume", *point.extra_arguments, "--run-id", run_directory.name])
    exit_message = None
    try:
        sdf_run.main()
    except SystemExit as stopped:
        exit_message = str(stopped.code)
    final_path = run_directory / "final" / "sdf_corpus.jsonl"
    outcome = support.ScenarioOutcome(
        scenario=scenario, recorder=recorder, run_directory=run_directory, configs_used=[config],
        exit_messages_by_invocation=[exit_message],
        final_corpus_lines_after_each_invocation=[final_path.read_text(encoding="utf-8").splitlines()
                                                  if final_path.exists() else []],
        stage_files_after_each_invocation=[support.snapshot_stage_files(run_directory)])
    return outcome, run_directory


def record_for_resume(outcome: support.ScenarioOutcome, run_directory: Path) -> dict:
    """The compact-format record of a resume, plus the names of the files in the run folder afterwards."""
    record = support.record_for_outcome(outcome)
    record["run_folder_file_names"] = sorted(
        name for name in files_of_run_folder(run_directory)
        if not _is_left_out_of_stored_folder(name)
        and not name.startswith("inputs/"))
    return record


# ---------------------------------------------------------------------------------- recording files

def write_recorded_resume(point: InterruptionPoint, record: dict, metadata: dict) -> Path:
    """Write what the resume of one stored folder did, in the prompt-equivalence test's compact form."""
    extra_keys = {"run_folder_file_names": record.pop("run_folder_file_names")}
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    content = {"format_version": RECORDED_FILE_FORMAT_VERSION, **metadata,
               "interruption_point": point.as_recorded_description(),
               **support.recorded_form_of_record(record), **extra_keys}
    text = support.json_text_with_short_scalar_containers_on_one_line(content) + "\n"
    assert json.loads(text) == content, "the recorded file's text does not parse back to the recorded content"
    path = recorded_resume_path(point)
    path.write_text(text, encoding="utf-8")
    return path


def load_recorded_resume(point: InterruptionPoint) -> dict:
    """The recording of one point's resume on origin/main, with dealt hands and repeated values expanded
    again, so the comparison sees every record in full."""
    path = recorded_resume_path(point)
    assert path.is_file(), f"no recorded resume for interruption point {point.name!r} at {path}. {REGENERATION_RULE}"
    recorded = json.loads(path.read_text(encoding="utf-8"))
    assert recorded.get("format_version") == RECORDED_FILE_FORMAT_VERSION, (
        f"{path.name} has format version {recorded.get('format_version')!r}, this test reads "
        f"{RECORDED_FILE_FORMAT_VERSION}. {REGENERATION_RULE}")
    dealt_hands = recorded.pop("dealt_hands_by_identifier")
    repeated_values = recorded.pop("repeated_values_by_identifier")
    for key in support.SECTIONS_WHOSE_REPEATED_VALUES_ARE_STORED_ONCE:
        recorded[key] = support._expand_dealt_hand_references(
            support._expand_repeated_value_references(recorded[key], repeated_values), dealt_hands)
    return recorded


# ---------------------------------------------------------------------------------- comparing

def label_of(point: InterruptionPoint) -> str:
    """How the difference messages name the point (the support's describers put it in square brackets)."""
    return f"resume of a run interrupted {point.description}; resumed with {point.resume_command_description}"


def differences_between_recorded_and_current_resume(point: InterruptionPoint, recorded: dict,
                                                    current: dict) -> dict:
    """``{"requests", "stage files", "final corpus", "files in the run folder"}``, each a list of difference messages."""
    name = label_of(point)
    run_folder_differences = []
    recorded_names, current_names = set(recorded["run_folder_file_names"]), set(current["run_folder_file_names"])
    for missing in sorted(recorded_names - current_names):
        run_folder_differences.append(f"[{name}] file {missing} is in the run folder after the resume on "
                                      "origin/main but not now")
    for extra in sorted(current_names - recorded_names):
        run_folder_differences.append(f"[{name}] file {extra} is in the run folder after the resume now but "
                                      "not on origin/main")
    return {
        "requests": support.differences_between_recorded_and_current_requests(name, recorded, current),
        "stage files": support.differences_between_recorded_and_current_stage_files(name, recorded, current),
        "final corpus": support.differences_between_recorded_and_current_final_corpora(name, recorded, current),
        "files in the run folder": run_folder_differences,
    }


def sha256_of_files(files: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + files[name] + b"\0")
    return digest.hexdigest()
