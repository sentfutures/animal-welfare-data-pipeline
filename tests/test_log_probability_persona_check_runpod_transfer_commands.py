"""Integration test of the launcher's real tar transfer commands. ssh is replaced by a local shell (`bash -c`) and the
pod's /workspace by a temporary folder, so the same tar pipelines used against a pod run for real. No RunPod calls."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from evals.log_probability_persona_check import launch_and_stop_runpod_scoring_pod as launcher

LOGS_FOLDER_NAME = "pod_testpod_logs_2026-10-22_12-00"


@pytest.fixture
def local_run_folder(tmp_path):
    folder = tmp_path / "local_run_folder"
    (folder / "scoring_inputs").mkdir(parents=True)
    (folder / "scoring_inputs" / "documents.jsonl").write_text('{"document_id": "d1"}\n')
    (folder / "scoring_inputs" / "conditions.jsonl").write_text('{"condition_id": "c1"}\n')
    (folder / "run_configuration_2026-10-22_12-00.json").write_text('{"model_name": "test_model"}')
    return folder


@pytest.fixture
def fake_pod_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "fake_pod_workspace" / "scoring_copy"
    monkeypatch.setattr(launcher, "REMOTE_WORKING_DIRECTORY", str(workspace))
    monkeypatch.setattr(launcher, "REMOTE_RUN_FOLDER", f"{workspace}/run_folder")
    return workspace


@pytest.fixture
def transfer(local_run_folder, fake_pod_workspace):
    return launcher.SshTarTransfer("unused_key", local_run_folder, ssh_timeout_seconds=60,
                                   remote_shell_prefix=["bash", "-c"])


def write_files(root, files):
    for relative_path, text in files.items():
        path = Path(root) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def run_python_in_pod_copy(fake_pod_workspace, code):
    """An interpreter whose only source of the `evals` package is the uploaded copy."""
    return subprocess.run([sys.executable, "-c", code], cwd=fake_pod_workspace, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(fake_pod_workspace)})


# ---- what is uploaded

def test_upload_sends_run_configuration_scoring_inputs_scorer_package_and_prompts(transfer, fake_pod_workspace):
    transfer.upload_project_files()
    remote_run_folder = fake_pod_workspace / "run_folder"
    assert (remote_run_folder / "run_configuration_2026-10-22_12-00.json").read_text() == '{"model_name": "test_model"}'
    assert (remote_run_folder / "scoring_inputs" / "documents.jsonl").exists()
    package_folder = fake_pod_workspace / "evals" / "log_probability_persona_check"
    for name in ("score_document_log_probabilities_on_pod.py", "prompt_rendering.py", "persona_sets.py",
                 "scoring_model_defaults.py", "timed_progress_and_jsonl_helpers.py",
                 "requirements_for_gemma_4_and_qwen3_scoring_pod.txt"):
        assert (package_folder / name).is_file(), name
    assert (fake_pod_workspace / "evals" / "__init__.py").is_file()
    assert (fake_pod_workspace / "prompts" / "tools" / "persona_checks" / "model_type_framing").is_dir()
    assert (fake_pod_workspace / launcher.REQUIREMENTS_FILE_RELATIVE_PATH).is_file()
    # The repository's own shared package would collide with any other `shared` on the pod and is not needed.
    assert not (fake_pod_workspace / "shared").exists()
    assert not list(fake_pod_workspace.rglob("._*")) and not list(fake_pod_workspace.rglob("__pycache__"))


def test_upload_leaves_out_unfinished_partial_files_of_an_earlier_scorer(transfer, local_run_folder, fake_pod_workspace):
    write_files(local_run_folder / "scoring_outputs", {
        "per_token_log_probabilities/d1.npz": "finished", "per_token_log_probabilities/d2.npz.partial": "half written"})
    transfer.upload_project_files()
    remote_outputs = fake_pod_workspace / "run_folder" / "scoring_outputs" / "per_token_log_probabilities"
    assert (remote_outputs / "d1.npz").exists() and not (remote_outputs / "d2.npz.partial").exists()


def test_pod_copy_imports_prompt_rendering_and_persona_sets_without_the_rest_of_the_repository(transfer,
                                                                                               fake_pod_workspace):
    transfer.upload_project_files()
    code = ("import sys, pathlib\n"
            "import evals.log_probability_persona_check.prompt_rendering as prompt_rendering\n"
            "import evals.log_probability_persona_check.timed_progress_and_jsonl_helpers\n"
            "from evals.log_probability_persona_check import persona_sets\n"
            f"pod_root = pathlib.Path({str(fake_pod_workspace)!r}).resolve()\n"
            "assert pathlib.Path(prompt_rendering.__file__).resolve().is_relative_to(pod_root)\n"
            "assert 'shared' not in sys.modules\n"
            "framing = persona_sets.load_model_type_framing()\n"
            "assert persona_sets.PERSONA_CHECKS_PROMPT_DIRECTORY.resolve().is_relative_to(pod_root)\n"
            "print('imported_from_pod_copy')\n")
    completed = run_python_in_pod_copy(fake_pod_workspace, code)
    assert completed.returncode == 0 and "imported_from_pod_copy" in completed.stdout, completed.stderr


def test_scorer_module_imports_from_the_pod_copy_and_reads_the_uploaded_run_configuration(transfer, fake_pod_workspace):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    transfer.upload_project_files()
    code = ("from evals.log_probability_persona_check import score_document_log_probabilities_on_pod as scorer\n"
            f"print(scorer.load_first_run_configuration({str(fake_pod_workspace / 'run_folder')!r})['model_name'])\n")
    completed = run_python_in_pod_copy(fake_pod_workspace, code)
    assert completed.returncode == 0 and completed.stdout.strip() == "test_model", completed.stderr


# ---- periodic pull of finished files only

def test_periodic_pull_treats_missing_scoring_outputs_as_nothing_to_pull(transfer, local_run_folder, fake_pod_workspace):
    transfer.upload_project_files()
    assert not (fake_pod_workspace / "run_folder" / "scoring_outputs").exists()
    transfer.pull_scoring_outputs(False)
    transfer.pull_scoring_outputs(True)
    assert not (local_run_folder / "scoring_outputs").exists()


def test_periodic_pull_copies_finished_files_and_skips_unfinished_ones(transfer, local_run_folder, fake_pod_workspace):
    transfer.upload_project_files()
    write_files(fake_pod_workspace / "run_folder" / "scoring_outputs", {
        "per_token_log_probabilities/d1.npz": "finished one",
        "per_token_log_probabilities/d2.npz.partial": "being written",
        "scoring_completion_records.jsonl": '{"document_id": "d1"}\n'})
    transfer.pull_scoring_outputs(False)
    pulled = local_run_folder / "scoring_outputs"
    assert (pulled / "per_token_log_probabilities" / "d1.npz").read_text() == "finished one"
    assert (pulled / "scoring_completion_records.jsonl").exists()
    assert not list(pulled.rglob("*.partial"))
    assert not list(local_run_folder.glob("tmp*"))  # staging folder removed


def test_incremental_pull_sends_only_files_changed_since_the_previous_pull(transfer, local_run_folder, fake_pod_workspace):
    transfer.upload_project_files()
    outputs = fake_pod_workspace / "run_folder" / "scoring_outputs"
    write_files(outputs, {"per_token_log_probabilities/d1.npz": "first"})
    transfer.pull_scoring_outputs(False)
    local_first = local_run_folder / "scoring_outputs" / "per_token_log_probabilities" / "d1.npz"
    local_first.write_text("local marker that a second pull must not overwrite")
    time.sleep(1.1)
    write_files(outputs, {"per_token_log_probabilities/d2.npz": "second"})
    transfer.pull_scoring_outputs(False)
    assert (local_run_folder / "scoring_outputs" / "per_token_log_probabilities" / "d2.npz").read_text() == "second"
    assert local_first.read_text() == "local marker that a second pull must not overwrite"
    transfer.pull_scoring_outputs(True)  # a full pull sends everything again
    assert local_first.read_text() == "first"


def test_periodic_pull_never_fails_with_file_changed_while_the_scorer_keeps_writing(transfer, local_run_folder,
                                                                                    fake_pod_workspace):
    """The legacy run's problem: tar of a folder whose files are being appended to, or written under a temporary
    name and renamed, reported 'file changed as we read it'. A writer thread imitates the scorer while pulls run."""
    transfer.upload_project_files()
    outputs = fake_pod_workspace / "run_folder" / "scoring_outputs"
    (outputs / "per_token_log_probabilities").mkdir(parents=True)
    stop_writing = threading.Event()

    def write_like_the_scorer():
        document_number = 0
        while not stop_writing.is_set():
            document_number += 1
            partial = outputs / "per_token_log_probabilities" / f"d{document_number}.npz.partial"
            partial.write_bytes(bytes(range(256)) * 40)
            os.replace(partial, outputs / "per_token_log_probabilities" / f"d{document_number}.npz")
            with open(outputs / "scoring_completion_records.jsonl", "a") as handle:
                for _ in range(20):
                    handle.write('{"document_id": "d%d", "condition_id": "c"}\n' % document_number)
            time.sleep(0.005)

    writer = threading.Thread(target=write_like_the_scorer)
    writer.start()
    try:
        for pull_number in range(15):
            transfer.pull_scoring_outputs(pull_number % 5 == 0)
    finally:
        stop_writing.set()
        writer.join()
    transfer.pull_scoring_outputs(True)
    pulled_names = {path.name for path in (local_run_folder / "scoring_outputs" / "per_token_log_probabilities").glob("*")}
    assert pulled_names and all(name.endswith(".npz") for name in pulled_names)


# ---- final pull

def test_final_pull_with_everything_missing_but_the_scorer_logs_succeeds(transfer, local_run_folder, fake_pod_workspace):
    """The scorer crashed before creating scoring_outputs/."""
    transfer.upload_project_files()
    write_files(fake_pod_workspace / "run_folder", {"scorer_standard_output.txt": "Traceback\n", "scorer_exit_code.txt": "1\n"})
    transfer.pull_final_copy_and_verify(LOGS_FOLDER_NAME, 60)
    assert (local_run_folder / LOGS_FOLDER_NAME / "scorer_exit_code.txt").read_text() == "1\n"
    assert (local_run_folder / LOGS_FOLDER_NAME / "scorer_standard_output.txt").read_text() == "Traceback\n"
    assert not (local_run_folder / "scoring_outputs").exists()


def populate_finished_pod(fake_pod_workspace):
    write_files(fake_pod_workspace, {
        "run_folder/scoring_outputs/per_token_log_probabilities/d1.npz": "binary-ish content" * 50,
        "run_folder/scoring_outputs/per_token_log_probabilities/d2.npz.partial": "left by a killed scorer",
        "run_folder/scoring_outputs/scoring_completion_records.jsonl": '{"document_id": "d1"}\n',
        "run_folder/scoring_outputs/scoring_progress_log_2026-10-22_12-05.txt": "done\n",
        "run_folder/scorer_standard_output.txt": "all good\n",
        "run_folder/scorer_exit_code.txt": "0\n",
        "run_folder/.last_pull_marker": "",
        "watchdog_log.txt": "",
        "pod_setup_pip_install_log.txt": "installed\n",
        "pod_setup_environment_versions.txt": "torch 2.8\n",
        "pod_secret_environment.sh": "export HF_TOKEN=secret\n",
    })


def test_full_final_pull_copies_everything_but_scoring_inputs_secrets_and_unfinished_files_and_verifies(
        transfer, local_run_folder, fake_pod_workspace):
    transfer.upload_project_files()
    populate_finished_pod(fake_pod_workspace)
    files_verified = transfer.pull_final_copy_and_verify(LOGS_FOLDER_NAME, 60)
    # 11 files written, minus the secret file, the marker and the partial file, plus the uploaded run configuration
    assert files_verified == 11 - 3 + 1
    outputs = local_run_folder / "scoring_outputs"
    assert (outputs / "per_token_log_probabilities" / "d1.npz").stat().st_size == len("binary-ish content") * 50
    assert (outputs / "scoring_completion_records.jsonl").exists()
    assert not list(local_run_folder.rglob("*.partial"))
    logs = local_run_folder / LOGS_FOLDER_NAME
    for name in ("scorer_standard_output.txt", "scorer_exit_code.txt", "watchdog_log.txt",
                 "pod_setup_pip_install_log.txt", "pod_setup_environment_versions.txt"):
        assert (logs / name).exists(), name
    assert not (logs / "pod_secret_environment.sh").exists() and not list(local_run_folder.rglob("pod_secret*"))
    assert not list(local_run_folder.rglob(".last_pull_marker*"))
    assert not (logs / "scoring_inputs").exists()
    assert not list(local_run_folder.glob("tmp*"))  # staging folder removed


def test_final_pull_verification_reports_files_that_did_not_arrive(transfer, local_run_folder, fake_pod_workspace,
                                                                    monkeypatch):
    transfer.upload_project_files()
    populate_finished_pod(fake_pod_workspace)

    def pull_that_delivers_an_empty_archive(command, output_file_path, timeout_seconds):
        Path(output_file_path).write_bytes(b"")
        return 255, "connection reset"

    monkeypatch.setattr(transfer, "_run_remote_command_into_file", pull_that_delivers_an_empty_archive)
    with pytest.raises(launcher.FinalCopyIncompleteError) as excinfo:
        transfer.pull_final_copy_and_verify(LOGS_FOLDER_NAME, 60)
    assert any("scorer_exit_code.txt" in name and "did not arrive" in name for name in excinfo.value.problem_files)


def test_final_pull_verification_reports_size_mismatch(transfer, local_run_folder, fake_pod_workspace, monkeypatch):
    transfer.upload_project_files()
    populate_finished_pod(fake_pod_workspace)
    real_run_remote_command = transfer.run_remote_command

    def listing_that_claims_larger_files(command, input_text=None, timeout_seconds=None):
        exit_code, output = real_run_remote_command(command, input_text, timeout_seconds)
        if "wc -c" in command:
            output = output.replace("10\tpod_setup_pip_install_log.txt", "999\tpod_setup_pip_install_log.txt")
        return exit_code, output

    monkeypatch.setattr(transfer, "run_remote_command", listing_that_claims_larger_files)
    with pytest.raises(launcher.FinalCopyIncompleteError) as excinfo:
        transfer.pull_final_copy_and_verify(LOGS_FOLDER_NAME, 60)
    assert any("pod_setup_pip_install_log.txt (pod 999 bytes" in name for name in excinfo.value.problem_files)
