"""Tests for the RunPod launcher using a fake RunPod client and fake transfer layer. No real RunPod calls."""

import json
import subprocess
from pathlib import Path

import pytest

from evals.log_probability_persona_check import launch_and_stop_runpod_scoring_pod as launcher

TEST_COST_CENTER_NAME = "test-cost-center"


class FakeClock:
    def __init__(self):
        self.now = 1_000_000.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeRunPodClient:
    def __init__(self, hourly_price=2.5):
        self.calls = []
        self.hourly_price = hourly_price
        self.delete_failures_remaining = 0

    def create_pod(self, creation_request):
        self.calls.append(("create_pod", creation_request))
        return {"id": "fakepod123", "costPerHr": self.hourly_price, "gpu": {"displayName": "H100 SXM"}}

    def get_pod(self, pod_id):
        self.calls.append(("get_pod", pod_id))
        return {"id": pod_id, "costPerHr": self.hourly_price, "publicIp": "1.2.3.4", "portMappings": {"22": 4000}}

    def delete_pod(self, pod_id):
        self.calls.append(("delete_pod", pod_id))
        if self.delete_failures_remaining > 0:
            self.delete_failures_remaining -= 1
            raise RuntimeError("temporary failure")

    def call_names(self):
        return [name for name, _ in self.calls]


class FakeTransfer:
    def __init__(self, scorer_finishes_after_seconds=None, scorer_exit_code="0", clock=None, fail_pulls=0,
                 raise_during_install=False, fail_final_pulls=0, scorer_vanishes_after_seconds=None,
                 scorer_output_text="Traceback ... FileNotFoundError: boom", install_failures=0, ssh_never_ready=False,
                 scorer_attempts=None, scorer_idle_before_first_start=False, requirements_marker_text=""):
        self.commands, self.pulls, self.uploads = [], [], 0
        self.final_pull_attempts, self.final_pull_clock_times = 0, []
        self.host = "1.2.3.4"
        self.clock, self.scorer_finishes_after_seconds = clock, scorer_finishes_after_seconds
        self.scorer_exit_code, self.fail_pulls, self.raise_during_install = scorer_exit_code, fail_pulls, raise_during_install
        self.fail_final_pulls, self.scorer_vanishes_after_seconds = fail_final_pulls, scorer_vanishes_after_seconds
        self.scorer_output_text = scorer_output_text
        self.install_failures_remaining, self.ssh_never_ready = install_failures, ssh_never_ready
        # scorer_attempts: one (seconds until finished, exit code) per scorer start, measured from that start
        self.scorer_attempts, self.scorer_start_times = scorer_attempts, []
        self.scorer_idle_before_first_start, self.requirements_marker_text = scorer_idle_before_first_start, requirements_marker_text

    def wait_until_ready(self, pod, deadline, get_pod_function, sleep_function):
        if self.ssh_never_ready:
            raise RuntimeError("Pod did not become reachable over ssh before the startup deadline")

    def upload_project_files(self):
        self.uploads += 1

    def run_remote_command(self, command, input_text=None, timeout_seconds=None):
        self.commands.append((command, input_text))
        if "pip install" in command and (self.raise_during_install or self.install_failures_remaining > 0):
            self.install_failures_remaining -= 1
            raise RuntimeError("install exploded")
        if "nohup bash -c" in command and "score_document_log_probabilities" in command:
            self.scorer_start_times.append(self.clock.time())
        if command.startswith("cat ") and launcher.REQUIREMENTS_INSTALLED_MARKER_FILE_NAME in command:
            return 0, self.requirements_marker_text
        if command.startswith("pgrep"):
            if self.scorer_idle_before_first_start and not self.scorer_start_times:
                return 1, ""
            vanished = (self.scorer_vanishes_after_seconds is not None
                        and self.clock.time() - 1_000_000.0 >= self.scorer_vanishes_after_seconds)
            return (1, "") if vanished else (0, "")
        if command.startswith("tail -n") and "scorer_standard_output.txt" in command:
            return 0, self.scorer_output_text
        if "scorer_exit_code.txt 2>/dev/null" in command and self.scorer_attempts is not None:
            if not self.scorer_start_times:
                return 1, ""
            finish_after_seconds, exit_code = self.scorer_attempts[min(len(self.scorer_start_times), len(self.scorer_attempts)) - 1]
            finished = self.clock.time() - self.scorer_start_times[-1] >= finish_after_seconds
            return (0, exit_code) if finished else (1, "")
        if "scorer_exit_code.txt 2>/dev/null" in command:
            finished = (self.scorer_finishes_after_seconds is not None
                        and self.clock.time() - 1_000_000.0 >= self.scorer_finishes_after_seconds)
            return (0, self.scorer_exit_code) if finished else (1, "")
        return 0, "7\nscoring progress line"

    def pull_final_copy_and_verify(self, logs_folder_name, timeout_seconds):
        self.final_pull_attempts += 1
        self.final_pull_clock_times.append(self.clock.time() if self.clock else None)
        self.logs_folder_name = logs_folder_name
        if self.fail_final_pulls > 0:
            self.fail_final_pulls -= 1
            raise launcher.FinalCopyIncompleteError("simulated mismatch", ["run_folder/scoring_outputs/a.npz (did not arrive)"])
        return 5

    def pull_scoring_outputs(self, pull_everything):
        self.pulls.append(pull_everything)
        if self.fail_pulls > 0:
            self.fail_pulls -= 1
            raise RuntimeError("pull failed")


@pytest.fixture
def run_folder(tmp_path):
    folder = tmp_path / "small_run_2026-10-22_12-00"
    (folder / "scoring_inputs").mkdir(parents=True)
    return folder


def make_arguments(run_folder, *extra, cost_center_name=TEST_COST_CENTER_NAME):
    """Most tests run with a cost center name; pass cost_center_name=None for the launcher without one."""
    cost_center_arguments = ["--cost-center-name", cost_center_name] if cost_center_name else []
    return launcher.parse_arguments(["--run-folder", str(run_folder), "--maximum-running-hours", "2",
                                     "--ssh-private-key-path", str(run_folder / "no_such_key"),
                                     *cost_center_arguments, *extra])


SECRETS = {"RUNPOD_API_KEY": "rp_test", "HF_TOKEN": "hf_test"}


class FakeConfirmationSource:
    """Stands in for the terminal listener: confirms once the fake clock passes `confirm_after_seconds`."""

    def __init__(self, clock, confirm_after_seconds=None):
        self.clock, self.confirm_after_seconds = clock, confirm_after_seconds

    @property
    def confirmed(self):
        return (self.confirm_after_seconds is not None
                and self.clock.time() - 1_000_000.0 >= self.confirm_after_seconds)


def launch(run_folder, client, transfer, clock, *extra, confirm_after_seconds=None, cost_center_name=TEST_COST_CENTER_NAME):
    return launcher.run_launcher(make_arguments(run_folder, *extra, cost_center_name=cost_center_name),
                                 client=client, transfer=transfer,
                                 secrets=SECRETS, clock=clock.time, sleep=clock.sleep,
                                 confirmation_source=FakeConfirmationSource(clock, confirm_after_seconds))


def read_cost_log(run_folder):
    return json.loads(next(run_folder.glob("pod_cost_log_*.json")).read_text())


def read_progress_log(run_folder):
    return next(run_folder.glob("pod_progress_log_*.txt")).read_text()


def test_api_key_comes_from_env_file_not_os_environ(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "key_from_os_environment")
    monkeypatch.setenv("HF_TOKEN", "token_from_os_environment")
    environment_file = tmp_path / ".env"
    environment_file.write_text("RUNPOD_API_KEY=key_from_dot_env\n")
    secrets = launcher.read_secrets_from_environment_file(environment_file)
    assert secrets == {"RUNPOD_API_KEY": "key_from_dot_env", "HF_TOKEN": None, "RUNPOD_COST_CENTER_NAME": None}
    environment_file.write_text("HF_TOKEN=token_from_dot_env\nRUNPOD_COST_CENTER_NAME=name_from_dot_env\n")
    monkeypatch.setenv("RUNPOD_COST_CENTER_NAME", "name_from_os_environment")
    secrets = launcher.read_secrets_from_environment_file(environment_file)
    assert secrets["RUNPOD_COST_CENTER_NAME"] == "name_from_dot_env" and secrets["HF_TOKEN"] == "token_from_dot_env"
    environment_file.write_text("OTHER=1\n")
    assert launcher.read_secrets_from_environment_file(environment_file)["RUNPOD_API_KEY"] is None


def test_environment_file_is_the_repository_root_one_and_parent_folders_are_not_searched(tmp_path, monkeypatch):
    assert launcher.ENVIRONMENT_FILE_PATH == launcher.REPOSITORY_ROOT / ".env"
    (tmp_path / ".env").write_text("RUNPOD_API_KEY=key_in_a_parent_folder\n")
    nested_folder = tmp_path / "nested"
    nested_folder.mkdir()
    monkeypatch.chdir(nested_folder)
    secrets = launcher.read_secrets_from_environment_file(nested_folder / ".env")  # no such file
    assert secrets["RUNPOD_API_KEY"] is None


def test_missing_api_key_fails_clearly_without_calls(run_folder):
    client = FakeRunPodClient()
    with pytest.raises(SystemExit) as excinfo:
        launcher.run_launcher(make_arguments(run_folder), client=client, transfer=FakeTransfer(),
                              secrets={"RUNPOD_API_KEY": None, "HF_TOKEN": None})
    assert "RUNPOD_API_KEY is missing" in str(excinfo.value)
    assert client.calls == []


def test_pod_terminated_on_success_and_cost_log_written(run_folder):
    clock, client = FakeClock(), FakeRunPodClient(hourly_price=2.5)
    transfer = FakeTransfer(scorer_finishes_after_seconds=600, clock=clock)
    assert launch(run_folder, client, transfer, clock) == 0
    assert client.call_names().count("delete_pod") == 1
    assert transfer.final_pull_attempts == 1  # the final pull copies everything and is verified before deletion
    log = json.loads(next(run_folder.glob("pod_cost_log_*.json")).read_text())
    assert log["pod_id"] == "fakepod123" and log["hourly_price_dollars"] == 2.5
    assert log["termination_reason"] == "scorer_finished" and log["pod_terminated"] is True
    assert log["total_cost_dollars"] == pytest.approx(2.5 * (clock.time() - 1_000_000.0) / 3600, abs=0.05)
    assert any(command.startswith("umask 077; cat >") and "HF_TOKEN" in (text or "") for command, text in transfer.commands)


def test_gpu_type_missing_from_the_creation_response_is_read_once_the_pod_is_running(run_folder):
    class ClientWhoseCreationResponseNamesNoGpu(FakeRunPodClient):
        def create_pod(self, creation_request):
            self.calls.append(("create_pod", creation_request))
            return {"id": "fakepod123", "costPerHr": self.hourly_price}

        def get_pod(self, pod_id):
            return {**super().get_pod(pod_id), "machine": {"gpuTypeId": "NVIDIA H100 80GB HBM3"}}

    clock, client = FakeClock(), ClientWhoseCreationResponseNamesNoGpu()
    transfer = FakeTransfer(scorer_finishes_after_seconds=600, clock=clock)
    assert launch(run_folder, client, transfer, clock) == 0
    assert read_cost_log(run_folder)["gpu_type"] == "NVIDIA H100 80GB HBM3"


def test_pod_terminated_on_exception(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(clock=clock, ssh_never_ready=True)
    assert launch(run_folder, client, transfer, clock) == 1
    assert client.call_names().count("delete_pod") == 1


def test_pod_terminated_on_keyboard_interrupt(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(clock=clock)
    original_sleep = clock.sleep

    def interrupting_sleep(seconds):
        original_sleep(seconds)
        if clock.time() > 1_000_000.0 + 200:
            raise KeyboardInterrupt()

    exit_code = launcher.run_launcher(make_arguments(run_folder), client=client, transfer=transfer, secrets=SECRETS,
                                      clock=clock.time, sleep=interrupting_sleep)
    assert exit_code == 130
    assert client.call_names().count("delete_pod") == 1


def test_pod_terminated_at_time_limit(run_folder):
    clock, client = FakeClock(), FakeRunPodClient(hourly_price=1.0)
    transfer = FakeTransfer(clock=clock)  # scorer never finishes
    assert launch(run_folder, client, transfer, clock) == 1
    assert client.call_names().count("delete_pod") == 1
    assert clock.time() - 1_000_000.0 == pytest.approx(2 * 3600, abs=120)
    assert json.loads(next(run_folder.glob("pod_cost_log_*.json")).read_text())["termination_reason"] == "running_hours_limit_reached"


def test_pod_terminated_at_dollar_limit(run_folder):
    clock, client = FakeClock(), FakeRunPodClient(hourly_price=10.0)
    transfer = FakeTransfer(clock=clock)
    assert launch(run_folder, client, transfer, clock, "--maximum-dollars", "5") == 1
    elapsed_hours = (clock.time() - 1_000_000.0) / 3600
    assert elapsed_hours == pytest.approx(0.5, abs=0.05)  # 5 dollars at 10 dollars/hour
    assert json.loads(next(run_folder.glob("pod_cost_log_*.json")).read_text())["termination_reason"] == "dollar_limit_reached"


def test_final_pull_failing_retries_within_bounds_then_deletes_with_loud_warning(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(scorer_finishes_after_seconds=300, clock=clock, fail_final_pulls=10_000)
    assert launch(run_folder, client, transfer, clock, confirm_after_seconds=0) == 1  # data may be missing
    assert 2 <= transfer.final_pull_attempts <= 20
    duration = transfer.final_pull_clock_times[-1] - transfer.final_pull_clock_times[0]
    assert duration <= launcher.FINAL_PULL_TOTAL_BUDGET_SECONDS
    assert client.call_names().count("delete_pod") == 1
    progress = read_progress_log(run_folder)
    assert "DATA MAY BE LOST" in progress and "a.npz (did not arrive)" in progress
    assert read_cost_log(run_folder)["final_pull_verified"] is False


def test_final_pull_that_succeeds_on_a_retry_is_followed_by_deletion(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(scorer_finishes_after_seconds=300, clock=clock, fail_final_pulls=2)
    assert launch(run_folder, client, transfer, clock, confirm_after_seconds=0) == 0
    assert transfer.final_pull_attempts == 3
    assert "DATA MAY BE LOST" not in read_progress_log(run_folder)
    assert client.call_names().count("delete_pod") == 1


def test_final_pull_retries_never_extend_past_the_hours_limit(run_folder):
    clock, client = FakeClock(), FakeRunPodClient(hourly_price=1.0)
    transfer = FakeTransfer(clock=clock, fail_final_pulls=10_000)  # scorer never finishes
    assert launch(run_folder, client, transfer, clock) == 1
    assert transfer.final_pull_attempts == 1  # the limit is already reached: one attempt, no retry window
    assert clock.time() - 1_000_000.0 == pytest.approx(2 * 3600, abs=120)
    assert client.call_names().count("delete_pod") == 1


def test_scorer_failure_prints_output_tail_pulls_then_waits_and_deletes_without_instruction(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(scorer_finishes_after_seconds=30, scorer_exit_code="1", clock=clock)
    assert launch(run_folder, client, transfer, clock) == 1  # cost center unconfirmed
    assert transfer.final_pull_attempts == 1
    waited = clock.time() - 1_000_000.0
    assert 20 * 60 <= waited < 20 * 60 + 200  # the default wait, and no 2-minute cost center window
    progress = read_progress_log(run_folder)
    assert "scorer> Traceback ... FileNotFoundError: boom" in progress
    assert "WAITING FOR A FIX" in progress and "No instruction within 20 minutes; deleting the pod" in progress
    reason = "scorer_failed_exit_code_1_and_no_instruction_within_20_minutes"
    assert f"Deleting immediately ({reason})" in progress
    assert read_cost_log(run_folder)["termination_reason"] == reason
    assert client.call_names().count("delete_pod") == 1


def test_early_crash_check_detects_an_exited_scorer_about_a_minute_after_start(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(scorer_finishes_after_seconds=10, scorer_exit_code="3", clock=clock)
    launch(run_folder, client, transfer, clock, "--failure-wait-minutes", "0")
    progress = read_progress_log(run_folder)
    assert "Scorer finished with exit code 3 (early crash check)" in progress
    assert clock.time() - 1_000_000.0 <= launcher.EARLY_CRASH_CHECK_DELAY_SECONDS + 30 + 10


def test_early_crash_check_reports_a_still_running_scorer_and_detects_a_vanished_one(run_folder):
    clock = FakeClock()
    transfer = FakeTransfer(scorer_finishes_after_seconds=500, clock=clock)
    launch(run_folder, FakeRunPodClient(), transfer, clock, confirm_after_seconds=0)
    assert "Early crash check: the scorer is still running" in read_progress_log(run_folder)
    clock = FakeClock()
    transfer = FakeTransfer(scorer_vanishes_after_seconds=20, clock=clock)
    client = FakeRunPodClient()
    assert launch(run_folder, client, transfer, clock, "--failure-wait-minutes", "0") == 1
    assert read_cost_log(run_folder)["termination_reason"].startswith("scorer_vanished_without_exit_code")
    assert transfer.final_pull_attempts == 1 and client.call_names().count("delete_pod") == 1


def test_termination_retries_after_failure(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    client.delete_failures_remaining = 2
    transfer = FakeTransfer(scorer_finishes_after_seconds=60, clock=clock)
    assert launch(run_folder, client, transfer, clock) == 0
    assert client.call_names().count("delete_pod") == 3


def test_failed_scorer_exit_code_still_pulls_and_terminates(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(scorer_finishes_after_seconds=60, scorer_exit_code="2", clock=clock)
    assert launch(run_folder, client, transfer, clock) == 1
    assert client.call_names().count("delete_pod") == 1 and transfer.final_pull_attempts == 1


def test_cost_computation():
    assert launcher.cost_so_far_dollars(3.0, 0, 5400) == pytest.approx(4.5)
    assert launcher.cost_so_far_dollars(3.0, 100, 50) == 0.0
    assert launcher.effective_limit_seconds(3, 25, 2.0) == 3 * 3600  # hours limit binds
    assert launcher.effective_limit_seconds(10, 25, 5.0) == 5 * 3600  # dollar limit binds
    with pytest.raises(RuntimeError):
        launcher.pod_cost_per_hour({})


def test_dry_run_makes_no_client_or_transfer_calls(run_folder, capsys):
    client, transfer = FakeRunPodClient(), FakeTransfer()
    exit_code = launcher.run_launcher(make_arguments(run_folder, "--dry-run"), client=client, transfer=transfer,
                                      secrets={"RUNPOD_API_KEY": None, "HF_TOKEN": None})
    output = capsys.readouterr().out
    assert exit_code == 0 and client.calls == [] and transfer.commands == [] and transfer.uploads == 0
    assert "gpuTypeIds" in output and "Cost ceiling: 25.00 dollars" in output and "small_run_2026-10-22_12-00" in output
    assert "Projected cost:" in output


def test_pod_name_includes_run_folder_name(run_folder):
    request = launcher.build_pod_creation_request(make_arguments(run_folder), "ssh-ed25519 AAAA")
    assert "small_run_2026-10-22_12-00" in request["name"]
    assert request["gpuTypeIds"][0] == "NVIDIA H100 80GB HBM3" and request["gpuTypePriority"] == "custom"


def test_scorer_command_line_matches_scorer_cli(run_folder):
    arguments = make_arguments(run_folder, "--model-name", "google/gemma-4-12b-it", "--device", "cuda", "--dtype",
                               "bfloat16", "--maximum-documents", "20", "--maximum-tokens-per-batch", "8192",
                               "--model-revision", "abc123", "--model-class", "auto_image_text_to_text",
                               "--positions-per-head-chunk", "512", "--verify-first-document-against-full-logits")
    command = launcher.build_scorer_command_line(arguments)
    assert command[:5] == ["python", "-m", launcher.SCORER_MODULE_NAME, "--run-folder", launcher.REMOTE_RUN_FOLDER]
    assert command[5:] == ["--model-name", "google/gemma-4-12b-it", "--device", "cuda", "--dtype", "bfloat16",
                           "--maximum-documents", "20", "--maximum-tokens-per-batch", "8192",
                           "--model-revision", "abc123", "--model-class", "auto_image_text_to_text",
                           "--positions-per-head-chunk", "512", "--verify-first-document-against-full-logits"]
    assert launcher.build_scorer_command_line(make_arguments(run_folder)) == command[:5]
    shell_command = launcher.build_scorer_launch_shell_command(arguments)
    assert shell_command.startswith("nohup bash -c") and shell_command.endswith("&")


def test_watchdog_uses_pod_environment_and_sleep_duration():
    command = launcher.build_self_termination_watchdog_command(7200)
    assert "sleep 7200" in command and "RUNPOD_POD_ID" in command and "runpodctl" in command


def test_missing_cost_center_name_does_not_block_or_wait_or_print_instructions(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(scorer_finishes_after_seconds=300, clock=clock)
    assert launch(run_folder, client, transfer, clock, cost_center_name=None) == 0
    assert clock.time() - 1_000_000.0 < 300 + 90  # no final cost center window
    progress = read_progress_log(run_folder)
    assert "ACTION NEEDED NOW" not in progress and "REMINDER" not in progress and "cost center" not in progress
    assert client.call_names().count("delete_pod") == 1
    log = read_cost_log(run_folder)
    assert log["cost_center_name"] is None and log["cost_center_assignment_confirmed_by_user"] is False


def test_cost_center_name_from_env_file_value_is_used_when_the_flag_is_absent(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(scorer_finishes_after_seconds=300, clock=clock)
    secrets = {**SECRETS, "RUNPOD_COST_CENTER_NAME": "name-from-dot-env"}
    launcher.run_launcher(make_arguments(run_folder, cost_center_name=None), client=client, transfer=transfer,
                          secrets=secrets, clock=clock.time, sleep=clock.sleep,
                          confirmation_source=FakeConfirmationSource(clock, 0))
    assert read_cost_log(run_folder)["cost_center_name"] == "name-from-dot-env"
    assert "name-from-dot-env" in read_progress_log(run_folder)


def test_dry_run_reports_cost_center_setting_without_failing(run_folder, capsys):
    client = FakeRunPodClient()
    for name, expected in ((None, "No cost center name given"), (TEST_COST_CENTER_NAME, f"Cost center '{TEST_COST_CENTER_NAME}'")):
        exit_code = launcher.run_launcher(make_arguments(run_folder, "--dry-run", cost_center_name=name), client=client,
                                          transfer=FakeTransfer(), secrets={"RUNPOD_API_KEY": None, "HF_TOKEN": None})
        assert exit_code == 0 and expected in capsys.readouterr().out
    assert client.calls == []


def test_reminder_at_creation_and_in_every_progress_report_until_confirmed(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(scorer_finishes_after_seconds=1000, clock=clock)
    assert launch(run_folder, client, transfer, clock, confirm_after_seconds=500) == 0
    progress = read_progress_log(run_folder)
    assert "ACTION NEEDED NOW" in progress and "fakepod123" in progress and TEST_COST_CENTER_NAME in progress
    assert "https://www.console.runpod.io/user/cost-center/manage" in progress and "log-probability-scoring-" in progress
    assert progress.count("REMINDER:") >= 2  # reports at about 180 and 360 seconds precede the confirmation
    assert "confirmed by the user; reminders stopped" in progress
    reminders_after_confirmation = progress.split("reminders stopped")[1].count("REMINDER:")
    assert reminders_after_confirmation == 0
    assert "WARNING" not in progress
    log = read_cost_log(run_folder)
    assert log["cost_center_name"] == TEST_COST_CENTER_NAME and log["cost_center_assignment_confirmed_by_user"] is True
    assert log["cost_center_assignment_confirmation_time"] is not None
    assert client.call_names().count("delete_pod") == 1


def test_unconfirmed_normal_completion_waits_final_window_then_deletes(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(scorer_finishes_after_seconds=300, clock=clock)
    assert launch(run_folder, client, transfer, clock) == 0
    log = read_cost_log(run_folder)
    finished_at = clock.time() - 1_000_000.0
    assert 300 + launcher.FINAL_COST_CENTER_ASSIGNMENT_WINDOW_SECONDS <= finished_at <= 300 + 120 + 90
    assert log["stop_time"] and log["cost_center_assignment_confirmed_by_user"] is False
    assert log["cost_center_assignment_confirmation_time"] is None
    assert "will appear as uncategorized" in read_progress_log(run_folder)
    assert client.call_names().count("delete_pod") == 1


def test_confirmation_during_final_window_deletes_promptly(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(scorer_finishes_after_seconds=300, clock=clock)
    assert launch(run_folder, client, transfer, clock, confirm_after_seconds=360) == 0
    assert clock.time() - 1_000_000.0 < 300 + 90  # well short of the full window
    assert read_cost_log(run_folder)["cost_center_assignment_confirmed_by_user"] is True


def test_limits_delete_without_waiting_when_unconfirmed(run_folder):
    clock, client = FakeClock(), FakeRunPodClient(hourly_price=1.0)
    assert launch(run_folder, client, FakeTransfer(clock=clock), clock) == 1
    assert clock.time() - 1_000_000.0 == pytest.approx(2 * 3600, abs=120)  # no extra window after the hours limit
    assert "Deleting immediately (running_hours_limit_reached)" in read_progress_log(run_folder)
    clock, client = FakeClock(), FakeRunPodClient(hourly_price=10.0)
    assert launch(run_folder, client, FakeTransfer(clock=clock), clock, "--maximum-dollars", "5") == 1
    assert (clock.time() - 1_000_000.0) / 3600 == pytest.approx(0.5, abs=0.05)
    assert client.call_names().count("delete_pod") == 1


def test_exception_and_interrupt_delete_without_waiting_when_unconfirmed(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    assert launch(run_folder, client, FakeTransfer(clock=clock, ssh_never_ready=True), clock) == 1
    assert clock.time() - 1_000_000.0 < 60
    assert client.call_names().count("delete_pod") == 1
    clock, client = FakeClock(), FakeRunPodClient()
    original_sleep = clock.sleep

    def interrupting_sleep(seconds):
        original_sleep(seconds)
        if clock.time() > 1_000_000.0 + 200:
            raise KeyboardInterrupt()

    exit_code = launcher.run_launcher(make_arguments(run_folder), client=client, transfer=FakeTransfer(clock=clock),
                                      secrets=SECRETS, clock=clock.time, sleep=interrupting_sleep,
                                      confirmation_source=FakeConfirmationSource(clock))
    assert exit_code == 130 and clock.time() - 1_000_000.0 < 260
    assert client.call_names().count("delete_pod") == 1


def test_attach_to_existing_pod_gets_the_same_reminders(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(scorer_finishes_after_seconds=300, clock=clock)
    assert launch(run_folder, client, transfer, clock, "--attach-to-existing-pod", "fakepod123") == 0
    progress = read_progress_log(run_folder)
    assert "ACTION NEEDED NOW" in progress and "fakepod123" in progress and "will appear as uncategorized" in progress


def test_stdin_listener_confirms_on_typed_word_and_survives_end_of_input():
    import io
    listener = launcher.StdinConfirmationListener(io.StringIO("hello\n  Assigned \n"))
    listener.start()
    listener.thread.join(timeout=5)
    assert listener.confirmed is True
    no_input_listener = launcher.StdinConfirmationListener(io.StringIO(""))
    no_input_listener.start()
    no_input_listener.thread.join(timeout=5)
    assert no_input_listener.confirmed is False and not no_input_listener.thread.is_alive()


def test_confirmation_file_appearing_mid_run_confirms_and_stops_reminders(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(scorer_finishes_after_seconds=1000, clock=clock)
    confirmation_file = run_folder / launcher.COST_CENTER_CONFIRMATION_FILE_NAME

    def sleeping_then_creating_file(seconds):
        clock.sleep(seconds)
        if clock.time() - 1_000_000.0 >= 400:
            confirmation_file.write_text("assigned")

    exit_code = launcher.run_launcher(make_arguments(run_folder), client=client, transfer=transfer, secrets=SECRETS,
                                      clock=clock.time, sleep=sleeping_then_creating_file,
                                      confirmation_source=FakeConfirmationSource(clock))
    assert exit_code == 0
    progress = read_progress_log(run_folder)
    assert str(confirmation_file.resolve()) in progress and "type `assigned`" in progress
    assert "reminders stopped" in progress and progress.split("reminders stopped")[1].count("REMINDER:") == 0
    assert "WARNING" not in progress
    log = read_cost_log(run_folder)
    assert log["cost_center_assignment_confirmed_by_user"] is True and log["cost_center_assignment_confirmation_time"]


def test_confirmation_file_present_during_final_window_ends_it_early(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(scorer_finishes_after_seconds=300, clock=clock)
    confirmation_file = run_folder / launcher.COST_CENTER_CONFIRMATION_FILE_NAME

    def sleeping_then_creating_file(seconds):
        clock.sleep(seconds)
        if clock.time() - 1_000_000.0 >= 340:
            confirmation_file.write_text("assigned")

    exit_code = launcher.run_launcher(make_arguments(run_folder), client=client, transfer=transfer, secrets=SECRETS,
                                      clock=clock.time, sleep=sleeping_then_creating_file,
                                      confirmation_source=FakeConfirmationSource(clock))
    assert exit_code == 0 and clock.time() - 1_000_000.0 < 300 + 90
    assert read_cost_log(run_folder)["cost_center_assignment_confirmed_by_user"] is True
    assert client.call_names().count("delete_pod") == 1


def test_pod_setup_logs_torch_and_transformers_versions_before_scoring_starts(run_folder):
    clock = FakeClock()
    transfer = FakeTransfer(scorer_finishes_after_seconds=60, clock=clock)
    launch(run_folder, FakeRunPodClient(), transfer, clock, confirm_after_seconds=0)
    commands = [command for command, _ in transfer.commands]
    version_index = next(i for i, command in enumerate(commands) if "import torch, transformers" in command)
    install_index = next(i for i, command in enumerate(commands) if "pip install" in command)
    scorer_index = next(i for i, command in enumerate(commands) if "score_document_log_probabilities" in command)
    assert install_index < version_index < scorer_index
    assert "Pod environment:" in read_progress_log(run_folder)


# ---- failure waiting, instruction files, keeping the pod, reusing a pod ----

def sleep_creating_files_at(clock, files_by_second):
    """A fake sleep that creates each file once the fake clock passes its second (counted from the start)."""
    created = set()

    def sleeping(seconds):
        clock.sleep(seconds)
        for file_path, at_second in files_by_second.items():
            if file_path not in created and clock.time() - 1_000_000.0 >= at_second:
                created.add(file_path)
                Path(file_path).write_text("go")
    return sleeping


def launch_with_sleep(run_folder, client, transfer, clock, sleep_function, *extra, confirm_after_seconds=None):
    return launcher.run_launcher(make_arguments(run_folder, *extra), client=client, transfer=transfer, secrets=SECRETS,
                                 clock=clock.time, sleep=sleep_function,
                                 confirmation_source=FakeConfirmationSource(clock, confirm_after_seconds))


def scorer_start_commands(transfer):
    return [command for command, _ in transfer.commands if "nohup bash -c" in command and "score_document_log_probabilities" in command]


def test_keep_pod_after_run_leaves_the_pod_running_and_logs_it(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(scorer_finishes_after_seconds=300, clock=clock)
    assert launch(run_folder, client, transfer, clock, "--keep-pod-after-run") == 0
    assert "delete_pod" not in client.call_names() and transfer.final_pull_attempts == 1
    log = read_cost_log(run_folder)
    assert log["pod_kept_after_run"] is True and log["pod_terminated"] is False and log["termination_reason"] == "kept_after_run"
    progress = read_progress_log(run_folder)
    assert "KEPT RUNNING" in progress and "hard limits still apply" in progress
    assert clock.time() - 1_000_000.0 < 400  # no cost center window when keeping


def test_success_without_the_flag_deletes_the_pod(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    assert launch(run_folder, client, FakeTransfer(scorer_finishes_after_seconds=300, clock=clock), clock) == 0
    assert client.call_names().count("delete_pod") == 1 and "pod_kept_after_run" not in read_cost_log(run_folder)


def test_scorer_failure_waits_then_retry_file_restarts_the_scorer_on_the_same_pod_and_run_completes(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(clock=clock, scorer_attempts=[(30, "1"), (100, "0")])
    retry_file = run_folder / launcher.RETRY_SCORER_INSTRUCTION_FILE_NAME
    sleep_function = sleep_creating_files_at(clock, {str(retry_file): 400})
    assert launch_with_sleep(run_folder, client, transfer, clock, sleep_function) == 0
    assert client.call_names().count("create_pod") == 1 and client.call_names().count("delete_pod") == 1
    assert transfer.uploads == 2 and len(scorer_start_commands(transfer)) == 2
    assert not retry_file.exists() and len(list(run_folder.glob(retry_file.name + ".consumed_*"))) == 1
    assert transfer.final_pull_attempts == 2  # after the failure and again after the finish
    progress = read_progress_log(run_folder)
    assert str(retry_file.resolve()) in progress and str(run_folder.resolve() / launcher.DELETE_POD_INSTRUCTION_FILE_NAME) in progress
    assert progress.count("WAITING FOR A FIX") >= 2  # at the start of the wait and in a later progress report
    assert read_cost_log(run_folder)["termination_reason"] == "scorer_finished"
    assert sum("rm -f" in command and "scorer_exit_code.txt" in command for command, _ in transfer.commands) == 2


def test_delete_file_deletes_the_pod_after_a_scorer_failure(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(clock=clock, scorer_attempts=[(30, "1")])
    delete_file = run_folder / launcher.DELETE_POD_INSTRUCTION_FILE_NAME
    sleep_function = sleep_creating_files_at(clock, {str(delete_file): 200})
    assert launch_with_sleep(run_folder, client, transfer, clock, sleep_function) == 1
    assert client.call_names().count("delete_pod") == 1 and transfer.uploads == 1
    assert clock.time() - 1_000_000.0 < 300
    assert read_cost_log(run_folder)["termination_reason"] == "instruction_delete_pod"
    assert len(list(run_folder.glob(delete_file.name + ".consumed_*"))) == 1


def test_no_instruction_deletes_after_the_configured_wait(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(clock=clock, scorer_attempts=[(30, "1")])
    assert launch(run_folder, client, transfer, clock, "--failure-wait-minutes", "5") == 1
    assert 5 * 60 <= clock.time() - 1_000_000.0 < 5 * 60 + 100
    assert client.call_names().count("delete_pod") == 1


def test_hard_limits_delete_immediately_while_waiting_for_an_instruction(run_folder):
    clock, client = FakeClock(), FakeRunPodClient(hourly_price=1.0)
    transfer = FakeTransfer(clock=clock, scorer_attempts=[(30, "1")])
    assert launch(run_folder, client, transfer, clock, "--failure-wait-minutes", "600") == 1
    assert clock.time() - 1_000_000.0 == pytest.approx(2 * 3600, abs=60)
    assert read_cost_log(run_folder)["termination_reason"] == "running_hours_limit_reached"
    clock, client = FakeClock(), FakeRunPodClient(hourly_price=10.0)
    transfer = FakeTransfer(clock=clock, scorer_attempts=[(30, "1")])
    assert launch(run_folder, client, transfer, clock, "--failure-wait-minutes", "600", "--maximum-dollars", "5") == 1
    assert (clock.time() - 1_000_000.0) / 3600 == pytest.approx(0.5, abs=0.02)
    assert client.call_names().count("delete_pod") == 1 and transfer.final_pull_attempts == 1


def test_ctrl_c_deletes_immediately_while_waiting_for_an_instruction(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(clock=clock, scorer_attempts=[(30, "1")])
    original_sleep = clock.sleep

    def interrupting_sleep(seconds):
        original_sleep(seconds)
        if clock.time() > 1_000_000.0 + 300:
            raise KeyboardInterrupt()

    assert launch_with_sleep(run_folder, client, transfer, clock, interrupting_sleep) == 130
    assert clock.time() - 1_000_000.0 < 400 and client.call_names().count("delete_pod") == 1


def test_setup_failure_waits_and_retry_setup_file_redoes_setup_on_the_same_pod(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(clock=clock, install_failures=1, scorer_attempts=[(100, "0")])
    retry_file = run_folder / launcher.RETRY_SETUP_INSTRUCTION_FILE_NAME
    sleep_function = sleep_creating_files_at(clock, {str(retry_file): 100})
    assert launch_with_sleep(run_folder, client, transfer, clock, sleep_function) == 0
    assert client.call_names().count("create_pod") == 1 and client.call_names().count("delete_pod") == 1
    commands = [command for command, _ in transfer.commands]
    assert transfer.uploads == 2 and sum("pip install" in command for command in commands) == 2
    assert len(scorer_start_commands(transfer)) == 1
    assert "Setup failed: install exploded" in read_progress_log(run_folder)
    assert len(list(run_folder.glob(retry_file.name + ".consumed_*"))) == 1


def test_setup_failure_without_instruction_deletes_after_the_wait_and_ssh_failure_deletes_at_once(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    assert launch(run_folder, client, FakeTransfer(clock=clock, raise_during_install=True), clock) == 1
    assert 20 * 60 <= clock.time() - 1_000_000.0 < 20 * 60 + 100 and client.call_names().count("delete_pod") == 1
    clock, client = FakeClock(), FakeRunPodClient()
    assert launch(run_folder, client, FakeTransfer(clock=clock, ssh_never_ready=True), clock) == 1
    assert clock.time() - 1_000_000.0 < 60 and client.call_names().count("delete_pod") == 1


def test_attach_to_existing_pod_follows_the_failure_waiting_behaviour(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(clock=clock, scorer_attempts=[(30, "1")])
    # attached monitoring sees the scorer that was already started before the attach
    transfer.scorer_start_times.append(clock.time())
    delete_file = run_folder / launcher.DELETE_POD_INSTRUCTION_FILE_NAME
    sleep_function = sleep_creating_files_at(clock, {str(delete_file): 200})
    assert launch_with_sleep(run_folder, client, transfer, clock, sleep_function,
                             "--attach-to-existing-pod", "fakepod123") == 1
    assert "create_pod" not in client.call_names() and client.call_names().count("delete_pod") == 1
    assert "WAITING FOR A FIX" in read_progress_log(run_folder)


def test_reuse_existing_pod_skips_creation_and_pip_replaces_watchdog_before_scorer_and_does_not_remind(
        run_folder, tmp_path):
    earlier_runs = tmp_path / "some_earlier_run"  # a sibling of the run folder
    earlier_runs.mkdir(parents=True)
    (earlier_runs / "pod_cost_log_2026-10-22_01-00.json").write_text(json.dumps(
        {"pod_id": "fakepod123", "cost_center_assignment_confirmed_by_user": True}))
    (earlier_runs / "pod_cost_log_2026-10-22_01-01.json").write_text("not json")
    clock, client = FakeClock(), FakeRunPodClient(hourly_price=3.0)
    transfer = FakeTransfer(clock=clock, scorer_finishes_after_seconds=300, scorer_idle_before_first_start=True,
                            requirements_marker_text=(launcher.requirements_file_hash() or "") + "\n")
    assert launch(run_folder, client, transfer, clock, "--reuse-existing-pod", "fakepod123", "--keep-pod-after-run") == 0
    assert "create_pod" not in client.call_names() and "delete_pod" not in client.call_names()
    commands = [command for command, _ in transfer.commands]
    assert not any("pip install" in command for command in commands)
    kill_index = next(i for i, command in enumerate(commands) if command.startswith("pkill -f '" + launcher.WATCHDOG_PROCESS_PATTERN))
    watchdog_index = next(i for i, command in enumerate(commands) if "sleep " in command and "runpodctl" in command and command.startswith("nohup"))
    scorer_index = next(i for i, command in enumerate(commands) if "nohup bash -c" in command and "score_document_log_probabilities" in command)
    assert kill_index < watchdog_index < scorer_index
    assert any("_before_reuse_" in command for command in commands)  # the pod's old run folder is moved aside
    progress = read_progress_log(run_folder)
    assert "ACTION NEEDED NOW" not in progress and "REMINDER" not in progress and "skipping pip install" in progress
    log = read_cost_log(run_folder)
    assert log["reused_existing_pod"] is True and log["hourly_price_dollars"] == 3.0 and log["pod_kept_after_run"] is True
    assert log["cost_center_assignment_confirmed_in_earlier_cost_log"] is True
    assert log["total_cost_dollars"] == pytest.approx(3.0 * 300 / 3600, abs=0.1)  # counted from this invocation


def test_reuse_existing_pod_installs_when_marker_differs_and_reminds_when_unconfirmed(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(clock=clock, scorer_finishes_after_seconds=300, scorer_idle_before_first_start=True,
                            requirements_marker_text="some other hash")
    assert launch(run_folder, client, transfer, clock, "--reuse-existing-pod", "fakepod123", "--keep-pod-after-run") == 0
    assert any("pip install" in command for command, _ in transfer.commands)
    assert "ACTION NEEDED NOW" in read_progress_log(run_folder)


def test_reuse_existing_pod_refuses_when_a_scorer_is_already_running_and_leaves_the_pod_alone(run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeTransfer(clock=clock)  # pgrep reports a running scorer
    with pytest.raises(SystemExit):
        launch(run_folder, client, transfer, clock, "--reuse-existing-pod", "fakepod123")
    assert "delete_pod" not in client.call_names() and transfer.uploads == 0


def test_dry_run_describes_waiting_instruction_files_keeping_and_reuse(run_folder, capsys):
    launcher.run_launcher(make_arguments(run_folder, "--dry-run", "--keep-pod-after-run"), client=FakeRunPodClient(),
                          transfer=FakeTransfer(), secrets={"RUNPOD_API_KEY": None, "HF_TOKEN": None})
    output = capsys.readouterr().out
    for expected in (launcher.RETRY_SCORER_INSTRUCTION_FILE_NAME, launcher.RETRY_SETUP_INSTRUCTION_FILE_NAME,
                     launcher.DELETE_POD_INSTRUCTION_FILE_NAME, "20 minutes", "KEPT running", "--reuse-existing-pod"):
        assert expected in output


def test_ssh_readiness_check_retries_after_a_timeout_until_it_succeeds_or_the_deadline(run_folder, monkeypatch):
    import subprocess
    transfer = launcher.SshTarTransfer("unused_key", run_folder)
    outcomes = [subprocess.TimeoutExpired("ssh", 40), subprocess.TimeoutExpired("ssh", 40), (0, "ssh_ready")]

    def fake_run_remote_command(command, input_text=None, timeout_seconds=None):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(transfer, "run_remote_command", fake_run_remote_command)
    pod = {"publicIp": "1.2.3.4", "portMappings": {"22": 4000}}
    transfer.wait_until_ready(pod, 1e18, lambda: pod, lambda seconds: None)
    assert outcomes == [] and transfer.host == "1.2.3.4"
    monkeypatch.setattr(transfer, "run_remote_command",
                        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("ssh", 40)))
    with pytest.raises(RuntimeError, match="did not become reachable"):
        transfer.wait_until_ready(pod, 0, lambda: pod, lambda seconds: None)


# ---- new in the pipeline repository: projected cost, local tar flags, REST client, upload list

def test_projected_cost_range_scales_with_tokens_and_is_ordered():
    low_small, high_small = launcher.project_cost_range_dollars(1_000_000)
    low_large, high_large = launcher.project_cost_range_dollars(100_000_000)
    assert 0 < low_small < high_small and low_small < low_large and high_small < high_large
    only_setup_low, only_setup_high = launcher.project_cost_range_dollars(0)
    assert only_setup_low == pytest.approx(launcher.PRICE_RANGE_DOLLARS_PER_HOUR[0] * launcher.PROJECTED_SETUP_SECONDS / 3600)
    assert only_setup_high == pytest.approx(launcher.PRICE_RANGE_DOLLARS_PER_HOUR[1] * launcher.PROJECTED_SETUP_SECONDS / 3600)


def test_dry_run_prints_projected_cost_range_when_tokens_are_countable(run_folder, capsys, monkeypatch):
    monkeypatch.setattr(launcher, "count_total_tokens_through_model", lambda folder: 50_000_000)
    launcher.run_launcher(make_arguments(run_folder, "--dry-run"), secrets={"RUNPOD_API_KEY": None, "HF_TOKEN": None})
    low, high = launcher.project_cost_range_dollars(50_000_000)
    assert f"Projected cost: {low:.2f} to {high:.2f} dollars for 50,000,000 tokens" in capsys.readouterr().out


def test_neutral_defaults_are_flags(run_folder):
    arguments = launcher.parse_arguments(["--run-folder", str(run_folder), "--maximum-running-hours", "1"])
    assert arguments.cost_center_name is None and arguments.cloud_type == "SECURE"
    assert arguments.ssh_private_key_path == str(Path.home() / ".ssh" / "id_ed25519")
    assert arguments.gpu_type_ids == launcher.DEFAULT_GPU_TYPE_IDS and arguments.image_name == launcher.DEFAULT_POD_IMAGE_NAME
    custom = launcher.parse_arguments(["--run-folder", str(run_folder), "--maximum-running-hours", "1",
                                       "--cloud-type", "COMMUNITY", "--image-name", "some/image:tag",
                                       "--gpu-type-ids", "NVIDIA A100 80GB PCIe", "--ssh-private-key-path", "/keys/k"])
    request = launcher.build_pod_creation_request(custom, "ssh-ed25519 AAAA")
    assert request["cloudType"] == "COMMUNITY" and request["imageName"] == "some/image:tag"
    assert request["gpuTypeIds"] == ["NVIDIA A100 80GB PCIe"] and custom.ssh_private_key_path == "/keys/k"


def test_upload_list_has_the_scorer_package_and_prompts_but_not_the_shared_package():
    paths = launcher.list_repository_paths_to_upload()
    assert paths == ["evals/__init__.py", "evals/log_probability_persona_check", "prompts/tools/persona_checks"]
    assert not any(path.startswith("shared") for path in paths)
    assert launcher.REQUIREMENTS_FILE_RELATIVE_PATH.startswith("evals/log_probability_persona_check/")
    assert (launcher.REPOSITORY_ROOT / launcher.REQUIREMENTS_FILE_RELATIVE_PATH).is_file()


def test_local_tar_command_uses_only_flags_the_local_tar_supports(monkeypatch, tmp_path):
    for supported_flags in ([], ["--no-mac-metadata"], ["--no-mac-metadata", "--no-xattrs"]):
        monkeypatch.setattr(launcher, "local_tar_supports_flag", lambda flag, supported_flags=supported_flags: flag in supported_flags)
        command = launcher.build_local_tar_command(tmp_path, ["a", "b"])
        assert [flag for flag in ("--no-mac-metadata", "--no-xattrs") if flag in command] == supported_flags
        assert command[-4:] == ["-C", str(tmp_path), "a", "b"] and "--exclude=*.partial" in command
    assert isinstance(launcher.local_tar_supports_flag("--no-such-tar-flag-at-all"), bool)
    assert launcher.local_tar_supports_flag("--no-such-tar-flag-at-all") is False


def test_copyfile_disable_is_set_only_on_macos(monkeypatch):
    monkeypatch.setattr(launcher.platform, "system", lambda: "Darwin")
    assert launcher.build_local_tar_environment()["COPYFILE_DISABLE"] == "1"
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.delenv("COPYFILE_DISABLE", raising=False)
    assert "COPYFILE_DISABLE" not in launcher.build_local_tar_environment()


def test_rest_client_sends_bearer_token_and_raises_on_http_errors(monkeypatch):
    import io
    import urllib.error
    requests_seen = []

    class FakeResponse:
        status = 201

        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *arguments):
            return False

    def fake_urlopen(request, timeout):
        requests_seen.append(request)
        if request.method == "DELETE":
            raise urllib.error.HTTPError(request.full_url, 500, "boom", {}, io.BytesIO(b"server error"))
        return FakeResponse(json.dumps({"id": "abc", "costPerHr": 2.0}).encode())

    monkeypatch.setattr(launcher.urllib.request, "urlopen", fake_urlopen)
    client = launcher.RunPodRestClient("rp_secret")
    assert client.create_pod({"name": "n"})["id"] == "abc"
    assert requests_seen[0].get_header("Authorization") == "Bearer rp_secret"
    # urllib.request.Request stores header names capitalized, e.g. "User-agent".
    assert requests_seen[0].get_header("User-agent") == launcher.RUNPOD_REST_CLIENT_USER_AGENT
    assert requests_seen[0].full_url == f"{launcher.RUNPOD_REST_BASE_URL}/pods" and json.loads(requests_seen[0].data) == {"name": "n"}
    assert client.get_pod("abc")["costPerHr"] == 2.0
    with pytest.raises(RuntimeError, match="deletion failed .500."):
        client.delete_pod("abc")


def test_pod_volume_holds_both_models_and_any_gpu_type_can_be_passed(run_folder):
    arguments = launcher.parse_arguments(["--run-folder", str(run_folder), "--maximum-running-hours", "1",
                                          "--gpu-type-ids", "NVIDIA H200"])
    request = launcher.build_pod_creation_request(arguments, "ssh-ed25519 AAAA")
    assert request["gpuTypeIds"] == ["NVIDIA H200"]
    assert request["volumeInGb"] == 200 and request["containerDiskInGb"] == 50
    assert request["env"]["HF_HOME"].startswith("/workspace/")  # the cache is on the volume, not the container disk
