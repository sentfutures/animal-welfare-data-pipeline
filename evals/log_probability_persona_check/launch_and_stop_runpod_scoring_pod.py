"""Launch one RunPod GPU pod, run the log-probability scorer on it, pull results back, and always terminate the pod.

Run from the repository root:
    python evals/log_probability_persona_check/launch_and_stop_runpod_scoring_pod.py \\
        --run-folder outputs/sdf/runs/<run id>/audit/<run folder name> --maximum-running-hours 3 --dry-run
Drop --dry-run to launch (a paid pod).

Talks to RunPod's documented REST API (https://rest.runpod.io/v1, bearer token) with the standard library; the
`runpod` Python package is not needed. Files move over SSH on the pod's exposed TCP port 22 using tar streams,
because RunPod's proxied SSH does not support scp/sftp and rsync is not guaranteed to exist in the image.

Cost safety, in layers:
  1. This launcher terminates (deletes) the pod at the running-hours limit, the dollar limit, Ctrl-C, SIGTERM, pod
     creation failure, ssh never becoming ready, and a normal finish (unless --keep-pod-after-run).
  2. A watchdog started on the pod deletes the pod itself after the limit, even if the laptop sleeps.
A scorer failure or a quick-to-fix setup failure does not delete the pod: after the verified final pull the launcher
waits for an instruction file in the run folder (pod_instruction_retry_scorer.txt, pod_instruction_retry_setup.txt or
pod_instruction_delete_pod.txt), and deletes the pod if none arrives within --failure-wait-minutes.

Keys: RUNPOD_API_KEY and HF_TOKEN (only if the model is gated) are read only from the repository-root .env file,
which is parsed directly; the OS environment and parent folders are never consulted.

What goes to the pod: the scorer's package folder (evals/log_probability_persona_check/ and evals/__init__.py), the
prompt files (prompts/tools/persona_checks/), and from the run folder the run_configuration_*.json files,
scoring_inputs/ and any existing scoring_outputs/. The repository's own `shared` package is not uploaded: the scorer
does not import it.

Optional cost center: RunPod's APIs cannot assign a pod to a cost center, so it is done by hand in the console while
the pod is running. With --cost-center-name (or RUNPOD_COST_CENTER_NAME in .env) the launcher prints the manual
instructions at pod creation and in every progress report until the user types `assigned` and Enter in this
terminal (or creates cost_center_assignment_confirmed_by_user.txt in the run folder, for launchers running in a
background shell), gives one final window before a normally completed pod is deleted, and records the name and the
confirmation in the pod cost log. Without a name nothing is printed, awaited or blocked.
"""

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evals.log_probability_persona_check.persona_sets import PERSONA_CHECKS_PROMPT_DIRECTORY  # noqa: E402
from evals.log_probability_persona_check.prompt_rendering import count_tokens_for_scoring_inputs  # noqa: E402
from evals.log_probability_persona_check.timed_progress_and_jsonl_helpers import (  # noqa: E402
    PROGRESS_REPORT_INTERVAL_SECONDS, format_duration, read_jsonl, timestamp_now)

RUNPOD_REST_BASE_URL = "https://rest.runpod.io/v1"
ENVIRONMENT_FILE_PATH = REPOSITORY_ROOT / ".env"
PACKAGE_FOLDER_RELATIVE_PATH = "evals/log_probability_persona_check"
SCORER_MODULE_NAME = "evals.log_probability_persona_check.score_document_log_probabilities_on_pod"
REQUIREMENTS_FILE_RELATIVE_PATH = f"{PACKAGE_FOLDER_RELATIVE_PATH}/requirements_for_gemma_4_scoring_pod.txt"
# Relative to the repository root, exactly what the pod needs to import the scorer, prompt_rendering and
# persona_sets, and to read the prompt files: nothing else of the repository (in particular not `shared`).
REPOSITORY_PATHS_UPLOADED_TO_POD = ["evals/__init__.py", PACKAGE_FOLDER_RELATIVE_PATH, "prompts/tools/persona_checks"]
FILE_NAME_PATTERNS_NEVER_UPLOADED = ["__pycache__", "*.pyc", ".DS_Store"]
FILE_NAME_PATTERN_OF_UNFINISHED_FILES = "*.partial"  # the scorer writes under this suffix, then renames

# Defaults for the flags of the same names. GPU type ids are RunPod's REST gpuTypeIds enum strings, in priority order.
DEFAULT_GPU_TYPE_IDS = [
    "NVIDIA H100 80GB HBM3",     # H100 SXM
    "NVIDIA H100 PCIe",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
]
DEFAULT_CLOUD_TYPE = "SECURE"
DEFAULT_SSH_PRIVATE_KEY_PATH = Path.home() / ".ssh" / "id_ed25519"
DEFAULT_POD_IMAGE_NAME = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
POD_VERSION_CHECK_COMMAND = ("python -c 'import torch, transformers; print(\"torch\", torch.__version__, "
                             "\"transformers\", transformers.__version__, \"cuda available\", "
                             "torch.cuda.is_available())'")
REMOTE_WORKING_DIRECTORY = "/workspace/log_probability_persona_check_scoring_copy"
REMOTE_RUN_FOLDER = f"{REMOTE_WORKING_DIRECTORY}/run_folder"
WATCHDOG_GRACE_SECONDS = 600  # the pod's own deletion fires this long after the launcher's limit
LOOP_SLEEP_SECONDS = 30
SCORER_FINISHED_CHECK_INTERVAL_SECONDS = 60
EARLY_CRASH_CHECK_DELAY_SECONDS = 60  # after starting the scorer, check once that it has not already crashed
SCORER_OUTPUT_TAIL_LINES = 40
FINAL_PULL_TOTAL_BUDGET_SECONDS = 300  # attempts plus waits, also never past the hours and dollar limits
FINAL_PULL_RETRY_WAIT_SECONDS = 20
FINAL_PULL_MINIMUM_ATTEMPT_TIMEOUT_SECONDS = 60
SCORER_PROCESS_PATTERN = "[s]core_document_log_probabilities_on_pod"  # bracket keeps pgrep and pkill from matching themselves
PIP_INSTALL_LOG_FILE_NAME = "pod_setup_pip_install_log.txt"
ENVIRONMENT_VERSIONS_LOG_FILE_NAME = "pod_setup_environment_versions.txt"
TERMINATION_ATTEMPTS = 5
COST_CENTER_CONSOLE_URL = "https://www.console.runpod.io/user/cost-center/manage"
COST_CENTER_CONFIRMATION_WORD = "assigned"
COST_CENTER_NAME_ENVIRONMENT_FILE_KEY = "RUNPOD_COST_CENTER_NAME"
COST_CENTER_CONFIRMATION_FILE_NAME = "cost_center_assignment_confirmed_by_user.txt"
FINAL_COST_CENTER_ASSIGNMENT_WINDOW_SECONDS = 120
COST_CENTER_CONFIRMATION_CHECK_INTERVAL_SECONDS = 5
RETRY_SCORER_INSTRUCTION_FILE_NAME = "pod_instruction_retry_scorer.txt"
RETRY_SETUP_INSTRUCTION_FILE_NAME = "pod_instruction_retry_setup.txt"
DELETE_POD_INSTRUCTION_FILE_NAME = "pod_instruction_delete_pod.txt"
INSTRUCTION_CHECK_INTERVAL_SECONDS = 10
# Rough planning figures for the dry run's projected cost (not measurements): dollars per hour for the default GPU
# types, input tokens per second for a 12-billion-parameter model on one such GPU, and pod setup time.
PRICE_RANGE_DOLLARS_PER_HOUR = (2.0, 4.0)
THROUGHPUT_RANGE_TOKENS_PER_SECOND = (5_000, 20_000)
PROJECTED_SETUP_SECONDS = 900
DEFAULT_FAILURE_WAIT_MINUTES = 20
# Matches the command line of the launcher's watchdog and of hand-started ones that delete via $RUNPOD_POD_ID; the
# brackets keep pkill from matching its own command line.
WATCHDOG_PROCESS_PATTERN = "[r]unpodctl pod delete|[R]UNPOD_POD_ID"
REQUIREMENTS_INSTALLED_MARKER_FILE_NAME = "pod_setup_requirements_installed_marker.txt"


class RunPodRestClient:
    """The three RunPod REST calls the launcher needs (documented at docs.runpod.io/api-reference/pods)."""

    def __init__(self, api_key):
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _request(self, method, path, body=None):
        request = urllib.request.Request(f"{RUNPOD_REST_BASE_URL}{path}", method=method, headers=self.headers,
                                         data=json.dumps(body).encode("utf-8") if body is not None else None)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode("utf-8", errors="replace")

    def create_pod(self, creation_request):
        status, text = self._request("POST", "/pods", creation_request)
        if status >= 300:
            raise RuntimeError(f"RunPod pod creation failed ({status}): {text[:500]}")
        return json.loads(text)

    def get_pod(self, pod_id):
        status, text = self._request("GET", f"/pods/{pod_id}")
        if status >= 300:
            raise RuntimeError(f"RunPod get pod failed ({status}): {text[:500]}")
        return json.loads(text)

    def delete_pod(self, pod_id):
        status, text = self._request("DELETE", f"/pods/{pod_id}")
        if status not in (200, 204, 404):
            raise RuntimeError(f"RunPod pod deletion failed ({status}): {text[:500]}")


class StdinConfirmationListener:
    """Reads lines from the terminal in a daemon thread; `confirmed` becomes True once the user types `assigned`.
    If there is no terminal or input ends, the thread just stops and the launcher keeps reminding."""

    def __init__(self, input_stream=None):
        self.input_stream = input_stream if input_stream is not None else sys.stdin
        self.confirmed = False
        self.thread = threading.Thread(target=self._read_until_confirmed, daemon=True)

    def start(self):
        self.thread.start()

    def _read_until_confirmed(self):
        try:
            for line in iter(self.input_stream.readline, ""):
                if line.strip().lower() == COST_CENTER_CONFIRMATION_WORD:
                    self.confirmed = True
                    return
        except (OSError, ValueError):
            return


class FinalCopyIncompleteError(RuntimeError):
    """The final pull ran but some files did not arrive intact. `problem_files` names them (None if unknown)."""

    def __init__(self, message, problem_files=None):
        super().__init__(message)
        self.problem_files = problem_files


# Shell that prints, one per line, every file of the final copy relative to REMOTE_WORKING_DIRECTORY: everything in the
# remote run folder except scoring_inputs/ (the local copy of those is the original), plus the watchdog and setup logs.
# pod_secret_environment.sh holds the Hugging Face token and is deliberately never copied back.
FINAL_COPY_FILE_LIST_SHELL = (
    "find run_folder -type f -not -path 'run_folder/scoring_inputs/*' -not -name '.last_pull_marker*' "
    f"-not -name '{FILE_NAME_PATTERN_OF_UNFINISHED_FILES}'; "
    "for log_file in watchdog_log.txt pod_setup_*.txt; do [ -f \"$log_file\" ] && echo \"$log_file\"; done; true")


def local_tar_supports_flag(flag):
    """True if the local tar accepts the flag (bsdtar on macOS has --no-mac-metadata and --no-xattrs; GNU tar may not)."""
    probe = subprocess.run(["tar", flag, "-cf", os.devnull, "--files-from", os.devnull], capture_output=True)
    return probe.returncode == 0


def build_local_tar_command(base_directory, relative_paths):
    """tar command that writes a gzip stream of `relative_paths` (under base_directory) to standard output. The
    macOS-only flags that stop tar adding extended attributes and AppleDouble ._ files (which the pod's tar warns
    about) are included only when the local tar supports them."""
    command = ["tar", "czf", "-"]
    command += [f"--exclude={pattern}" for pattern in FILE_NAME_PATTERNS_NEVER_UPLOADED]
    command += [f"--exclude={FILE_NAME_PATTERN_OF_UNFINISHED_FILES}"]
    command += [flag for flag in ("--no-mac-metadata", "--no-xattrs") if local_tar_supports_flag(flag)]
    return command + ["-C", str(base_directory), *relative_paths]


def build_local_tar_environment():
    environment = dict(os.environ)
    if platform.system() == "Darwin":
        environment["COPYFILE_DISABLE"] = "1"  # macOS tar: no AppleDouble ._ files
    return environment


def list_repository_paths_to_upload():
    return [path for path in REPOSITORY_PATHS_UPLOADED_TO_POD if (REPOSITORY_ROOT / path).exists()]


class SshTarTransfer:
    """Runs commands on the pod and moves files, all through ssh to root@<public ip> -p <mapped port 22>.
    `remote_shell_prefix` replaces the ssh command (the integration test passes ["bash", "-c"] so that the real tar
    pipelines run against local folders)."""

    def __init__(self, private_key_path, run_folder, ssh_timeout_seconds=900, remote_shell_prefix=None):
        self.private_key_path = str(private_key_path)
        self.run_folder = Path(run_folder)
        self.ssh_timeout_seconds = ssh_timeout_seconds
        self.remote_shell_prefix = remote_shell_prefix
        self.host = None
        self.port = None

    def _ssh_base(self):
        if self.remote_shell_prefix is not None:
            return list(self.remote_shell_prefix)
        return ["ssh", "-i", self.private_key_path, "-p", str(self.port), "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
                f"root@{self.host}"]

    def wait_until_ready(self, pod, deadline_epoch_seconds, get_pod_function, sleep_function=time.sleep):
        """Poll until the pod has a public IP and mapped port 22 and accepts an ssh command."""
        while True:
            ip_address, port_mappings = pod.get("publicIp"), pod.get("portMappings") or {}
            if ip_address and port_mappings.get("22"):
                self.host, self.port = ip_address, int(port_mappings["22"])
                try:
                    exit_code, _ = self.run_remote_command("echo ssh_ready", timeout_seconds=40)
                except subprocess.TimeoutExpired:
                    exit_code = None  # a slow first connection is retried until the deadline, not fatal
                if exit_code == 0:
                    return
            if time.time() > deadline_epoch_seconds:
                raise RuntimeError("Pod did not become reachable over ssh before the startup deadline")
            sleep_function(15)
            pod = get_pod_function()

    def run_remote_command(self, command, input_text=None, timeout_seconds=None):
        completed = subprocess.run(self._ssh_base() + [command], input=input_text, capture_output=True, text=True,
                                   timeout=timeout_seconds or self.ssh_timeout_seconds)
        return completed.returncode, completed.stdout + completed.stderr

    def upload_project_files(self):
        """Stream the scorer package, the prompt files, the run folder's run_configuration_*.json files,
        scoring_inputs/ and any scoring_outputs/ to the pod."""
        self.run_remote_command(f"mkdir -p {REMOTE_WORKING_DIRECTORY} {REMOTE_RUN_FOLDER}")
        self._stream_local_tar_to_remote(build_local_tar_command(REPOSITORY_ROOT, list_repository_paths_to_upload()),
                                         REMOTE_WORKING_DIRECTORY)
        # The scorer reads run_configuration_*.json (model, model type, prompt hashes) from the run folder as well as
        # scoring_inputs/. Unfinished files (*.partial) of an earlier scorer are not sent.
        run_folder_items = [path.name for path in sorted(self.run_folder.glob("run_configuration_*.json"))]
        run_folder_items += [name for name in ("scoring_inputs", "scoring_outputs") if (self.run_folder / name).exists()]
        self._stream_local_tar_to_remote(build_local_tar_command(self.run_folder, run_folder_items), REMOTE_RUN_FOLDER)

    def _stream_local_tar_to_remote(self, local_tar_command, remote_directory):
        tar_process = subprocess.Popen(local_tar_command, stdout=subprocess.PIPE, env=build_local_tar_environment())
        # The pod's /workspace volume refuses chown to the local user id, so extract as the pod's own user.
        ssh_result = subprocess.run(self._ssh_base() + [f"tar xzf - --no-same-owner --no-same-permissions "
                                                        f"-C {remote_directory}"], stdin=tar_process.stdout,
                                    capture_output=True, text=True, timeout=self.ssh_timeout_seconds)
        tar_process.stdout.close()
        if tar_process.wait() != 0 or ssh_result.returncode != 0:
            raise RuntimeError(f"Upload to {remote_directory} failed: {ssh_result.stderr[:300]}")

    def _run_remote_command_into_file(self, command, output_file_path, timeout_seconds):
        """Runs a remote command with its standard output written to a local file; returns (exit code, stderr)."""
        with open(output_file_path, "wb") as output_handle:
            completed = subprocess.run(self._ssh_base() + [command], stdout=output_handle, stderr=subprocess.PIPE,
                                       timeout=timeout_seconds)
        return completed.returncode, completed.stderr.decode(errors="replace")

    def pull_scoring_outputs(self, pull_everything):
        """Periodic pull of scoring_outputs/ into the local run folder. Only finished files are sent: unfinished
        ones (the scorer writes `*.partial` and then renames) are skipped, and every file is first copied to a
        staging folder on the pod, so a file that is appended to (the completion records, the progress log) cannot
        change while it is being read. Incremental pulls send only files newer than the previous successful pull's
        marker. A missing scoring_outputs/ on the pod is nothing to pull, not a failure."""
        marker = f"{REMOTE_RUN_FOLDER}/.last_pull_marker"
        newer_option = "" if pull_everything else f"$( [ -f {marker} ] && echo -newer {marker} )"
        unfinished = FILE_NAME_PATTERN_OF_UNFINISHED_FILES
        remote_command = (
            f"cd {REMOTE_RUN_FOLDER} && if [ -d scoring_outputs ]; then touch {marker}.next && "
            f"staging=$(mktemp -d) && "
            f"find scoring_outputs -type f -not -name '{unfinished}' {newer_option} | "
            f"while IFS= read -r file_path; do mkdir -p \"$staging/$(dirname \"$file_path\")\" && "
            f"cp -p \"$file_path\" \"$staging/$file_path\"; done; "
            f"if [ -d \"$staging/scoring_outputs\" ]; then tar czf - -C \"$staging\" scoring_outputs; fi; "
            f"rm -rf \"$staging\"; fi")
        with tempfile.TemporaryDirectory(dir=self.run_folder) as staging_directory:
            archive_path = Path(staging_directory) / "pulled_scoring_outputs.tar.gz"
            exit_code, ssh_error = self._run_remote_command_into_file(remote_command, archive_path,
                                                                      self.ssh_timeout_seconds)
            if exit_code != 0:
                raise RuntimeError(f"Pull of scoring_outputs failed: {ssh_error[:300]}")
            if archive_path.stat().st_size == 0:
                self.run_remote_command(f"[ -f {marker}.next ] && mv -f {marker}.next {marker}; true")
                return  # nothing new (or no scoring_outputs/ on the pod yet)
            extraction_directory = Path(staging_directory) / "extracted"
            extraction_directory.mkdir()
            extraction = subprocess.run(["tar", "xzf", str(archive_path), "-C", str(extraction_directory)],
                                        capture_output=True, text=True, timeout=self.ssh_timeout_seconds)
            if extraction.returncode != 0:
                raise RuntimeError(f"Extracting pulled scoring_outputs failed: {extraction.stderr[:300]}")
            staged_outputs = extraction_directory / "scoring_outputs"
            if staged_outputs.exists():
                shutil.copytree(staged_outputs, self.run_folder / "scoring_outputs", dirs_exist_ok=True)
        self.run_remote_command(f"mv -f {marker}.next {marker}")

    @staticmethod
    def local_destination_for_final_copy(run_folder, remote_relative_path, logs_folder_name):
        """scoring_outputs/ files go back into the run folder; every other file goes into the logs subfolder."""
        parts = Path(remote_relative_path).parts
        if parts[0] == "run_folder":
            parts = parts[1:]
            if parts[0] == "scoring_outputs":
                return Path(run_folder).joinpath(*parts)
        return Path(run_folder).joinpath(logs_folder_name, *parts)

    def pull_final_copy_and_verify(self, logs_folder_name, timeout_seconds):
        """Copy back everything on the pod except scoring_inputs/ (see FINAL_COPY_FILE_LIST_SHELL), then compare the
        remote file list and sizes with what arrived. scoring_outputs/ files land in the run folder; the other files
        (scorer output and exit code, watchdog and setup logs) land in <run folder>/<logs_folder_name>/.
        Returns the number of files verified; raises FinalCopyIncompleteError or RuntimeError otherwise."""
        listing_command = (f"cd {REMOTE_WORKING_DIRECTORY} && {{ {FINAL_COPY_FILE_LIST_SHELL}; }} | "
                           f"while IFS= read -r file_path; do printf '%s\\t%s\\n' "
                           f"\"$(wc -c < \"$file_path\" | tr -d ' ')\" \"$file_path\"; done")
        exit_code, listing_output = self.run_remote_command(listing_command, timeout_seconds=timeout_seconds)
        if exit_code != 0:
            raise RuntimeError(f"Could not list the files on the pod: {listing_output[-300:]}")
        remote_sizes = {}
        for line in listing_output.splitlines():
            size_text, separator, file_path = line.partition("\t")
            if separator and size_text.isdigit():
                remote_sizes[file_path] = int(size_text)
        if not remote_sizes:
            return 0
        pull_command = (f"cd {REMOTE_WORKING_DIRECTORY} && {{ {FINAL_COPY_FILE_LIST_SHELL}; }} | tar czf - -T -")
        with tempfile.TemporaryDirectory(dir=self.run_folder) as staging_directory:
            archive_path = Path(staging_directory) / "final_copy_from_pod.tar.gz"
            extraction_directory = Path(staging_directory) / "extracted"
            extraction_directory.mkdir()
            _, ssh_error = self._run_remote_command_into_file(pull_command, archive_path, timeout_seconds)
            # Extract whatever arrived even if the stream broke; the size comparison below decides success.
            subprocess.run(["tar", "xzf", str(archive_path), "-C", str(extraction_directory)], capture_output=True,
                           timeout=timeout_seconds)
            problem_files = []
            for file_path, expected_size in sorted(remote_sizes.items()):
                staged_file = extraction_directory / file_path
                if not staged_file.is_file():
                    problem_files.append(f"{file_path} (did not arrive)")
                    continue
                destination = self.local_destination_for_final_copy(self.run_folder, file_path, logs_folder_name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staged_file, destination)
                if destination.stat().st_size != expected_size:
                    problem_files.append(f"{file_path} (pod {expected_size} bytes, local {destination.stat().st_size})")
        if problem_files:
            raise FinalCopyIncompleteError(f"{len(problem_files)} of {len(remote_sizes)} files did not arrive intact; "
                                           f"ssh said: {ssh_error[:200]}", problem_files)
        return len(remote_sizes)


def read_secrets_from_environment_file(environment_file_path=ENVIRONMENT_FILE_PATH):
    """Parses the given .env file (by default the repository-root one, no parent folders searched) with the same
    python-dotenv parser shared/api.py uses; os.environ is never consulted."""
    from dotenv import dotenv_values
    values = dotenv_values(environment_file_path) if Path(environment_file_path).is_file() else {}
    return {"RUNPOD_API_KEY": values.get("RUNPOD_API_KEY") or None, "HF_TOKEN": values.get("HF_TOKEN") or None,
            COST_CENTER_NAME_ENVIRONMENT_FILE_KEY: values.get(COST_CENTER_NAME_ENVIRONMENT_FILE_KEY) or None}


def build_scorer_command_line(arguments):
    command = ["python", "-m", SCORER_MODULE_NAME, "--run-folder", REMOTE_RUN_FOLDER]
    optional_flags = [("--model-name", arguments.model_name), ("--device", arguments.device),
                      ("--dtype", arguments.dtype), ("--maximum-documents", arguments.maximum_documents),
                      ("--maximum-tokens-per-batch", arguments.maximum_tokens_per_batch),
                      ("--model-revision", arguments.model_revision), ("--model-class", arguments.model_class),
                      ("--positions-per-head-chunk", arguments.positions_per_head_chunk)]
    for flag, value in optional_flags:
        if value is not None:
            command += [flag, str(value)]
    if arguments.verify_first_document_against_full_logits:
        command.append("--verify-first-document-against-full-logits")
    return command


def build_scorer_launch_shell_command(arguments):
    """Background launch: survives the ssh connection dropping. Writes stdout and, on exit, the exit code."""
    scorer = " ".join(shlex.quote(part) for part in build_scorer_command_line(arguments))
    inner = (f"cd {REMOTE_WORKING_DIRECTORY} && . {REMOTE_WORKING_DIRECTORY}/pod_secret_environment.sh && "
             f"{scorer} > {REMOTE_RUN_FOLDER}/scorer_standard_output.txt 2>&1; "
             f"echo $? > {REMOTE_RUN_FOLDER}/scorer_exit_code.txt")
    return f"nohup bash -c {shlex.quote(inner)} > /dev/null 2>&1 &"


def build_self_termination_watchdog_command(seconds_until_termination):
    """Deletes this pod from the inside. Tries runpodctl (both command spellings) then the REST API using the
    pod-scoped RUNPOD_API_KEY that RunPod places in the pod's environment (read from /proc/1/environ because
    ssh sessions do not inherit container variables)."""
    terminate = (
        "export $(tr '\\0' '\\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | xargs); "
        "runpodctl pod delete \"$RUNPOD_POD_ID\" || runpodctl remove pod \"$RUNPOD_POD_ID\" || "
        "curl -s -X DELETE -H \"Authorization: Bearer $RUNPOD_API_KEY\" "
        f"{RUNPOD_REST_BASE_URL}/pods/\"$RUNPOD_POD_ID\""
    )
    inner = f"sleep {int(seconds_until_termination)}; {terminate}"
    return f"nohup bash -c {shlex.quote(inner)} > {REMOTE_WORKING_DIRECTORY}/watchdog_log.txt 2>&1 &"


def build_pod_creation_request(arguments, public_key_text):
    pod_name = f"log-probability-scoring-{Path(arguments.run_folder).name}"[:191]
    return {
        "name": pod_name,
        "imageName": arguments.image_name,
        "cloudType": arguments.cloud_type,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": arguments.gpu_type_ids,
        "gpuTypePriority": "custom",
        "containerDiskInGb": arguments.container_disk_gigabytes,
        "volumeInGb": arguments.volume_gigabytes,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "interruptible": False,
        "env": {"PUBLIC_KEY": public_key_text, "SSH_PUBLIC_KEY": public_key_text,
                "HF_HOME": "/workspace/huggingface_cache"},
    }


def pod_cost_per_hour(pod):
    price = pod.get("costPerHr")
    if price is None:
        raise RuntimeError("RunPod did not report costPerHr for the pod; cannot enforce the dollar limit")
    return float(price)


def cost_so_far_dollars(hourly_price, start_epoch_seconds, now_epoch_seconds):
    return hourly_price * max(0.0, now_epoch_seconds - start_epoch_seconds) / 3600.0


def effective_limit_seconds(maximum_running_hours, maximum_dollars, hourly_price):
    """The earlier of the running-hours limit and the time at which the dollar limit would be reached."""
    return min(maximum_running_hours * 3600.0, maximum_dollars / hourly_price * 3600.0)


class PodCostLog:
    def __init__(self, path, record):
        self.path, self.record = Path(path), record

    def update(self, **fields):
        self.record.update(fields)
        self.path.write_text(json.dumps(self.record, indent=2), encoding="utf-8")


def iso_from_epoch(epoch_seconds):
    return datetime.fromtimestamp(epoch_seconds).isoformat(timespec="seconds") if epoch_seconds else None


def requirements_file_hash():
    requirements_path = REPOSITORY_ROOT / REQUIREMENTS_FILE_RELATIVE_PATH
    return hashlib.sha256(requirements_path.read_bytes()).hexdigest() if requirements_path.exists() else None


def cost_center_assignment_confirmed_in_earlier_cost_log(pod_id, search_folder):
    """True if a pod cost log in a sibling run folder of `search_folder` (the folder holding the run folders, for
    example the audit/ folder) records a confirmed assignment for this pod."""
    for path in Path(search_folder).glob("*/pod_cost_log_*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if record.get("pod_id") == pod_id and record.get("cost_center_assignment_confirmed_by_user") is True:
            return True
    return False


class PodSession:
    """Owns one pod's lifecycle. `terminate` is idempotent and is always reached from the finally block."""

    def __init__(self, arguments, client, transfer, secrets, cost_log, clock=time.time, sleep=time.sleep,
                 confirmation_source=None):
        self.arguments, self.client, self.transfer, self.secrets = arguments, client, transfer, secrets
        self.cost_log, self.clock, self.sleep = cost_log, clock, sleep
        self.confirmation_source = confirmation_source
        self.cost_center_name = cost_log.record.get("cost_center_name")  # None: no cost center, nothing to confirm
        self.cost_center_confirmation_recorded = False
        self.pod_id = None
        self.pod_name = None
        self.hourly_price = None
        self.start_epoch_seconds = None
        self.terminated = False
        self.final_pull_succeeded = None  # None until a final pull has been settled, then True or False
        self.final_logs_folder_name = None
        self.waiting_message = None  # set while the pod waits for an instruction file; repeated in each report
        self.run_folder = Path(arguments.run_folder)
        self.confirmation_file_path = self.run_folder / COST_CENTER_CONFIRMATION_FILE_NAME
        self.progress_log_path = self.run_folder / f"pod_progress_log_{cost_log.record['launcher_timestamp']}.txt"

    def say(self, message):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(self.progress_log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def cost_center_instructions(self):
        return (f"Assign pod '{self.pod_name}' (id {self.pod_id}) to cost center '{self.cost_center_name}' at "
                f"{COST_CENTER_CONSOLE_URL} (Uncategorized resources, tick the pod, Add resources to cost center), "
                f"then EITHER type `{COST_CENTER_CONFIRMATION_WORD}` and Enter in this terminal OR create the file "
                f"{self.confirmation_file_path.resolve()}")

    def announce_cost_center_assignment_needed(self):
        if not self.cost_center_name:
            return
        self.say("=" * 100)
        self.say("ACTION NEEDED NOW: " + self.cost_center_instructions())
        self.say("Setup and scoring continue meanwhile; the pod bills until it is deleted.")
        self.say("=" * 100)

    def record_confirmation_if_given(self):
        """True once the user has confirmed (always True when no cost center name was given); records the
        confirmation in the cost log the first time."""
        if not self.cost_center_name or self.cost_center_confirmation_recorded:
            return True
        confirmed_in_terminal = self.confirmation_source is not None and self.confirmation_source.confirmed
        if confirmed_in_terminal or self.confirmation_file_path.exists():
            self.cost_center_confirmation_recorded = True
            self.cost_log.update(cost_center_assignment_confirmed_by_user=True,
                                 cost_center_assignment_confirmation_time=iso_from_epoch(self.clock()))
            self.say(f"Cost center assignment to '{self.cost_center_name}' confirmed by the user; reminders stopped")
            return True
        return False

    def remind_if_cost_center_unconfirmed(self):
        if not self.record_confirmation_if_given():
            self.say("REMINDER: " + self.cost_center_instructions())

    def warn_and_wait_for_cost_center_assignment_before_deletion(self, reason):
        """Final chance to assign the pod. Only normal completion waits; limits, errors and Ctrl-C delete at once."""
        if self.record_confirmation_if_given():
            return
        self.say("!" * 100)
        self.say(f"WARNING: pod {self.pod_id} is about to be deleted without being assigned to cost center "
                 f"'{self.cost_center_name}'. Its cost will appear as uncategorized.")
        if reason != "scorer_finished":
            self.say(f"Deleting immediately ({reason}); the cost limits are never extended.")
            self.say("!" * 100)
            return
        self.say(f"Waiting up to {format_duration(FINAL_COST_CENTER_ASSIGNMENT_WINDOW_SECONDS)} more (billing continues). "
                 + self.cost_center_instructions())
        self.say("!" * 100)
        deadline = self.clock() + FINAL_COST_CENTER_ASSIGNMENT_WINDOW_SECONDS
        try:
            while not self.record_confirmation_if_given() and self.clock() < deadline:
                self.sleep(COST_CENTER_CONFIRMATION_CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            self.say("Interrupted during the final assignment window; deleting the pod now")
        if not self.cost_center_confirmation_recorded:
            self.say("No confirmation received; deleting the pod, its cost will be uncategorized")

    def current_cost(self):
        return cost_so_far_dollars(self.hourly_price, self.start_epoch_seconds, self.clock())

    def terminate(self, reason):
        if self.terminated or self.pod_id is None:
            return
        self.warn_and_wait_for_cost_center_assignment_before_deletion(reason)
        last_error = None
        for attempt in range(1, TERMINATION_ATTEMPTS + 1):
            try:
                self.client.delete_pod(self.pod_id)
                self.terminated = True
                break
            except Exception as error:  # keep trying; billing continues until this succeeds
                last_error = error
                self.say(f"Termination attempt {attempt} failed: {error}")
                self.sleep(5)
        stop_epoch_seconds = self.clock()
        if not self.terminated:
            self.say(f"!!! COULD NOT TERMINATE POD {self.pod_id}. Delete it now at https://console.runpod.io/pods "
                     f"(the on-pod watchdog will also delete it). Last error: {last_error}")
        self.cost_log.update(stop_time=iso_from_epoch(stop_epoch_seconds), termination_reason=reason,
                             pod_terminated=self.terminated,
                             total_cost_dollars=round(cost_so_far_dollars(
                                 self.hourly_price or 0.0, self.start_epoch_seconds or stop_epoch_seconds,
                                 stop_epoch_seconds), 4))
        if self.terminated:
            self.say(f"Pod {self.pod_id} terminated ({reason}); total cost about "
                     f"{self.cost_log.record['total_cost_dollars']:.2f} dollars")

    def keep_pod_running(self):
        """After a successful run with --keep-pod-after-run: leave the pod running for the next run. The hard limits
        still apply through the on-pod watchdog."""
        self.record_confirmation_if_given()
        remaining = format_duration(self.seconds_until_a_limit() + WATCHDOG_GRACE_SECONDS)
        self.cost_log.update(stop_time=iso_from_epoch(self.clock()), termination_reason="kept_after_run",
                             pod_terminated=False, pod_kept_after_run=True,
                             total_cost_dollars=round(self.current_cost(), 4))
        self.say(f"Pod {self.pod_id} is KEPT RUNNING after the run (--keep-pod-after-run); it keeps billing "
                 f"{self.hourly_price:.2f} dollars/hour, cost so far about {self.current_cost():.2f} dollars. The hard "
                 f"limits still apply ({self.arguments.maximum_running_hours} hours, {self.arguments.maximum_dollars:.2f} "
                 f"dollars, counted from this launch): the pod's own watchdog deletes it in about {remaining}.")
        self.remind_if_cost_center_unconfirmed()

    def start_new_pod(self, creation_request):
        self.start_epoch_seconds = self.clock()
        pod = self.client.create_pod(creation_request)
        self.pod_id = pod["id"]
        self.pod_name = creation_request["name"]
        self.hourly_price = pod_cost_per_hour(pod)
        self.cost_log.update(pod_id=self.pod_id, gpu_type=self._gpu_name(pod), hourly_price_dollars=self.hourly_price,
                             start_time=iso_from_epoch(self.start_epoch_seconds), pod_creation_request_name=creation_request["name"])
        self.say(f"Created pod {self.pod_id} ({self._gpu_name(pod)}) at {self.hourly_price:.2f} dollars/hour")
        self.announce_cost_center_assignment_needed()
        return pod

    def attach_to_pod(self, pod_id, previous_start_epoch_seconds):
        self.pod_id = pod_id
        pod = self.client.get_pod(pod_id)
        self.pod_name = pod.get("name") or "unknown name"
        self.hourly_price = pod_cost_per_hour(pod)
        self.start_epoch_seconds = previous_start_epoch_seconds or self.clock()
        self.cost_log.update(pod_id=pod_id, gpu_type=self._gpu_name(pod), hourly_price_dollars=self.hourly_price,
                             start_time=iso_from_epoch(self.start_epoch_seconds), attached_to_existing_pod=True)
        self.announce_cost_center_assignment_needed()
        return pod

    def reuse_pod(self, pod_id):
        """Run on a pod an earlier launcher invocation created: nothing is created, the price comes from the API, and
        cost is counted from this invocation's start."""
        self.pod_id = pod_id
        self.start_epoch_seconds = self.clock()
        already_confirmed = bool(self.cost_center_name) and cost_center_assignment_confirmed_in_earlier_cost_log(
            pod_id, self.run_folder.parent)
        pod = self.client.get_pod(pod_id)
        self.pod_name = pod.get("name") or "unknown name"
        self.hourly_price = pod_cost_per_hour(pod)
        self.cost_log.update(pod_id=pod_id, gpu_type=self._gpu_name(pod), hourly_price_dollars=self.hourly_price,
                             start_time=iso_from_epoch(self.start_epoch_seconds), reused_existing_pod=True,
                             cost_counted_from="start of this launcher invocation, not from the pod's creation")
        self.say(f"Reusing pod {pod_id} ({self._gpu_name(pod)}) at {self.hourly_price:.2f} dollars/hour; cost is "
                 f"counted from this launch")
        if not self.cost_center_name:
            pass
        elif already_confirmed:
            self.cost_center_confirmation_recorded = True
            self.cost_log.update(cost_center_assignment_confirmed_by_user=True,
                                 cost_center_assignment_confirmed_in_earlier_cost_log=True)
            self.say(f"An earlier cost log records the cost center assignment of pod {pod_id}; not asking again")
        else:
            self.announce_cost_center_assignment_needed()
        return pod

    @staticmethod
    def _gpu_name(pod):
        gpu = pod.get("gpu") or {}
        return gpu.get("displayName") or gpu.get("id") or "unknown"

    def prepare_and_start_scorer(self, pod):
        """Returns None once the scorer is running, or a termination reason if the pod is to be deleted instead. ssh
        never becoming ready raises (the pod is then deleted); a failed setup step waits for an instruction file."""
        self.say("Waiting for pod ssh")
        self.transfer.wait_until_ready(pod, self.clock() + 1200, lambda: self.client.get_pod(self.pod_id), self.sleep)
        if self.arguments.reuse_existing_pod:
            self.prepare_reused_pod_run_folder()
        while True:
            try:
                self.upload_install_and_start_scorer()
                return None
            except Exception as error:
                self.say(f"Setup failed: {error}")
                outcome = self.wait_for_instruction("setup_failed", RETRY_SETUP_INSTRUCTION_FILE_NAME)
                if outcome != "retry":
                    return outcome
                self.say("Instruction received: redoing the upload, install and scorer start on the same pod")

    def prepare_reused_pod_run_folder(self):
        """A reused pod still holds the earlier run's folder. Refuse if a scorer is running on it (use
        --attach-to-existing-pod for that); otherwise move the folder aside (never deleted) so that this run starts
        from its own local files."""
        exit_code, _ = self.transfer.run_remote_command(f"pgrep -f '{SCORER_PROCESS_PATTERN}' > /dev/null")
        if exit_code == 0:
            self.pod_id = None  # a scorer of another run is working there: leave the pod alone
            raise SystemExit("A scorer is already running on that pod. Use --attach-to-existing-pod to monitor it; "
                             "the pod has not been touched or deleted.")
        self.transfer.run_remote_command(
            f"[ -d {REMOTE_RUN_FOLDER} ] && mv {REMOTE_RUN_FOLDER} {REMOTE_RUN_FOLDER}_before_reuse_{timestamp_now()}; true")

    def upload_install_and_start_scorer(self):
        self.say("Uploading scripts and scoring inputs")
        self.transfer.upload_project_files()
        secret_lines = f"export HF_TOKEN={shlex.quote(self.secrets['HF_TOKEN'])}\n" if self.secrets.get("HF_TOKEN") else ""
        secret_lines += "export HF_HOME=/workspace/huggingface_cache\n"
        self.transfer.run_remote_command(
            f"umask 077; cat > {REMOTE_WORKING_DIRECTORY}/pod_secret_environment.sh", input_text=secret_lines)
        self.install_requirements_unless_already_installed()
        _, versions_output = self.transfer.run_remote_command(
            f"{POD_VERSION_CHECK_COMMAND} 2>&1 | tee {REMOTE_WORKING_DIRECTORY}/{ENVIRONMENT_VERSIONS_LOG_FILE_NAME}")
        self.say(f"Pod environment: {versions_output.strip()}")
        self.start_watchdog()
        self.start_scorer()

    def install_requirements_unless_already_installed(self):
        marker_path = f"{REMOTE_WORKING_DIRECTORY}/{REQUIREMENTS_INSTALLED_MARKER_FILE_NAME}"
        requirements_hash = requirements_file_hash()
        if requirements_hash:
            _, marker_text = self.transfer.run_remote_command(f"cat {marker_path} 2>/dev/null")
            if marker_text.strip() == requirements_hash:
                self.say("Requirements already installed on this pod for the same requirements file; skipping pip install")
                return
        self.say("Installing requirements on the pod")
        pip_log_path = f"{REMOTE_WORKING_DIRECTORY}/{PIP_INSTALL_LOG_FILE_NAME}"
        exit_code, output = self.transfer.run_remote_command(
            f"cd {REMOTE_WORKING_DIRECTORY} && pip install -q -r {REQUIREMENTS_FILE_RELATIVE_PATH} > {pip_log_path} 2>&1; "
            f"pip_exit_code=$?; tail -n 20 {pip_log_path}; exit $pip_exit_code")
        if exit_code != 0:
            raise RuntimeError(f"Installing requirements on the pod failed: {output[-500:]}")
        if requirements_hash:
            self.transfer.run_remote_command(f"echo {requirements_hash} > {marker_path}")

    def start_scorer(self):
        self.say("Starting scorer in the background")
        # A previous attempt's exit code would otherwise be read as this attempt's result.
        self.transfer.run_remote_command(f"rm -f {REMOTE_RUN_FOLDER}/scorer_exit_code.txt")
        exit_code, output = self.transfer.run_remote_command(build_scorer_launch_shell_command(self.arguments))
        if exit_code != 0:
            raise RuntimeError(f"Starting the scorer failed: {output[-500:]}")

    def restart_scorer_on_same_pod(self):
        """Re-uploads everything the normal upload sends (fixed scripts included) and restarts the scorer, which
        resumes and skips finished documents."""
        self.say("Re-uploading scripts and scoring inputs, then restarting the scorer on the same pod")
        self.transfer.upload_project_files()
        self.start_scorer()

    def wait_for_instruction(self, failure_description, retry_instruction_file_name):
        """The pod stays up (and billing) while the user or Claude Code fixes something. Returns "retry" when the retry
        file appears, otherwise the termination reason: the delete file, the hours or dollar limit (which never wait),
        or no instruction within --failure-wait-minutes. Each instruction file is consumed (renamed) once acted on."""
        retry_path = self.run_folder / retry_instruction_file_name
        delete_path = self.run_folder / DELETE_POD_INSTRUCTION_FILE_NAME
        wait_minutes = self.arguments.failure_wait_minutes
        wait_start = last_report = self.clock()
        deadline = wait_start + wait_minutes * 60
        limit_seconds = effective_limit_seconds(self.arguments.maximum_running_hours, self.arguments.maximum_dollars,
                                                self.hourly_price)
        self.waiting_message = (f"POD {self.pod_id} IS KEPT RUNNING, WAITING FOR A FIX ({failure_description}). To retry, "
                                f"create {retry_path.resolve()} ; to delete the pod, create {delete_path.resolve()} . "
                                f"With no instruction within {wait_minutes:g} minutes the pod is deleted; the cost limits "
                                f"and Ctrl-C delete it immediately.")
        try:
            self.say(self.waiting_message)
            while True:
                self.record_confirmation_if_given()
                if self.current_cost() >= self.arguments.maximum_dollars:
                    self.say("Dollar limit reached while waiting for an instruction")
                    return "dollar_limit_reached"
                if self.clock() - self.start_epoch_seconds >= self.arguments.maximum_running_hours * 3600:
                    self.say("Running-hours limit reached while waiting for an instruction")
                    return "running_hours_limit_reached"
                for instruction_path, outcome in ((delete_path, "instruction_delete_pod"), (retry_path, "retry")):
                    if instruction_path.exists():
                        consumed_path = instruction_path.with_name(
                            f"{instruction_path.name}.consumed_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
                        instruction_path.rename(consumed_path)
                        self.say(f"Instruction {instruction_path.name} received and consumed (renamed {consumed_path.name})")
                        return outcome
                if self.clock() >= deadline:
                    self.say(f"No instruction within {wait_minutes:g} minutes; deleting the pod")
                    return f"{failure_description}_and_no_instruction_within_{wait_minutes:g}_minutes"
                self.sleep(INSTRUCTION_CHECK_INTERVAL_SECONDS)
                if self.clock() - last_report >= PROGRESS_REPORT_INTERVAL_SECONDS:
                    last_report = self.clock()
                    self.report(limit_seconds)
        finally:
            self.waiting_message = None

    def start_watchdog(self):
        """Replaces any earlier watchdog on the pod (from a previous launcher invocation), so that an old one cannot
        delete the pod mid-run; the new one must exist before the scorer starts."""
        self.transfer.run_remote_command(f"pkill -f '{WATCHDOG_PROCESS_PATTERN}'; true")
        limit = effective_limit_seconds(self.arguments.maximum_running_hours, self.arguments.maximum_dollars,
                                        self.hourly_price)
        elapsed = self.clock() - self.start_epoch_seconds
        seconds_remaining = max(60, limit - elapsed + WATCHDOG_GRACE_SECONDS)
        self.transfer.run_remote_command(build_self_termination_watchdog_command(seconds_remaining))
        self.say(f"On-pod self-termination watchdog set for {format_duration(seconds_remaining)} from now")

    def scorer_exit_code(self):
        exit_code, output = self.transfer.run_remote_command(f"cat {REMOTE_RUN_FOLDER}/scorer_exit_code.txt 2>/dev/null")
        text = output.strip()
        return int(text) if exit_code == 0 and text.lstrip("-").isdigit() else None

    def scorer_has_vanished_without_exit_code(self):
        """True when no exit code was written and no scorer process exists (for example, it was killed outright)."""
        exit_code, _ = self.transfer.run_remote_command(f"pgrep -f '{SCORER_PROCESS_PATTERN}' > /dev/null")
        return exit_code == 1

    def print_scorer_output_tail(self):
        _, output = self.transfer.run_remote_command(
            f"tail -n {SCORER_OUTPUT_TAIL_LINES} {REMOTE_RUN_FOLDER}/scorer_standard_output.txt 2>&1")
        self.say(f"--- last {SCORER_OUTPUT_TAIL_LINES} lines of scorer_standard_output.txt ---")
        for line in output.splitlines():
            self.say(f"scorer> {line}")
        self.say("--- end of scorer_standard_output.txt ---")

    def stop_scorer_if_running(self):
        """Before the final copy of a run that is being cut short, so that no file changes while it is verified."""
        try:
            self.transfer.run_remote_command(f"pkill -f '{SCORER_PROCESS_PATTERN}'; sleep 2; true", timeout_seconds=60)
        except Exception as error:
            self.say(f"Could not stop the scorer before the final copy: {error}")

    def seconds_until_a_limit(self):
        limit_seconds = effective_limit_seconds(self.arguments.maximum_running_hours, self.arguments.maximum_dollars,
                                                self.hourly_price)
        return max(0.0, limit_seconds - (self.clock() - self.start_epoch_seconds))

    def final_pull_with_verification(self):
        """Copies everything back from the pod (see SshTarTransfer.pull_final_copy_and_verify) and verifies it. On
        failure retries after a short wait, within FINAL_PULL_TOTAL_BUDGET_SECONDS and never past the hours or dollar
        limits. Always returns (never waits unboundedly); False means the caller deletes the pod with files missing."""
        if self.final_pull_succeeded is not None:
            return self.final_pull_succeeded
        if self.final_logs_folder_name is None:
            self.final_logs_folder_name = f"pod_{self.pod_id}_logs_{timestamp_now()}"
        budget_seconds = min(FINAL_PULL_TOTAL_BUDGET_SECONDS, self.seconds_until_a_limit())
        pull_start = self.clock()
        last_error, attempt = None, 0
        while True:
            attempt += 1
            attempt_timeout = max(FINAL_PULL_MINIMUM_ATTEMPT_TIMEOUT_SECONDS, budget_seconds - (self.clock() - pull_start))
            try:
                files_verified = self.transfer.pull_final_copy_and_verify(self.final_logs_folder_name, attempt_timeout)
                self.say(f"Final pull verified: {files_verified} files on the pod match the local copy "
                         f"(scoring_outputs in the run folder, everything else in {self.final_logs_folder_name}/)")
                self.final_pull_succeeded = True
                self.cost_log.update(final_pull_verified=True, final_pull_logs_folder=self.final_logs_folder_name)
                return True
            except Exception as error:
                last_error = error
                self.say(f"Final pull attempt {attempt} failed: {error}")
            if self.clock() - pull_start + FINAL_PULL_RETRY_WAIT_SECONDS >= budget_seconds:
                break
            self.sleep(FINAL_PULL_RETRY_WAIT_SECONDS)
        problem_files = getattr(last_error, "problem_files", None)
        if problem_files:
            missing_description = "these files are missing or incomplete locally: " + "; ".join(problem_files)
        else:
            missing_description = (f"the pod's file list could not be compared, so ANY of these may be missing locally: "
                                   f"everything in the pod's run_folder except scoring_inputs (scoring_outputs, "
                                   f"scorer_standard_output.txt, scorer_exit_code.txt), watchdog_log.txt, and the "
                                   f"pod_setup_*.txt logs")
        self.say("!" * 100)
        self.say(f"!!! FINAL PULL FAILED after {attempt} attempts; DELETING THE POD ANYWAY. DATA MAY BE LOST: "
                 f"{missing_description}. Last error: {last_error}")
        self.say("!" * 100)
        self.final_pull_succeeded = False
        self.cost_log.update(final_pull_verified=False, final_pull_possibly_missing=missing_description)
        return False

    def report(self, limit_seconds):
        elapsed = self.clock() - self.start_epoch_seconds
        _, status = self.transfer.run_remote_command(
            f"ls {REMOTE_RUN_FOLDER}/scoring_outputs/per_token_log_probabilities 2>/dev/null | wc -l; "
            f"tail -n 1 {REMOTE_RUN_FOLDER}/scoring_outputs/scoring_progress_log_*.txt 2>/dev/null | tail -n 1")
        status_lines = status.strip().splitlines()
        documents_scored = status_lines[0].strip() if status_lines else "?"
        scorer_line = status_lines[1].strip() if len(status_lines) > 1 else ""
        cost = self.current_cost()
        self.say(f"elapsed {format_duration(elapsed)} of limit {format_duration(limit_seconds)}; cost so far "
                 f"{cost:.2f} of {self.arguments.maximum_dollars:.2f} dollars; documents scored on pod "
                 f"{documents_scored}; scorer: {scorer_line}")
        self.cost_log.update(total_cost_dollars=round(cost, 4), last_report_time=iso_from_epoch(self.clock()))
        self.remind_if_cost_center_unconfirmed()
        if self.waiting_message:
            self.say(self.waiting_message)

    def try_pull(self, pull_everything):
        try:
            self.transfer.pull_scoring_outputs(pull_everything)
            return True
        except Exception as error:
            self.say(f"Pull of scoring outputs failed: {error}")
            return False

    def handle_scorer_failure(self, reason):
        """Scorer crashed or vanished: show why, copy everything back, then keep the pod and wait for an instruction.
        Returns None after the scorer was restarted on the same pod (monitoring continues), else (reason, pull ok)."""
        self.print_scorer_output_tail()
        self.final_pull_with_verification()
        failure_description = reason
        while True:
            outcome = self.wait_for_instruction(failure_description, RETRY_SCORER_INSTRUCTION_FILE_NAME)
            if outcome != "retry":
                return outcome, self.final_pull_succeeded
            try:
                self.restart_scorer_on_same_pod()
            except Exception as error:
                self.say(f"Restarting the scorer failed: {error}")
                failure_description = "scorer_restart_failed"
                continue
            self.final_pull_succeeded = self.final_logs_folder_name = None  # the next failure or finish pulls afresh
            return None

    def monitor_until_finished_or_limit(self):
        """Returns (reason, final_pull_succeeded). The final pull always happens here, whatever the reason."""
        limit_seconds = effective_limit_seconds(self.arguments.maximum_running_hours, self.arguments.maximum_dollars,
                                                self.hourly_price)
        monitoring_start = last_pull = last_finished_check = self.clock()
        early_crash_check_done = False
        while True:
            failure_reason = None
            elapsed = self.clock() - self.start_epoch_seconds
            self.record_confirmation_if_given()
            if self.current_cost() >= self.arguments.maximum_dollars:
                self.say("Dollar limit reached")
                self.stop_scorer_if_running()
                return "dollar_limit_reached", self.final_pull_with_verification()
            if elapsed >= self.arguments.maximum_running_hours * 3600:
                self.say("Running-hours limit reached")
                self.stop_scorer_if_running()
                return "running_hours_limit_reached", self.final_pull_with_verification()
            now = self.clock()
            early_check_due = not early_crash_check_done and now - monitoring_start >= EARLY_CRASH_CHECK_DELAY_SECONDS
            if early_check_due or now - last_finished_check >= SCORER_FINISHED_CHECK_INTERVAL_SECONDS:
                last_finished_check = now
                scorer_exit = self.scorer_exit_code()
                if early_check_due:
                    early_crash_check_done = True
                if scorer_exit is None and self.scorer_has_vanished_without_exit_code():
                    self.say("Scorer process is gone and left no exit code")
                    failure_reason = "scorer_vanished_without_exit_code"
                if early_check_due and scorer_exit is None and failure_reason is None:
                    self.say("Early crash check: the scorer is still running")
                if scorer_exit is not None:
                    self.say(f"Scorer finished with exit code {scorer_exit}"
                             + (" (early crash check)" if early_check_due and scorer_exit != 0 else ""))
                    if scorer_exit != 0:
                        failure_reason = "scorer_failed_exit_code_%d" % scorer_exit
                    else:
                        return "scorer_finished", self.final_pull_with_verification()
            if failure_reason:
                outcome = self.handle_scorer_failure(failure_reason)
                if outcome is not None:
                    return outcome
                monitoring_start = last_pull = last_finished_check = self.clock()
                early_crash_check_done = False
                continue
            if now - last_pull >= PROGRESS_REPORT_INTERVAL_SECONDS:
                last_pull = now
                self.try_pull(False)
                self.report(limit_seconds)
            self.sleep(LOOP_SLEEP_SECONDS)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--maximum-running-hours", type=float, required=True)
    parser.add_argument("--maximum-dollars", type=float, default=25.0)
    parser.add_argument("--gpu-type-ids", nargs="+", default=DEFAULT_GPU_TYPE_IDS,
                        help="RunPod gpuTypeIds in priority order")
    parser.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default=DEFAULT_CLOUD_TYPE)
    parser.add_argument("--image-name", default=DEFAULT_POD_IMAGE_NAME)
    parser.add_argument("--container-disk-gigabytes", type=int, default=50)
    parser.add_argument("--volume-gigabytes", type=int, default=100)
    parser.add_argument("--ssh-private-key-path", default=str(DEFAULT_SSH_PRIVATE_KEY_PATH),
                        help="Private key whose .pub file is given to the pod")
    parser.add_argument("--cost-center-name", default=None,
                        help=f"Optional. Name of the RunPod cost center to assign the pod to by hand (RunPod has no API "
                             f"for it). Default: {COST_CENTER_NAME_ENVIRONMENT_FILE_KEY} in .env, else none, in which case "
                             f"nothing is printed, awaited or recorded about a cost center")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--attach-to-existing-pod", default=None, metavar="POD_ID")
    parser.add_argument("--reuse-existing-pod", default=None, metavar="POD_ID",
                        help="Run this run folder on an already running pod from an earlier launcher invocation")
    parser.add_argument("--keep-pod-after-run", action="store_true",
                        help="After a successful run with a verified final pull, leave the pod running (limits still apply)")
    parser.add_argument("--failure-wait-minutes", type=float, default=DEFAULT_FAILURE_WAIT_MINUTES,
                        help="How long a failed pod waits for an instruction file before it is deleted")
    parser.add_argument("--model-name", default=None, help="Passed to the scorer (overrides the run configuration)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--maximum-documents", type=int, default=None)
    parser.add_argument("--maximum-tokens-per-batch", type=int, default=None)
    parser.add_argument("--model-revision", default=None, help="Passed to the scorer (overrides the run configuration)")
    parser.add_argument("--model-class", default=None, choices=["auto_causal_lm", "auto_image_text_to_text"])
    parser.add_argument("--positions-per-head-chunk", type=int, default=None)
    parser.add_argument("--verify-first-document-against-full-logits", action="store_true",
                        help="Passed to the scorer: check its first document against plain full logits")
    return parser.parse_args(argv)


def read_public_key_text(private_key_path, required):
    public_key_path = Path(str(private_key_path) + ".pub")
    if public_key_path.exists():
        return public_key_path.read_text().strip()
    if required:
        raise SystemExit(f"No ssh public key at {public_key_path}. Create one with: ssh-keygen -t ed25519 "
                         f"-f {private_key_path}   (or pass --ssh-private-key-path)")
    return "<contents of " + str(public_key_path) + ">"


def resolve_cost_center_name(arguments, secrets):
    return arguments.cost_center_name or secrets.get(COST_CENTER_NAME_ENVIRONMENT_FILE_KEY) or None


def read_model_type_and_model_name_from_run_folder(run_folder):
    """(model_type, model_name, model_revision) from the first run configuration, or None if there is none."""
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if not paths:
        return None
    configuration = json.loads(paths[0].read_text(encoding="utf-8"))
    return configuration.get("model_type"), configuration.get("model_name"), configuration.get("model_revision")


def count_total_tokens_through_model(run_folder):
    """Tokens the scorer would push through the model for this run folder, counted with the model's tokenizer if
    transformers is installed and the tokenizer files are already in the local Hugging Face cache (nothing is
    downloaded); otherwise None."""
    model_description = read_model_type_and_model_name_from_run_folder(run_folder)
    if model_description is None:
        return None
    model_type, model_name, model_revision = model_description
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision, local_files_only=True)
        documents = read_jsonl(Path(run_folder) / "scoring_inputs" / "documents.jsonl")
        conditions = read_jsonl(Path(run_folder) / "scoring_inputs" / "conditions.jsonl")
        return count_tokens_for_scoring_inputs(tokenizer, documents, conditions, model_type, model_name)[
            "total_tokens_through_model"]
    except Exception:
        return None


def project_cost_range_dollars(total_tokens_through_model, price_range_dollars_per_hour=PRICE_RANGE_DOLLARS_PER_HOUR,
                               tokens_per_second_range=THROUGHPUT_RANGE_TOKENS_PER_SECOND,
                               setup_seconds=PROJECTED_SETUP_SECONDS):
    """(lowest, highest) projected cost: the fastest throughput at the lowest price, and the slowest at the highest.
    The throughput and price ranges are rough planning figures, not measurements; a run's own scoring environment
    record and pod cost log hold the real ones."""
    lowest_seconds = setup_seconds + total_tokens_through_model / tokens_per_second_range[1]
    highest_seconds = setup_seconds + total_tokens_through_model / tokens_per_second_range[0]
    return (price_range_dollars_per_hour[0] * lowest_seconds / 3600.0,
            price_range_dollars_per_hour[1] * highest_seconds / 3600.0)


def print_cost_ceiling_and_projection(arguments):
    print(f"Cost ceiling: {arguments.maximum_dollars:.2f} dollars or {arguments.maximum_running_hours} hours, whichever "
          f"comes first (the hourly price is known only when RunPod reports costPerHr at creation; "
          f"{PRICE_RANGE_DOLLARS_PER_HOUR[0]:.0f} to {PRICE_RANGE_DOLLARS_PER_HOUR[1]:.0f} dollars/hour is typical for the "
          f"default GPU types). The pod is deleted when either is reached, also while waiting for an instruction or "
          f"when kept after the run (then by the on-pod watchdog).")
    total_tokens = count_total_tokens_through_model(arguments.run_folder)
    if total_tokens is None:
        print("Projected cost: not available (the run's tokenizer is not in the local Hugging Face cache, or the run "
              "folder has no run configuration, so tokens could not be counted; nothing is downloaded)")
        return None
    lowest, highest = project_cost_range_dollars(total_tokens)
    print(f"Projected cost: {lowest:.2f} to {highest:.2f} dollars for {total_tokens:,} tokens through the model "
          f"(rough planning range: {THROUGHPUT_RANGE_TOKENS_PER_SECOND[0]:,} to {THROUGHPUT_RANGE_TOKENS_PER_SECOND[1]:,} "
          f"input tokens/s at {PRICE_RANGE_DOLLARS_PER_HOUR[0]:.0f} to {PRICE_RANGE_DOLLARS_PER_HOUR[1]:.0f} dollars/hour, "
          f"plus {PROJECTED_SETUP_SECONDS // 60} minutes of setup)")
    return lowest, highest


def print_dry_run(arguments, secrets):
    request = build_pod_creation_request(arguments, read_public_key_text(arguments.ssh_private_key_path, False))
    request["env"] = {key: ("<public key>" if "KEY" in key else value) for key, value in request["env"].items()}
    print("DRY RUN: no RunPod calls are made.")
    print(f"RUNPOD_API_KEY found in .env: {bool(secrets['RUNPOD_API_KEY'])}; HF_TOKEN found in .env: {bool(secrets['HF_TOKEN'])}")
    cost_center_name = resolve_cost_center_name(arguments, secrets)
    if cost_center_name:
        print(f"Cost center '{cost_center_name}': the manual assignment instructions (console at "
              f"{COST_CENTER_CONSOLE_URL}) are printed at pod creation and in every progress report until confirmed, and "
              f"the name and the confirmation are recorded in the pod cost log")
    else:
        print("No cost center name given (--cost-center-name or RUNPOD_COST_CENTER_NAME in .env): nothing is "
              "printed, awaited or recorded about a cost center")
    print(f"POST {RUNPOD_REST_BASE_URL}/pods\n{json.dumps(request, indent=2)}")
    print("Commands that would run on the pod:")
    print(f"  upload (tar over ssh, unfinished *.partial files left out): {', '.join(REPOSITORY_PATHS_UPLOADED_TO_POD)}, "
          f"and from the run folder run_configuration_*.json, scoring_inputs and scoring_outputs")
    print(f"  pip install -q -r {REQUIREMENTS_FILE_RELATIVE_PATH} (skipped when {REQUIREMENTS_INSTALLED_MARKER_FILE_NAME} "
          f"on the pod holds the same requirements file hash)")
    print(f"  version check, logged before scoring: {POD_VERSION_CHECK_COMMAND}")
    print(f"  earlier watchdogs on the pod are killed first: pkill -f '{WATCHDOG_PROCESS_PATTERN}'")
    print(f"  watchdog: {build_self_termination_watchdog_command(arguments.maximum_running_hours * 3600 + WATCHDOG_GRACE_SECONDS)}")
    print(f"  scorer:   {build_scorer_launch_shell_command(arguments)}")
    print(f"  pull every {PROGRESS_REPORT_INTERVAL_SECONDS} seconds: the finished files of {REMOTE_RUN_FOLDER}/scoring_outputs "
          f"(copied to a staging folder first, so nothing changes while it is read; a missing folder is nothing to pull)")
    print(f"  early crash check {EARLY_CRASH_CHECK_DELAY_SECONDS} seconds after the scorer starts")
    run_folder_path = Path(arguments.run_folder).resolve()
    print(f"  scorer failure (non-zero exit, vanished, early crash): the tail of scorer_standard_output.txt is printed, "
          f"everything is pulled and verified, and the pod is NOT deleted. It waits, checking every "
          f"{INSTRUCTION_CHECK_INTERVAL_SECONDS} seconds, for an instruction file in the run folder:\n"
          f"    {run_folder_path / RETRY_SCORER_INSTRUCTION_FILE_NAME}  re-upload the project scripts and restart the "
          f"scorer on the same pod (it skips finished documents)\n"
          f"    {run_folder_path / DELETE_POD_INSTRUCTION_FILE_NAME}  delete the pod\n"
          f"  with neither within {arguments.failure_wait_minutes:g} minutes (--failure-wait-minutes) the pod is deleted. "
          f"Each instruction file is renamed with a timestamp once acted on.")
    print(f"  quick-to-fix setup failure (upload, pip install, scorer start): same waiting, with "
          f"{run_folder_path / RETRY_SETUP_INSTRUCTION_FILE_NAME} meaning redo upload, install and scorer start on the "
          f"same pod. Pod creation failure and ssh never becoming ready delete the pod.")
    print(f"  the hours and dollar limits and Ctrl-C delete the pod immediately, also while it waits")
    print(f"  --reuse-existing-pod <pod id>: skip creation, take ssh details and the hourly price from the API, move "
          f"the pod's old run folder aside, kill earlier watchdogs and set a new one before starting the scorer, "
          f"cost counted from this launch; refuses if a scorer is already running on the pod")
    print(f"  final pull, whatever ends the run: everything in {REMOTE_RUN_FOLDER} except scoring_inputs, plus "
          f"watchdog_log.txt and pod_setup_*.txt, into the run folder (non-scoring_outputs files in "
          f"pod_<pod id>_logs_<timestamp>/); file list and sizes are verified, retried for at most "
          f"{FINAL_PULL_TOTAL_BUDGET_SECONDS} seconds and never past the limits")
    print(f"  after a successful run (final pull verified): "
          + ("the pod is KEPT running (--keep-pod-after-run), limits still apply through the on-pod watchdog; "
             "pod_kept_after_run: true goes in the pod cost log" if arguments.keep_pod_after_run else
             "the pod is deleted (after the 2-minute cost center window if a cost center name was given and the assignment is unconfirmed); "
             "--keep-pod-after-run would keep it instead"))
    print_cost_ceiling_and_projection(arguments)


def find_previous_start_epoch_seconds(run_folder, pod_id):
    for path in sorted(Path(run_folder).glob("pod_cost_log_*.json")):
        record = json.loads(path.read_text())
        if record.get("pod_id") == pod_id and record.get("start_time"):
            return datetime.fromisoformat(record["start_time"]).timestamp()
    return None


def transfer_is_connected(transfer):
    return getattr(transfer, "host", True) is not None


def run_launcher(arguments, client=None, transfer=None, secrets=None, clock=time.time, sleep=time.sleep,
                 confirmation_source=None):
    """Returns the process exit code. All RunPod access goes through `client` and `transfer` (fakes in tests)."""
    secrets = secrets if secrets is not None else read_secrets_from_environment_file()
    if arguments.dry_run:
        print_dry_run(arguments, secrets)
        return 0
    if not secrets["RUNPOD_API_KEY"]:
        raise SystemExit(f"RUNPOD_API_KEY is missing from {ENVIRONMENT_FILE_PATH}. Add a line RUNPOD_API_KEY=... to "
                         f"the project's .env (the OS environment is deliberately not used).")
    if arguments.attach_to_existing_pod and arguments.reuse_existing_pod:
        raise SystemExit("Use either --attach-to-existing-pod (monitor a run in progress) or --reuse-existing-pod (start "
                         "this run on an idle pod), not both")
    run_folder = Path(arguments.run_folder)
    if not (run_folder / "scoring_inputs").is_dir():
        raise SystemExit(f"{run_folder}/scoring_inputs does not exist; run build_scoring_inputs.py first")
    public_key_text = read_public_key_text(arguments.ssh_private_key_path, required=client is None)
    client = client or RunPodRestClient(secrets["RUNPOD_API_KEY"])
    transfer = transfer or SshTarTransfer(arguments.ssh_private_key_path, run_folder)
    launcher_timestamp = timestamp_now()
    cost_center_name = resolve_cost_center_name(arguments, secrets)
    cost_log = PodCostLog(run_folder / f"pod_cost_log_{launcher_timestamp}.json",
                          {"launcher_timestamp": launcher_timestamp, "run_folder": str(run_folder),
                           "maximum_running_hours": arguments.maximum_running_hours,
                           "maximum_dollars": arguments.maximum_dollars, "pod_id": None,
                           "cost_center_name": cost_center_name,
                           "cost_center_assignment_confirmed_by_user": False,
                           "cost_center_assignment_confirmation_time": None})
    if confirmation_source is None and cost_center_name:
        confirmation_source = StdinConfirmationListener()
        confirmation_source.start()
    session = PodSession(arguments, client, transfer, secrets, cost_log, clock, sleep, confirmation_source)

    def raise_keyboard_interrupt_on_terminate_signal(signal_number, frame):
        raise KeyboardInterrupt()

    previous_sigterm_handler = signal.signal(signal.SIGTERM, raise_keyboard_interrupt_on_terminate_signal)
    reason, exit_code = "exception", 1
    try:
        if arguments.attach_to_existing_pod:
            pod = session.attach_to_pod(arguments.attach_to_existing_pod,
                                        find_previous_start_epoch_seconds(run_folder, arguments.attach_to_existing_pod))
            transfer.wait_until_ready(pod, clock() + 600, lambda: client.get_pod(session.pod_id), sleep)
            session.say(f"Attached to pod {session.pod_id}; monitoring only")
            setup_ended_with_reason = None
        else:
            if arguments.reuse_existing_pod:
                pod = session.reuse_pod(arguments.reuse_existing_pod)
            else:
                pod = session.start_new_pod(build_pod_creation_request(arguments, public_key_text))
            setup_ended_with_reason = session.prepare_and_start_scorer(pod)
        if setup_ended_with_reason:
            reason, final_pull_succeeded = setup_ended_with_reason, False
        else:
            reason, final_pull_succeeded = session.monitor_until_finished_or_limit()
        exit_code = 0 if reason == "scorer_finished" and final_pull_succeeded else 1
    except KeyboardInterrupt:
        reason, exit_code = "interrupted_by_user", 130
        session.say("Interrupted; pulling what exists and terminating the pod")
    except Exception as error:
        reason, exit_code = f"exception: {error}", 1
        session.say(f"Error: {error}")
    finally:
        if (reason == "interrupted_by_user" or reason.startswith("exception")) and transfer_is_connected(transfer):
            try:
                session.stop_scorer_if_running()
                session.final_pull_with_verification()
            except BaseException as error:
                session.say(f"Final pull could not be completed: {error!r}")
        if exit_code == 0 and arguments.keep_pod_after_run:
            session.keep_pod_running()
        else:
            session.terminate(reason)
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
    return exit_code


def main():
    sys.exit(run_launcher(parse_arguments()))


if __name__ == "__main__":
    main()
