"""Small standard-library helpers shared by the scorer (which runs on the pod without the rest of the repository)
and the pod launcher: timestamps, durations, timed progress reports, and append-only JSONL records.

Nothing here imports from the rest of the repository, so the whole `evals/log_probability_persona_check/` folder
can be uploaded to a pod on its own.
"""

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROGRESS_REPORT_INTERVAL_SECONDS = 180


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def timestamp_unused_in_folder(folder, file_name_templates):
    """The usual timestamp, or that with `_2`, `_3`... added, such that none of the files named by the templates (each
    containing `{timestamp}`) exists in the folder yet. Two runs started within the same minute then never overwrite
    each other's output files."""
    base_timestamp = timestamp_now()
    timestamp, attempt = base_timestamp, 1
    while any((Path(folder) / template.format(timestamp=timestamp)).exists() for template in file_name_templates):
        attempt += 1
        timestamp = f"{base_timestamp}_{attempt}"
    return timestamp


def format_duration(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def read_jsonl(path):
    """All records of a JSON lines file. A final line that is not valid JSON (a file copied from the pod while the
    scorer was appending to it, or a scorer killed mid-write) is ignored with a warning; an invalid line anywhere
    else still raises."""
    with open(path, encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    records = []
    for line_number, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if line_number < len(lines) - 1:
                raise
            print(f"Ignoring the truncated final line of {path}", file=sys.stderr, flush=True)
    return records


def append_jsonl_record(path, record):
    """Appends one record on its own line. A final line left unfinished by an interrupted write is first removed
    (or, if it is a complete record that lacks its newline, terminated), so the new record is never glued onto it."""
    finish_unterminated_final_line(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def finish_unterminated_final_line(path):
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        return
    with open(path, "rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) == b"\n":
            return
        handle.seek(0)
        content = handle.read()
        start_of_final_line = content.rfind(b"\n") + 1
        try:
            json.loads(content[start_of_final_line:].decode("utf-8"))
        except ValueError:
            handle.truncate(start_of_final_line)
        else:
            handle.write(b"\n")


def find_completed_keys(jsonl_path, key_field_names):
    """Set of tuples of `key_field_names` values over every record in the file (empty if it does not exist)."""
    if not Path(jsonl_path).is_file():
        return set()
    return {tuple(record[name] for name in key_field_names) for record in read_jsonl(jsonl_path)}


class TimedProgressReporter:
    """Counts finished tasks and prints and logs elapsed time, rate and a remaining-time estimate, at most every
    `report_interval_seconds` and once at the end. `status_suffix_function` returns extra text for each line."""

    def __init__(self, log_path, total_new_tasks, already_done_count=0, status_suffix_function=None,
                 report_interval_seconds=PROGRESS_REPORT_INTERVAL_SECONDS):
        self.log_path = log_path
        self.total_new_tasks = total_new_tasks
        self.already_done_count = already_done_count
        self.status_suffix_function = status_suffix_function
        self.report_interval_seconds = report_interval_seconds
        self.completed_new_tasks = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.last_report_time = self.start_time

    def note_task_finished(self):
        with self.lock:
            self.completed_new_tasks += 1
            if time.time() - self.last_report_time >= self.report_interval_seconds:
                self._report(final=False)

    def finish(self):
        with self.lock:
            self._report(final=True)

    def _report(self, final):
        now = time.time()
        self.last_report_time = now
        elapsed = now - self.start_time
        rate = self.completed_new_tasks / elapsed if elapsed > 0 else 0.0
        remaining = (self.total_new_tasks - self.completed_new_tasks) / rate if rate > 0 else float("nan")
        suffix = f"; {self.status_suffix_function()}" if self.status_suffix_function else ""
        line = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {'FINAL ' if final else ''}"
                f"done {self.completed_new_tasks}/{self.total_new_tasks} this session "
                f"({self.already_done_count} done before resume); elapsed {format_duration(elapsed)}; "
                f"rate {rate * 60:.1f} documents/min; estimated remaining "
                f"{format_duration(remaining) if remaining == remaining else 'unknown'}{suffix}")
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
