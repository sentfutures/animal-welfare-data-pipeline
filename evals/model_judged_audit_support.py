"""Shared building blocks for the model-judged corpus audits (plausibility_audit_sdf.py and
shoehorning_audit_sdf.py) and their post-processors.

Holds the resumable JSONL records, run-folder management, strict JSON schema helpers, quote verification,
timed progress reporting, CSV writing, stratified estimate sample and partial-run folder rule. Model calls go through
shared/openai_api.py (and so into the repo's cost log and model-call log), keys through
shared/api_keys.py. See the "Model-judged audits" section of evals/README.md.
"""

import csv
import difflib
import hashlib
import json
import random
import re
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from shared import openai_api

# ----------------------------------------------------------------------------
# Corpus input
# ----------------------------------------------------------------------------

DEFAULT_INPUT = "outputs/sdf/latest"


def corpus_path_of(input_argument):
    """The sdf_corpus.jsonl an --input names: the file itself, or <run directory>/final/sdf_corpus.jsonl."""
    path = Path(input_argument)
    if path.is_dir():
        path = path / "final" / "sdf_corpus.jsonl"
    if not path.is_file():
        raise SystemExit(f"Corpus not found: {path}")
    return path


def run_directory_of(input_argument):
    """The SDF run directory an --input points at (a run directory, or a corpus in <run>/final/), else None."""
    path = Path(input_argument)
    if path.is_dir():
        return path
    if path.parent.name == "final":
        return path.parent.parent
    return None


# ----------------------------------------------------------------------------
# Append-only JSONL files
# ----------------------------------------------------------------------------

def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class AppendOnlyJsonlWriter:
    def __init__(self):
        self.lock = threading.Lock()

    def append(self, path, record):
        with self.lock:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_all_records(folder, file_name_pattern):
    records = []
    for path in sorted(Path(folder).glob(file_name_pattern)):
        records.extend(read_jsonl(path))
    return records


def find_completed_keys(folder, file_name_pattern, key_field_names):
    """Set of tuples of `key_field_names` values over every record already in the folder."""
    return {tuple(record[name] for name in key_field_names)
            for record in read_all_records(folder, file_name_pattern)}


# ----------------------------------------------------------------------------
# Run folders, run configuration, resume safety check
# ----------------------------------------------------------------------------

SMALL_SAMPLE_CHECKS_DIRECTORY_NAME = "small_sample_checks"
TIMESTAMP_SUFFIX_PATTERN = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$")


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def name_with_timestamp_appended_once(run_name):
    """Appends the timestamp unless run_name already ends with one."""
    if TIMESTAMP_SUFFIX_PATTERN.search(run_name):
        return run_name
    return f"{run_name}_{timestamp_now()}"


def sha256_of_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_run_folder(runs_directory, run_name, resume_folder):
    """Returns (folder, is_resume). New folders are created inside runs_directory and named
    <run_name>_<timestamp>, with the timestamp appended exactly once."""
    runs_directory = Path(runs_directory)
    runs_directory.mkdir(parents=True, exist_ok=True)
    if resume_folder:
        folder = Path(resume_folder)
        if not folder.is_dir():
            raise SystemExit(f"--resume-folder {folder} does not exist")
        return folder, True
    base_name = TIMESTAMP_SUFFIX_PATTERN.sub("", run_name)
    existing = sorted(runs_directory.glob(f"{base_name}_????-??-??_??-??"))
    if existing:
        raise SystemExit(f"A folder for run name '{run_name}' already exists: {existing[-1]}\n"
                         f"Pass --resume-folder {existing[-1]} to resume it, or choose a new --run-name.")
    folder = runs_directory / name_with_timestamp_appended_once(run_name)
    folder.mkdir(parents=True)
    return folder, False


def choose_runs_parent_directory(arguments, runs_directory):
    """Audit rule. Returns (kind_description, parent_directory): full-corpus runs go in the runs
    directory itself; every partial run (estimate-only, --document-ids, --document-limit) goes into its
    small_sample_checks subfolder."""
    runs_directory = Path(runs_directory)
    partial_reasons = []
    if arguments.estimate_only:
        partial_reasons.append("estimate-only")
    if arguments.document_ids:
        partial_reasons.append("document-ids subset")
    if arguments.document_limit:
        partial_reasons.append("document-limit subset")
    if partial_reasons:
        return ("partial run (" + ", ".join(partial_reasons) + ")",
                runs_directory / SMALL_SAMPLE_CHECKS_DIRECTORY_NAME)
    return "full-corpus run", runs_directory


def write_run_configuration(run_folder, timestamp, configuration):
    path = Path(run_folder) / f"run_configuration_{timestamp}.json"
    path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    return path


def check_resume_is_compatible(run_folder, new_configuration, hashes_key="prompt_file_sha256",
                               compared_settings=()):
    """Raise RuntimeError, saying what changed, if the new configuration differs from any of the run's
    recorded configurations in a file hash, the corpus hash (when both record one), or any of the
    named settings."""
    configuration_paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if not configuration_paths:
        raise RuntimeError(f"No run_configuration file in {run_folder}; cannot verify prompts.")
    for path in configuration_paths:
        old_configuration = json.loads(path.read_text(encoding="utf-8"))
        old_hashes = old_configuration[hashes_key]
        for file_name, new_hash in new_configuration[hashes_key].items():
            if file_name in old_hashes and old_hashes[file_name] != new_hash:
                raise RuntimeError(
                    f"Refusing to resume: prompt file {file_name} changed since the run started "
                    f"({path.name}).")
        old_corpus_hash = old_configuration.get("corpus_sha256")
        new_corpus_hash = new_configuration.get("corpus_sha256")
        if old_corpus_hash and new_corpus_hash and old_corpus_hash != new_corpus_hash:
            raise RuntimeError(
                f"Refusing to resume: the corpus changed since the run started (recorded corpus "
                f"{old_configuration.get('corpus_path')}, now {new_configuration.get('corpus_path')}; "
                f"corpus_sha256 differs; {path.name}).")
        for setting in compared_settings:
            if old_configuration.get(setting) != new_configuration[setting]:
                raise RuntimeError(
                    f"Refusing to resume: {setting} changed from {old_configuration.get(setting)!r} "
                    f"to {new_configuration[setting]!r} ({path.name}).")


def select_documents(all_documents, document_ids, document_limit, corpus_description):
    """Apply --document-ids (unknown ids exit with an error naming them) and then --document-limit."""
    documents = all_documents
    if document_ids:
        wanted = set(document_ids)
        unknown_ids = sorted(wanted - {d["doc_id"] for d in all_documents})
        if unknown_ids:
            raise SystemExit(f"error: --document-ids not found in corpus {corpus_description}: "
                             + ", ".join(unknown_ids))
        documents = [d for d in documents if d["doc_id"] in wanted]
    if document_limit:
        documents = documents[:document_limit]
    return documents


def choose_stratified_sample(documents, sample_size, seed=0):
    """Round-robin over strata (English/non-English x short/long by median length), fixed seed."""
    median_length = sorted(len(d["content"]) for d in documents)[len(documents) // 2]
    strata = {}
    for document in documents:
        key = (document["language"] == "English", len(document["content"]) >= median_length)
        strata.setdefault(key, []).append(document)
    generator = random.Random(seed)
    for members in strata.values():
        members.sort(key=lambda d: d["doc_id"])
        generator.shuffle(members)
    ordered_keys = sorted(strata.keys())
    sample = []
    while len(sample) < sample_size and any(strata.values()):
        for key in ordered_keys:
            if strata[key] and len(sample) < sample_size:
                sample.append(strata[key].pop())
    return sample


# ----------------------------------------------------------------------------
# Progress reports
# ----------------------------------------------------------------------------

PROGRESS_REPORT_INTERVAL_SECONDS = 180


def format_duration(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


class TimedProgressReporter:
    """Counts finished tasks and prints/logs elapsed time, rate and remaining-time estimate, at most
    every few minutes and once at the end.

    `status_suffix_function` returns extra text appended to each line (for example cost so far).
    """

    def __init__(self, log_path, total_new_tasks, already_done_count=0, status_suffix_function=None,
                 report_interval_seconds=PROGRESS_REPORT_INTERVAL_SECONDS):
        self.log_path = log_path
        self.total_new_tasks = total_new_tasks
        self.already_done_count = already_done_count
        self.status_suffix_function = status_suffix_function
        self.report_interval_seconds = report_interval_seconds
        self.completed_new_tasks = 0
        self.failed_new_tasks = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.last_report_time = self.start_time

    def note_task_finished(self, failed):
        with self.lock:
            if failed:
                self.failed_new_tasks += 1
            else:
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
        done_new = self.completed_new_tasks + self.failed_new_tasks
        rate = done_new / elapsed if elapsed > 0 else 0.0
        remaining = (self.total_new_tasks - done_new) / rate if rate > 0 else float("nan")
        suffix = f"; {self.status_suffix_function()}" if self.status_suffix_function else ""
        line = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{'FINAL ' if final else ''}done {done_new}/{self.total_new_tasks} this session "
                f"({self.already_done_count} done before resume, {self.failed_new_tasks} failed); "
                f"elapsed {format_duration(elapsed)}; rate {rate * 60:.1f} calls/min; "
                f"estimated remaining {format_duration(remaining) if remaining == remaining else 'unknown'}"
                f"{suffix}")
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# ----------------------------------------------------------------------------
# Strict JSON schemas
# ----------------------------------------------------------------------------

def build_strict_object_schema(properties):
    """Object schema where every property is required and no extras are allowed.

    The order of `properties` is preserved, and with structured outputs it is the generation order.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def build_array_schema(items_schema, maximum_items=None):
    array_schema = {"type": "array", "items": items_schema}
    if maximum_items is not None:
        array_schema["maxItems"] = maximum_items
    return array_schema


def find_strict_schema_violations(schema, path="schema"):
    """Return a list of human-readable problems (empty list means the schema is strict-valid)."""
    problems = []
    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            problems.append(f"{path}: additionalProperties is not False")
        if sorted(schema.get("required", [])) != sorted(schema.get("properties", {}).keys()):
            problems.append(f"{path}: required does not list every property")
        for name, sub_schema in schema.get("properties", {}).items():
            problems.extend(find_strict_schema_violations(sub_schema, f"{path}.{name}"))
    elif schema.get("type") == "array":
        problems.extend(find_strict_schema_violations(schema["items"], f"{path}[]"))
    return problems


def assert_schema_is_strict_valid(schema):
    problems = find_strict_schema_violations(schema)
    assert not problems, "; ".join(problems)


# ----------------------------------------------------------------------------
# Quote verification
# ----------------------------------------------------------------------------

FUZZY_MATCH_THRESHOLD = 90.0

QUOTE_CHARACTER_TRANSLATION = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"', "「": '"', "」": '"'})

try:
    from rapidfuzz import fuzz as rapidfuzz_module
except ImportError:
    rapidfuzz_module = None


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).translate(QUOTE_CHARACTER_TRANSLATION)
    return re.sub(r"\s+", " ", text).strip()


def partial_match_score(needle, haystack):
    """Best partial similarity (0-100) of needle against a needle-length window of haystack.

    A needle at least as long as the haystack is compared with the whole haystack, so a long quote
    that merely contains a short paragraph never scores as a match for it.
    """
    if not needle:
        return 0.0
    if len(haystack) <= len(needle):
        if rapidfuzz_module is not None:
            return float(rapidfuzz_module.ratio(needle, haystack))
        return 100.0 * difflib.SequenceMatcher(None, needle, haystack, autojunk=False).ratio()
    if rapidfuzz_module is not None:
        return float(rapidfuzz_module.partial_ratio(needle, haystack))
    # Windows start where a block of the needle matches the haystack, so the best alignment is
    # found wherever the quote sits in the paragraph.
    matching_blocks = difflib.SequenceMatcher(None, needle, haystack, autojunk=False).get_matching_blocks()
    latest_window_start = len(haystack) - len(needle)
    window_starts = {min(max(0, haystack_position - needle_position), latest_window_start)
                     for needle_position, haystack_position, block_length in matching_blocks if block_length}
    best = 0.0
    for start in sorted(window_starts):
        window = haystack[start:start + len(needle)]
        best = max(best, 100.0 * difflib.SequenceMatcher(None, needle, window, autojunk=False).ratio())
    return best


PARAGRAPH_BREAK_PATTERN = re.compile(r"\n\s*\n")


def split_into_normalized_paragraphs(document_text):
    """Paragraphs are separated by blank lines (two or more newlines with only whitespace between)."""
    paragraphs = (normalize_text(paragraph) for paragraph in PARAGRAPH_BREAK_PATTERN.split(document_text))
    return [paragraph for paragraph in paragraphs if paragraph]


def verify_quote(quote, document_text):
    """Returns (found, exact_substring, fuzzy_score, quote_spans_paragraph_break).

    A quote only counts as found if it matches within a single paragraph of the document. The
    last value is True when the quote would match only if paragraphs were joined together.
    """
    normalized_quote = normalize_text(quote)
    paragraphs = split_into_normalized_paragraphs(document_text)
    if normalized_quote and any(normalized_quote in paragraph for paragraph in paragraphs):
        return True, True, 100.0, False
    best_paragraph_score = max(
        (partial_match_score(normalized_quote, paragraph) for paragraph in paragraphs), default=0.0)
    found = best_paragraph_score >= FUZZY_MATCH_THRESHOLD
    spans_paragraph_break = False
    if not found and normalized_quote and len(paragraphs) > 1:
        joined_document = " ".join(paragraphs)
        spans_paragraph_break = (normalized_quote in joined_document
                                 or partial_match_score(normalized_quote, joined_document) >= FUZZY_MATCH_THRESHOLD)
    return found, False, round(best_paragraph_score, 1), spans_paragraph_break


# ----------------------------------------------------------------------------
# CSV output
# ----------------------------------------------------------------------------

def to_csv_value(value):
    """None becomes empty, lists are joined with ';', dicts are JSON-encoded."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(element) for element in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


def derive_csv_columns(rows, preferred_columns=()):
    """Preferred columns present in the rows first, then every other field in first-seen order."""
    seen_fields = []
    for row in rows:
        for field in row:
            if field not in seen_fields:
                seen_fields.append(field)
    preferred = [column for column in preferred_columns if column in seen_fields]
    return preferred + [field for field in seen_fields if field not in preferred]


def write_rows_to_csv(path, rows, preferred_columns=()):
    """Write dict rows to CSV with columns derived from the rows, so new fields can never crash it."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=derive_csv_columns(rows, preferred_columns), restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_csv_value(value) for key, value in row.items()})


# ----------------------------------------------------------------------------
# Usage and cost totals (for progress lines and the run's usage_and_cost_log file; the repo's own
# cost log, written by shared/openai_api.py per call, stays the record of spend)
# ----------------------------------------------------------------------------

USAGE_FIELD_NAMES = ["input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"]


def compute_cost_in_dollars(model_name, input_tokens, cached_input_tokens, output_tokens):
    return openai_api.compute_cost_usd(model_name, input_tokens, cached_input_tokens, output_tokens)


class RunningUsageTotals:
    """Thread-safe token totals over a run's records (including records from before a resume)."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.lock = threading.Lock()
        self.calls = 0
        self.tokens = {field_name: 0 for field_name in USAGE_FIELD_NAMES}

    def add_usage(self, usage):
        with self.lock:
            self.calls += 1
            for field_name in USAGE_FIELD_NAMES:
                self.tokens[field_name] += usage.get(field_name, 0)

    def totals(self):
        with self.lock:
            tokens = dict(self.tokens)
            calls = self.calls
        cost = compute_cost_in_dollars(self.model_name, tokens["input_tokens"], tokens["cached_input_tokens"],
                                       tokens["output_tokens"])
        cache_rate = tokens["cached_input_tokens"] / tokens["input_tokens"] if tokens["input_tokens"] else 0.0
        return {"calls": calls, **tokens, "cache_rate": cache_rate, "cost_dollars": cost}

    def status_text(self):
        totals = self.totals()
        return f"total cost so far ${totals['cost_dollars']:.4f}; cache rate {totals['cache_rate']:.1%}"

    def write_log_file(self, path, note):
        Path(path).write_text(json.dumps({"model": self.model_name, "note": note, "totals": self.totals()},
                                         indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------
# Structured-output requests through shared/openai_api.py
# ----------------------------------------------------------------------------

# shared/openai_api.py requires an output cap, and it includes
# reasoning tokens; this is set well above anything the audits' answers have needed.
MAXIMUM_OUTPUT_TOKENS = 64000
EMPTY_USAGE = {field_name: 0 for field_name in USAGE_FIELD_NAMES}


def sum_usages(usages):
    return {field_name: sum(usage[field_name] for usage in usages) for field_name in USAGE_FIELD_NAMES}


def usage_paid_before_transport_failure(error):
    """Usage of the attempts that completed before a transport error escaped request_structured_output."""
    return dict(getattr(error, "paid_usage_of_earlier_attempts", EMPTY_USAGE))


class StructuredOutputParseFailure(Exception):
    """Both parse attempts failed; carries what the error log needs."""

    def __init__(self, description, raw_output_text, usage, latency_seconds):
        super().__init__(description)
        self.description = description
        self.raw_output_text = raw_output_text
        self.usage = usage
        self.latency_seconds = latency_seconds


def request_structured_output(model_name, instructions_text, input_text, schema_name, schema,
                              reasoning_effort, stage, item_id, validate_parsed_result=None,
                              prompt_cache_key=None, maximum_output_tokens=MAXIMUM_OUTPUT_TOKENS):
    """Call the model through shared.openai_api.call_openai, parse the JSON, validate it, and retry
    once on a parse or validation failure (a refused or cut-off answer counts as a failure).

    Returns a dict with parsed_result, raw_output_text, usage (summed over attempts), parse_attempts
    and latency_seconds. Raises StructuredOutputParseFailure if both attempts fail; transport errors
    propagate (call_openai retries the transient ones itself), carrying the usage of any earlier
    attempt (see usage_paid_before_transport_failure).
    """
    attempt_usages = []
    last_failure_description = None
    last_raw_output_text = None
    started = time.time()
    for parse_attempt_number in (1, 2):
        usage = {}
        try:
            raw_output_text, stop_reason = openai_api.call_openai(
                input_text, instructions_text, model=model_name, max_output_tokens=maximum_output_tokens,
                reasoning_effort=reasoning_effort, stage=stage, item_id=item_id, json_schema=schema,
                schema_name=schema_name, prompt_cache_key=prompt_cache_key, usage_out=usage)
        except Exception as error:
            error.paid_usage_of_earlier_attempts = sum_usages(attempt_usages)
            raise
        attempt_usages.append({**EMPTY_USAGE, **usage})
        last_raw_output_text = raw_output_text
        try:
            if stop_reason != "end_turn":
                raise ValueError(f"response stop reason {stop_reason}")
            parsed_result = json.loads(raw_output_text)
            if validate_parsed_result is not None:
                validate_parsed_result(parsed_result)
        except (ValueError, TypeError) as failure:
            last_failure_description = f"attempt {parse_attempt_number}: {failure}"
            continue
        return {"parsed_result": parsed_result, "raw_output_text": raw_output_text,
                "usage": sum_usages(attempt_usages), "parse_attempts": parse_attempt_number,
                "latency_seconds": round(time.time() - started, 3)}
    raise StructuredOutputParseFailure(last_failure_description, last_raw_output_text,
                                       sum_usages(attempt_usages), round(time.time() - started, 3))
