#!/usr/bin/env python3
"""LLM judge that scores every document of an SDF corpus for scenario plausibility (paid, one call per document).

Calls go through shared/openai_api.py (cost log, model-call log) and keys through shared/api_keys.py.

One pass per document. The model receives ONLY the instructions
(prompts/tools/plausibility_audit_judge_instructions.txt) and the document's `content` text. Nothing else
about the document (description, variables, language, doc_id, scores) is ever sent.

Runs are resumable: finished doc_ids are read back from the run folder. Run
evals/plausibility_audit_postprocess_sdf.py on the run folder afterwards for the per-document scores,
per-problem rows and summaries.

Usage examples:
    python evals/plausibility_audit_sdf.py --input outputs/sdf/latest --run-name full_plausibility_audit_medium_effort
    python evals/plausibility_audit_sdf.py --input outputs/sdf/latest --run-name x --estimate-only
    python evals/plausibility_audit_sdf.py --input outputs/sdf/latest --run-name x --document-ids matrix_000003
    python evals/plausibility_audit_sdf.py --input outputs/sdf/latest --run-name x \\
        --resume-folder outputs/sdf/latest/audit/plausibility_audit_runs/x_2026-10-07_12-00

Run folders go to <run>/audit/plausibility_audit_runs/ (--runs-directory to change); every partial run
(--estimate-only, --document-ids, --document-limit) goes to its small_sample_checks subfolder.
"""

import argparse
import concurrent.futures
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.model_judged_audit_support import (  # noqa: E402
    DEFAULT_INPUT, MAXIMUM_OUTPUT_TOKENS, AppendOnlyJsonlWriter, RunningUsageTotals, StructuredOutputParseFailure,
    TimedProgressReporter,
    build_array_schema, build_strict_object_schema, check_resume_is_compatible, choose_run_folder,
    choose_runs_parent_directory, choose_stratified_sample, compute_cost_in_dollars, corpus_path_of,
    find_completed_keys, format_duration, read_all_records, read_jsonl, request_structured_output,
    run_directory_of, select_documents, sha256_of_file, timestamp_now, usage_paid_before_transport_failure,
    write_run_configuration)
from shared import api, openai_api  # noqa: E402
from shared.api_keys import add_api_key_source_flags, apply_api_key_source_flags  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
PROMPT_DIRECTORY = REPO_ROOT / "prompts" / "tools"
INSTRUCTIONS_FILE_NAME = "plausibility_audit_judge_instructions.txt"

MODEL_NAME = "gpt-5.6-luna"
STAGE_TAG = "eval_plausibility_audit"
SAMPLING_TEMPERATURE = openai_api.SAMPLING_TEMPERATURE
SCHEMA_NAME = "plausibility_audit_judgment"
MAXIMUM_ENTRIES_PER_LIST = 3
SEVERITY_VALUES = ["minor", "noticeable", "breaks_believability"]
SCORE_VALUES = [1, 2, 3, 4, 5]
EXPECTED_AXIS_COUNT = 10


# ----------------------------------------------------------------------------
# Prompt parsing and schema construction
# ----------------------------------------------------------------------------

def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def instructions_file_path(prompt_directory=PROMPT_DIRECTORY):
    return Path(prompt_directory) / INSTRUCTIONS_FILE_NAME


def parse_axis_keys(instructions_text):
    """Ordered axis keys from the '### axis_..' headers (the prompt is the single source of truth)."""
    return [match.group(1) for match in re.finditer(r"^###\s+(axis_\w+)\s*$", instructions_text, re.MULTILINE)]


def build_problem_schema():
    return build_strict_object_schema({
        "quote": {"type": "string"},
        "english_gloss": {"type": ["string", "null"]},
        "explanation": {"type": "string"},
        "severity": {"type": "string", "enum": SEVERITY_VALUES},
    })


def build_score_schema():
    # Strict mode is not relied on for numeric bounds; an enum of the five allowed integers is accepted in
    # strict mode and constrains the output. validate_parsed_result re-checks the range after parsing anyway.
    return {"type": "integer", "enum": SCORE_VALUES}


def build_axis_schema():
    return build_strict_object_schema({  # dict order = generation order: reasoning before score
        "applicable": {"type": "boolean"},
        "problems": build_array_schema(build_problem_schema(), MAXIMUM_ENTRIES_PER_LIST),
        "score_reasoning": {"type": "string"},
        "score": build_score_schema(),
    })


def build_other_implausibility_schema():
    return build_strict_object_schema({
        "suggested_axis_name": {"type": "string"},
        "description": {"type": "string"},
        "quote": {"type": "string"},
        "english_gloss": {"type": ["string", "null"]},
        "explanation": {"type": "string"},
        "severity": {"type": "string", "enum": SEVERITY_VALUES},
    })


def build_judgment_schema(axis_keys):
    return build_strict_object_schema({
        "axes": build_strict_object_schema({axis_key: build_axis_schema() for axis_key in axis_keys}),
        "other_implausibilities": build_array_schema(build_other_implausibility_schema(),
                                                     MAXIMUM_ENTRIES_PER_LIST),
        "overall_reasoning": {"type": "string"},
        "overall_plausibility": build_score_schema(),
    })


class JudgeDefinition:
    def __init__(self, prompt_directory=PROMPT_DIRECTORY):
        self.instructions_text = read_text(instructions_file_path(prompt_directory))
        self.axis_keys = parse_axis_keys(self.instructions_text)
        self.schema = build_judgment_schema(self.axis_keys)
        self.schema_name = SCHEMA_NAME


def build_request_arguments(judge_definition, document_content, reasoning_effort):
    """The complete payload sent to client.responses.create by shared.openai_api.call_openai.
    Only instructions and content go in."""
    return openai_api.build_request_arguments(
        MODEL_NAME, judge_definition.instructions_text, document_content,
        max_output_tokens=MAXIMUM_OUTPUT_TOKENS, reasoning_effort=reasoning_effort,
        json_schema=judge_definition.schema, schema_name=judge_definition.schema_name,
        prompt_cache_key=judge_definition.schema_name)


def is_valid_score(value):
    return isinstance(value, int) and not isinstance(value, bool) and value in SCORE_VALUES


def validate_quoted_entry(entry, description, extra_text_fields=()):
    if not isinstance(entry, dict):
        raise ValueError(f"{description} is not an object")
    for field_name in ("quote", "explanation", *extra_text_fields):
        if not isinstance(entry.get(field_name), str):
            raise ValueError(f"{description} has no text field {field_name}")
    if not (entry.get("english_gloss") is None or isinstance(entry["english_gloss"], str)):
        raise ValueError(f"{description} english_gloss is neither text nor null")
    if entry.get("severity") not in SEVERITY_VALUES:
        raise ValueError(f"{description} has invalid severity {entry.get('severity')!r}")


def validate_parsed_result(parsed_result, judge_definition):
    """Raise ValueError if the parsed JSON is not a complete, in-range judgment. Nothing is defaulted."""
    if not isinstance(parsed_result, dict) or not isinstance(parsed_result.get("axes"), dict):
        raise ValueError("missing top-level 'axes' object")
    axes = parsed_result["axes"]
    missing_axes = [key for key in judge_definition.axis_keys if key not in axes]
    unexpected_axes = [key for key in axes if key not in judge_definition.axis_keys]
    if missing_axes or unexpected_axes:
        raise ValueError(f"axis keys mismatch: missing {missing_axes}, unexpected {unexpected_axes}")
    for axis_key in judge_definition.axis_keys:
        axis = axes[axis_key]
        if not isinstance(axis, dict):
            raise ValueError(f"{axis_key} is not an object")
        if not isinstance(axis.get("applicable"), bool):
            raise ValueError(f"{axis_key} applicable is not a boolean")
        if not isinstance(axis.get("score_reasoning"), str):
            raise ValueError(f"{axis_key} has no score_reasoning text")
        if not is_valid_score(axis.get("score")):
            raise ValueError(f"{axis_key} score {axis.get('score')!r} is not an integer from 1 to 5")
        problems = axis.get("problems")
        if not isinstance(problems, list) or len(problems) > MAXIMUM_ENTRIES_PER_LIST:
            raise ValueError(f"{axis_key} problems is not a list of at most {MAXIMUM_ENTRIES_PER_LIST}")
        for problem in problems:
            validate_quoted_entry(problem, f"{axis_key} problem")
    others = parsed_result.get("other_implausibilities")
    if not isinstance(others, list) or len(others) > MAXIMUM_ENTRIES_PER_LIST:
        raise ValueError(f"other_implausibilities is not a list of at most {MAXIMUM_ENTRIES_PER_LIST}")
    for other in others:
        validate_quoted_entry(other, "other implausibility", ("suggested_axis_name", "description"))
    if not isinstance(parsed_result.get("overall_reasoning"), str):
        raise ValueError("missing overall_reasoning text")
    if not is_valid_score(parsed_result.get("overall_plausibility")):
        raise ValueError(f"overall_plausibility {parsed_result.get('overall_plausibility')!r} "
                         "is not an integer from 1 to 5")


def judge_one_document(judge_definition, document, reasoning_effort):
    """Returns (record, error_record). Retries once on parse/schema failure."""
    try:
        result = request_structured_output(
            MODEL_NAME, judge_definition.instructions_text, document["content"],
            judge_definition.schema_name, judge_definition.schema, reasoning_effort,
            stage=STAGE_TAG, item_id=document["doc_id"],
            validate_parsed_result=lambda parsed: validate_parsed_result(parsed, judge_definition),
            prompt_cache_key=judge_definition.schema_name)
    except StructuredOutputParseFailure as failure:
        return None, {
            "doc_id": document["doc_id"], "error_kind": "parse_or_schema_failure",
            "description": failure.description, "raw_output_text": failure.raw_output_text,
            "usage": failure.usage, "latency_seconds": failure.latency_seconds}
    return {
        "doc_id": document["doc_id"],
        "model": MODEL_NAME,
        "reasoning_effort": reasoning_effort,
        "parsed_result": result["parsed_result"],
        "raw_output_text": result["raw_output_text"],
        "usage": result["usage"],
        "parse_attempts": result["parse_attempts"],
        "latency_seconds": result["latency_seconds"],
    }, None


# ----------------------------------------------------------------------------
# Run folder, resume, configuration
# ----------------------------------------------------------------------------

JUDGMENT_FILE_PATTERN = "judgments_*.jsonl"


def load_corpus(corpus_path):
    return read_jsonl(corpus_path)


def find_completed_document_ids(run_folder):
    return {key[0] for key in find_completed_keys(run_folder, JUDGMENT_FILE_PATTERN, ["doc_id"])}


def load_existing_records(run_folder):
    return read_all_records(run_folder, JUDGMENT_FILE_PATTERN)


def build_run_configuration(reasoning_effort, corpus_path, prompt_directory=PROMPT_DIRECTORY):
    input_price, cached_input_price, output_price = openai_api._PRICING[MODEL_NAME]
    return {
        "model": MODEL_NAME,
        "reasoning_effort": reasoning_effort,
        "temperature": SAMPLING_TEMPERATURE,
        "prompt_file_sha256": {INSTRUCTIONS_FILE_NAME: sha256_of_file(instructions_file_path(prompt_directory))},
        "axis_keys": parse_axis_keys(read_text(instructions_file_path(prompt_directory))),
        "corpus_path": str(Path(corpus_path).resolve()),
        "corpus_sha256": sha256_of_file(corpus_path),
        "prices_per_million_tokens": {
            "input": input_price, "cached_input": cached_input_price, "output": output_price},
    }


def check_resume_configuration_matches(run_folder, new_configuration):
    """Refuse a resume with a changed prompt (shared check), model or reasoning effort."""
    check_resume_is_compatible(run_folder, new_configuration)
    for path in sorted(Path(run_folder).glob("run_configuration_*.json")):
        old_configuration = json.loads(path.read_text(encoding="utf-8"))
        for setting in ("model", "reasoning_effort"):
            if old_configuration.get(setting) != new_configuration[setting]:
                raise RuntimeError(f"Refusing to resume: {setting} changed from "
                                   f"{old_configuration.get(setting)!r} to {new_configuration[setting]!r} "
                                   f"({path.name}).")


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------

class RunReporter:
    """Writes judgments, errors, progress reports and the usage log for one run."""

    def __init__(self, run_folder, timestamp, total_new_tasks, already_done_count, prior_records):
        self.run_folder = Path(run_folder)
        self.timestamp = timestamp
        self.record_writer = AppendOnlyJsonlWriter()
        self.usage_totals = RunningUsageTotals(MODEL_NAME)
        for record in prior_records:
            self.usage_totals.add_usage(record["usage"])
        self.judgments_path = self.run_folder / f"judgments_{timestamp}.jsonl"
        self.errors_path = self.run_folder / f"errors_{timestamp}.jsonl"
        self.usage_log_path = self.run_folder / f"usage_and_cost_log_{timestamp}.json"
        self.progress_reporter = TimedProgressReporter(
            self.run_folder / f"progress_log_{timestamp}.txt", total_new_tasks, already_done_count,
            status_suffix_function=self.usage_totals.status_text)

    @property
    def completed_new_tasks(self):
        return self.progress_reporter.completed_new_tasks

    @property
    def failed_new_tasks(self):
        return self.progress_reporter.failed_new_tasks

    def record_success(self, record):
        self.record_writer.append(self.judgments_path, record)
        self.usage_totals.add_usage(record["usage"])
        self.progress_reporter.note_task_finished(failed=False)

    def record_failure(self, error_record):
        self.record_writer.append(self.errors_path, error_record)
        self.usage_totals.add_usage(error_record["usage"])
        self.progress_reporter.note_task_finished(failed=True)

    def finish(self):
        self.progress_reporter.finish()
        self.usage_totals.write_log_file(self.usage_log_path, "totals include records from before any resume")


def run_judging(run_folder, timestamp, judge_definition, documents, reasoning_effort, worker_count,
                already_completed_document_ids, prior_records):
    """Judge every not-yet-done document. Returns the reporter."""
    tasks = [document for document in documents if document["doc_id"] not in already_completed_document_ids]
    reporter = RunReporter(run_folder, timestamp, len(tasks), len(already_completed_document_ids), prior_records)
    print(f"{len(tasks)} calls to make ({len(already_completed_document_ids)} documents already done).",
          flush=True)

    def work(document):
        try:
            record, error_record = judge_one_document(judge_definition, document, reasoning_effort)
        except Exception as error:  # noqa: BLE001 - transport failure after all retries
            record, error_record = None, {
                "doc_id": document["doc_id"], "error_kind": "transport_failure", "description": repr(error),
                "raw_output_text": None,
                "usage": usage_paid_before_transport_failure(error),
                "latency_seconds": None}
        if record:
            reporter.record_success(record)
        else:
            reporter.record_failure(error_record)

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(work, tasks))
    reporter.finish()
    return reporter


# ----------------------------------------------------------------------------
# Estimate-only
# ----------------------------------------------------------------------------

def extrapolate_estimate(sample_records, sample_documents, all_documents, error_records=()):
    """Scale uncached input by content characters, output and cached input by document count.

    Scales are taken over the sample documents that were judged successfully; the tokens of failed
    attempts count towards the totals, as they are paid for. Returns None if no sample call succeeded.
    """
    judged_document_ids = {r["doc_id"] for r in sample_records}
    judged_documents = [d for d in sample_documents if d["doc_id"] in judged_document_ids]
    if not judged_documents:
        return None
    sample_ids = {d["doc_id"] for d in sample_documents}
    all_attempt_records = list(sample_records) + [e for e in error_records if e["doc_id"] in sample_ids]
    sample_characters = sum(len(d["content"]) for d in judged_documents)
    corpus_characters = sum(len(d["content"]) for d in all_documents)
    character_scale = corpus_characters / sample_characters
    document_scale = len(all_documents) / len(judged_documents)
    uncached = sum(r["usage"]["input_tokens"] - r["usage"]["cached_input_tokens"] for r in all_attempt_records)
    cached = sum(r["usage"]["cached_input_tokens"] for r in all_attempt_records)
    output = sum(r["usage"]["output_tokens"] for r in all_attempt_records)
    total_uncached = uncached * character_scale
    total_cached = cached * document_scale
    total_output = output * document_scale
    total_seconds = sum(r["latency_seconds"] or 0 for r in all_attempt_records) * document_scale
    return {"character_scale": character_scale, "document_scale": document_scale,
            "uncached_input_tokens": total_uncached, "cached_input_tokens": total_cached,
            "output_tokens": total_output, "total_call_seconds_across_all_calls": total_seconds,
            "total_cost_dollars": compute_cost_in_dollars(
                MODEL_NAME, total_uncached + total_cached, total_cached, total_output),
            "note": ("Sample uncached input includes the instruction prefix on cache-missing calls, "
                     "so uncached input (and cost) is likely overestimated; scaled by corpus chars / "
                     "sample chars. Cached input and output scaled by document count.")}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="SDF run directory or sdf_corpus.jsonl path")
    parser.add_argument("--run-name", required=True, help="Descriptive name for this run")
    parser.add_argument("--runs-directory", default=None,
                        help="Where run folders go (default: <run>/audit/plausibility_audit_runs)")
    parser.add_argument("--resume-folder", default=None, help="Existing run folder to resume")
    parser.add_argument("--document-limit", type=int, default=None, help="Use only the first N documents")
    parser.add_argument("--document-ids", nargs="+", default=None, help="Use only these doc_ids")
    parser.add_argument("--reasoning-effort", default="medium",
                        choices=["minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--estimate-sample-size", type=int, default=6)
    add_api_key_source_flags(parser)
    return parser.parse_args(argv)


def default_runs_directory(input_argument):
    run_directory = run_directory_of(input_argument)
    audit_folder = (run_directory if run_directory is not None else Path(input_argument).parent) / "audit"
    return audit_folder / "plausibility_audit_runs"


def main(argv=None):
    arguments = parse_arguments(argv)
    apply_api_key_source_flags(arguments)
    corpus_path = corpus_path_of(arguments.input)
    all_documents = read_jsonl(corpus_path)
    documents = select_documents(all_documents, arguments.document_ids, arguments.document_limit, corpus_path)

    runs_directory = Path(arguments.runs_directory) if arguments.runs_directory else default_runs_directory(
        arguments.input)
    run_kind, runs_parent_directory = choose_runs_parent_directory(arguments, runs_directory)
    run_folder, is_resume = choose_run_folder(runs_parent_directory, arguments.run_name, arguments.resume_folder)
    print(f"Run kind: {run_kind}{' (resuming)' if is_resume else ''}; run folder: {run_folder}")
    timestamp = timestamp_now()
    configuration = build_run_configuration(arguments.reasoning_effort, corpus_path)
    if is_resume:
        check_resume_configuration_matches(run_folder, configuration)
    write_run_configuration(run_folder, timestamp, configuration)

    judge_definition = JudgeDefinition()
    if len(judge_definition.axis_keys) != EXPECTED_AXIS_COUNT:
        sys.exit(f"error: expected {EXPECTED_AXIS_COUNT} axes in the prompt, found {len(judge_definition.axis_keys)}")
    (run_folder / f"instructions_sent_{timestamp}.txt").write_text(
        judge_definition.instructions_text, encoding="utf-8")

    already_completed_document_ids = find_completed_document_ids(run_folder)
    prior_records = load_existing_records(run_folder) + read_all_records(run_folder, "errors_*.jsonl")
    api.init_cost_log_only(run_folder / f"model_call_cost_log_{timestamp}.jsonl")

    if arguments.estimate_only:
        documents = choose_stratified_sample(documents, arguments.estimate_sample_size)
        print("Estimate sample: " + ", ".join(f"{d['doc_id']} ({d['language']}, {len(d['content'])} chars)"
                                              for d in documents), flush=True)

    run_judging(run_folder, timestamp, judge_definition, documents, arguments.reasoning_effort,
                arguments.workers, already_completed_document_ids, prior_records)

    if arguments.estimate_only:
        sample_ids = {d["doc_id"] for d in documents}
        sample_records = [r for r in load_existing_records(run_folder) if r["doc_id"] in sample_ids]
        error_records = read_all_records(run_folder, "errors_*.jsonl")
        estimate = extrapolate_estimate(sample_records, documents, all_documents, error_records)
        if estimate is None:
            print("ESTIMATE full corpus: no estimate is available, every sample call failed "
                  f"(see the error records in {run_folder})")
            print(f"Run folder: {run_folder}")
            return run_folder
        estimate["sample_doc_ids"] = sorted(sample_ids)
        estimate["corpus_document_count"] = len(all_documents)
        estimate["workers_assumed_for_wall_clock"] = arguments.workers
        estimate["estimated_wall_clock_seconds"] = (
            estimate["total_call_seconds_across_all_calls"] / arguments.workers)
        (run_folder / f"estimate_{timestamp}.json").write_text(json.dumps(estimate, indent=2), encoding="utf-8")
        print(f"ESTIMATE full corpus: ${estimate['total_cost_dollars']:.2f}, about "
              f"{format_duration(estimate['estimated_wall_clock_seconds'])} with {arguments.workers} workers")
    print(f"Run folder: {run_folder}")
    return run_folder


if __name__ == "__main__":
    main()
