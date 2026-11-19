"""The runner: the work every SDF model-calling step repeats (pick pending groups, call the model on a thread pool,
write stage files and print progress on the main thread in input order). The runner is the only thing that writes
files, counts progress and counts attempts; a step only builds requests and reads replies."""

import threading
import time
from dataclasses import dataclass, field, replace

from shared import api, openai_api, utils
from sdf_pipeline.compatibility_with_runs_made_by_earlier_code.checkpoint_marks import keep_marks_up_to_date, marked_as_finished
from sdf_pipeline.shared_step_machinery.step_record import Run, Step, document_id_of
from sdf_pipeline.shared_step_machinery.update_steps import (
    Failed, FailureKind, ModelReply, ModelRequest, OutcomeRecords, StepProgress, WhatAFailedDocumentDoes,
    WhereAnErrorIsAThrownCall)


@dataclass
class StepRunResult:
    """Stage-file records of the finished input documents, then those written this run (empty for a grouped step), and the groups that failed."""
    records_in_input_order: list = field(default_factory=list)
    failed_groups: list = field(default_factory=list)
    failed_document_ids: list = field(default_factory=list)
    thrown_call_document_ids: list = field(default_factory=list)  # failed with a thrown call (any place the policy counts)
    pending_document_ids: list = field(default_factory=list)
    progress: StepProgress = field(default_factory=StepProgress)  # the counts and clock of this run


def send_model_request(request: ModelRequest) -> ModelReply:
    """Send the request through ``shared.api.call_claude`` with exactly the keyword arguments the
    step sets: settings left at their default are not passed."""
    if request.openai_schema_name is not None:
        # The description rewrite's OpenAI route (a ``gpt-`` model).
        text, stop_reason = openai_api.call_openai(
            request.user_message,
            system_prompt=request.system_prompt,
            model=request.model,
            max_output_tokens=request.max_tokens,
            reasoning_effort=request.openai_reasoning_effort,
            stage=request.stage,
            item_id=request.item_id,
            json_schema=request.json_schema,
            schema_name=request.openai_schema_name,
        )
        return ModelReply(text=text, stop_reason=stop_reason)
    keyword_arguments = dict(
        system_prompt=request.system_prompt,
        model=request.model,
        stage=request.stage,
        item_id=request.item_id,
        return_stop_reason=True,
    )
    if request.max_tokens is not None:
        keyword_arguments["max_tokens"] = request.max_tokens
    if request.cache_system:
        keyword_arguments["cache_system"] = True
    if request.temperature is not None:
        keyword_arguments["temperature"] = request.temperature
    if request.json_schema is not None:
        keyword_arguments["json_schema"] = request.json_schema
    text, stop_reason = api.call_claude(request.user_message, **keyword_arguments)
    return ModelReply(text=text, stop_reason=stop_reason)


def resolve_settings_of_model_request(request: ModelRequest, config: dict | None = None) -> api.ResolvedRequestSettings:
    """The settings ``send_model_request`` will send ``request`` with (read-only).
    The OpenAI route sends the request's own max tokens and reasoning effort, with no thinking headroom."""
    if request.openai_schema_name is None:
        return api.resolve_request_settings(
            request.model, request.stage, request.max_tokens, request.temperature,
            json_schema=request.json_schema, cache_system=request.cache_system, config=config)
    return api.ResolvedRequestSettings(
        provider="openai", model=request.model, max_tokens=request.max_tokens, thinking_headroom_added=0,
        requested_temperature=openai_api.SAMPLING_TEMPERATURE, temperature=openai_api.SAMPLING_TEMPERATURE,
        effort=request.openai_reasoning_effort, effort_configured=request.openai_reasoning_effort,
        thinking=None, structured_output=request.json_schema is not None, cache_system=False)


def failure_for_exception(error: Exception) -> Failed:
    """The ``Failed`` a step's declared exception policy turns an exception into."""
    return Failed(FailureKind.THROWN_CALL, f"error: {type(error).__name__}: {error}")





def maximum_attempts(step: Step, run: Run) -> int:
    """How many times the step may send the same request in one run (the policy's, or the configured setting)."""
    policy = step.policy
    attempts = policy.maximum_attempts_in_one_run
    if policy.maximum_attempts_setting:
        attempts = int(run.config["sdf"].get(policy.maximum_attempts_setting, attempts))
    if attempts < 1:
        raise ValueError(f"maximum_attempts_in_one_run is {attempts}; it must be at least 1")
    return attempts


def _records_of_finished_documents(step: Step, run: Run) -> dict:
    """{document id: record} of the step's stage file (empty for a grouped step)."""
    if step.group is not None:
        return {}
    return {document_id_of(record): record for record in run.records(step.writes[0])}


def find_pending(step: Step, run: Run, documents: list) -> tuple[dict, dict]:
    """({group key: group} for every group the step would run on ``documents``, the same for the groups still
    pending). A group is one document that applies to the step, unless the step groups."""
    groups = step.group(documents, run) if step.group else {
        document_id_of(document): [document] for document in documents
        if step.applies(document, run) and not step.finished_elsewhere(document, run)}
    marks = marked_as_finished(step.name, run.stage_directories[step.folder_key])
    if step.finished_group_keys:
        finished = step.finished_group_keys(run) if marks is None else marks
    else:
        finished = set(_records_of_finished_documents(step, run)) | (marks or set())
    pending = {key: group for key, group in groups.items() if key not in finished}
    return groups, pending


def _read_group(step: Step, run: Run, send, documents: list) -> tuple[dict, float]:
    """Worker: build the request, send it and read the reply; an unusable answer is resent up to the policy's attempts."""
    counted_places = step.policy.errors_counted_as_thrown_calls

    def every_document_failed(error: Exception) -> dict:
        failure = failure_for_exception(error)
        return {document_id_of(document): failure for document in documents}

    try:
        request = step.build_request(documents, run)
    except Exception as error:
        if WhereAnErrorIsAThrownCall.BUILD not in counted_places:
            raise
        return every_document_failed(error), 0.0
    started = time.monotonic()
    for attempt in range(1, maximum_attempts(step, run) + 1):
        try:
            reply = replace(send(request), attempt=attempt)
        except Exception as error:
            # One poison document must not kill the step and discard its siblings' in-flight work.
            if WhereAnErrorIsAThrownCall.SEND not in counted_places:
                raise
            outcomes = every_document_failed(error)
        else:
            try:
                outcomes = step.read_reply(documents, reply, run)
            except Exception as error:
                if WhereAnErrorIsAThrownCall.INTERPRET not in counted_places:
                    raise
                outcomes = every_document_failed(error)
        if not any(isinstance(outcome, Failed) and outcome.kind is FailureKind.UNUSABLE_ANSWER for outcome in outcomes.values()):
            break
    return outcomes, time.monotonic() - started


def _write_outcomes_and_print_line(
        step, run, stage_directory, progress, documents, outcomes, seconds) -> OutcomeRecords:
    """Write the group's records (dropping the earlier ones it replaces), count the group in the run's progress and
    print the step's line."""
    outcome_records = step.records(documents, outcomes, run)
    for file_name, field_name, value in outcome_records.replacing:
        earlier = utils.load_jsonl(stage_directory / file_name)
        kept = [record for record in earlier if record[field_name] != value]
        if len(kept) != len(earlier):
            utils.save_jsonl(kept, stage_directory / file_name)
    for file_name, record in outcome_records.records:
        utils.append_jsonl(record, stage_directory / file_name)
    if step.finished_group_keys:
        keep_marks_up_to_date(step.name, stage_directory, lambda: step.finished_group_keys(run))
    progress.record(outcomes.values(), seconds)
    line = step.progress_line(documents, outcomes, progress, run) if step.progress_line else None
    if line is not None:
        print(line)
    return outcome_records


def run_step(step: Step, run: Run, documents: list, send=send_model_request) -> StepRunResult:
    """Run a model-calling step over ``documents`` (see the module docstring)."""
    maximum_attempts(step, run)   # refuses a step that may send no request, before any call
    stage_directory = run.stage_directories[step.folder_key]
    finished = _records_of_finished_documents(step, run)
    _, pending = find_pending(step, run, documents)
    progress = StepProgress(pending_count=len(pending))
    result = StepRunResult(
        records_in_input_order=[finished[document_id_of(d)] for d in documents if document_id_of(d) in finished],
        pending_document_ids=[document_id_of(document) for group in pending.values() for document in group],
        progress=progress)
    write_lock = threading.Lock()

    def answer(keyed_group):
        group = keyed_group[1]
        outcomes, seconds = _read_group(step, run, send, group)
        if step.writes_as_replies_arrive:
            with write_lock:
                return outcomes, seconds, _write_outcomes_and_print_line(
                    step, run, stage_directory, progress, group, outcomes, seconds)
        return outcomes, seconds, None

    counted_failures = 0
    pending_groups = list(pending.items())
    for (_, group), (outcomes, seconds, outcome_records) in zip(
            pending_groups, utils.parallel_map(answer, pending_groups, run.workers)):
        if outcome_records is None:   # otherwise the worker wrote its files and printed its line
            outcome_records = _write_outcomes_and_print_line(
                step, run, stage_directory, progress, group, outcomes, seconds)
        failures = {document_id: outcome for document_id, outcome in outcomes.items() if isinstance(outcome, Failed)}
        if failures:
            result.failed_groups.append(group)
            result.failed_document_ids.extend(failures)
            result.thrown_call_document_ids.extend(
                document_id for document_id, failure in failures.items() if failure.kind is FailureKind.THROWN_CALL)
            if next(iter(failures.values())).kind in step.policy.failure_kinds_counting_towards_the_systemic_stop:
                counted_failures += 1
        if step.group is None:
            result.records_in_input_order.extend(
                record for file_name, record in outcome_records.records if file_name == step.writes[0].file_name)
    if pending and counted_failures == len(pending):
        raise SystemExit(step.systemic_stop_message)
    if step.after_run:
        step.after_run(documents, result, run)
    if result.failed_document_ids and step.policy.failed_document is WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP:
        raise SystemExit(step.stop_message(documents, result, run) if step.stop_message else
                         f"{step.name}: {len(result.failed_document_ids)} document(s) failed; fix and --resume.")
    return result
