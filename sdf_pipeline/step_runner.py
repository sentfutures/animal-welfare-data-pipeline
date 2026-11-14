"""``StepRunner``: the work every SDF layer repeats, for a ``DocumentStep``.

Pick the pending documents, fan out over a thread pool, call the model, append an event as each
reply arrives (under the event log's lock), then on the calling thread, in input order, write the
step's stage file, mark the checkpoint, print progress and apply the step's stop rules. Workers
only build the request, call the model and interpret the reply; every stage file write and
checkpoint mark stays on the main thread, in input order, exactly as the layers did before.

A document is pending when it applies to the step, has no record in the stage file and is not
marked done in the checkpoint. ``Failed`` marks nothing done, so a resume retries it. ``Removed``
marks nothing done unless the step supplies a record through ``stage_file_record_for_removal``.
"""

import threading
import time
from dataclasses import dataclass, field

from shared import api, openai_api, utils
from sdf_pipeline.document_event_log import (
    DOCUMENT_EVENT_LOG_FILE_NAME, DocumentEventLog, fold_document_events, read_document_events)
from sdf_pipeline.update_steps import (
    Accepted, BatchStep, CorpusStep, DocumentStep, Failed, FailureKind, ModelReply, ModelRequest, Removed,
    WhatAFailedDocumentDoes)


@dataclass
class StepRunResult:
    """``records_in_input_order``: the stage-file records of every input document finished before this
    run, whether or not the step applies to it now (in input order), followed by the ones written in
    this run (in input order)."""
    records_in_input_order: list = field(default_factory=list)
    failed_document_ids: list = field(default_factory=list)
    pending_document_ids: list = field(default_factory=list)


def send_model_request(request: ModelRequest) -> ModelReply:
    """Send the request through ``shared.api.call_claude`` with exactly the keyword arguments a
    layer passes today: settings left at their default are not passed."""
    if request.openai_schema_name is not None:
        # The description rewrite's OpenAI route (a ``gpt-`` model), with the arguments it always passed.
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
    """The settings ``send_model_request`` will send ``request`` with, by the route it takes.

    A request without ``openai_schema_name`` goes through ``shared.api.call_claude``, so its settings
    are ``api.resolve_request_settings`` (the very call ``call_claude`` makes). The description rewrite's
    OpenAI route calls ``openai_api.call_openai`` directly with the request's own max tokens and
    reasoning effort, so no thinking headroom is added and its settings are those values as they are
    (the temperature is the OpenAI wrapper's constant). Read-only; ``config`` defaults to the loaded one."""
    if request.openai_schema_name is None:
        return api.resolve_request_settings(
            request.model, request.stage, request.max_tokens, request.temperature,
            json_schema=request.json_schema, cache_system=request.cache_system, config=config)
    return api.ResolvedRequestSettings(
        provider="openai", model=request.model, max_tokens=request.max_tokens, thinking_headroom_added=0,
        requested_temperature=openai_api.SAMPLING_TEMPERATURE, temperature=openai_api.SAMPLING_TEMPERATURE,
        effort=request.openai_reasoning_effort, effort_configured=request.openai_reasoning_effort,
        thinking=None, structured_output=request.json_schema is not None, cache_system=False)


def outcome_name_and_event_detail(outcome, update_as_recorded=lambda update: update) -> tuple[str, dict]:
    """The ``outcome`` name and the detail fields of the event an accepted, removed or failed outcome
    is recorded as. ``update_as_recorded`` turns an accepted update into what the event keeps (a
    document step records the stage file record; a batch or corpus step records the update itself)."""
    if isinstance(outcome, Accepted):
        return "accepted", {"update": update_as_recorded(outcome.update)}
    if isinstance(outcome, Removed):
        return "removed", {"reason_code": outcome.reason_code, "detail": outcome.detail}
    return "failed", {"failure_kind": outcome.kind.value, "reason": outcome.reason}


class StepRunner:
    """Runs a ``DocumentStep`` over a stage directory. The event log is the run folder's
    ``document_events.jsonl``, beside the layer folders, unless ``event_log`` is given.
    ``existing_records_by_document_id`` defaults to the stage file's records keyed by the step's
    ``stage_file_id_field``. The checkpoint defaults to the stage directory's ``_checkpoint.json``;
    the options for a step that does things differently:

    * ``uses_checkpoint=False``: no checkpoint is read or written (the stage file is the only record);
    * ``marks_nothing_done=True``: the checkpoint is read but an outcome never marks a document done
      (only a later step finishes the document);
    * ``also_finished_document_ids``: documents treated as finished, so they are not pending.
    """

    def __init__(self, step: DocumentStep, stage_directory, event_log: DocumentEventLog | None = None,
                 workers: int = 1, send=send_model_request, existing_records_by_document_id: dict | None = None,
                 checkpoint: utils.Checkpoint | None = None, uses_checkpoint: bool = True,
                 marks_nothing_done: bool = False, also_finished_document_ids=frozenset()):
        self.step = step
        self._write_lock = threading.Lock()
        self._removal_record_by_document_id: dict = {}
        self.stage_directory = stage_directory
        self.event_log = (
            event_log if event_log is not None
            else DocumentEventLog(stage_directory.parent / DOCUMENT_EVENT_LOG_FILE_NAME))
        self.workers = workers
        self.send = send
        self.stage_file_path = stage_directory / step.stage_file_name
        self.checkpoint = (
            None if not uses_checkpoint
            else checkpoint if checkpoint is not None else utils.Checkpoint(stage_directory / "_checkpoint.json"))
        self.marks_nothing_done = marks_nothing_done
        self.also_finished_document_ids = also_finished_document_ids
        self.existing_records_by_document_id = (
            existing_records_by_document_id if existing_records_by_document_id is not None
            else {record[step.stage_file_id_field]: record for record in utils.load_jsonl(self.stage_file_path)})

    def _is_finished_by_checkpoint(self, document_id) -> bool:
        return document_id in self.also_finished_document_ids or (
            self.checkpoint is not None and self.checkpoint.is_done(document_id))

    def _mark_done(self, document_id) -> None:
        if self.checkpoint is not None and not self.marks_nothing_done:
            self.checkpoint.mark_done(document_id)

    def select_pending(self, documents: list) -> list:
        step = self.step
        return [
            document for document in documents
            if step.applies_to(document)
            and step.document_id_of(document) not in self.existing_records_by_document_id
            and not self._is_finished_by_checkpoint(step.document_id_of(document))
        ]

    def _answer_one(self, document):
        """Worker: send (up to the policy's attempts), interpret and record an event per attempt;
        returns the final outcome. An unusable answer is sent again, the same request, up to
        ``maximum_attempts_in_one_run``; a thrown call and any other outcome end the attempts. A
        step that writes as replies arrive has its files written here, under a lock."""
        outcome = self._attempt_loop(document)
        if self.step.writes_stage_file_as_replies_arrive:
            with self._write_lock:
                self._write_outcome_files(document, outcome)
                line = self.step.progress_line(document, outcome)
                if line is not None:
                    print(line)
        return outcome

    def _attempt_loop(self, document):
        step = self.step
        try:
            request = step.build_request(document)
        except Exception as error:
            if not step.build_request_errors_count_as_thrown_calls:
                raise
            outcome = Failed(FailureKind.THROWN_CALL, f"error: {type(error).__name__}: {error}")
            self._append_event(document, None, None, outcome)
            return outcome
        request_hash = request.request_hash()
        maximum_attempts = step.policy.maximum_attempts_in_one_run
        if maximum_attempts < 1:
            return step.failure_when_no_attempt_is_allowed(document)
        for attempt in range(1, maximum_attempts + 1):
            try:
                reply = self.send(request)
            except Exception as error:
                # One poison document must not kill the step and discard its siblings' in-flight work.
                outcome = Failed(FailureKind.THROWN_CALL, f"error: {type(error).__name__}: {error}")
            else:
                outcome = step.interpret(document, reply)
            self._append_event(document, request, request_hash, outcome)
            if not (isinstance(outcome, Failed) and outcome.kind is FailureKind.UNUSABLE_ANSWER):
                break
        return outcome

    def _removal_record(self, document, removed):
        """The step's record for a removed document, asked for once (the step may have side effects)."""
        document_id = self.step.document_id_of(document)
        if document_id not in self._removal_record_by_document_id:
            self._removal_record_by_document_id[document_id] = self.step.stage_file_record_for_removal(document, removed)
        return self._removal_record_by_document_id[document_id]

    def _write_outcome_files(self, document, outcome):
        """The stage file record and checkpoint mark of an accepted (or removed-with-record)
        outcome, and the failure record of a failed one."""
        step = self.step
        document_id = step.document_id_of(document)
        if isinstance(outcome, Failed):
            failure_record = step.failure_record(document, outcome)
            if failure_record is not None:
                utils.append_jsonl(failure_record, self.stage_directory / step.failure_file_name)
        elif isinstance(outcome, Removed):
            record = self._removal_record(document, outcome)
            if record is not None:
                utils.append_jsonl(record, self.stage_file_path)
                self._mark_done(document_id)
        elif isinstance(outcome, Accepted):
            utils.append_jsonl(step.stage_file_record(document, outcome.update), self.stage_file_path)
            self._mark_done(document_id)

    def _append_event(self, document, request, request_hash, outcome):
        step = self.step
        outcome_name, detail = outcome_name_and_event_detail(
            outcome, lambda update: step.stage_file_record(document, update))
        self.event_log.append_outcome_event(
            document_id=step.document_id_of(document), step_name=step.step_name, outcome=outcome_name,
            request_hash=request_hash, stage=request.stage if request is not None else None, detail=detail)

    def run(self, documents: list) -> StepRunResult:
        step = self.step
        policy = step.policy
        pending = self.select_pending(documents)
        result = StepRunResult(
            records_in_input_order=[
                self.existing_records_by_document_id[step.document_id_of(d)] for d in documents
                if step.document_id_of(d) in self.existing_records_by_document_id],
            pending_document_ids=[step.document_id_of(d) for d in pending])
        counted_failures = 0
        writes_as_replies_arrive = step.writes_stage_file_as_replies_arrive
        for document, outcome in zip(pending, utils.parallel_map(self._answer_one, pending, self.workers)):
            document_id = step.document_id_of(document)
            if isinstance(outcome, Failed):
                result.failed_document_ids.append(document_id)
                if outcome.kind in policy.failure_kinds_counting_towards_the_systemic_stop:
                    counted_failures += 1
            elif isinstance(outcome, Removed):
                record = self._removal_record(document, outcome)
                if record is not None:
                    result.records_in_input_order.append(record)
            elif isinstance(outcome, Accepted):
                result.records_in_input_order.append(step.stage_file_record(document, outcome.update))
            if not writes_as_replies_arrive:  # otherwise the worker wrote its files and printed its line
                self._write_outcome_files(document, outcome)
                line = step.progress_line(document, outcome)
                if line is not None:
                    print(line)
        if pending and counted_failures == len(pending):
            raise SystemExit(step.systemic_stop_message)
        step.after_documents_are_processed(result)
        if result.failed_document_ids and policy.failed_document is WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP:
            raise SystemExit(step.message_when_failed_documents_stop_the_run(result))
        return result


def run_corpus_step(step: CorpusStep, documents: list, event_log: DocumentEventLog) -> dict:
    """Apply a corpus step (no model call) and record its outcomes: {document id: StepOutcome}.

    The step runs every time (it is deterministic and cheap). An event is appended for a document
    only when its outcome differs from the latest recorded one for this step, so a resume that
    changes nothing appends nothing and a new threshold appends exactly the changed outcomes.
    """
    outcomes = step.apply(documents)
    folded = fold_document_events(read_document_events(event_log.path))
    for document_id, outcome in outcomes.items():
        if not isinstance(outcome, (Accepted, Removed)):
            raise TypeError(f"{step.step_name}: a corpus step returned {type(outcome).__name__} for {document_id}")
        outcome_name, detail = outcome_name_and_event_detail(outcome)
        latest = (outcome_name, detail["update"] if isinstance(outcome, Accepted) else detail)
        recorded = folded.get(document_id)
        if recorded is not None and recorded.latest_outcome_by_step.get(step.step_name) == latest:
            continue
        event_log.append_outcome_event(
            document_id=document_id, step_name=step.step_name, outcome=outcome_name,
            request_hash=None, stage=None, detail=detail)
    return outcomes


class BatchStepRunner:
    """Run a ``BatchStep``: one model call per batch, fanned out over the thread pool.

    A batch object has ``batch_id`` and ``document_ids``. A batch is pending when the checkpoint
    does not mark its ``batch_id`` done. Workers build the request, send it, interpret the reply
    and append one event per member as the reply arrives; the main thread then, in batch order,
    hands each finished batch's outcomes to ``write_batch_outcomes`` (the layer's stage-file
    writes), marks the checkpoint and prints the step's line. A batch whose members came back
    ``Failed`` is not written or checkpointed, so a resume retries exactly it.

    Stop rules, from the step's policy: when every pending batch failed with a kind the policy
    counts, the run stops with ``step.systemic_stop_message``; otherwise, when the policy says a
    failed batch stops the run after the step, it stops with
    ``step.stop_message_for_failed_batches(failed_batch_count, failed_batches)``.
    """

    def __init__(self, step: BatchStep, event_log: DocumentEventLog, checkpoint: utils.Checkpoint,
                 write_batch_outcomes, workers: int = 1, send=send_model_request):
        self.step = step
        self.event_log = event_log
        self.checkpoint = checkpoint
        self.write_batch_outcomes = write_batch_outcomes
        self.workers = workers
        self.send = send

    def select_pending(self, batches: list) -> list:
        return [batch for batch in batches if not self.checkpoint.is_done(batch.batch_id)]

    def _answer_one(self, batch):
        """Worker: build, send, interpret, record one event per member. Returns (outcomes, seconds)."""
        step = self.step
        request_hash = None
        seconds = 0.0
        try:
            request = step.build_request(batch)
            request_hash = request.request_hash()
            started = time.monotonic()
            reply = self.send(request)
            outcomes = step.interpret(batch, reply)
            seconds = time.monotonic() - started
        except Exception as error:
            # One failing call must not discard its siblings' in-flight work.
            reason = f"error: {type(error).__name__}: {error}"
            outcomes = {document_id: Failed(FailureKind.THROWN_CALL, reason) for document_id in batch.document_ids}
        for document_id, outcome in outcomes.items():
            self._append_event(document_id, request_hash, outcome)
        return outcomes, seconds

    def _append_event(self, document_id, request_hash, outcome):
        outcome_name, detail = outcome_name_and_event_detail(outcome)
        self.event_log.append_outcome_event(
            document_id=document_id, step_name=self.step.step_name, outcome=outcome_name,
            request_hash=request_hash, stage=self.step.stage, detail=detail)

    def run(self, batches: list) -> None:
        step = self.step
        pending = self.select_pending(batches)
        counted_failures = 0
        failed_batches = []
        for done, (batch, (outcomes, seconds)) in enumerate(
                zip(pending, utils.parallel_map(self._answer_one, pending, self.workers)), 1):
            failures = [outcome for outcome in outcomes.values() if isinstance(outcome, Failed)]
            if failures:
                failed_batches.append(batch)
                if failures[0].kind in step.policy.failure_kinds_counting_towards_the_systemic_stop:
                    counted_failures += 1
                print(step.failed_batch_line(batch, failures[0]))
                continue
            self.write_batch_outcomes(batch, outcomes)
            self.checkpoint.mark_done(batch.batch_id)
            print(step.progress_line(batch, outcomes, done, len(pending), seconds))
        if pending and counted_failures == len(pending):
            raise SystemExit(step.systemic_stop_message)
        if failed_batches and step.policy.failed_document is WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP:
            raise SystemExit(step.stop_message_for_failed_batches(len(failed_batches), failed_batches))
