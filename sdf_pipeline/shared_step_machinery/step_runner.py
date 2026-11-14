"""``StepRunner``: the work every SDF layer repeats, for a ``DocumentStep``.

Pick the pending documents, fan out over a thread pool, call the model, then on the calling
thread, in input order, write the step's stage file, mark the checkpoint, print progress and apply
the step's stop rules. Workers only build the request, call the model and interpret the reply;
every stage file write and checkpoint mark stays on the main thread, in input order, exactly as the
layers did before.

A document is pending when it applies to the step, has no record in the stage file and is not
marked done in the checkpoint. ``Failed`` marks nothing done, so a resume retries it. ``Removed``
marks nothing done unless the step's ``outcome_records`` says it finishes the document.

The runner is the only thing that writes files, counts progress (``StepProgress``) and counts attempts
(``ModelReply.attempt``); a step builds requests and interprets replies.
"""

import threading
import time
from dataclasses import dataclass, field, replace

from shared import api, openai_api, utils
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, BatchStep, CorpusStep, DocumentStep, Failed, FailureKind, ModelReply, ModelRequest, Removed,
    StepProgress, WhatAFailedDocumentDoes, WhereAnErrorIsAThrownCall)


@dataclass
class StepRunResult:
    """``records_in_input_order``: the stage-file records of every input document finished before this
    run, whether or not the step applies to it now (in input order), followed by the ones written in
    this run (in input order)."""
    records_in_input_order: list = field(default_factory=list)
    failed_document_ids: list = field(default_factory=list)
    thrown_call_document_ids: list = field(default_factory=list)  # failed with a thrown call (any place the policy counts)
    pending_document_ids: list = field(default_factory=list)
    progress: StepProgress = field(default_factory=StepProgress)  # the counts and clock of this run


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


def failure_for_exception(error: Exception) -> Failed:
    """The ``Failed`` a step's declared exception policy turns an exception into."""
    return Failed(FailureKind.THROWN_CALL, f"error: {type(error).__name__}: {error}")


class StepRunner:
    """Runs a ``DocumentStep`` over a stage directory.
    ``existing_records_by_document_id`` defaults to the stage file's records keyed by the step's
    ``stage_file_id_field``. The checkpoint defaults to the stage directory's ``_checkpoint.json``;
    the options for a step that does things differently:

    * ``uses_checkpoint=False``: no checkpoint is read or written (the stage file is the only record);
    * ``marks_nothing_done=True``: the checkpoint is read but an outcome never marks a document done
      (only a later step finishes the document);
    * ``also_finished_document_ids``: documents treated as finished, so they are not pending.
    """

    def __init__(self, step: DocumentStep, stage_directory,
                 workers: int = 1, send=send_model_request, existing_records_by_document_id: dict | None = None,
                 checkpoint: utils.Checkpoint | None = None, uses_checkpoint: bool = True,
                 marks_nothing_done: bool = False, also_finished_document_ids=frozenset()):
        self.step = step
        self._write_lock = threading.Lock()
        self.progress = StepProgress()
        self.stage_directory = stage_directory
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
        """Worker: send (up to the policy's attempts) and interpret; returns the final outcome. An unusable answer is sent again, the same request, up to
        ``maximum_attempts_in_one_run``; a thrown call and any other outcome end the attempts. A
        step that writes as replies arrive has its files written here, under a lock."""
        outcome = self._attempt_loop(document)
        if self.step.writes_stage_file_as_replies_arrive:
            with self._write_lock:
                self._write_outcome_files_and_print_line(document, outcome)
        return outcome

    def _attempt_loop(self, document):
        step = self.step
        counted_places = step.policy.errors_counted_as_thrown_calls
        try:
            request = step.build_request(document)
        except Exception as error:
            if WhereAnErrorIsAThrownCall.BUILD not in counted_places:
                raise
            return failure_for_exception(error)
        for attempt in range(1, step.policy.maximum_attempts_in_one_run + 1):
            try:
                reply = replace(self.send(request), attempt=attempt)
            except Exception as error:
                # One poison document must not kill the step and discard its siblings' in-flight work.
                if WhereAnErrorIsAThrownCall.SEND not in counted_places:
                    raise
                outcome = failure_for_exception(error)
            else:
                try:
                    outcome = step.interpret(document, reply)
                except Exception as error:
                    if WhereAnErrorIsAThrownCall.INTERPRET not in counted_places:
                        raise
                    outcome = failure_for_exception(error)
            if not (isinstance(outcome, Failed) and outcome.kind is FailureKind.UNUSABLE_ANSWER):
                break
        return outcome

    def _write_outcome_files(self, document, outcome):
        """The records the step says the outcome produces, in its order, then the checkpoint mark when
        it finishes the document."""
        outcome_records = self.step.outcome_records(document, outcome)
        for file_name, record in outcome_records.records:
            utils.append_jsonl(record, self.stage_directory / file_name)
        if outcome_records.finishes_the_document:
            self._mark_done(self.step.document_id_of(document))

    def _write_outcome_files_and_print_line(self, document, outcome):
        """Write the outcome's files, count it in the run's progress and print the step's line."""
        self._write_outcome_files(document, outcome)
        self.progress.record(outcome)
        line = self.step.progress_line(document, outcome, self.progress)
        if line is not None:
            print(line)

    def run(self, documents: list) -> StepRunResult:
        step = self.step
        policy = step.policy
        pending = self.select_pending(documents)
        result = StepRunResult(
            records_in_input_order=[
                self.existing_records_by_document_id[step.document_id_of(d)] for d in documents
                if step.document_id_of(d) in self.existing_records_by_document_id],
            pending_document_ids=[step.document_id_of(d) for d in pending],
            progress=StepProgress(pending_count=len(pending)))
        self.progress = result.progress
        counted_failures = 0
        writes_as_replies_arrive = step.writes_stage_file_as_replies_arrive
        for document, outcome in zip(pending, utils.parallel_map(self._answer_one, pending, self.workers)):
            document_id = step.document_id_of(document)
            if isinstance(outcome, Failed):
                result.failed_document_ids.append(document_id)
                if outcome.kind is FailureKind.THROWN_CALL:
                    result.thrown_call_document_ids.append(document_id)
                if outcome.kind in policy.failure_kinds_counting_towards_the_systemic_stop:
                    counted_failures += 1
            elif isinstance(outcome, (Removed, Accepted)):
                result.records_in_input_order.extend(
                    record for file_name, record in step.outcome_records(document, outcome).records
                    if file_name == step.stage_file_name)
            if not writes_as_replies_arrive:  # otherwise the worker wrote its files and printed its line
                self._write_outcome_files_and_print_line(document, outcome)
        if pending and counted_failures == len(pending):
            raise SystemExit(step.systemic_stop_message)
        step.after_documents_are_processed(result)
        if result.failed_document_ids and policy.failed_document is WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP:
            raise SystemExit(step.message_when_failed_documents_stop_the_run(result))
        return result


def run_corpus_step(step: CorpusStep, documents: list) -> dict:
    """Apply a corpus step (no model call): {document id: StepOutcome}. The step runs every time
    (it is deterministic and cheap); every outcome must be accepted or removed."""
    outcomes = step.apply(documents)
    for document_id, outcome in outcomes.items():
        if not isinstance(outcome, (Accepted, Removed)):
            raise TypeError(f"{step.step_name}: a corpus step returned {type(outcome).__name__} for {document_id}")
    return outcomes


class BatchStepRunner:
    """Run a ``BatchStep``: one model call per batch, fanned out over the thread pool.

    A batch object has ``batch_id`` and ``document_ids``. A batch is pending when the checkpoint
    does not mark its ``batch_id`` done. Workers build the request, send it, interpret the reply
    ; the main thread then, in batch order,
    hands each finished batch's outcomes to ``write_batch_outcomes`` (the layer's stage-file
    writes), marks the checkpoint and prints the step's line. A batch whose members came back
    ``Failed`` is not written or checkpointed, so a resume retries exactly it.

    Stop rules, from the step's policy: when every pending batch failed with a kind the policy
    counts, the run stops with ``step.systemic_stop_message``; otherwise, when the policy says a
    failed batch stops the run after the step, it stops with
    ``step.stop_message_for_failed_batches(failed_batch_count, failed_batches)``.
    """

    def __init__(self, step: BatchStep, checkpoint: utils.Checkpoint,
                 write_batch_outcomes, workers: int = 1, send=send_model_request):
        self.step = step
        self.checkpoint = checkpoint
        self.write_batch_outcomes = write_batch_outcomes
        self.workers = workers
        self.send = send

    def select_pending(self, batches: list) -> list:
        return [batch for batch in batches if not self.checkpoint.is_done(batch.batch_id)]

    def _answer_one(self, batch):
        """Worker: build, send, interpret. Returns (outcomes, seconds)."""
        step = self.step
        counted_places = step.policy.errors_counted_as_thrown_calls
        seconds = 0.0
        stage_of_failure = WhereAnErrorIsAThrownCall.BUILD
        try:
            request = step.build_request(batch)
            started = time.monotonic()
            stage_of_failure = WhereAnErrorIsAThrownCall.SEND
            reply = self.send(request)
            stage_of_failure = WhereAnErrorIsAThrownCall.INTERPRET
            outcomes = step.interpret(batch, reply)
            seconds = time.monotonic() - started
        except Exception as error:
            # One failing call must not discard its siblings' in-flight work.
            if stage_of_failure not in counted_places:
                raise
            failure = failure_for_exception(error)
            outcomes = {document_id: failure for document_id in batch.document_ids}
        return outcomes, seconds

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
